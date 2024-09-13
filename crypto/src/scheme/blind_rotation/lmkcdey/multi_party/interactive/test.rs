use crate::{
    core::{
        lwe::{test::LweParam, LweCiphertext, LwePlaintext, LweSecretKey, LweSecretKeyOwned},
        rgsw::{test::RgswParam, RgswDecompositionParam},
        rlwe::{
            RlweCiphertext, RlwePlaintext, RlwePlaintextOwned, RlwePublicKey, RlwePublicKeyOwned,
            RlweSecretKey, SeededRlwePublicKey,
        },
    },
    scheme::blind_rotation::lmkcdey::{
        self,
        interactive::{self, LmkcdeyMpiCrs, LmkcdeyMpiKeyShare, LmkcdeyMpiParam},
        test::nand_lut,
        LmkcdeyKeyOwned, LmkcdeyParam,
    },
    util::{
        distribution::{NoiseDistribution, SecretDistribution},
        rng::StdLweRng,
    },
};
use core::{
    array::from_fn,
    iter::{repeat, repeat_with},
};
use itertools::{izip, Itertools};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::{DistributionVariance, Gaussian, Sampler, Ternary},
    izip_eq,
    modulus::{Modulus, ModulusOps, Native, NonNativePowerOfTwo, Prime},
    poly::automorphism::AutomorphismMap,
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
    util::{
        dev::{time_consuming_test_repetition, Stats, StatsSampler},
        scratch::ScratchOwned,
    },
};
use rand::{rngs::StdRng, seq::SliceRandom, thread_rng, RngCore, SeedableRng};

fn test_param(modulus: impl Into<Modulus>) -> LmkcdeyMpiParam {
    let ring_size = 2048;
    LmkcdeyMpiParam {
        param: LmkcdeyParam {
            message_bits: 2,
            modulus: modulus.into(),
            ring_size,
            sk_distribution: Ternary.into(),
            noise_distribution: Gaussian(3.19).into(),
            u_distribution: Ternary.into(),
            auto_decomposition_param: DecompositionParam {
                log_base: 24,
                level: 1,
            },
            rlwe_by_rgsw_decomposition_param: RgswDecompositionParam {
                log_base: 17,
                level_a: 1,
                level_b: 1,
            },
            lwe_modulus: NonNativePowerOfTwo::new(16).into(),
            lwe_dimension: 620,
            lwe_sk_distribution: Ternary.into(),
            lwe_noise_distribution: Gaussian(3.19).into(),
            lwe_ks_decomposition_param: DecompositionParam {
                log_base: 1,
                level: 13,
            },
            q: 2 * ring_size,
            g: 5,
            w: 10,
        },
        rgsw_by_rgsw_decomposition_param: RgswDecompositionParam {
            log_base: 6,
            level_a: 7,
            level_b: 6,
        },
        total_shares: 4,
    }
}

const P4_MSG1_LUT3_PADD1_128: LmkcdeyMpiParam = LmkcdeyMpiParam {
    param: LmkcdeyParam {
        message_bits: 4,
        modulus: Modulus::PowerOfTwo(64),
        ring_size: 1 << 13,
        sk_distribution: SecretDistribution::Ternary(Ternary),
        noise_distribution: NoiseDistribution::Gaussian(Gaussian(3.19)),
        u_distribution: SecretDistribution::Ternary(Ternary),
        auto_decomposition_param: DecompositionParam {
            log_base: 17,
            level: 2,
        },
        rlwe_by_rgsw_decomposition_param: RgswDecompositionParam {
            log_base: 17,
            level_a: 2,
            level_b: 1,
        },
        lwe_modulus: Modulus::PowerOfTwo(25),
        lwe_dimension: 925,
        lwe_sk_distribution: SecretDistribution::Ternary(Ternary),
        lwe_noise_distribution: NoiseDistribution::Gaussian(Gaussian(5.1)),
        lwe_ks_decomposition_param: DecompositionParam {
            log_base: 9,
            level: 2,
        },
        q: 1 << 14,
        g: 5,
        w: 20,
    },
    rgsw_by_rgsw_decomposition_param: RgswDecompositionParam {
        log_base: 9,
        level_a: 5,
        level_b: 4,
    },
    total_shares: 4,
};

#[allow(clippy::type_complexity)]
fn bs_key_gen<R: RingOps>(
    param: LmkcdeyMpiParam,
    crs: LmkcdeyMpiCrs<StdRng>,
    mut rng: impl RngCore,
) -> (
    LweSecretKeyOwned<i32>,
    LweSecretKeyOwned<i32>,
    RlwePublicKeyOwned<R::Elem>,
    LmkcdeyKeyOwned<R::Elem, u64>,
) {
    let rgsw = RgswParam::from(*param).build::<R>();
    let rlwe = rgsw.rlwe();
    let lwe_ks = LweParam::from(*param).build::<NonNativePowerOfTwoRing>();
    let ring = rgsw.ring();
    let mod_ks = lwe_ks.modulus();

    let mut rngs = repeat_with(|| StdRng::from_rng(&mut rng).unwrap())
        .take(param.total_shares)
        .collect_vec();
    let sk_shares = rngs.iter_mut().map(|rng| rlwe.sk_gen(rng)).collect_vec();
    let sk_ks_shares = rngs.iter_mut().map(|rng| lwe_ks.sk_gen(rng)).collect_vec();
    let pk_shares = izip!(&sk_shares, &mut rngs)
        .map(|(sk_share, rng)| {
            let mut pk_share = SeededRlwePublicKey::allocate(param.ring_size);
            let mut scratch = ring.allocate_scratch(2, 2, 0);
            interactive::pk_share_gen(
                ring,
                &mut pk_share,
                &param,
                &crs,
                sk_share,
                scratch.borrow_mut(),
                rng,
            );
            pk_share
        })
        .collect_vec();
    let pk = {
        let mut pk = RlwePublicKey::allocate(ring.ring_size());
        interactive::aggregate_pk_shares(ring, &mut pk, &crs, &pk_shares);
        pk
    };
    let bs_key_shares = izip!(0.., &sk_shares, &sk_ks_shares, &mut rngs)
        .map(|(share_idx, sk_share, sk_ks_share, rng)| {
            let mut bs_key_share = LmkcdeyMpiKeyShare::allocate(param, crs, share_idx);
            let mut scratch = ring.allocate_scratch(2, 3, 0);
            interactive::bs_key_share_gen(
                ring,
                mod_ks,
                &mut bs_key_share,
                sk_share,
                &pk,
                sk_ks_share,
                scratch.borrow_mut(),
                rng,
            );
            bs_key_share
        })
        .collect_vec();
    let bs_key = {
        let mut bs_key = LmkcdeyKeyOwned::allocate(*param);
        let mut scratch = ring.allocate_scratch(
            2,
            3,
            2 * (param.rgsw_by_rgsw_decomposition_param.level_a
                + param.rgsw_by_rgsw_decomposition_param.level_b),
        );
        interactive::aggregate_bs_key_shares(
            ring,
            mod_ks,
            &mut bs_key,
            &crs,
            &bs_key_shares,
            scratch.borrow_mut(),
        );
        bs_key
    };
    let sk = sk_shares
        .into_iter()
        .map(LweSecretKey::from)
        .reduce(|mut sk, sk_share| {
            izip_eq!(sk.as_mut(), sk_share.as_ref()).for_each(|(a, b)| *a += b);
            sk
        })
        .unwrap();
    let sk_ks = sk_ks_shares
        .into_iter()
        .reduce(|mut sk_ks, sk_ks_share| {
            izip_eq!(sk_ks.as_mut(), sk_ks_share.as_ref()).for_each(|(a, b)| *a += b);
            sk_ks
        })
        .unwrap();
    (sk, sk_ks, pk, bs_key)
}

#[test]
fn bootstrap_nand() {
    fn run<R1: RingOps, R2: RingOps<Elem = R1::Elem>>(modulus: impl Into<Modulus>) {
        let param = test_param(modulus);
        let crs = LmkcdeyMpiCrs::sample(thread_rng());
        let rgsw = RgswParam::from(*param).build::<R2>();
        let rlwe = rgsw.rlwe();
        let lwe = RgswParam::from(*param).rlwe.to_lwe().build::<R2>();
        let lwe_ks = LweParam::from(*param).build::<NonNativePowerOfTwoRing>();
        let ring = rgsw.ring();
        let mod_ks = lwe_ks.modulus();

        let (sk, _, pk, bs_key) = bs_key_gen::<R1>(param, crs, thread_rng());
        let bs_key = {
            let mut scratch = ring.allocate_scratch(0, 3, 0);
            let mut bs_key_prep = LmkcdeyKeyOwned::allocate_eval(*bs_key.param(), ring.eval_size());
            lmkcdey::prepare_bs_key(ring, &mut bs_key_prep, &bs_key, scratch.borrow_mut());
            bs_key_prep
        };

        let encoded_half = ring.elem_from(param.encoded_half());
        let nand_lut = nand_lut(&param, ring);
        let mut scratch = ScratchOwned::allocate(bs_key.param().scratch_bytes(ring, mod_ks));
        let mut rng = StdLweRng::from_entropy();
        for m in 0..1 << 2 {
            let [a, b] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct_a, ct_b] = [a, b].map(|m| {
                let pt = rlwe.encode(&vec![m as u64; ring.ring_size()]);
                rlwe.sample_extract(&rlwe.pk_encrypt(&pk, &pt, &mut rng), 0)
            });
            let mut ct = lwe.add(&ct_a, &ct_b);
            lmkcdey::bootstrap(
                ring,
                mod_ks,
                &mut ct,
                &bs_key,
                &nand_lut,
                scratch.borrow_mut(),
            );
            *ct.b_mut() = ring.add(ct.b(), &encoded_half);
            assert_eq!(!(a & b) as u64, lwe.decode(lwe.decrypt(&sk, &ct)));
        }
    }

    run::<NativeRing, NoisyNativeRing>(Native::native());
    run::<NonNativePowerOfTwoRing, NoisyNonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(54));
    run::<PrimeRing, NoisyPrimeRing>(Prime::gen(54, 12));
}

fn general_lut<R: RingOps, const N: usize>(
    param: &LmkcdeyParam,
    ring: &R,
    table: [u64; N],
) -> RlwePlaintextOwned<R::Elem> {
    debug_assert_eq!(param.q % N, 0);
    let lut = {
        let repetition = param.q / 2 / N;
        let encoded = [ring.zero(), ring.elem_from(param.encoded_one())];
        let mut lut = table
            .into_iter()
            .flat_map(|v| repeat(encoded[v as usize]).take(repetition))
            .collect_vec();
        ring.poly_mul_assign_monomial(&mut lut, -(repetition as i64 / 2));
        AutomorphismMap::new(param.q / 2, param.q - param.g)
            .apply(&lut, |v| ring.neg(v))
            .collect()
    };
    RlwePlaintext::new(lut, param.q / 2)
}

#[test]
fn bootstrap_three_way() {
    fn run<R1: RingOps, R2: RingOps<Elem = R1::Elem>>(param: LmkcdeyMpiParam) {
        let crs = LmkcdeyMpiCrs::sample(thread_rng());
        let rgsw = RgswParam::from(*param).build::<R2>();
        let rlwe = rgsw.rlwe();
        let lwe = RgswParam::from(*param).rlwe.to_lwe().build::<R2>();
        let lwe_ks = LweParam::from(*param).build::<NonNativePowerOfTwoRing>();
        let ring = rgsw.ring();
        let mod_ks = lwe_ks.modulus();

        let (sk, _, pk, bs_key) = bs_key_gen::<R1>(param, crs, thread_rng());
        let bs_key = {
            let mut scratch = ring.allocate_scratch(0, 3, 0);
            let mut bs_key_prep = LmkcdeyKeyOwned::allocate_eval(*bs_key.param(), ring.eval_size());
            lmkcdey::prepare_bs_key(ring, &mut bs_key_prep, &bs_key, scratch.borrow_mut());
            bs_key_prep
        };

        let table = {
            let mut table = [1, 1, 1, 1, 0, 0, 0, 0];
            table[1..7].shuffle(&mut thread_rng());
            table
        };
        let general_lut = general_lut(&param, ring, table);
        let scalar_fma = |a: &[_], b: &[_]| izip_eq!(a, b).map(|(a, b)| *a * *b).sum::<u64>();
        let mut scratch = ScratchOwned::allocate(bs_key.param().scratch_bytes(ring, mod_ks));
        let mut rng = StdLweRng::from_entropy();
        let [mut a, mut b, mut c] = [1, 1, 0];
        let [mut ct_a, mut ct_b, mut ct_c] = [a, b, c].map(|m| {
            let pt = rlwe.encode(&vec![m; ring.ring_size()]);
            rlwe.sample_extract(&rlwe.pk_encrypt(&pk, &pt, &mut rng), 0)
        });
        for _ in 0..time_consuming_test_repetition() {
            [(a, ct_a), (b, ct_b), (c, ct_c)] = [[1, 2, 4], [2, 4, 1], [4, 1, 2]].map(|scalars| {
                let m = scalar_fma(&[a, b, c], &scalars) as usize;
                let mut ct = lwe.scalar_fma([&ct_a, &ct_b, &ct_c], scalars);
                lmkcdey::bootstrap(
                    ring,
                    mod_ks,
                    &mut ct,
                    &bs_key,
                    &general_lut,
                    scratch.borrow_mut(),
                );
                assert_eq!(table[m], lwe.decode(lwe.decrypt(&sk, &ct)));
                (table[m], ct)
            });
        }
    }

    run::<NativeRing, NoisyNativeRing>(P4_MSG1_LUT3_PADD1_128);
}

#[test]
fn bs_key_gen_determinism() {
    fn run<R: RingOps>(modulus: impl Into<Modulus>) {
        let param = test_param(modulus);
        let crs = LmkcdeyMpiCrs::sample(thread_rng());
        let rng = StdRng::from_entropy();
        assert_eq!(
            bs_key_gen::<R>(param, crs, rng.clone()),
            bs_key_gen::<R>(param, crs, rng.clone()),
        );
    }

    run::<NativeRing>(Native::native());
    run::<NonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(54));
    run::<PrimeRing>(Prime::gen(54, 12));
}

struct NoiseStdDev {
    log2_brk: f64,
    log2_ct_ks: f64,
    log2_ct_auto: f64,
    log2_ct_rlwe_by_rgsw: f64,
}

fn run_noise_stats<R: RingOps>(param: LmkcdeyMpiParam, noise_std_dev: NoiseStdDev) {
    let crs = LmkcdeyMpiCrs::sample(thread_rng());
    let rgsw = &RgswParam::from(*param).build::<R>();
    let rlwe = rgsw.rlwe();
    let lwe_ks = LweParam::from(*param).build::<NonNativePowerOfTwoRing>();
    let ring = rgsw.ring();
    let mod_ks = lwe_ks.modulus();

    let (sk, sk_ks, _, bs_key) = &bs_key_gen::<R>(param, crs, thread_rng());
    let sk = &RlweSecretKey::from(sk.cloned());

    let noise_var_ks_key = param.total_shares as f64 * param.lwe_noise_distribution.variance();
    let noise_ks_key = lwe_ks
        .ks_key_noise(&sk.clone().into(), sk_ks, &bs_key.ks_key().cloned())
        .into_iter()
        .flatten()
        .collect::<Stats<_>>();
    assert_eq!(noise_ks_key.mean().round(), 0.0);
    assert!((noise_ks_key.log2_std_dev() - noise_var_ks_key.sqrt().log2()).abs() < 0.1);

    let noise_var_ak = param.total_shares as f64 * param.noise_distribution.variance();
    let noise_ak = bs_key
        .aks()
        .flat_map(|ak| rlwe.noise_auto_key(sk, &ak.cloned()))
        .flatten()
        .collect::<Stats<_>>();
    assert_eq!(noise_ak.mean().round(), 0.0);
    assert!((noise_ak.log2_std_dev() - noise_var_ak.sqrt().log2()).abs() < 0.1);

    let noise_brk = izip_eq!(sk_ks.as_ref(), bs_key.brks())
        .flat_map(|(sk_ks_i, brk_i)| {
            let exp = param.embedding_factor() as i64 * *sk_ks_i as i64;
            let mut pt = RlwePlaintext::allocate(ring.ring_size());
            ring.poly_set_monomial(pt.as_mut(), exp);
            rgsw.noise(sk, &pt, &brk_i.cloned())
        })
        .flatten()
        .collect::<Stats<_>>();
    assert!((noise_brk.log2_std_dev() - noise_std_dev.log2_brk).abs() < 0.1);

    let noise_ct_ks = StatsSampler::default()
        .sample_size(1000)
        .without_timeout()
        .sample(|rng| {
            let mut ct = LweCiphertext::allocate(param.ring_size);
            mod_ks.sample_uniform_into(ct.a_mut(), rng);
            let pt = mod_ks.neg(&mod_ks.slice_dot_elem_from(ct.a(), sk.as_ref()));
            let ct_ks = lwe_ks.key_switch(&bs_key.ks_key().cloned(), &ct);
            [lwe_ks.noise(sk_ks, &LwePlaintext(pt), &ct_ks)]
        });
    assert!((noise_ct_ks.log2_std_dev() - noise_std_dev.log2_ct_ks).abs() < 0.1);

    let noise_ct_auto = StatsSampler::default().sample(|rng| {
        let mut scratch = ring.allocate_scratch(0, 2, 0);
        let mut pt = RlwePlaintext::allocate(param.ring_size);
        let mut ct = RlweCiphertext::allocate(param.ring_size);
        ring.sample_uniform_into(ct.a_mut(), rng);
        ring.poly_mul_elem_from(pt.as_mut(), ct.a(), sk.as_ref(), scratch.borrow_mut());
        ring.slice_neg_assign(pt.as_mut());
        bs_key.aks().flat_map(move |ak| {
            let mut pt_auto = RlwePlaintext::allocate(param.ring_size);
            ring.poly_add_auto(pt_auto.as_mut(), pt.as_ref(), ak.auto_map());
            let ct_auto = rlwe.automorphism(&ak.cloned(), &ct);
            rlwe.noise(sk, &pt_auto, &ct_auto)
        })
    });
    assert!((noise_ct_auto.log2_std_dev() - noise_std_dev.log2_ct_auto).abs() < 0.1);

    let noise_ct_rlwe_by_rgsw = StatsSampler::default().sample(|rng| {
        let mut scratch = ring.allocate_scratch(0, 2, 0);
        let mut pt = RlwePlaintext::allocate(param.ring_size);
        let mut ct = RlweCiphertext::allocate(param.ring_size);
        ring.sample_uniform_into(ct.a_mut(), rng);
        ring.poly_mul_elem_from(pt.as_mut(), ct.a(), sk.as_ref(), scratch.borrow_mut());
        ring.slice_neg_assign(pt.as_mut());
        izip_eq!(sk_ks.as_ref(), bs_key.brks()).flat_map(move |(sk_ks_i, brk_i)| {
            let mut pt_by_brk_i = pt.clone();
            let exp = param.embedding_factor() as i64 * *sk_ks_i as i64;
            ring.poly_mul_assign_monomial(pt_by_brk_i.as_mut(), exp);
            let ct_by_brk_i = rgsw.rlwe_by_rgsw(&ct, &brk_i.cloned());
            rlwe.noise(sk, &pt_by_brk_i, &ct_by_brk_i)
        })
    });
    // TODO: The sampled std dev is always slightly smaller than the one sage says.
    assert!(
        (noise_ct_rlwe_by_rgsw.log2_std_dev() - noise_std_dev.log2_ct_rlwe_by_rgsw).abs() < 0.5
    );
}

#[test]
fn noise_stats_test_param_native() {
    run_noise_stats::<NativeRing>(
        test_param(Native::native()),
        NoiseStdDev {
            log2_brk: 27.500045507048082,
            log2_ct_ks: 9.273649333233724,
            log2_ct_auto: 44.20751875305761,
            log2_ct_rlwe_by_rgsw: 51.18317420521815,
        },
    );
}

#[test]
fn noise_stats_test_param_non_native_power_of_two() {
    run_noise_stats::<NonNativePowerOfTwoRing>(
        test_param(NonNativePowerOfTwo::new(54)),
        NoiseStdDev {
            log2_brk: 20.53468701198804,
            log2_ct_ks: 9.273649333233724,
            log2_ct_auto: 34.211094105080484,
            log2_ct_rlwe_by_rgsw: 41.75913415851187,
        },
    );
}

#[test]
fn noise_stats_test_param_prime() {
    run_noise_stats::<PrimeRing>(
        test_param(Prime::gen(54, 12)),
        NoiseStdDev {
            log2_brk: 20.53468701198804,
            log2_ct_ks: 9.273649333233724,
            log2_ct_auto: 34.211094105080484,
            log2_ct_rlwe_by_rgsw: 41.73815728744025,
        },
    );
}

#[test]
fn noise_stats_p4_msg1_lut3_padd1_128() {
    run_noise_stats::<NativeRing>(
        P4_MSG1_LUT3_PADD1_128,
        NoiseStdDev {
            log2_brk: 27.1020334812903,
            log2_ct_ks: 17.5584492020649,
            log2_ct_auto: 35.2075187926351,
            log2_ct_rlwe_by_rgsw: 49.6036623760187,
        },
    );
}

#[cfg(feature = "serde")]
#[test]
fn serialize_deserialize() {
    use phantom_zone_math::util::serde::dev::assert_serde_eq;

    fn run<R: RingOps>(modulus: impl Into<Modulus>) {
        let mut rng = StdLweRng::from_entropy();
        let param = test_param(modulus);
        let crs = LmkcdeyMpiCrs::<StdRng>::sample(&mut rng);
        let rgsw = RgswParam::from(*param).build::<R>();
        let rlwe = rgsw.rlwe();
        let lwe_ks = LweParam::from(*param).build::<NonNativePowerOfTwoRing>();
        let ring = rgsw.ring();
        let mod_ks = lwe_ks.modulus();

        let sk_share = rlwe.sk_gen(&mut rng);
        let sk_ks_share = lwe_ks.sk_gen(&mut rng);
        let pk_share = {
            let mut pk_share = SeededRlwePublicKey::allocate(param.ring_size);
            let mut scratch = ring.allocate_scratch(2, 2, 0);
            interactive::pk_share_gen(
                ring,
                &mut pk_share,
                &param,
                &crs,
                &sk_share,
                scratch.borrow_mut(),
                &mut rng,
            );
            pk_share
        };
        let pk = {
            let mut pk = RlwePublicKey::allocate(ring.ring_size());
            interactive::aggregate_pk_shares(ring, &mut pk, &crs, &[pk_share.clone()]);
            pk
        };
        let bs_key_share = {
            let mut bs_key_share = LmkcdeyMpiKeyShare::allocate(param, crs, 0);
            let mut scratch = ring.allocate_scratch(2, 3, 0);
            interactive::bs_key_share_gen(
                ring,
                mod_ks,
                &mut bs_key_share,
                &sk_share,
                &pk,
                &sk_ks_share,
                scratch.borrow_mut(),
                &mut rng,
            );
            bs_key_share
        };
        assert_serde_eq(&crs);
        assert_serde_eq(&pk_share);
        assert_serde_eq(&pk);
        assert_serde_eq(&bs_key_share);
        assert_serde_eq(&bs_key_share.compact(ring, mod_ks));
        assert_eq!(
            &bs_key_share,
            &bs_key_share.compact(ring, mod_ks).uncompact(ring, mod_ks)
        );
    }

    run::<NativeRing>(Native::native());
    run::<NonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(54));
    run::<PrimeRing>(Prime::gen(54, 12));
}
