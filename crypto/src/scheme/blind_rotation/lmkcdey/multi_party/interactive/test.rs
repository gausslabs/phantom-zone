use crate::{
    core::{
        lwe::{test::LweParam, LweCiphertext, LwePlaintext, LweSecretKey, LweSecretKeyOwned},
        rgsw::{test::RgswParam, RgswDecompositionParam},
        rlwe::{
            test::RlweParam, RlweCiphertext, RlwePlaintext, RlwePublicKey, RlwePublicKeyOwned,
            RlweSecretKey, SeededRlwePublicKey,
        },
    },
    scheme::blind_rotation::lmkcdey::{
        self,
        interactive::{
            self, LmkcdeyInteractiveCrs, LmkcdeyInteractiveKeyShare, LmkcdeyInteractiveParam,
        },
        test::nand_nc_lut,
        LmkcdeyKeyOwned, LmkcdeyParam,
    },
    util::rng::StdLweRng,
};
use core::{array::from_fn, iter::repeat_with};
use itertools::{izip, Itertools};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::{Gaussian, Sampler, Ternary},
    izip_eq,
    modulus::{Modulus, ModulusOps, Native, NonNativePowerOfTwo, Prime},
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
    util::{dev::Stats, scratch::ScratchOwned},
};
use rand::{rngs::StdRng, thread_rng, RngCore, SeedableRng};

impl From<LmkcdeyInteractiveParam> for RgswParam {
    fn from(param: LmkcdeyInteractiveParam) -> Self {
        RgswParam {
            rlwe: RlweParam {
                message_modulus: 1 << param.message_bits,
                ciphertext_modulus: param.modulus,
                ring_size: param.ring_size,
                sk_distribution: param.sk_distribution,
                noise_distribution: param.noise_distribution,
                u_distribution: param.u_distribution,
                ks_decomposition_param: param.auto_decomposition_param,
            },
            decomposition_param: param.rlwe_by_rgsw_decomposition_param,
        }
    }
}

fn test_param(modulus: impl Into<Modulus>) -> LmkcdeyInteractiveParam {
    let ring_size = 2048;
    LmkcdeyInteractiveParam {
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

#[allow(clippy::type_complexity)]
fn bs_key_gen<R: RingOps>(
    param: LmkcdeyInteractiveParam,
    crs: LmkcdeyInteractiveCrs<StdRng>,
    mut rng: impl RngCore,
) -> (
    LweSecretKeyOwned<i32>,
    LweSecretKeyOwned<i32>,
    RlwePublicKeyOwned<R::Elem>,
    LmkcdeyKeyOwned<R::Elem, u64>,
) {
    let rgsw = RgswParam::from(param).build::<R>();
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
            let mut bs_key_share = LmkcdeyInteractiveKeyShare::allocate(param, crs, share_idx);
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
fn interactive() {
    fn run<R1: RingOps, R2: RingOps<Elem = R1::Elem>>(modulus: impl Into<Modulus>) {
        let param = test_param(modulus);
        let crs = LmkcdeyInteractiveCrs::sample(thread_rng());
        let rgsw = RgswParam::from(param).build::<R2>();
        let rlwe = rgsw.rlwe();
        let lwe = RgswParam::from(param).rlwe.to_lwe().build::<R2>();
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

        let encoded_half = ring.elem_from(ring.modulus().as_f64() / 8f64);
        let nand_nc_lut = nand_nc_lut(ring, param.q, param.g, encoded_half);
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
                &nand_nc_lut,
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

#[test]
fn bs_key_gen_determinism() {
    fn run<R: RingOps>(modulus: impl Into<Modulus>) {
        let param = test_param(modulus);
        let crs = LmkcdeyInteractiveCrs::sample(thread_rng());
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

#[test]
fn noise_stats() {
    struct NoiseStdDev {
        log2_ks_key: f64,
        log2_ak: f64,
        log2_brk: f64,
        log2_ct_ks: f64,
        log2_ct_auto: f64,
    }

    fn run<R: RingOps>(modulus: impl Into<Modulus>, noise_std_dev: NoiseStdDev) {
        let param = test_param(modulus);
        let crs = LmkcdeyInteractiveCrs::sample(thread_rng());
        let rgsw = RgswParam::from(param).build::<R>();
        let rlwe = rgsw.rlwe();
        let lwe_ks = LweParam::from(*param).build::<NonNativePowerOfTwoRing>();
        let ring = rgsw.ring();
        let mod_ks = lwe_ks.modulus();

        let (sk, sk_ks, _, bs_key) = bs_key_gen::<R>(param, crs, thread_rng());
        let sk = RlweSecretKey::from(sk);

        let mut noise_ks_key = Stats::default();
        noise_ks_key.extend(
            lwe_ks
                .ks_key_noise(&sk.clone().into(), &sk_ks, &bs_key.ks_key().cloned())
                .into_iter()
                .flatten(),
        );
        assert_eq!(noise_ks_key.mean().round(), 0.0);
        assert!((noise_ks_key.log2_std_dev() - noise_std_dev.log2_ks_key).abs() < 0.1);

        let mut noise_ak = Stats::default();
        bs_key.aks().for_each(|ak| {
            noise_ak.extend(rlwe.noise_auto_key(&sk, &ak.cloned()).into_iter().flatten())
        });
        assert_eq!(noise_ak.mean().round(), 0.0);
        assert!((noise_ak.log2_std_dev() - noise_std_dev.log2_ak).abs() < 0.1);

        let mut noise_brk = Stats::default();
        izip_eq!(sk_ks.as_ref(), bs_key.brks()).for_each(|(sk_ks_i, brk_i)| {
            let brk_i = brk_i.cloned();
            let exp = param.embedding_factor() as i64 * *sk_ks_i as i64;
            let mut pt = RlwePlaintext::allocate(ring.ring_size());
            ring.poly_set_monomial(pt.as_mut(), exp);
            noise_brk.extend(rgsw.noise(&sk, &pt, &brk_i).into_iter().flatten());
        });
        assert!((noise_brk.log2_std_dev() - noise_std_dev.log2_brk).abs() < 0.1);

        let mut noise_ct_ks = Stats::default();
        for _ in 0..1000 {
            let mut rng = StdLweRng::from_entropy();
            let mut ct = LweCiphertext::allocate(param.ring_size);
            mod_ks.sample_uniform_into(ct.a_mut(), &mut rng);
            let pt = mod_ks.neg(&mod_ks.slice_dot_elem_from(ct.a(), sk.as_ref()));
            let ct_ks = lwe_ks.key_switch(&bs_key.ks_key().cloned(), &ct);
            noise_ct_ks.push(lwe_ks.noise(&sk_ks, &LwePlaintext(pt), &ct_ks));
        }
        assert!((noise_ct_ks.log2_std_dev() - noise_std_dev.log2_ct_ks).abs() < 0.1);

        let mut noise_ct_auto = Stats::default();
        for _ in 0..1000 {
            let mut rng = StdLweRng::from_entropy();
            let mut scratch = ring.allocate_scratch(0, 2, 0);
            let mut pt = RlwePlaintext::allocate(param.ring_size);
            let mut ct = RlweCiphertext::allocate(param.ring_size);
            ring.sample_uniform_into(ct.a_mut(), &mut rng);
            ring.poly_mul_elem_from(pt.as_mut(), ct.a(), sk.as_ref(), scratch.borrow_mut());
            ring.slice_neg_assign(pt.as_mut());
            bs_key.aks().for_each(|ak| {
                let mut pt_auto = RlwePlaintext::allocate(param.ring_size);
                ring.poly_add_auto(pt_auto.as_mut(), pt.as_ref(), ak.auto_map());
                let ct_auto = rlwe.automorphism(&ak.cloned(), &ct);
                noise_ct_auto.extend(rlwe.noise(&sk, &pt_auto, &ct_auto));
            });
        }
        assert!((noise_ct_auto.log2_std_dev() - noise_std_dev.log2_ct_auto).abs() < 0.1);
    }

    run::<NativeRing>(
        Native::native(),
        NoiseStdDev {
            log2_ks_key: 2.673556423990145,
            log2_ak: 2.673556423990145,
            log2_brk: 27.500045507048082,
            log2_ct_ks: 9.273649333233724,
            log2_ct_auto: 44.20751875305761,
        },
    );
    run::<NonNativePowerOfTwoRing>(
        NonNativePowerOfTwo::new(54),
        NoiseStdDev {
            log2_ks_key: 2.673556423990145,
            log2_ak: 2.673556423990145,
            log2_brk: 20.53468701198804,
            log2_ct_ks: 9.273649333233724,
            log2_ct_auto: 34.211094105080484,
        },
    );
    run::<PrimeRing>(
        Prime::gen(54, 12),
        NoiseStdDev {
            log2_ks_key: 2.673556423990145,
            log2_ak: 2.673556423990145,
            log2_brk: 20.53468701198804,
            log2_ct_ks: 9.273649333233724,
            log2_ct_auto: 34.211094105080484,
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
        let crs = LmkcdeyInteractiveCrs::<StdRng>::sample(&mut rng);
        let rgsw = RgswParam::from(param).build::<R>();
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
            let mut bs_key_share = LmkcdeyInteractiveKeyShare::allocate(param, crs, 0);
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
