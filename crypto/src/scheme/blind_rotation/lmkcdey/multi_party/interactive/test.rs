use crate::{
    core::{
        lwe::{test::LweParam, LweCiphertext, LwePlaintext, LweSecretKey, LweSecretKeyOwned},
        rgsw::{test::RgswParam, RgswDecompositionParam},
        rlwe::{
            RlweCiphertext, RlwePlaintext, RlwePublicKey, RlwePublicKeyOwned, RlweSecretKey,
            SeededRlwePublicKey,
        },
    },
    scheme::blind_rotation::lmkcdey::{
        self,
        interactive::{
            self, LmkcdeyInteractiveCrs, LmkcdeyInteractiveKeyShare, LmkcdeyInteractiveParam,
        },
        test::nand_lut,
        LmkcdeyKeyOwned, LmkcdeyParam,
    },
    util::rng::StdLweRng,
};
use core::{array::from_fn, iter::repeat_with};
use itertools::{izip, Itertools};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::{DistributionVariance, Gaussian, Sampler, Ternary},
    izip_eq,
    modulus::{Modulus, ModulusOps, Native, NonNativePowerOfTwo, Prime},
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
    util::{dev::Stats, scratch::ScratchOwned},
};
use rand::{rngs::StdRng, thread_rng, RngCore, SeedableRng};

fn test_param(modulus: impl Into<Modulus>) -> LmkcdeyInteractiveParam {
    let ring_size = 2048;
    LmkcdeyInteractiveParam {
        param: LmkcdeyParam {
            modulus: modulus.into(),
            ring_size,
            sk_distribution: Ternary.into(),
            noise_distribution: Gaussian(3.19).into(),
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
        u_distribution: Ternary.into(),
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

        let big_q_by_8 = ring.elem_from(ring.modulus().as_f64() / 8f64);
        let nand_lut = nand_lut(ring, param.q, param.g);
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
            *ct.b_mut() = ring.add(ct.b(), &big_q_by_8);
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
    fn run<R: RingOps>(modulus: impl Into<Modulus>) {
        let param = test_param(modulus);
        let crs = LmkcdeyInteractiveCrs::sample(thread_rng());
        let rgsw = RgswParam::from(*param).build::<R>();
        let rlwe = rgsw.rlwe();
        let lwe_ks = LweParam::from(*param).build::<NonNativePowerOfTwoRing>();
        let ring = rgsw.ring();
        let mod_ks = lwe_ks.modulus();

        let (sk, sk_ks, _, bs_key) = bs_key_gen::<R>(param, crs, thread_rng());
        let sk = RlweSecretKey::from(sk);
        let embedding_factor = param.embedding_factor() as i64;
        let mut noise_ks_key = Stats::default();
        let mut noise_ak = Stats::default();
        let mut noise_brk = Stats::default();
        noise_ks_key.extend(
            lwe_ks
                .ks_key_noise(&sk.clone().into(), &sk_ks, &bs_key.ks_key().cloned())
                .into_iter()
                .flatten(),
        );
        bs_key.aks().for_each(|ak| {
            noise_ak.extend(rlwe.noise_auto_key(&sk, &ak.cloned()).into_iter().flatten())
        });
        izip_eq!(sk_ks.as_ref(), bs_key.brks()).for_each(|(sk_ks_i, brk_i)| {
            let brk_i = brk_i.cloned();
            let mut pt = RlwePlaintext::allocate(ring.ring_size());
            ring.poly_set_monomial(pt.as_mut(), embedding_factor * *sk_ks_i as i64);
            noise_brk.extend(rgsw.noise(&sk, &pt, &brk_i).into_iter().flatten());
        });

        let total_shares = param.total_shares as f64;
        let ring_size = param.ring_size as f64;
        let var_noise = param.noise_distribution.variance();
        let var_u = param.u_distribution.variance();
        let var_sk = total_shares * param.sk_distribution.variance();
        let var_noise_pk = total_shares * var_noise;
        let var_noise_pk_ct = ring_size * (var_u * var_noise_pk + var_sk * var_noise) + var_noise;
        let var_noise_ks_key = total_shares * var_noise;
        let var_noise_ak = total_shares * var_noise;
        let var_noise_brk = {
            let level_a = param.rgsw_by_rgsw_decomposition_param.level_a;
            let level_b = param.rgsw_by_rgsw_decomposition_param.level_b;
            let log_base = param.rgsw_by_rgsw_decomposition_param.log_base;
            let log_ignored_a = param.modulus.bits().saturating_sub(log_base * level_a);
            let log_ignored_b = param.modulus.bits().saturating_sub(log_base * level_b);
            let var_uniform_base = (0..1 << log_base).variance();
            let var_uniform_ignored_a = (0..1u64 << log_ignored_a).variance();
            let var_uniform_ignored_b = (0..1u64 << log_ignored_b).variance();
            let var_noise_a = level_a as f64 * ring_size * var_uniform_base * var_noise_pk_ct;
            let var_noise_b = level_b as f64 * ring_size * var_uniform_base * var_noise_pk_ct;
            let var_noise_ignored_a = ring_size * var_uniform_ignored_a * var_sk;
            let var_noise_ignored_b = var_uniform_ignored_b;
            total_shares * (var_noise_a + var_noise_b + var_noise_ignored_a + var_noise_ignored_b)
        };
        assert_eq!(noise_ks_key.mean().round(), 0.0);
        assert_eq!(noise_ak.mean().round(), 0.0);
        assert!((noise_ks_key.log2_std_dev() - var_noise_ks_key.sqrt().log2()).abs() < 0.1);
        assert!((noise_ak.log2_std_dev() - var_noise_ak.sqrt().log2()).abs() < 0.1);
        assert!(noise_brk.log2_std_dev() < var_noise_brk.sqrt().log2());

        let mut noise_ct_ks = Stats::default();
        for _ in 0..1000 {
            let mut rng = StdLweRng::from_entropy();
            let mut ct = LweCiphertext::allocate(param.ring_size);
            mod_ks.sample_uniform_into(ct.a_mut(), &mut rng);
            let pt = mod_ks.neg(&mod_ks.slice_dot_elem_from(ct.a(), sk.as_ref()));
            let ct_ks = lwe_ks.key_switch(&bs_key.ks_key().cloned(), &ct);
            noise_ct_ks.push(lwe_ks.noise(&sk_ks, &LwePlaintext(pt), &ct_ks));
        }
        let var_noise_ct_ks = {
            let level = param.lwe_ks_decomposition_param.level;
            let log_base = param.lwe_ks_decomposition_param.log_base;
            let log_ignored = param.lwe_modulus.bits().saturating_sub(log_base * level);
            let var_uniform_base = (0..1u64 << log_base).variance();
            let var_uniform_ignored = (0..1u64 << log_ignored).variance();
            let var_noise = level as f64 * ring_size * var_uniform_base * var_noise_ks_key;
            let var_noise_ignored = ring_size * var_uniform_ignored * var_sk;
            var_noise + var_noise_ignored
        };
        assert!((noise_ct_ks.log2_std_dev() - var_noise_ct_ks.sqrt().log2()).abs() < 0.1);

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
        let var_noise_ct_auto = {
            let level = param.auto_decomposition_param.level;
            let log_base = param.auto_decomposition_param.log_base;
            let log_ignored = param.modulus.bits().saturating_sub(log_base * level);
            let var_uniform_base = (0..1u64 << log_base).variance();
            let var_uniform_ignored = (0..1u64 << log_ignored).variance();
            let var_noise = level as f64 * ring_size * var_uniform_base * var_noise_ak;
            let var_noise_ignored = ring_size * var_uniform_ignored * var_sk;
            var_noise + var_noise_ignored
        };
        assert!((noise_ct_auto.log2_std_dev() - var_noise_ct_auto.sqrt().log2()).abs() < 0.05);
    }

    run::<NativeRing>(Native::native());
    run::<NonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(54));
    run::<PrimeRing>(Prime::gen(54, 12));
}

#[cfg(feature = "serde")]
#[test]
fn serialize_deserialize() {
    use phantom_zone_math::util::serde::dev::assert_serde_eq;

    fn run<R: RingOps>(modulus: impl Into<Modulus>) {
        let mut rng = StdLweRng::from_entropy();
        let param = test_param(modulus);
        let crs = LmkcdeyInteractiveCrs::<StdRng>::sample(&mut rng);
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
