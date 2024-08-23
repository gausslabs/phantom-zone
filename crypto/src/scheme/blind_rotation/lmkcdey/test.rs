use crate::{
    core::{
        lwe::{test::LweParam, LweSecretKey, LweSecretKeyOwned},
        rgsw::{test::RgswParam, RgswDecompositionParam},
        rlwe::{
            test::RlweParam, RlwePlaintext, RlwePlaintextOwned, RlwePublicKey, RlwePublicKeyOwned,
            RlweSecretKey,
        },
    },
    scheme::blind_rotation::lmkcdey::{
        self, LmkcdeyInteractiveCrs, LmkcdeyInteractiveParam, LmkcdeyKey, LmkcdeyKeyShare,
        LmkcdeyParam,
    },
    util::rng::StdLweRng,
};
use core::{array::from_fn, iter::repeat_with};
use itertools::{izip, Itertools};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::{DistributionVariance, Gaussian, Ternary},
    izip_eq,
    modulus::{Modulus, Native, NonNativePowerOfTwo, Prime},
    poly::automorphism::AutomorphismMap,
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
    util::{dev::Stats, scratch::ScratchOwned},
};
use rand::{rngs::StdRng, thread_rng, RngCore, SeedableRng};

impl From<LmkcdeyParam> for RgswParam {
    fn from(param: LmkcdeyParam) -> Self {
        RgswParam {
            rlwe: RlweParam {
                message_modulus: 4,
                ciphertext_modulus: param.modulus,
                ring_size: param.ring_size,
                sk_distribution: param.sk_distribution,
                noise_distribution: param.noise_distribution,
                u_distribution: Ternary.into(),
                ks_decomposition_param: param.auto_decomposition_param,
            },
            decomposition_param: param.rlwe_by_rgsw_decomposition_param,
        }
    }
}

impl From<LmkcdeyParam> for LweParam {
    fn from(param: LmkcdeyParam) -> Self {
        LweParam {
            message_modulus: 4,
            ciphertext_modulus: param.lwe_modulus,
            dimension: param.lwe_dimension,
            sk_distribution: param.lwe_sk_distribution,
            noise_distribution: param.lwe_noise_distribution,
            ks_decomposition_param: param.lwe_ks_decomposition_param,
        }
    }
}

fn test_param(modulus: impl Into<Modulus>, embedding_factor: usize) -> LmkcdeyParam {
    let ring_size = 1024;
    LmkcdeyParam {
        modulus: modulus.into(),
        ring_size,
        sk_distribution: Gaussian(3.19).into(),
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
        lwe_dimension: 100,
        lwe_sk_distribution: Gaussian(3.19).into(),
        lwe_noise_distribution: Gaussian(3.19).into(),
        lwe_ks_decomposition_param: DecompositionParam {
            log_base: 1,
            level: 13,
        },
        q: 2 * ring_size / embedding_factor,
        g: 5,
        w: 10,
    }
}

fn nand_lut<R: RingOps>(ring: &R, q: usize, g: usize) -> RlwePlaintextOwned<R::Elem> {
    let big_q_by_8 = ring.elem_from(ring.modulus().as_f64() / 8f64);
    let auto_map = AutomorphismMap::new(q / 2, -(g as i64));
    let lut_value = [big_q_by_8, ring.neg(&big_q_by_8)];
    let log_q_by_8 = (q / 8).ilog2() as usize;
    let f = |(sign, idx)| lut_value[sign as usize ^ [0, 0, 0, 1][idx >> log_q_by_8]];
    RlwePlaintext::new(auto_map.iter().map(f).collect(), q / 2)
}

#[test]
fn bootstrap() {
    fn run<R: RingOps>(modulus: impl Into<Modulus>, embedding_factor: usize) {
        let mut rng = StdLweRng::from_entropy();
        let param = test_param(modulus, embedding_factor);
        let rgsw = RgswParam::from(param).build::<R>();
        let lwe = RgswParam::from(param).rlwe.to_lwe().build::<R>();
        let lwe_ks = LweParam::from(param).build::<NonNativePowerOfTwoRing>();
        let ring = rgsw.ring();
        let mod_ks = lwe_ks.modulus();

        let sk = rgsw.rlwe().sk_gen(&mut rng);
        let sk_ks = lwe_ks.sk_gen(&mut rng);
        let bs_key = {
            let mut scratch = ring.allocate_scratch(0, 3, 0);
            let mut bs_key = LmkcdeyKey::allocate(param);
            lmkcdey::bs_key_gen(
                ring,
                mod_ks,
                &mut bs_key,
                &sk,
                &sk_ks,
                scratch.borrow_mut(),
                &mut rng,
            );
            let mut bs_key_prep = LmkcdeyKey::allocate_eval(*bs_key.param(), ring.eval_size());
            lmkcdey::prepare_bs_key(ring, &mut bs_key_prep, &bs_key, scratch.borrow_mut());
            bs_key_prep
        };
        let sk = sk.into();

        let mut scratch = ScratchOwned::allocate(bs_key.param().scratch_bytes(ring, mod_ks));
        let big_q_by_8 = ring.elem_from(ring.modulus().as_f64() / 8f64);
        let nand_lut = nand_lut(ring, param.q, param.g);
        for m in 0..1 << 2 {
            let [a, b] = from_fn(|i| (m >> i) & 1 == 1);
            let ct_a = lwe.sk_encrypt(&sk, lwe.encode(a as _), &mut rng);
            let ct_b = lwe.sk_encrypt(&sk, lwe.encode(b as _), &mut rng);
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

    for embedding_factor in [1, 2] {
        run::<NoisyNativeRing>(Native::native(), embedding_factor);
        run::<NoisyNonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(50), embedding_factor);
        run::<NativeRing>(Native::native(), embedding_factor);
        run::<NonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(50), embedding_factor);
        run::<NoisyPrimeRing>(Prime::gen(50, 12), embedding_factor);
        run::<PrimeRing>(Prime::gen(50, 12), embedding_factor);
    }
}

fn test_interactive_param(modulus: impl Into<Modulus>) -> LmkcdeyInteractiveParam {
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
fn interactive_bs_key_gen<R: RingOps>(
    param: LmkcdeyInteractiveParam,
    crs: LmkcdeyInteractiveCrs<StdRng>,
    mut rng: impl RngCore,
) -> (
    LweSecretKeyOwned<i32>,
    LweSecretKeyOwned<i32>,
    RlwePublicKeyOwned<R::Elem>,
    LmkcdeyKey<R::Elem, u64>,
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
        .map(|(sk, rng)| rlwe.seeded_pk_gen(sk, &mut crs.pk_rng(rng)))
        .collect_vec();
    let pk = {
        let mut pk = RlwePublicKey::allocate(ring.ring_size());
        lmkcdey::aggregate_pk_shares(ring, &mut pk, &crs, &pk_shares);
        pk
    };
    let bs_key_shares = izip!(0.., &sk_shares, &sk_ks_shares, &mut rngs)
        .map(|(share_idx, sk, sk_ks, rng)| {
            let mut bs_key_share = LmkcdeyKeyShare::allocate(param, crs, share_idx);
            let mut scratch = ring.allocate_scratch(2, 3, 0);
            lmkcdey::bs_key_share_gen(
                ring,
                mod_ks,
                &mut bs_key_share,
                sk,
                &pk,
                sk_ks,
                scratch.borrow_mut(),
                rng,
            );
            bs_key_share
        })
        .collect_vec();
    let bs_key = {
        let mut bs_key = LmkcdeyKey::allocate(*param);
        let mut scratch = ring.allocate_scratch(
            2,
            3,
            2 * (param.rgsw_by_rgsw_decomposition_param.level_a
                + param.rgsw_by_rgsw_decomposition_param.level_b),
        );
        lmkcdey::aggregate_bs_key_shares(
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
        .reduce(|mut sk, sk_share| {
            izip_eq!(sk.as_mut(), sk_share.as_ref()).for_each(|(a, b)| *a += b);
            sk
        })
        .unwrap();
    (sk, sk_ks, pk, bs_key)
}

#[test]
fn interactive() {
    fn run<R1: RingOps, R2: RingOps<Elem = R1::Elem>>(modulus: impl Into<Modulus>) {
        let param = test_interactive_param(modulus);
        let crs = LmkcdeyInteractiveCrs::sample(thread_rng());
        let rgsw = RgswParam::from(*param).build::<R2>();
        let rlwe = rgsw.rlwe();
        let lwe = RgswParam::from(*param).rlwe.to_lwe().build::<R2>();
        let lwe_ks = LweParam::from(*param).build::<NonNativePowerOfTwoRing>();
        let ring = rgsw.ring();
        let mod_ks = lwe_ks.modulus();

        let (sk, _, pk, bs_key) = interactive_bs_key_gen::<R1>(param, crs, thread_rng());
        let bs_key = {
            let mut scratch = ring.allocate_scratch(0, 3, 0);
            let mut bs_key_prep = LmkcdeyKey::allocate_eval(*bs_key.param(), ring.eval_size());
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
fn interactive_key_gen_determinism() {
    fn run<R: RingOps>(modulus: impl Into<Modulus>) {
        let param = test_interactive_param(modulus);
        let crs = LmkcdeyInteractiveCrs::sample(thread_rng());
        let rng = StdRng::from_entropy();
        assert_eq!(
            interactive_bs_key_gen::<R>(param, crs, rng.clone()),
            interactive_bs_key_gen::<R>(param, crs, rng.clone()),
        );
    }

    run::<PrimeRing>(Prime::gen(54, 12));
}

#[test]
fn interactive_key_gen_stats() {
    fn run<R: RingOps>(modulus: impl Into<Modulus>) {
        let param = test_interactive_param(modulus);
        let crs = LmkcdeyInteractiveCrs::sample(thread_rng());
        let rgsw = RgswParam::from(*param).build::<R>();
        let ring = rgsw.ring();

        let (sk, sk_ks, _, bs_key) = interactive_bs_key_gen::<R>(param, crs, thread_rng());
        let sk = RlweSecretKey::from(sk);
        let embedding_factor = param.embedding_factor() as i64;
        let mut noise = Stats::default();
        izip_eq!(sk_ks.as_ref(), bs_key.brks()).for_each(|(sk_ks_i, brk_i)| {
            let brk_i = brk_i.cloned();
            let mut pt = RlwePlaintext::allocate(ring.ring_size());
            ring.poly_set_monomial(pt.as_mut(), embedding_factor * *sk_ks_i as i64);
            noise.extend(rgsw.noise(&sk, &pt, &brk_i).into_iter().flatten());
        });

        let total_shares = param.total_shares as f64;
        let ring_size = param.ring_size as f64;
        let var_sk = total_shares * param.sk_distribution.variance();
        let var_noise = param.noise_distribution.variance();
        let var_noise_ct_pk =
            ring_size * var_sk * var_noise + (ring_size * var_sk + 1.0) * var_noise;
        let var_noise_brk = {
            let log_base = param.rgsw_by_rgsw_decomposition_param.log_base;
            let level_a = param.rgsw_by_rgsw_decomposition_param.level_a;
            let level_b = param.rgsw_by_rgsw_decomposition_param.level_b;
            let ignored_a = 1u64 << param.modulus.bits().saturating_sub(log_base * level_a);
            let ignored_b = 1u64 << param.modulus.bits().saturating_sub(log_base * level_b);
            let base = 1 << log_base;
            let var_noise_a = ((level_a * base * base) as f64 * ring_size * var_noise_ct_pk) / 12.0;
            let var_noise_b = ((level_b * base * base) as f64 * ring_size * var_noise_ct_pk) / 12.0;
            let var_ignored_a = ((ignored_a * ignored_a) as f64 * ring_size * var_sk) / 12.0;
            let var_ignored_b = ((ignored_b * ignored_b) as f64) / 12.0;
            total_shares * (var_noise_a + var_noise_b + var_ignored_a + var_ignored_b)
        };
        assert!(noise.log2_std_dev() < var_noise_brk.sqrt().log2());
    }

    run::<PrimeRing>(Prime::gen(54, 12));
}
