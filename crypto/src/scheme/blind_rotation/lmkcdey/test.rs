use crate::{
    core::{
        lwe::test::LweParam,
        rgsw::{test::RgswParam, RgswDecompositionParam},
        rlwe::{test::RlweParam, RlwePlaintext, RlwePlaintextOwned},
    },
    scheme::blind_rotation::lmkcdey::{self, LmkcdeyKey, LmkcdeyParam},
    util::rng::StdLweRng,
};
use core::array::from_fn;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::{Gaussian, Ternary},
    modulus::{Modulus, Native, NonNativePowerOfTwo, Prime},
    poly::automorphism::AutomorphismMap,
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
    util::scratch::ScratchOwned,
};
use rand::SeedableRng;

impl From<LmkcdeyParam> for RgswParam {
    fn from(param: LmkcdeyParam) -> Self {
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

impl From<LmkcdeyParam> for LweParam {
    fn from(param: LmkcdeyParam) -> Self {
        LweParam {
            message_modulus: 1 << param.message_bits,
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
        message_bits: 2,
        modulus: modulus.into(),
        ring_size,
        sk_distribution: Gaussian(3.19).into(),
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

pub fn nand_lut<R: RingOps>(
    ring: &R,
    q: usize,
    g: usize,
    encoded_half: R::Elem,
) -> RlwePlaintextOwned<R::Elem> {
    let auto_map = AutomorphismMap::new(q / 2, q - g);
    let lut_value = [ring.neg(&encoded_half), encoded_half];
    let log_q_by_8 = (q / 8).ilog2() as usize;
    let f = |(sign, idx)| lut_value[sign as usize ^ [1, 1, 1, 0][idx >> log_q_by_8]];
    RlwePlaintext::new(auto_map.iter().map(f).collect(), q / 2)
}

#[test]
fn bootstrap_nand() {
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
        let encoded_half = ring.elem_from(param.encoded_half());
        let nand_lut = nand_lut(ring, param.q, param.g, encoded_half);
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
            *ct.b_mut() = ring.add(ct.b(), &encoded_half);
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

#[cfg(feature = "serde")]
#[test]
fn serialize_deserialize() {
    use phantom_zone_math::util::serde::dev::assert_serde_eq;

    fn run<R: RingOps>(modulus: impl Into<Modulus>) {
        let mut rng = StdLweRng::from_entropy();
        let param = test_param(modulus, 1);
        let rgsw = RgswParam::from(param).build::<R>();
        let lwe_ks = LweParam::from(param).build::<NonNativePowerOfTwoRing>();
        let ring = rgsw.ring();
        let mod_ks = lwe_ks.modulus();

        let sk = rgsw.rlwe().sk_gen(&mut rng);
        let sk_ks = lwe_ks.sk_gen(&mut rng);
        let (bs_key, bs_key_prep) = {
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
            (bs_key, bs_key_prep)
        };
        assert_serde_eq(&sk);
        assert_serde_eq(&sk_ks);
        assert_serde_eq(&bs_key);
        assert_serde_eq(&bs_key_prep);
        assert_serde_eq(&bs_key.compact(ring, mod_ks));
        assert_eq!(
            &bs_key,
            &bs_key.compact(ring, mod_ks).uncompact(ring, mod_ks)
        );
    }

    run::<NoisyNativeRing>(Native::native());
    run::<NoisyNonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(50));
    run::<NativeRing>(Native::native());
    run::<NonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(50));
    run::<NoisyPrimeRing>(Prime::gen(50, 12));
    run::<PrimeRing>(Prime::gen(50, 12));
}
