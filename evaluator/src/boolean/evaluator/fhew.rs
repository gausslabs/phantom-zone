//! Implementation of `BoolEvaluator` using boolean gates in 2020/086 and
//! blind rotation in 2022/198.

use crate::boolean::evaluator::BoolEvaluator;
use core::marker::PhantomData;
use phantom_zone_crypto::{
    core::{
        lwe::{
            self, LweCiphertext, LweCiphertextMutView, LweCiphertextOwned, LweCiphertextView,
            LwePlaintext, LweSecretKeyView,
        },
        rlwe::{RlwePlaintext, RlwePlaintextOwned},
    },
    scheme::blind_rotation::lmkcdey::{self, LmkcdeyKeyOwned, LmkcdeyParam},
    util::{distribution::NoiseDistribution, rng::LweRng},
};
use phantom_zone_math::{
    modulus::{ElemFrom, ModulusOps, NonNativePowerOfTwo},
    poly::automorphism::AutomorphismMap,
    ring::RingOps,
    util::scratch::ScratchOwned,
};
use rand::RngCore;

pub type FhewBoolParam = LmkcdeyParam;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FhewBoolCiphertext<R: RingOps> {
    ct: LweCiphertextOwned<R::Elem>,
    #[cfg_attr(feature = "serde", serde(skip))]
    _marker: PhantomData<R>,
}

impl<R: RingOps> FhewBoolCiphertext<R> {
    pub fn new(ct: LweCiphertextOwned<R::Elem>) -> Self {
        Self {
            ct,
            _marker: PhantomData,
        }
    }

    pub fn encrypt<'a, T>(
        ring: &R,
        sk: impl Into<LweSecretKeyView<'a, T>>,
        m: bool,
        noise_distribution: NoiseDistribution,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> Self
    where
        R: ElemFrom<T>,
        T: 'a + Copy,
    {
        let pt = LwePlaintext(NonNativePowerOfTwo::new(2).mod_switch(&(m as u64), ring));
        let mut ct = LweCiphertext::allocate(ring.ring_size());
        lwe::sk_encrypt(ring, &mut ct, sk, pt, noise_distribution, rng);
        Self::new(ct)
    }

    pub fn decrypt<'a, T>(&self, ring: &R, sk: impl Into<LweSecretKeyView<'a, T>>) -> bool
    where
        R: ElemFrom<T>,
        T: 'a + Copy,
    {
        let pt = lwe::decrypt(ring, sk, &self.ct);
        let m = ring.mod_switch(&pt.0, &NonNativePowerOfTwo::new(2));
        assert!(m == 0 || m == 1);
        m == 1
    }
}

pub struct FhewBoolEvaluator<R: RingOps, M: ModulusOps> {
    ring: R,
    mod_ks: M,
    big_q_by_4: R::Elem,
    big_q_by_8: R::Elem,
    bs_key: LmkcdeyKeyOwned<R::EvalPrep, M::Elem>,
    /// Contains tables for AND, NAND, OR (XOR), NOR (XNOR).
    luts: [RlwePlaintextOwned<R::Elem>; 4],
    scratch_bytes: usize,
}

impl<R: RingOps, M: ModulusOps> FhewBoolEvaluator<R, M> {
    pub fn new(bs_key: LmkcdeyKeyOwned<R::EvalPrep, M::Elem>) -> Self {
        let param = bs_key.param();
        let ring = <R as RingOps>::new(param.modulus, param.ring_size);
        let mod_ks = M::new(param.lwe_modulus);
        let big_q_by_4 = ring.elem_from(param.modulus.as_f64() / 4f64);
        let big_q_by_8 = ring.elem_from(param.modulus.as_f64() / 8f64);
        let luts = {
            let auto_map = AutomorphismMap::new(param.q / 2, param.q - param.g);
            let lut_value = [big_q_by_8, ring.neg(&big_q_by_8)];
            let log_q_by_8 = (param.q / 8).ilog2() as usize;
            [[1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 1]].map(|lut| {
                let f = |(sign, idx)| lut_value[sign as usize ^ lut[idx >> log_q_by_8]];
                RlwePlaintext::new(auto_map.iter().map(f).collect(), param.q / 2)
            })
        };
        let scratch_bytes = param.scratch_bytes(&ring, &mod_ks);
        Self {
            ring,
            mod_ks,
            big_q_by_4,
            big_q_by_8,
            bs_key,
            luts,
            scratch_bytes,
        }
    }

    pub fn param(&self) -> &LmkcdeyParam {
        self.bs_key.param()
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }

    fn bitop_assign<const XOR: bool>(
        &self,
        lut_idx: usize,
        mut a: LweCiphertextMutView<R::Elem>,
        b: LweCiphertextView<R::Elem>,
    ) {
        if XOR {
            self.ring.slice_sub_assign(a.as_mut(), b.as_ref());
            self.ring.slice_double_assign(a.as_mut());
        } else {
            self.ring.slice_add_assign(a.as_mut(), b.as_ref())
        }
        let lut = &self.luts[lut_idx];
        lmkcdey::bootstrap(
            &self.ring,
            &self.mod_ks,
            &mut a,
            &self.bs_key,
            lut,
            ScratchOwned::allocate(self.scratch_bytes).borrow_mut(),
        );
        self.ring.add_assign(a.b_mut(), &self.big_q_by_8);
    }
}

impl<R: RingOps, M: ModulusOps> BoolEvaluator for FhewBoolEvaluator<R, M> {
    type Ciphertext = FhewBoolCiphertext<R>;

    fn bitnot_assign(&self, a: &mut Self::Ciphertext) {
        self.ring.slice_neg_assign(a.ct.as_mut());
        self.ring.add_assign(a.ct.b_mut(), &self.big_q_by_4);
    }

    fn bitand_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<false>(0, a.ct.as_mut_view(), b.ct.as_view())
    }

    fn bitnand_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<false>(1, a.ct.as_mut_view(), b.ct.as_view())
    }

    fn bitor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<false>(2, a.ct.as_mut_view(), b.ct.as_view())
    }

    fn bitnor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<false>(3, a.ct.as_mut_view(), b.ct.as_view())
    }

    fn bitxor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<true>(2, a.ct.as_mut_view(), b.ct.as_view())
    }

    fn bitxnor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<true>(3, a.ct.as_mut_view(), b.ct.as_view())
    }
}

#[cfg(any(test, feature = "dev"))]
mod dev {
    use crate::boolean::evaluator::fhew::{FhewBoolEvaluator, LmkcdeyParam};
    use phantom_zone_crypto::{
        core::lwe::LweSecretKeyOwned,
        scheme::blind_rotation::lmkcdey::{self, LmkcdeyKeyOwned},
        util::rng::StdLweRng,
    };
    use phantom_zone_math::{modulus::ModulusOps, ring::RingOps};
    use rand::RngCore;

    impl<R: RingOps, M: ModulusOps> FhewBoolEvaluator<R, M> {
        pub fn sample(param: LmkcdeyParam, sk: &LweSecretKeyOwned<i32>, rng: impl RngCore) -> Self {
            let mut rng = StdLweRng::from_rng(rng).unwrap();
            let ring = <R as RingOps>::new(param.modulus, param.ring_size);
            let mod_ks = M::new(param.lwe_modulus);
            let sk_ks = LweSecretKeyOwned::<i32>::sample(
                param.lwe_dimension,
                param.lwe_sk_distribution,
                &mut rng,
            );
            let mut scratch = ring.allocate_scratch(0, 3, 0);
            let mut bs_key = LmkcdeyKeyOwned::allocate(param);
            lmkcdey::bs_key_gen(
                &ring,
                &mod_ks,
                &mut bs_key,
                sk.as_view(),
                &sk_ks,
                scratch.borrow_mut(),
                &mut rng,
            );
            let mut bs_key_prep = LmkcdeyKeyOwned::allocate_eval(param, ring.eval_size());
            lmkcdey::prepare_bs_key(&ring, &mut bs_key_prep, &bs_key, scratch.borrow_mut());
            FhewBoolEvaluator::new(bs_key_prep)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::boolean::{
        evaluator::{
            fhew::{self, FhewBoolCiphertext, LmkcdeyParam},
            BoolEvaluator,
        },
        test::tt,
    };
    use core::array::from_fn;
    use phantom_zone_crypto::{
        core::{lwe::LweSecretKeyOwned, rgsw::RgswDecompositionParam},
        util::{
            distribution::{NoiseDistribution, SecretDistribution},
            rng::StdLweRng,
        },
    };
    use phantom_zone_math::{
        decomposer::DecompositionParam,
        distribution::Gaussian,
        modulus::{Modulus, Native, NonNativePowerOfTwo, Prime},
        ring::{
            NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
            NonNativePowerOfTwoRing, PrimeRing, RingOps,
        },
    };
    use rand::{thread_rng, SeedableRng};

    type FhewBoolEvaluator<R> = fhew::FhewBoolEvaluator<R, NonNativePowerOfTwoRing>;

    fn test_param(big_q: impl Into<Modulus>) -> LmkcdeyParam {
        let ring_size = 1024;
        LmkcdeyParam {
            modulus: big_q.into(),
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
            q: 2 * ring_size,
            g: 5,
            w: 10,
        }
    }

    fn sk_gen(ring_size: usize, sk_distribution: SecretDistribution) -> LweSecretKeyOwned<i32> {
        LweSecretKeyOwned::sample(ring_size, sk_distribution, thread_rng())
    }

    fn encrypt<R: RingOps>(
        ring: &R,
        sk: &LweSecretKeyOwned<i32>,
        m: bool,
        noise_distribution: NoiseDistribution,
    ) -> FhewBoolCiphertext<R> {
        let mut rng = StdLweRng::from_entropy();
        FhewBoolCiphertext::encrypt(ring, sk, m, noise_distribution, &mut rng)
    }

    #[test]
    fn encrypt_decrypt() {
        fn run<R: RingOps>(param: LmkcdeyParam) {
            let ring = <R as RingOps>::new(param.modulus, param.ring_size);
            let sk = sk_gen(param.ring_size, param.sk_distribution);
            for _ in 0..100 {
                for m in [false, true] {
                    let ct = encrypt(&ring, &sk, m, param.noise_distribution);
                    assert_eq!(m, ct.decrypt(&ring, &sk))
                }
            }
        }

        run::<NoisyNativeRing>(test_param(Native::native()));
        run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NativeRing>(test_param(Native::native()));
        run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NoisyPrimeRing>(test_param(Prime::gen(50, 11)));
        run::<PrimeRing>(test_param(Prime::gen(50, 11)));
    }

    #[test]
    fn bit_op() {
        fn run<R: RingOps>(param: LmkcdeyParam) {
            let sk = sk_gen(param.ring_size, param.sk_distribution);
            let evaluator = FhewBoolEvaluator::<R>::sample(param, &sk, thread_rng());
            let encrypt = |m| encrypt(&evaluator.ring, &sk, m, param.noise_distribution);
            macro_rules! assert_decrypted_to {
                ($ct_a:ident.$op:ident($($ct_b:ident)?), $c:expr) => {
                    paste::paste! {
                        let mut ct_a = $ct_a.clone();
                        evaluator.[<$op _assign>](&mut ct_a $(, $ct_b)?);
                        assert_eq!(ct_a.decrypt(&evaluator.ring, &sk), $c);
                    }
                };
            }
            for m in 0..1 << 1 {
                let m = m == 1;
                let ct = encrypt(m);
                assert_decrypted_to!(ct.bitnot(), !m);
            }
            for m in 0..1 << 2 {
                let [a, b] = from_fn(|i| (m >> i) & 1 == 1);
                let [ct_a, ct_b] = &[a, b].map(encrypt);
                assert_decrypted_to!(ct_a.bitand(ct_b), a & b);
                assert_decrypted_to!(ct_a.bitnand(ct_b), !(a & b));
                assert_decrypted_to!(ct_a.bitor(ct_b), a | b);
                assert_decrypted_to!(ct_a.bitnor(ct_b), !(a | b));
                assert_decrypted_to!(ct_a.bitxor(ct_b), a ^ b);
                assert_decrypted_to!(ct_a.bitxnor(ct_b), !(a ^ b));
            }
        }

        run::<NoisyNativeRing>(test_param(Native::native()));
        run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NativeRing>(test_param(Native::native()));
        run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NoisyPrimeRing>(test_param(Prime::gen(50, 11)));
        run::<PrimeRing>(test_param(Prime::gen(50, 11)));
    }

    #[test]
    fn add_sub() {
        fn run<R: RingOps>(param: LmkcdeyParam) {
            let sk = sk_gen(param.ring_size, param.sk_distribution);
            let evaluator = FhewBoolEvaluator::<R>::sample(param, &sk, thread_rng());
            let encrypt = |m| encrypt(&evaluator.ring, &sk, m, param.noise_distribution);
            macro_rules! assert_decrypted_to {
                ($ct_a:ident.$op:ident($ct_b:ident $(, $ct_c:ident)?), $c:expr) => {
                    paste::paste! {
                        let mut ct_a = $ct_a.clone();
                        let ct_d = evaluator.[<$op _assign>](&mut ct_a, $ct_b $(, $ct_c)?);
                        assert_eq!(
                            (
                                ct_a.decrypt(&evaluator.ring, &sk),
                                ct_d.decrypt(&evaluator.ring, &sk),
                            ),
                            $c,
                        );
                    }
                };
            }
            for m in 0..1 << 2 {
                let [a, b] = from_fn(|i| (m >> i) & 1 == 1);
                let [ct_a, ct_b] = &[a, b].map(encrypt);
                assert_decrypted_to!(ct_a.overflowing_add(ct_b), tt::OVERFLOWING_ADD[m]);
                assert_decrypted_to!(ct_a.overflowing_sub(ct_b), tt::OVERFLOWING_SUB[m]);
            }
            for m in 0..1 << 3 {
                let [a, b, c] = from_fn(|i| (m >> i) & 1 == 1);
                let [ct_a, ct_b, ct_c] = &[a, b, c].map(encrypt);
                assert_decrypted_to!(ct_a.carrying_add(ct_b, ct_c), tt::CARRYING_ADD[m]);
                assert_decrypted_to!(ct_a.borrowing_sub(ct_b, ct_c), tt::BORROWING_SUB[m]);
            }
        }

        run::<NoisyNativeRing>(test_param(Native::native()));
        run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NativeRing>(test_param(Native::native()));
        run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NoisyPrimeRing>(test_param(Prime::gen(50, 11)));
        run::<PrimeRing>(test_param(Prime::gen(50, 11)));
    }
}
