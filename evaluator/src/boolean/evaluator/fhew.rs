//! Implementation of `BoolEvaluator` using boolean gates in 2020/086 and
//! blind rotation in 2022/198.

use crate::boolean::evaluator::BoolEvaluator;
use core::{iter::repeat, marker::PhantomData};
use itertools::chain;
use phantom_zone_crypto::{
    core::{
        lwe::{
            self, LweCiphertext, LweCiphertextMutView, LweCiphertextOwned, LweCiphertextView,
            LweKeySwitchKey, LweKeySwitchKeyOwned, LwePlaintext, LweSecretKeyView,
        },
        rgsw::{RgswCiphertext, RgswCiphertextOwned, RgswDecompositionParam},
        rlwe::{RlweAutoKey, RlweAutoKeyOwned, RlwePlaintext, RlwePlaintextOwned},
    },
    scheme::blind_rotation::lmkcdey::{bootstrap, bootstrap_scratch_bytes, power_g_mod_q, LogGMap},
    util::{distribution::NoiseDistribution, rng::LweRng},
};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    modulus::{ElemFrom, Modulus, ModulusOps, NonNativePowerOfTwo},
    poly::automorphism::AutomorphismMap,
    ring::RingOps,
    util::scratch::ScratchOwned,
};
use rand::RngCore;

#[derive(Clone, Copy, Debug)]
pub struct FhewBoolParam {
    pub modulus: Modulus,
    pub ring_size: usize,
    pub auto_decomposition_param: DecompositionParam,
    pub rgsw_decomposition_param: RgswDecompositionParam,
    pub lwe_modulus: Modulus,
    pub lwe_dimension: usize,
    pub lwe_ks_decomposition_param: DecompositionParam,
    pub q: usize,
    pub g: usize,
    pub w: usize,
}

impl FhewBoolParam {
    pub fn embedding_factor(&self) -> usize {
        2 * self.ring_size / self.q
    }
}

#[derive(Clone, Debug)]
pub struct FhewBoolCiphertext<R: RingOps>(LweCiphertextOwned<R::Elem>, PhantomData<R>);

impl<R: RingOps> FhewBoolCiphertext<R> {
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
        Self(ct, PhantomData)
    }

    pub fn decrypt<'a, T>(&self, ring: &R, sk: impl Into<LweSecretKeyView<'a, T>>) -> bool
    where
        R: ElemFrom<T>,
        T: 'a + Copy,
    {
        let pt = lwe::decrypt(ring, sk, &self.0);
        let m = ring.mod_switch(&pt.0, &NonNativePowerOfTwo::new(2));
        debug_assert!(m == 0 || m == 1);
        m == 1
    }
}

pub struct FhewBoolEvaluator<R: RingOps, M: ModulusOps> {
    param: FhewBoolParam,
    ring: R,
    mod_ks: M,
    log_g_map: LogGMap,
    big_q_by_4: R::Elem,
    big_q_by_8: R::Elem,
    ks_key: LweKeySwitchKeyOwned<M::Elem>,
    brk: Vec<RgswCiphertextOwned<R::EvalPrep>>,
    ak: Vec<RlweAutoKeyOwned<R::EvalPrep>>,
    /// Contains tables for AND, NAND, OR (XOR), NOR (XNOR).
    luts: [RlwePlaintextOwned<R::Elem>; 4],
    scratch_bytes: usize,
}

impl<R: RingOps, M: ModulusOps> FhewBoolEvaluator<R, M> {
    pub fn new(
        param: FhewBoolParam,
        ks_key: LweKeySwitchKeyOwned<M::Elem>,
        brk: Vec<RgswCiphertextOwned<R::EvalPrep>>,
        ak: Vec<RlweAutoKeyOwned<R::EvalPrep>>,
    ) -> Self {
        let ring = <R as RingOps>::new(param.modulus, param.ring_size);
        let mod_ks = M::new(param.lwe_modulus);
        let log_g_map = LogGMap::new(param.g, param.q);
        let big_q_by_4 = ring.elem_from(param.modulus.as_f64() / 4f64);
        let big_q_by_8 = ring.elem_from(param.modulus.as_f64() / 8f64);
        let luts = {
            let auto_map = AutomorphismMap::new(param.q / 2, -(param.g as i64));
            let lut_value = [big_q_by_8, ring.neg(&big_q_by_8)];
            let log_q_by_8 = (param.q / 8).ilog2() as usize;
            [[1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 1]].map(|lut| {
                let f = |(sign, idx)| lut_value[sign as usize ^ lut[idx >> log_q_by_8]];
                RlwePlaintext::new(auto_map.iter().map(f).collect(), param.q / 2)
            })
        };
        let scratch_bytes = bootstrap_scratch_bytes(&ring, param.lwe_dimension);
        Self {
            param,
            ring,
            mod_ks,
            log_g_map,
            big_q_by_4,
            big_q_by_8,
            ks_key,
            brk,
            ak,
            luts,
            scratch_bytes,
        }
    }

    pub fn allocate(param: FhewBoolParam) -> Self {
        let mut evaluator = Self::new(
            param,
            LweKeySwitchKey::allocate(
                param.ring_size,
                param.lwe_dimension,
                param.lwe_ks_decomposition_param,
            ),
            Vec::new(),
            Vec::new(),
        );
        evaluator.brk = repeat(RgswCiphertext::allocate_eval(
            param.ring_size,
            evaluator.ring().eval_size(),
            param.rgsw_decomposition_param,
        ))
        .take(param.lwe_dimension)
        .collect();
        evaluator.ak = chain![[param.q - param.g], power_g_mod_q(param.g, param.q).skip(1)]
            .take(param.w + 1)
            .map(|k| {
                RlweAutoKey::allocate_eval(
                    param.ring_size,
                    evaluator.ring().eval_size(),
                    param.auto_decomposition_param,
                    k as _,
                )
            })
            .collect();
        evaluator
    }

    pub fn param(&self) -> FhewBoolParam {
        self.param
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
        bootstrap(
            &self.ring,
            &self.mod_ks,
            &self.log_g_map,
            &mut a,
            &self.ks_key,
            &self.brk,
            &self.ak,
            lut,
            ScratchOwned::allocate(self.scratch_bytes).borrow_mut(),
        );
        self.ring.add_assign(a.b_mut(), &self.big_q_by_8);
    }
}

impl<R: RingOps, M: RingOps> BoolEvaluator for FhewBoolEvaluator<R, M> {
    type Ciphertext = FhewBoolCiphertext<R>;

    fn bitnot_assign(&self, a: &mut Self::Ciphertext) {
        self.ring.slice_neg_assign(a.0.as_mut());
        self.ring.add_assign(a.0.b_mut(), &self.big_q_by_4);
    }

    fn bitand_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<false>(0, a.0.as_mut_view(), b.0.as_view())
    }

    fn bitnand_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<false>(1, a.0.as_mut_view(), b.0.as_view())
    }

    fn bitor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<false>(2, a.0.as_mut_view(), b.0.as_view())
    }

    fn bitnor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<false>(3, a.0.as_mut_view(), b.0.as_view())
    }

    fn bitxor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<true>(2, a.0.as_mut_view(), b.0.as_view())
    }

    fn bitxnor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.bitop_assign::<true>(3, a.0.as_mut_view(), b.0.as_view())
    }
}

#[cfg(any(test, feature = "dev"))]
mod dev {
    use crate::boolean::evaluator::fhew::{FhewBoolEvaluator, FhewBoolParam};
    use itertools::izip;
    use phantom_zone_crypto::{
        core::{
            lwe::{self, LweSecretKeyOwned},
            rgsw::{self, RgswCiphertext},
            rlwe::{self, RlweAutoKey, RlwePlaintext},
        },
        util::rng::StdLweRng,
    };
    use phantom_zone_math::{distribution::Gaussian, ring::RingOps};
    use rand::{RngCore, SeedableRng};

    impl<R: RingOps, M: RingOps> FhewBoolEvaluator<R, M> {
        pub fn sample(
            param: FhewBoolParam,
            sk: &LweSecretKeyOwned<i32>,
            rng: impl RngCore,
        ) -> Self {
            let mut evaluator = Self::allocate(param);
            let mut rng = StdLweRng::from_rng(rng).unwrap();
            let mut scratch = evaluator.ring.allocate_scratch(0, 3, 0);
            let mut scratch = scratch.borrow_mut();
            let sk_ks = LweSecretKeyOwned::<i32>::sample(
                param.lwe_dimension,
                Gaussian(3.2).into(),
                &mut rng,
            );
            lwe::ks_key_gen(
                &evaluator.mod_ks,
                &mut evaluator.ks_key,
                sk,
                &sk_ks,
                Gaussian(3.2).into(),
                &mut rng,
            );
            izip!(&mut evaluator.brk, sk_ks.as_ref()).for_each(|(brk_prep, sk_ks_i)| {
                let mut brk =
                    RgswCiphertext::allocate(brk_prep.ring_size(), brk_prep.decomposition_param());
                let mut pt = RlwePlaintext::allocate(brk_prep.ring_size());
                let exp = param.embedding_factor() as i32 * sk_ks_i;
                evaluator.ring.poly_set_monomial(pt.as_mut(), exp as _);
                rgsw::sk_encrypt(
                    &evaluator.ring,
                    &mut brk,
                    sk.as_view(),
                    &pt,
                    Gaussian(3.2).into(),
                    scratch.reborrow(),
                    &mut rng,
                );
                rgsw::prepare_rgsw(&evaluator.ring, brk_prep, &brk, scratch.reborrow());
            });
            evaluator.ak.iter_mut().for_each(|ak_prep| {
                let mut ak = RlweAutoKey::allocate(
                    ak_prep.ring_size(),
                    ak_prep.decomposition_param(),
                    ak_prep.k() as _,
                );
                rlwe::auto_key_gen(
                    &evaluator.ring,
                    &mut ak,
                    sk.as_view(),
                    Gaussian(3.2).into(),
                    scratch.reborrow(),
                    &mut rng,
                );
                rlwe::prepare_auto_key(&evaluator.ring, ak_prep, &ak, scratch.reborrow());
            });
            evaluator
        }
    }
}

#[cfg(test)]
mod test {
    use crate::boolean::{
        evaluator::{
            fhew::{self, FhewBoolCiphertext, FhewBoolParam},
            BoolEvaluator,
        },
        test::tt,
    };
    use core::array::from_fn;
    use phantom_zone_crypto::{
        core::{lwe::LweSecretKeyOwned, rgsw::RgswDecompositionParam},
        util::rng::StdLweRng,
    };
    use phantom_zone_math::{
        decomposer::DecompositionParam,
        distribution::Gaussian,
        modulus::{Modulus, NonNativePowerOfTwo, Prime},
        ring::{
            NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
            NonNativePowerOfTwoRing, PrimeRing, RingOps,
        },
    };
    use rand::{thread_rng, SeedableRng};

    type FhewBoolEvaluator<R> = fhew::FhewBoolEvaluator<R, NonNativePowerOfTwoRing>;

    fn test_param(big_q: impl Into<Modulus>) -> FhewBoolParam {
        let ring_size = 1024;
        FhewBoolParam {
            modulus: big_q.into(),
            ring_size,
            auto_decomposition_param: DecompositionParam {
                log_base: 24,
                level: 1,
            },
            rgsw_decomposition_param: RgswDecompositionParam {
                log_base: 17,
                level_a: 1,
                level_b: 1,
            },
            lwe_modulus: NonNativePowerOfTwo::new(16).into(),
            lwe_dimension: 100,
            lwe_ks_decomposition_param: DecompositionParam {
                log_base: 1,
                level: 13,
            },
            q: 2 * ring_size,
            g: 5,
            w: 10,
        }
    }

    fn sk_gen(ring_size: usize) -> LweSecretKeyOwned<i32> {
        LweSecretKeyOwned::sample(ring_size, Gaussian(3.2).into(), thread_rng())
    }

    fn encrypt<R: RingOps>(
        ring: &R,
        sk: &LweSecretKeyOwned<i32>,
        m: bool,
    ) -> FhewBoolCiphertext<R> {
        let mut rng = StdLweRng::from_entropy();
        FhewBoolCiphertext::encrypt(ring, sk, m, Gaussian(3.2).into(), &mut rng)
    }

    #[test]
    fn encrypt_decrypt() {
        fn run<R: RingOps>(param: FhewBoolParam) {
            let ring = <R as RingOps>::new(param.modulus, param.ring_size);
            let sk = sk_gen(ring.ring_size());
            for _ in 0..100 {
                for m in [false, true] {
                    let ct = encrypt(&ring, &sk, m);
                    assert_eq!(m, ct.decrypt(&ring, &sk))
                }
            }
        }

        run::<NoisyNativeRing>(test_param(Modulus::native()));
        run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NativeRing>(test_param(Modulus::native()));
        run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NoisyPrimeRing>(test_param(Prime::gen(50, 11)));
        run::<PrimeRing>(test_param(Prime::gen(50, 11)));
    }

    #[test]
    fn bit_op() {
        fn run<R: RingOps>(param: FhewBoolParam) {
            let sk = sk_gen(param.ring_size);
            let evaluator = FhewBoolEvaluator::<R>::sample(param, &sk, thread_rng());
            let encrypt = |m| encrypt(&evaluator.ring, &sk, m);
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

        run::<NoisyNativeRing>(test_param(Modulus::native()));
        run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NativeRing>(test_param(Modulus::native()));
        run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NoisyPrimeRing>(test_param(Prime::gen(50, 11)));
        run::<PrimeRing>(test_param(Prime::gen(50, 11)));
    }

    #[test]
    fn add_sub() {
        fn run<R: RingOps>(param: FhewBoolParam) {
            let sk = sk_gen(param.ring_size);
            let evaluator = FhewBoolEvaluator::<R>::sample(param, &sk, thread_rng());
            let encrypt = |m| encrypt(&evaluator.ring, &sk, m);
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

        run::<NoisyNativeRing>(test_param(Modulus::native()));
        run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NativeRing>(test_param(Modulus::native()));
        run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
        run::<NoisyPrimeRing>(test_param(Prime::gen(50, 11)));
        run::<PrimeRing>(test_param(Prime::gen(50, 11)));
    }
}
