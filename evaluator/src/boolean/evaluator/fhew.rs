//! Implementation of `BoolEvaluator` using boolean gates in 2020/086 and
//! blind rotation in 2022/198.

use crate::boolean::fhew::prelude::*;
use itertools::{izip, Itertools};
use phantom_zone_crypto::{
    core::{
        lwe::{self, LweCiphertext, LweCiphertextOwned, LwePlaintext, LweSecretKeyView},
        rlwe::{self, RlweCiphertext, RlwePlaintext, RlwePlaintextOwned, RlwePublicKeyView},
    },
    scheme::blind_rotation::lmkcdey,
};
use phantom_zone_math::{
    izip_eq, poly::automorphism::AutomorphismMap, util::scratch::ScratchOwned,
};
use rand::RngCore;

pub mod param;
pub mod prelude;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FhewBoolCiphertext<E>(LweCiphertextOwned<E>);

impl<E> FhewBoolCiphertext<E> {
    pub fn new(ct: LweCiphertextOwned<E>) -> Self {
        Self(ct)
    }

    pub fn allocate(ring_size: usize) -> Self
    where
        E: Default,
    {
        Self::new(LweCiphertext::allocate(ring_size))
    }

    pub fn sk_encrypt<'a, R, T>(
        param: &FhewBoolParam,
        ring: &R,
        sk: impl Into<LweSecretKeyView<'a, T>>,
        m: bool,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> Self
    where
        E: Default,
        R: RingOps<Elem = E> + ElemFrom<T>,
        T: 'a + Copy,
    {
        let pt = LwePlaintext(encode(ring, m));
        let mut ct = Self::allocate(ring.ring_size());
        lwe::sk_encrypt(ring, &mut ct.0, sk, pt, param.noise_distribution, rng);
        ct
    }

    pub fn pk_encrypt<'a, R: RingOps<Elem = E>>(
        param: &FhewBoolParam,
        ring: &R,
        pk: impl Into<RlwePublicKeyView<'a, E>>,
        m: bool,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> Self
    where
        E: 'a + Copy + Default,
    {
        Self::batched_pk_encrypt(param, ring, pk, [m], rng)
            .pop()
            .unwrap()
    }

    pub fn batched_pk_encrypt<'a, R: RingOps<Elem = E>>(
        param: &FhewBoolParam,
        ring: &R,
        pk: impl Into<RlwePublicKeyView<'a, E>>,
        ms: impl IntoIterator<Item = bool>,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> Vec<Self>
    where
        E: 'a + Copy + Default,
    {
        let pk = pk.into();
        let ms = ms.into_iter().collect_vec();
        let encoded_one = encode(ring, true);
        let mut cts = vec![Self::allocate(ring.ring_size()); ms.len()];
        let mut pt = RlwePlaintext::allocate(ring.ring_size());
        let mut ct_rlwe = RlweCiphertext::allocate(ring.ring_size());
        let mut scratch = ring.allocate_scratch(0, 2, 0);
        izip_eq!(
            cts.chunks_mut(ring.ring_size()),
            ms.chunks(ring.ring_size()),
        )
        .for_each(|(cts, ms)| {
            izip!(pt.as_mut(), ms)
                .for_each(|(pt, m)| *pt = if *m { encoded_one } else { ring.zero() });
            rlwe::pk_encrypt(
                ring,
                &mut ct_rlwe,
                pk,
                &pt,
                param.u_distribution,
                param.noise_distribution,
                scratch.borrow_mut(),
                rng,
            );
            izip!(0.., cts)
                .for_each(|(idx, ct)| rlwe::sample_extract(ring, &mut ct.0, &ct_rlwe, idx))
        });
        cts
    }

    pub fn decrypt<'a, R, T>(&self, ring: &R, sk: impl Into<LweSecretKeyView<'a, T>>) -> bool
    where
        R: RingOps<Elem = E> + ElemFrom<T>,
        T: 'a + Copy,
    {
        let pt = lwe::decrypt(ring, sk, &self.0);
        decode(ring, pt.0)
    }

    pub fn decrypt_share<'a, R, T>(
        &self,
        ring: &R,
        sk: impl Into<LweSecretKeyView<'a, T>>,
        noise_distribution: NoiseDistribution,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> LweDecryptionShare<E>
    where
        R: RingOps<Elem = E> + ElemFrom<T>,
        T: 'a + Copy,
    {
        lwe::decrypt_share(ring, sk, &self.0, noise_distribution, rng)
    }

    pub fn aggregate_decryption_shares<'a, R: RingOps<Elem = E>>(
        &self,
        ring: &R,
        dec_shares: impl IntoIterator<Item = &'a LweDecryptionShare<E>>,
    ) -> bool
    where
        E: 'a,
    {
        let pt = lwe::aggregate_decryption_shares(ring, &self.0, dec_shares);
        decode(ring, pt.0)
    }
}

fn encode<R: RingOps>(ring: &R, m: bool) -> R::Elem {
    if m {
        ring.elem_from(ring.modulus().as_f64() / 4f64)
    } else {
        ring.zero()
    }
}

fn decode<R: RingOps>(ring: &R, pt: R::Elem) -> bool {
    let delta = 4f64 / ring.modulus().as_f64();
    let m = (ring.to_u64(pt) as f64 * delta).round() as u64 & 0b11;
    assert!(m == 0 || m == 1);
    m == 1
}

pub struct FhewBoolEvaluator<R: RingOps, M: ModulusOps> {
    ring: R,
    mod_ks: M,
    bs_key: FhewBoolKey<R::EvalPrep, M::Elem>,
    /// Contains tables for AND, NAND, OR (XOR), NOR (XNOR).
    tables: [RlwePlaintextOwned<R::Elem>; 4],
    encoded_one: R::Elem,
    encoded_half: R::Elem,
    scratch_bytes: usize,
}

impl<R: RingOps, M: ModulusOps> FhewBoolEvaluator<R, M> {
    pub fn new(bs_key: FhewBoolKey<R::EvalPrep, M::Elem>) -> Self {
        assert_eq!(bs_key.param().message_bits, 2);
        let param = bs_key.param();
        let ring = <R as RingOps>::new(param.modulus, param.ring_size);
        let mod_ks = M::new(param.lwe_modulus);
        let encoded_one = ring.elem_from(param.encoded_one());
        let encoded_half = ring.elem_from(param.encoded_half());
        let tables = [[0, 0, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]]
            .map(|table| binary_lut(param, &ring, table));
        let scratch_bytes = param.scratch_bytes(&ring, &mod_ks);
        Self {
            ring,
            mod_ks,
            bs_key,
            tables,
            encoded_one,
            encoded_half,
            scratch_bytes,
        }
    }

    pub fn param(&self) -> &FhewBoolParam {
        self.bs_key.param()
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }

    pub fn mod_ks(&self) -> &M {
        &self.mod_ks
    }

    fn binary_op_assign<const XOR: bool>(
        &self,
        table_idx: usize,
        a: &mut FhewBoolCiphertext<R::Elem>,
        b: &FhewBoolCiphertext<R::Elem>,
    ) {
        let (mut a, b) = (a.0.as_mut_view(), b.0.as_view());
        if XOR {
            self.ring.slice_sub_assign(a.as_mut(), b.as_ref());
            self.ring.slice_double_assign(a.as_mut());
        } else {
            self.ring.slice_add_assign(a.as_mut(), b.as_ref())
        }
        lmkcdey::bootstrap(
            self.ring(),
            self.mod_ks(),
            &mut a,
            &self.bs_key,
            &self.tables[table_idx],
            ScratchOwned::allocate(self.scratch_bytes).borrow_mut(),
        );
        self.ring.add_assign(a.b_mut(), &self.encoded_half);
    }
}

impl<R: RingOps, M: ModulusOps> BoolEvaluator for FhewBoolEvaluator<R, M> {
    type Ciphertext = FhewBoolCiphertext<R::Elem>;

    fn bitnot_assign(&self, a: &mut Self::Ciphertext) {
        self.ring.slice_neg_assign(a.0.as_mut());
        self.ring.add_assign(a.0.b_mut(), &self.encoded_one);
    }

    fn bitand_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.binary_op_assign::<false>(0, a, b)
    }

    fn bitnand_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.binary_op_assign::<false>(1, a, b)
    }

    fn bitor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.binary_op_assign::<false>(2, a, b)
    }

    fn bitnor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.binary_op_assign::<false>(3, a, b)
    }

    fn bitxor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.binary_op_assign::<true>(2, a, b)
    }

    fn bitxnor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext) {
        self.binary_op_assign::<true>(3, a, b)
    }
}

fn binary_lut<R: RingOps>(
    param: &FhewBoolParam,
    ring: &R,
    table: [usize; 4],
) -> RlwePlaintextOwned<R::Elem> {
    let lut = {
        let encoded_half = ring.elem_from(param.encoded_half());
        let encoded = [ring.neg(&encoded_half), encoded_half];
        let log_q_by_8 = (param.q / 8).ilog2() as usize;
        AutomorphismMap::new(param.q / 2, param.q - param.g)
            .iter()
            .map(|(sign, idx)| encoded[sign as usize ^ table[idx >> log_q_by_8]])
            .collect()
    };
    RlwePlaintext::new(lut, param.q / 2)
}

#[cfg(any(test, feature = "dev"))]
mod dev {
    use crate::boolean::fhew::prelude::*;
    use phantom_zone_crypto::scheme::blind_rotation::lmkcdey::bs_key_gen;
    use rand::RngCore;

    impl<R: RingOps, M: ModulusOps> FhewBoolEvaluator<R, M> {
        pub fn sample(
            param: FhewBoolParam,
            sk: &LweSecretKeyOwned<i32>,
            rng: impl RngCore,
        ) -> Self {
            let mut rng = StdLweRng::from_rng(rng).unwrap();
            let ring = <R as RingOps>::new(param.modulus, param.ring_size);
            let mod_ks = M::new(param.lwe_modulus);
            let sk_ks = LweSecretKeyOwned::<i32>::sample(
                param.lwe_dimension,
                param.lwe_sk_distribution,
                &mut rng,
            );
            let mut bs_key = FhewBoolKey::allocate(param);
            bs_key_gen(&ring, &mod_ks, &mut bs_key, sk.as_view(), &sk_ks, &mut rng);
            let mut bs_key_prep = FhewBoolKey::allocate_eval(param, ring.eval_size());
            prepare_bs_key(&ring, &mut bs_key_prep, &bs_key);
            FhewBoolEvaluator::new(bs_key_prep)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::boolean::{
        evaluator::fhew::{self, prelude::*},
        test::tt,
    };
    use core::array::from_fn;
    use rand::{thread_rng, SeedableRng};

    type FhewBoolEvaluator<R> = fhew::FhewBoolEvaluator<R, NonNativePowerOfTwoRing>;

    fn test_param(modulus: impl Into<Modulus>) -> FhewBoolParam {
        let ring_size = 1024;
        FhewBoolParam {
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
            q: 2 * ring_size,
            g: 5,
            w: 10,
        }
    }

    fn sk_gen(ring_size: usize, sk_distribution: SecretDistribution) -> LweSecretKeyOwned<i32> {
        LweSecretKeyOwned::sample(ring_size, sk_distribution, thread_rng())
    }

    fn encrypt<R: RingOps>(
        param: &FhewBoolParam,
        ring: &R,
        sk: &LweSecretKeyOwned<i32>,
        m: bool,
    ) -> FhewBoolCiphertext<R::Elem> {
        let mut rng = StdLweRng::from_entropy();
        FhewBoolCiphertext::sk_encrypt(param, ring, sk, m, &mut rng)
    }

    #[test]
    fn encrypt_decrypt() {
        fn run<R: RingOps>(param: FhewBoolParam) {
            let ring = <R as RingOps>::new(param.modulus, param.ring_size);
            let sk = sk_gen(param.ring_size, param.sk_distribution);
            for _ in 0..100 {
                for m in [false, true] {
                    let ct = encrypt(&param, &ring, &sk, m);
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
        fn run<R: RingOps>(param: FhewBoolParam) {
            let sk = sk_gen(param.ring_size, param.sk_distribution);
            let evaluator = FhewBoolEvaluator::<R>::sample(param, &sk, thread_rng());
            let encrypt = |m| encrypt(evaluator.param(), evaluator.ring(), &sk, m);
            macro_rules! assert_decrypted_to {
                ($ct_a:ident.$op:ident($($ct_b:ident)?), $c:expr) => {
                    paste::paste! {
                        let mut ct_a = $ct_a.clone();
                        evaluator.[<$op _assign>](&mut ct_a $(, $ct_b)?);
                        assert_eq!(ct_a.decrypt(evaluator.ring(), &sk), $c);
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
        fn run<R: RingOps>(param: FhewBoolParam) {
            let sk = sk_gen(param.ring_size, param.sk_distribution);
            let evaluator = FhewBoolEvaluator::<R>::sample(param, &sk, thread_rng());
            let encrypt = |m| encrypt(evaluator.param(), evaluator.ring(), &sk, m);
            macro_rules! assert_decrypted_to {
                ($ct_a:ident.$op:ident($ct_b:ident $(, $ct_c:ident)?), $c:expr) => {
                    paste::paste! {
                        let mut ct_a = $ct_a.clone();
                        let ct_d = evaluator.[<$op _assign>](&mut ct_a, $ct_b $(, $ct_c)?);
                        assert_eq!(
                            (
                                ct_a.decrypt(evaluator.ring(), &sk),
                                ct_d.decrypt(evaluator.ring(), &sk),
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
