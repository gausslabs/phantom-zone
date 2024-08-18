use crate::{
    core::{
        lwe::{self, test::LweParam, LweCiphertextOwned, LwePlaintext},
        rlwe::{
            self, RlweAutoKey, RlweAutoKeyOwned, RlweCiphertext, RlweCiphertextOwned,
            RlweKeySwitchKey, RlweKeySwitchKeyOwned, RlwePlaintext, RlwePlaintextOwned,
            RlwePublicKey, RlwePublicKeyOwned, RlweSecretKey, RlweSecretKeyOwned,
        },
    },
    util::{
        distribution::{NoiseDistribution, SecretDistribution},
        rng::{LweRng, StdLweRng},
    },
};
use itertools::Itertools;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::{Gaussian, Sampler, Ternary},
    modulus::{Modulus, ModulusOps, NonNativePowerOfTwo, Prime},
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps, SliceOps,
    },
};
use rand::{thread_rng, RngCore, SeedableRng};

#[derive(Clone, Copy, Debug)]
pub struct RlweParam {
    pub message_modulus: u64,
    pub ciphertext_modulus: Modulus,
    pub ring_size: usize,
    pub sk_distribution: SecretDistribution,
    pub u_distribution: SecretDistribution,
    pub noise_distribution: NoiseDistribution,
    pub ks_decomposition_param: DecompositionParam,
}

impl RlweParam {
    pub fn to_lwe(self) -> LweParam {
        LweParam {
            message_modulus: self.message_modulus,
            ciphertext_modulus: self.ciphertext_modulus,
            dimension: self.ring_size,
            sk_distribution: self.sk_distribution,
            noise_distribution: self.noise_distribution,
            ks_decomposition_param: self.ks_decomposition_param,
        }
    }

    pub fn build<R: RingOps>(self) -> Rlwe<R> {
        let ring = R::new(self.ciphertext_modulus, self.ring_size);
        let message_ring = NonNativePowerOfTwoRing::new(
            NonNativePowerOfTwo::new(self.message_modulus.ilog2() as _).into(),
            self.ring_size,
        );
        Rlwe {
            param: self,
            ring,
            message_ring,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rlwe<R: RingOps> {
    param: RlweParam,
    ring: R,
    message_ring: NonNativePowerOfTwoRing,
}

impl<R: RingOps> Rlwe<R> {
    pub fn ring(&self) -> &R {
        &self.ring
    }

    pub fn ring_size(&self) -> usize {
        self.ring().ring_size()
    }

    pub fn message_ring(&self) -> &NonNativePowerOfTwoRing {
        &self.message_ring
    }

    pub fn encode(&self, m: &[u64]) -> RlwePlaintextOwned<R::Elem> {
        let mut pt = RlwePlaintext::allocate(self.ring_size());
        self.message_ring
            .slice_mod_switch(pt.as_mut(), m, self.ring());
        pt
    }

    pub fn decode(&self, pt: &RlwePlaintextOwned<R::Elem>) -> Vec<u64> {
        let mut m = vec![0; self.ring_size()];
        self.ring()
            .slice_mod_switch(&mut m, pt.as_ref(), &self.message_ring);
        m
    }

    pub fn decode_lwe(&self, pt: LwePlaintext<R::Elem>) -> u64 {
        self.ring().mod_switch(&pt.0, &self.message_ring)
    }

    pub fn sk_gen(&self) -> RlweSecretKeyOwned<i32> {
        RlweSecretKey::sample(self.ring_size(), self.param.sk_distribution, thread_rng())
    }

    pub fn pk_gen(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> RlwePublicKeyOwned<R::Elem> {
        let mut pk = RlwePublicKey::allocate(self.ring_size());
        let mut scratch = self.ring().allocate_scratch(0, 2, 0);
        rlwe::pk_gen(
            self.ring(),
            &mut pk,
            sk,
            self.param.noise_distribution,
            scratch.borrow_mut(),
            rng,
        );
        pk
    }

    pub fn sk_encrypt(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        pt: &RlwePlaintextOwned<R::Elem>,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = RlweCiphertext::allocate(self.ring_size());
        let mut scratch = self.ring().allocate_scratch(0, 2, 0);
        rlwe::sk_encrypt(
            self.ring(),
            &mut ct,
            sk,
            pt,
            self.param.noise_distribution,
            scratch.borrow_mut(),
            rng,
        );
        ct
    }

    pub fn pk_encrypt(
        &self,
        pk: &RlwePublicKeyOwned<R::Elem>,
        pt: &RlwePlaintextOwned<R::Elem>,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = RlweCiphertext::allocate(self.ring_size());
        let mut scratch = self.ring().allocate_scratch(0, 2, 0);
        rlwe::pk_encrypt(
            self.ring(),
            &mut ct,
            pk,
            pt,
            self.param.u_distribution,
            self.param.noise_distribution,
            scratch.borrow_mut(),
            rng,
        );
        ct
    }

    pub fn decrypt(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        ct: &RlweCiphertextOwned<R::Elem>,
    ) -> RlwePlaintextOwned<R::Elem> {
        let mut pt = RlwePlaintext::allocate(self.ring_size());
        let mut scratch = self.ring().allocate_scratch(0, 2, 0);
        rlwe::decrypt(self.ring(), &mut pt, sk, ct, scratch.borrow_mut());
        pt
    }

    pub fn decrypt_lwe(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        ct: &LweCiphertextOwned<R::Elem>,
    ) -> LwePlaintext<R::Elem> {
        lwe::decrypt(self.ring(), sk.as_view(), ct)
    }

    pub fn sample_extract(
        &self,
        ct_rlwe: &RlweCiphertextOwned<R::Elem>,
        idx: usize,
    ) -> LweCiphertextOwned<R::Elem> {
        let mut ct_lwe = LweCiphertextOwned::allocate(self.ring_size());
        rlwe::sample_extract(self.ring(), &mut ct_lwe, ct_rlwe, idx);
        ct_lwe
    }

    pub fn ks_key_gen(
        &self,
        sk_from: &RlweSecretKeyOwned<i32>,
        sk_to: &RlweSecretKeyOwned<i32>,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> RlweKeySwitchKeyOwned<R::Elem> {
        let mut ks_key =
            RlweKeySwitchKey::allocate(self.ring_size(), self.param.ks_decomposition_param);
        let mut scratch = self.ring().allocate_scratch(0, 2, 0);
        rlwe::ks_key_gen(
            self.ring(),
            &mut ks_key,
            sk_from,
            sk_to,
            self.param.noise_distribution,
            scratch.borrow_mut(),
            rng,
        );
        ks_key
    }

    pub fn key_switch(
        &self,
        ks_key: &RlweKeySwitchKeyOwned<R::Elem>,
        ct_from: &RlweCiphertextOwned<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = ct_from.clone();
        let mut scratch = self.ring().allocate_scratch(2, 4, 0);
        rlwe::key_switch_in_place(self.ring(), &mut ct, ks_key, scratch.borrow_mut());
        ct
    }

    pub fn auto_key_gen(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        k: i64,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> RlweAutoKeyOwned<R::Elem> {
        let mut auto_key =
            RlweAutoKey::allocate(self.ring_size(), self.param.ks_decomposition_param, k);
        let mut scratch = self.ring().allocate_scratch(0, 3, 0);
        rlwe::auto_key_gen(
            self.ring(),
            &mut auto_key,
            sk,
            self.param.noise_distribution,
            scratch.borrow_mut(),
            rng,
        );
        auto_key
    }

    pub fn automorphism(
        &self,
        auto_key: &RlweAutoKeyOwned<R::Elem>,
        ct: &RlweCiphertextOwned<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = ct.clone();
        let mut scratch = self.ring().allocate_scratch(2, 4, 0);
        rlwe::automorphism_in_place(self.ring(), &mut ct, auto_key, scratch.borrow_mut());
        ct
    }

    pub fn prepare_ks_key(
        &self,
        ks_key: &RlweKeySwitchKeyOwned<R::Elem>,
    ) -> RlweKeySwitchKeyOwned<R::EvalPrep> {
        let mut ks_key_prep = RlweKeySwitchKey::allocate_eval(
            self.ring_size(),
            self.ring().eval_size(),
            ks_key.decomposition_param(),
        );
        let mut scratch = self.ring().allocate_scratch(0, 1, 0);
        rlwe::prepare_ks_key(self.ring(), &mut ks_key_prep, ks_key, scratch.borrow_mut());
        ks_key_prep
    }

    pub fn prepare_auto_key(
        &self,
        auto_key: &RlweAutoKeyOwned<R::Elem>,
    ) -> RlweAutoKeyOwned<R::EvalPrep> {
        let mut auto_key_prep = RlweAutoKey::allocate_eval(
            self.ring_size(),
            self.ring().eval_size(),
            auto_key.decomposition_param(),
            auto_key.k() as _,
        );
        let mut scratch = self.ring().allocate_scratch(0, 1, 0);
        rlwe::prepare_auto_key(
            self.ring(),
            &mut auto_key_prep,
            auto_key,
            scratch.borrow_mut(),
        );
        auto_key_prep
    }

    pub fn key_switch_prep(
        &self,
        ks_key_prep: &RlweKeySwitchKeyOwned<R::EvalPrep>,
        ct_from: &RlweCiphertextOwned<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = ct_from.clone();
        let mut scratch = self.ring().allocate_scratch(2, 3, 0);
        rlwe::key_switch_prep_in_place(self.ring(), &mut ct, ks_key_prep, scratch.borrow_mut());
        ct
    }

    pub fn automorphism_prep(
        &self,
        auto_key_prep: &RlweAutoKeyOwned<R::EvalPrep>,
        ct: &RlweCiphertextOwned<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = ct.clone();
        let mut scratch = self.ring().allocate_scratch(2, 3, 0);
        rlwe::automorphism_prep_in_place(self.ring(), &mut ct, auto_key_prep, scratch.borrow_mut());
        ct
    }
}

pub fn test_param(ciphertext_modulus: impl Into<Modulus>) -> RlweParam {
    let ring_size = 256;
    RlweParam {
        message_modulus: 1 << 6,
        ciphertext_modulus: ciphertext_modulus.into(),
        ring_size,
        sk_distribution: Gaussian(3.2).into(),
        u_distribution: Ternary(ring_size / 2).into(),
        noise_distribution: Gaussian(3.2).into(),
        ks_decomposition_param: DecompositionParam {
            log_base: 8,
            level: 6,
        },
    }
}

#[test]
fn encrypt_decrypt() {
    fn run<R: RingOps>(param: RlweParam) {
        let mut rng = StdLweRng::from_entropy();
        let rlwe = param.build::<R>();
        let sk = rlwe.sk_gen();
        let pk = rlwe.pk_gen(&sk, &mut rng);
        for _ in 0..100 {
            let m = rlwe.message_ring.sample_uniform_poly(&mut rng);
            let pt = rlwe.encode(&m);
            let ct_sk = rlwe.sk_encrypt(&sk, &pt, &mut rng);
            let ct_pk = rlwe.pk_encrypt(&pk, &pt, &mut rng);
            assert_eq!(m, rlwe.decode(&pt));
            assert_eq!(m, rlwe.decode(&rlwe.decrypt(&sk, &ct_sk)));
            assert_eq!(m, rlwe.decode(&rlwe.decrypt(&sk, &ct_pk)));
        }
    }

    run::<NoisyNativeRing>(test_param(Modulus::native()));
    run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NoisyPrimeRing>(test_param(Prime::gen(50, 9)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}

#[test]
fn sample_extract() {
    fn run<R: RingOps>(param: RlweParam) {
        let mut rng = StdLweRng::from_entropy();
        let rlwe = param.build::<R>();
        let sk = rlwe.sk_gen();
        let m = rlwe.message_ring.sample_uniform_poly(&mut rng);
        let pt = rlwe.encode(&m);
        let ct_rlwe = rlwe.sk_encrypt(&sk, &pt, &mut rng);
        for (idx, m) in m.iter().enumerate() {
            let ct_lwe = rlwe.sample_extract(&ct_rlwe, idx);
            assert_eq!(*m, rlwe.decode_lwe(rlwe.decrypt_lwe(&sk, &ct_lwe)));
        }
    }

    run::<NoisyNativeRing>(test_param(Modulus::native()));
    run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NoisyPrimeRing>(test_param(Prime::gen(50, 9)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}

#[test]
fn key_switch() {
    fn run<R: RingOps>(param: RlweParam) {
        let mut rng = StdLweRng::from_entropy();
        let rlwe = param.build::<R>();
        let sk_from = rlwe.sk_gen();
        let sk_to = rlwe.sk_gen();
        let ks_key = rlwe.ks_key_gen(&sk_from, &sk_to, &mut rng);
        let ks_key_prep = rlwe.prepare_ks_key(&ks_key);
        for _ in 0..100 {
            let m = rlwe.message_ring.sample_uniform_poly(&mut rng);
            let pt = rlwe.encode(&m);
            let ct_from = rlwe.sk_encrypt(&sk_from, &pt, &mut rng);
            let ct_to = rlwe.key_switch(&ks_key, &ct_from);
            assert_eq!(m, rlwe.decode(&rlwe.decrypt(&sk_to, &ct_to)));
            let ct_to = rlwe.key_switch_prep(&ks_key_prep, &ct_from);
            assert_eq!(m, rlwe.decode(&rlwe.decrypt(&sk_to, &ct_to)));
        }
    }

    run::<NoisyNativeRing>(test_param(Modulus::native()));
    run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NoisyPrimeRing>(test_param(Prime::gen(50, 9)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}

#[test]
fn automorphism() {
    fn run<R: RingOps>(param: RlweParam) {
        let mut rng = StdLweRng::from_entropy();
        let rlwe = param.build::<R>();
        let sk = rlwe.sk_gen();
        for k in (1..rlwe.ring_size()).step_by(2) {
            let auto_key = rlwe.auto_key_gen(&sk, k as _, &mut rng);
            let auto_key_prep = rlwe.prepare_auto_key(&auto_key);
            let m = rlwe.message_ring.sample_uniform_poly(&mut rng);
            let ct = rlwe.sk_encrypt(&sk, &rlwe.encode(&m), &mut rng);
            let m_auto = {
                let neg = |v: &_| rlwe.message_ring.neg(v);
                auto_key.auto_map().apply(&m, neg).collect_vec()
            };
            let ct_auto = rlwe.automorphism(&auto_key, &ct);
            assert_eq!(m_auto, rlwe.decode(&rlwe.decrypt(&sk, &ct_auto)));
            let ct_auto = rlwe.automorphism_prep(&auto_key_prep, &ct);
            assert_eq!(m_auto, rlwe.decode(&rlwe.decrypt(&sk, &ct_auto)));
        }
    }

    run::<NoisyNativeRing>(test_param(Modulus::native()));
    run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NoisyPrimeRing>(test_param(Prime::gen(50, 9)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}
