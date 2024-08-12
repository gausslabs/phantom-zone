use crate::{
    core::lwe::{
        self, LweCiphertext, LweCiphertextOwned, LweKeySwitchKey, LweKeySwitchKeyOwned,
        LwePlaintext, LweSecretKey, LweSecretKeyOwned,
    },
    util::{
        distribution::{NoiseDistribution, SecretKeyDistribution},
        rng::{test::StdLweRng, LweRng},
    },
};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::Gaussian,
    modulus::{Modulus, ForeignPowerOfTwo, Prime},
    ring::{
        NativeRing, NoisyNativeRing, NoisyForeignPowerOfTwoRing, NoisyPrimeRing,
        ForeignPowerOfTwoRing, PrimeRing, RingOps,
    },
};
use rand::{thread_rng, RngCore};

#[derive(Clone, Copy, Debug)]
pub struct LweParam {
    pub message_modulus: u64,
    pub ciphertext_modulus: Modulus,
    pub dimension: usize,
    pub sk_distribution: SecretKeyDistribution,
    pub noise_distribution: NoiseDistribution,
    pub ks_decomposition_param: DecompositionParam,
}

impl LweParam {
    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    pub fn build<R: RingOps>(self) -> Lwe<R> {
        let delta = self.ciphertext_modulus.as_f64() / self.message_modulus as f64;
        let ring = RingOps::new(self.ciphertext_modulus, 1);
        Lwe {
            param: self,
            delta,
            ring,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Lwe<R: RingOps> {
    pub param: LweParam,
    pub delta: f64,
    pub ring: R,
}

impl<R: RingOps> Lwe<R> {
    pub fn ring(&self) -> &R {
        &self.ring
    }

    pub fn dimension(&self) -> usize {
        self.param.dimension
    }

    pub fn encode(&self, m: u64) -> LwePlaintext<R::Elem> {
        assert!(m < self.param.message_modulus);
        LwePlaintext(self.ring.elem_from((m as f64 * self.delta).round() as u64))
    }

    pub fn decode(&self, LwePlaintext(pt): LwePlaintext<R::Elem>) -> u64 {
        let pt: u64 = self.ring.elem_to(pt);
        (pt as f64 / self.delta).round() as u64 % self.param.message_modulus
    }

    pub fn sk_gen(&self) -> LweSecretKeyOwned<i32> {
        LweSecretKey::sample(self.dimension(), self.param.sk_distribution, thread_rng())
    }

    pub fn sk_encrypt(
        &self,
        sk: &LweSecretKeyOwned<i32>,
        pt: LwePlaintext<R::Elem>,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> LweCiphertextOwned<R::Elem> {
        let mut ct = LweCiphertext::allocate(self.dimension());
        lwe::sk_encrypt(
            self.ring(),
            &mut ct,
            sk,
            pt,
            self.param.noise_distribution,
            rng,
        );
        ct
    }

    pub fn decrypt(
        &self,
        sk: &LweSecretKeyOwned<i32>,
        ct: &LweCiphertextOwned<R::Elem>,
    ) -> LwePlaintext<R::Elem> {
        lwe::decrypt(self.ring(), sk, ct)
    }

    pub fn ks_key_gen(
        &self,
        sk_from: &LweSecretKeyOwned<i32>,
        sk_to: &LweSecretKeyOwned<i32>,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> LweKeySwitchKeyOwned<R::Elem> {
        assert_eq!(self.dimension(), sk_to.dimension());
        let mut ks_key = LweKeySwitchKey::allocate(
            sk_from.dimension(),
            sk_to.dimension(),
            self.param.ks_decomposition_param,
        );
        lwe::ks_key_gen(
            self.ring(),
            &mut ks_key,
            sk_from,
            sk_to,
            self.param.noise_distribution,
            rng,
        );
        ks_key
    }

    pub fn key_switch(
        &self,
        ks_key: &LweKeySwitchKeyOwned<R::Elem>,
        ct_from: &LweCiphertextOwned<R::Elem>,
    ) -> LweCiphertextOwned<R::Elem> {
        let mut ct_to = LweCiphertext::allocate(ks_key.to_dimension());
        lwe::key_switch(self.ring(), &mut ct_to, ks_key, ct_from);
        ct_to
    }

    pub fn add(
        &self,
        ct_a: &LweCiphertextOwned<R::Elem>,
        ct_b: &LweCiphertextOwned<R::Elem>,
    ) -> LweCiphertextOwned<R::Elem> {
        let mut ct_c = ct_a.clone();
        self.ring().slice_add_assign(ct_c.as_mut(), ct_b.as_ref());
        ct_c
    }
}

pub fn test_param(ciphertext_modulus: impl Into<Modulus>) -> LweParam {
    LweParam {
        message_modulus: 1 << 6,
        ciphertext_modulus: ciphertext_modulus.into(),
        dimension: 256,
        sk_distribution: Gaussian::new(3.2).into(),
        noise_distribution: Gaussian::new(3.2).into(),
        ks_decomposition_param: DecompositionParam {
            log_base: 8,
            level: 6,
        },
    }
}

#[test]
fn encrypt_decrypt() {
    fn run<R: RingOps>(param: LweParam) {
        let mut rng = StdLweRng::from_entropy();
        let lwe = param.build::<R>();
        let sk = lwe.sk_gen();
        for m in 0..param.message_modulus {
            let pt = lwe.encode(m);
            let ct = lwe.sk_encrypt(&sk, pt, &mut rng);
            assert_eq!(m, lwe.decode(pt));
            assert_eq!(m, lwe.decode(lwe.decrypt(&sk, &ct)));
        }
    }

    run::<NoisyNativeRing>(test_param(Modulus::native()));
    run::<NoisyForeignPowerOfTwoRing>(test_param(ForeignPowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<ForeignPowerOfTwoRing>(test_param(ForeignPowerOfTwo::new(50)));
    run::<NoisyPrimeRing>(test_param(Prime::gen(50, 0)));
    run::<PrimeRing>(test_param(Prime::gen(50, 0)));
}

#[test]
fn key_switch() {
    fn run<R: RingOps>(param: LweParam) {
        let mut rng = StdLweRng::from_entropy();
        let lwe_from = param.build::<R>();
        let lwe_to = param.dimension(2 * param.dimension).build::<R>();
        let sk_from = lwe_from.sk_gen();
        let sk_to = lwe_to.sk_gen();
        let ks_key = lwe_to.ks_key_gen(&sk_from, &sk_to, &mut rng);
        for m in 0..param.message_modulus {
            let pt = lwe_from.encode(m);
            let ct_from = lwe_from.sk_encrypt(&sk_from, pt, &mut rng);
            let ct_to = lwe_to.key_switch(&ks_key, &ct_from);
            assert_eq!(m, lwe_to.decode(lwe_to.decrypt(&sk_to, &ct_to)));
        }
    }

    run::<NoisyNativeRing>(test_param(Modulus::native()));
    run::<NoisyForeignPowerOfTwoRing>(test_param(ForeignPowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<ForeignPowerOfTwoRing>(test_param(ForeignPowerOfTwo::new(50)));
    run::<NoisyPrimeRing>(test_param(Prime::gen(50, 0)));
    run::<PrimeRing>(test_param(Prime::gen(50, 0)));
}
