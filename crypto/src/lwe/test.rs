use crate::{
    distribution::{NoiseDistribution, SecretKeyDistribution},
    lwe::{
        self, LweCiphertext, LweCiphertextOwned, LweKeySwitchKey, LweKeySwitchKeyOwned,
        LwePlaintext, LweSecretKey, LweSecretKeyOwned,
    },
};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::Gaussian,
    modulus::{Modulus, PowerOfTwo, Prime},
    ring::{
        power_of_two::{
            NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NonNativePowerOfTwoRing,
        },
        prime::PrimeRing,
        RingOps,
    },
};
use rand::{thread_rng, RngCore};

#[derive(Clone, Copy, Debug)]
struct LweParam {
    message_modulus: u64,
    ciphertext_modulus: Modulus,
    dimension: usize,
    sk_distribution: SecretKeyDistribution,
    noise_distribution: NoiseDistribution,
    ks_decomposition_param: DecompositionParam,
}

impl LweParam {
    fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    fn build<R: RingOps>(self) -> Lwe<R> {
        let delta = self.ciphertext_modulus.to_f64() / self.message_modulus as f64;
        let ring = RingOps::new(self.ciphertext_modulus, 1);
        Lwe {
            param: self,
            delta,
            ring,
        }
    }
}

#[derive(Clone, Debug)]
struct Lwe<R: RingOps> {
    param: LweParam,
    delta: f64,
    ring: R,
}

impl<R: RingOps> Lwe<R> {
    fn ring(&self) -> &R {
        &self.ring
    }

    fn dimension(&self) -> usize {
        self.param.dimension
    }

    fn encode(&self, m: u64) -> LwePlaintext<R::Elem> {
        assert!(m < self.param.message_modulus);
        LwePlaintext(self.ring.elem_from((m as f64 * self.delta).round() as u64))
    }

    fn decode(&self, LwePlaintext(pt): LwePlaintext<R::Elem>) -> u64 {
        let pt: u64 = self.ring.elem_to(pt);
        (pt as f64 / self.delta).round() as u64 % self.param.message_modulus
    }

    fn sk_gen(&self, rng: impl RngCore) -> LweSecretKeyOwned<i32> {
        LweSecretKey::sample(self.dimension(), self.param.sk_distribution, rng)
    }

    fn sk_encrypt(
        &self,
        sk: &LweSecretKeyOwned<i32>,
        pt: LwePlaintext<R::Elem>,
        rng: impl RngCore,
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

    fn decrypt(
        &self,
        sk: &LweSecretKeyOwned<i32>,
        ct: &LweCiphertextOwned<R::Elem>,
    ) -> LwePlaintext<R::Elem> {
        lwe::decrypt(self.ring(), sk, ct)
    }

    fn ks_key_gen(
        &self,
        sk_from: &LweSecretKeyOwned<i32>,
        sk_to: &LweSecretKeyOwned<i32>,
        rng: impl RngCore,
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

    fn key_switch(
        &self,
        ks_key: &LweKeySwitchKeyOwned<R::Elem>,
        ct_from: &LweCiphertextOwned<R::Elem>,
    ) -> LweCiphertextOwned<R::Elem> {
        let mut ct_to = LweCiphertext::allocate(ks_key.to_dimension());
        lwe::key_switch(self.ring(), &mut ct_to, ks_key, ct_from);
        ct_to
    }
}

fn test_param(ciphertext_modulus: impl Into<Modulus>) -> LweParam {
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
        let mut rng = thread_rng();
        let lwe = param.build::<R>();
        let sk = lwe.sk_gen(&mut rng);
        for m in 0..param.message_modulus {
            let pt = lwe.encode(m);
            let ct = lwe.sk_encrypt(&sk, pt, &mut rng);
            assert_eq!(m, lwe.decode(pt));
            assert_eq!(m, lwe.decode(lwe.decrypt(&sk, &ct)));
        }
    }

    run::<NoisyNativeRing>(test_param(Modulus::native()));
    run::<NoisyNonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<PrimeRing>(test_param(Prime::gen(50, 0)));
}

#[test]
fn key_switch() {
    fn run<R: RingOps>(param: LweParam) {
        let mut rng = thread_rng();
        let lwe_from = param.build::<R>();
        let lwe_to = param.dimension(2 * param.dimension).build::<R>();
        let sk_from = lwe_from.sk_gen(&mut rng);
        let sk_to = lwe_to.sk_gen(&mut rng);
        let ks_key = lwe_to.ks_key_gen(&sk_from, &sk_to, &mut rng);
        for m in 0..param.message_modulus {
            let pt = lwe_from.encode(m);
            let ct_from = lwe_from.sk_encrypt(&sk_from, pt, &mut rng);
            let ct_to = lwe_to.key_switch(&ks_key, &ct_from);
            assert_eq!(m, lwe_to.decode(lwe_to.decrypt(&sk_to, &ct_to)));
        }
    }

    run::<NoisyNativeRing>(test_param(Modulus::native()));
    run::<NoisyNonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<PrimeRing>(test_param(Prime::gen(50, 0)));
}
