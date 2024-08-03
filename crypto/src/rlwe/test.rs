use crate::{
    distribution::{NoiseDistribution, SecretKeyDistribution},
    lwe::{self, LweCiphertextOwned, LwePlaintext},
    rlwe::{
        self, RlweAutoKey, RlweAutoKeyOwned, RlweCiphertext, RlweCiphertextOwned, RlweKeySwitchKey,
        RlweKeySwitchKeyOwned, RlwePlaintext, RlwePlaintextOwned, RlwePublicKey,
        RlwePublicKeyOwned, RlweSecretKey, RlweSecretKeyOwned,
    },
};
use itertools::Itertools;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::{DistributionSized, Gaussian},
    izip_eq,
    modulus::{neg_mod, powers_mod, Modulus, PowerOfTwo, Prime},
    ring::{
        power_of_two::{NativeRing, NonNativePowerOfTwoRing},
        prime::PrimeRing,
        RingOps,
    },
};
use rand::{distributions::Uniform, thread_rng, RngCore};

#[derive(Clone, Copy, Debug)]
struct RlweParam {
    message_modulus: u64,
    ciphertext_modulus: Modulus,
    ring_size: usize,
    sk_distribution: SecretKeyDistribution,
    noise_distribution: NoiseDistribution,
    ks_decomposition_param: DecompositionParam,
}

impl RlweParam {
    fn build<R: RingOps>(self) -> Rlwe<R> {
        let delta = self.ciphertext_modulus.to_f64() / self.message_modulus as f64;
        let ring = RingOps::new(self.ciphertext_modulus, self.ring_size);
        Rlwe {
            param: self,
            delta,
            ring,
        }
    }
}

#[derive(Clone, Debug)]
struct Rlwe<R: RingOps> {
    param: RlweParam,
    delta: f64,
    ring: R,
}

impl<R: RingOps> Rlwe<R> {
    fn ring(&self) -> &R {
        &self.ring
    }

    fn ring_size(&self) -> usize {
        self.ring().ring_size()
    }

    fn encode(&self, m: &[u64]) -> RlwePlaintextOwned<R::Elem> {
        let encode = |m: &_| {
            assert!(*m < self.param.message_modulus);
            self.ring.elem_from((*m as f64 * self.delta).round() as u64)
        };
        let mut pt = RlwePlaintext::allocate(self.ring_size());
        izip_eq!(pt.as_mut(), m).for_each(|(pt, m)| *pt = encode(m));
        pt
    }

    fn decode(&self, pt: &RlwePlaintextOwned<R::Elem>) -> Vec<u64> {
        pt.as_ref()
            .iter()
            .map(|pt| self.decode_lwe(LwePlaintext(*pt)))
            .collect()
    }

    fn decode_lwe(&self, pt: LwePlaintext<R::Elem>) -> u64 {
        let pt: u64 = self.ring.elem_to(pt.0);
        (pt as f64 / self.delta).round() as u64 % self.param.message_modulus
    }

    fn sk_gen(&self, rng: impl RngCore) -> RlweSecretKeyOwned<i32> {
        RlweSecretKey::sample(self.ring_size(), self.param.sk_distribution, rng)
    }

    fn pk_gen(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        rng: impl RngCore,
    ) -> RlwePublicKeyOwned<R::Elem> {
        let mut pk = RlwePublicKey::allocate(self.ring_size());
        rlwe::pk_gen(self.ring(), &mut pk, sk, self.param.noise_distribution, rng);
        pk
    }

    fn sk_encrypt(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        pt: &RlwePlaintextOwned<R::Elem>,
        rng: impl RngCore,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = RlweCiphertext::allocate(self.ring_size());
        rlwe::sk_encrypt(
            self.ring(),
            &mut ct,
            sk,
            pt,
            self.param.noise_distribution,
            rng,
        );
        ct
    }

    fn pk_encrypt(
        &self,
        pk: &RlwePublicKeyOwned<R::Elem>,
        pt: &RlwePlaintextOwned<R::Elem>,
        rng: impl RngCore,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = RlweCiphertext::allocate(self.ring_size());
        rlwe::pk_encrypt(self.ring(), &mut ct, pk, pt, rng);
        ct
    }

    fn decrypt(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        ct: &RlweCiphertextOwned<R::Elem>,
    ) -> RlwePlaintextOwned<R::Elem> {
        let mut pt = RlwePlaintext::allocate(self.ring_size());
        rlwe::decrypt(self.ring(), &mut pt, sk, ct);
        pt
    }

    fn decrypt_lwe(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        ct: &LweCiphertextOwned<R::Elem>,
    ) -> LwePlaintext<R::Elem> {
        lwe::decrypt(self.ring(), sk.as_view(), ct)
    }

    fn ks_key_gen(
        &self,
        sk_from: &RlweSecretKeyOwned<i32>,
        sk_to: &RlweSecretKeyOwned<i32>,
        rng: impl RngCore,
    ) -> RlweKeySwitchKeyOwned<R::Elem> {
        let mut ks_key =
            RlweKeySwitchKey::allocate(self.ring_size(), self.param.ks_decomposition_param);
        rlwe::ks_key_gen(
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
        ks_key: &RlweKeySwitchKeyOwned<R::Elem>,
        ct_from: &RlweCiphertextOwned<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct_to = RlweCiphertext::allocate(self.ring_size());
        rlwe::key_switch(self.ring(), &mut ct_to, ks_key, ct_from);
        ct_to
    }

    fn sample_extract(
        &self,
        ct_rlwe: &RlweCiphertextOwned<R::Elem>,
        idx: usize,
    ) -> LweCiphertextOwned<R::Elem> {
        let mut ct_lwe = LweCiphertextOwned::allocate(self.ring_size());
        rlwe::sample_extract(self.ring(), &mut ct_lwe, ct_rlwe, idx);
        ct_lwe
    }

    fn auto_key_gen(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        k: i64,
        rng: impl RngCore,
    ) -> RlweAutoKeyOwned<R::Elem> {
        let mut auto_key =
            RlweAutoKey::allocate(self.ring_size(), self.param.ks_decomposition_param, k);
        rlwe::auto_key_gen(
            self.ring(),
            &mut auto_key,
            sk,
            self.param.noise_distribution,
            rng,
        );
        auto_key
    }

    fn automorphism(
        &self,
        auto_key: &RlweAutoKeyOwned<R::Elem>,
        ct: &RlweCiphertextOwned<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct_auto = RlweCiphertext::allocate(self.ring_size());
        rlwe::automorphism(self.ring(), &mut ct_auto, auto_key, ct);
        ct_auto
    }
}

fn test_param(ciphertext_modulus: impl Into<Modulus>) -> RlweParam {
    RlweParam {
        message_modulus: 1 << 6,
        ciphertext_modulus: ciphertext_modulus.into(),
        ring_size: 256,
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
    fn run<R: RingOps>(param: RlweParam) {
        let mut rng = thread_rng();
        let rlwe = param.build::<R>();
        let sk = rlwe.sk_gen(&mut rng);
        let pk = rlwe.pk_gen(&sk, &mut rng);
        for _ in 0..100 {
            let m = Uniform::new(0, param.message_modulus).sample_vec(param.ring_size, &mut rng);
            let pt = rlwe.encode(&m);
            let ct_sk = rlwe.sk_encrypt(&sk, &pt, &mut rng);
            let ct_pk = rlwe.pk_encrypt(&pk, &pt, &mut rng);
            assert_eq!(m, rlwe.decode(&pt));
            assert_eq!(m, rlwe.decode(&rlwe.decrypt(&sk, &ct_sk)));
            assert_eq!(m, rlwe.decode(&rlwe.decrypt(&sk, &ct_pk)));
        }
    }

    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}

#[test]
fn key_switch() {
    fn run<R: RingOps>(param: RlweParam) {
        let mut rng = thread_rng();
        let rlwe = param.build::<R>();
        let sk_from = rlwe.sk_gen(&mut rng);
        let sk_to = rlwe.sk_gen(&mut rng);
        let ks_key = rlwe.ks_key_gen(&sk_from, &sk_to, &mut rng);
        for _ in 0..100 {
            let m = Uniform::new(0, param.message_modulus).sample_vec(param.ring_size, &mut rng);
            let pt = rlwe.encode(&m);
            let ct_from = rlwe.sk_encrypt(&sk_from, &pt, &mut rng);
            let ct_to = rlwe.key_switch(&ks_key, &ct_from);
            assert_eq!(m, rlwe.decode(&rlwe.decrypt(&sk_to, &ct_to)));
        }
    }

    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}

#[test]
fn sample_extract() {
    fn run<R: RingOps>(param: RlweParam) {
        let mut rng = thread_rng();
        let rlwe = param.build::<R>();
        let sk = rlwe.sk_gen(&mut rng);
        let m = Uniform::new(0, param.message_modulus).sample_vec(param.ring_size, &mut rng);
        let pt = rlwe.encode(&m);
        let ct_rlwe = rlwe.sk_encrypt(&sk, &pt, &mut rng);
        for (idx, m) in m.iter().enumerate() {
            let ct_lwe = rlwe.sample_extract(&ct_rlwe, idx);
            assert_eq!(*m, rlwe.decode_lwe(rlwe.decrypt_lwe(&sk, &ct_lwe)));
        }
    }

    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}

#[test]
fn automorphism() {
    fn run<R: RingOps>(param: RlweParam) {
        let mut rng = thread_rng();
        let rlwe = param.build::<R>();
        let sk = rlwe.sk_gen(&mut rng);
        for k in powers_mod(5, 2 * rlwe.ring_size() as u64).take(rlwe.ring_size() / 2) {
            let auto_key = rlwe.auto_key_gen(&sk, k as _, &mut rng);
            let m = Uniform::new(0, param.message_modulus).sample_vec(param.ring_size, &mut rng);
            let pt = rlwe.encode(&m);
            let ct = rlwe.sk_encrypt(&sk, &pt, &mut rng);
            let ct_auto = rlwe.automorphism(&auto_key, &ct);
            let m_auto = {
                let neg = |v: &_| neg_mod(*v, param.message_modulus);
                auto_key.map().apply(&m, neg).collect_vec()
            };
            assert_eq!(m_auto, rlwe.decode(&rlwe.decrypt(&sk, &ct_auto)));
        }
    }

    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}
