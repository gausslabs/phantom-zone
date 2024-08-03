use crate::{
    distribution::{NoiseDistribution, SecretKeyDistribution},
    lwe::{self, LweCiphertextOwned, LweCiphertextView, LwePlaintext},
    rlwe::{
        self, RlweAutoKey, RlweAutoKeyOwned, RlweAutoKeyView, RlweCiphertext, RlweCiphertextOwned,
        RlweCiphertextView, RlweKeySwitchKey, RlweKeySwitchKeyOwned, RlweKeySwitchKeyView,
        RlwePlaintext, RlwePlaintextOwned, RlwePlaintextView, RlwePublicKey, RlwePublicKeyOwned,
        RlwePublicKeyView, RlweSecretKey, RlweSecretKeyOwned, RlweSecretKeyView,
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

    fn decode(&self, pt: RlwePlaintextView<R::Elem>) -> Vec<u64> {
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
        let mut sk = RlweSecretKey::allocate(self.ring_size());
        rlwe::sk_gen(sk.as_mut_view(), self.param.sk_distribution, rng);
        sk
    }

    fn pk_gen(&self, sk: RlweSecretKeyView<i32>, rng: impl RngCore) -> RlwePublicKeyOwned<R::Elem> {
        let mut pk = RlwePublicKey::allocate(self.ring_size());
        rlwe::pk_gen(
            self.ring(),
            pk.as_mut_view(),
            sk,
            self.param.noise_distribution,
            rng,
        );
        pk
    }

    fn sk_encrypt(
        &self,
        sk: RlweSecretKeyView<i32>,
        pt: RlwePlaintextView<R::Elem>,
        rng: impl RngCore,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = RlweCiphertext::allocate(self.ring_size());
        rlwe::sk_encrypt(
            self.ring(),
            ct.as_mut_view(),
            sk.as_view(),
            pt,
            self.param.noise_distribution,
            rng,
        );
        ct
    }

    fn pk_encrypt(
        &self,
        pk: RlwePublicKeyView<R::Elem>,
        pt: RlwePlaintextView<R::Elem>,
        rng: impl RngCore,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = RlweCiphertext::allocate(self.ring_size());
        rlwe::pk_encrypt(self.ring(), ct.as_mut_view(), pk.as_view(), pt, rng);
        ct
    }

    fn decrypt(
        &self,
        sk: RlweSecretKeyView<i32>,
        ct: RlweCiphertextView<R::Elem>,
    ) -> RlwePlaintextOwned<R::Elem> {
        let mut pt = RlwePlaintext::allocate(self.ring_size());
        rlwe::decrypt(self.ring(), pt.as_mut_view(), sk.as_view(), ct.as_view());
        pt
    }

    fn decrypt_lwe(
        &self,
        sk: RlweSecretKeyView<i32>,
        ct: LweCiphertextView<R::Elem>,
    ) -> LwePlaintext<R::Elem> {
        lwe::decrypt(self.ring(), sk.as_view().into(), ct.as_view())
    }

    fn ks_key_gen(
        &self,
        sk_from: RlweSecretKeyView<i32>,
        sk_to: RlweSecretKeyView<i32>,
        rng: impl RngCore,
    ) -> RlweKeySwitchKeyOwned<R::Elem> {
        let mut ks_key =
            RlweKeySwitchKey::allocate(self.ring_size(), self.param.ks_decomposition_param);
        rlwe::ks_key_gen(
            self.ring(),
            ks_key.as_mut_view(),
            sk_from,
            sk_to,
            self.param.noise_distribution,
            rng,
        );
        ks_key
    }

    fn key_switch(
        &self,
        ks_key: RlweKeySwitchKeyView<R::Elem>,
        ct_from: RlweCiphertextView<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct_to = RlweCiphertext::allocate(self.ring_size());
        rlwe::key_switch(self.ring(), ct_to.as_mut_view(), ks_key, ct_from);
        ct_to
    }

    fn sample_extract(
        &self,
        ct_rlwe: RlweCiphertextView<R::Elem>,
        idx: usize,
    ) -> LweCiphertextOwned<R::Elem> {
        let mut ct_lwe = LweCiphertextOwned::allocate(self.ring_size());
        rlwe::sample_extract(self.ring(), ct_lwe.as_mut_view(), ct_rlwe.as_view(), idx);
        ct_lwe
    }

    fn auto_key_gen(
        &self,
        sk: RlweSecretKeyView<i32>,
        k: i64,
        rng: impl RngCore,
    ) -> RlweAutoKeyOwned<R::Elem> {
        let mut auto_key =
            RlweAutoKey::allocate(self.ring_size(), self.param.ks_decomposition_param, k);
        rlwe::auto_key_gen(
            self.ring(),
            auto_key.as_mut_view(),
            sk,
            self.param.noise_distribution,
            rng,
        );
        auto_key
    }

    fn automorphism(
        &self,
        auto_key: RlweAutoKeyView<R::Elem>,
        ct: RlweCiphertextView<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct_auto = RlweCiphertext::allocate(self.ring_size());
        rlwe::automorphism(self.ring(), ct_auto.as_mut_view(), auto_key, ct);
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
        let pk = rlwe.pk_gen(sk.as_view(), &mut rng);
        for _ in 0..100 {
            let m = Uniform::new(0, param.message_modulus).sample_vec(param.ring_size, &mut rng);
            let pt = rlwe.encode(&m);
            let ct_sk = rlwe.sk_encrypt(sk.as_view(), pt.as_view(), &mut rng);
            let ct_pk = rlwe.pk_encrypt(pk.as_view(), pt.as_view(), &mut rng);
            assert_eq!(m, rlwe.decode(pt.as_view()));
            assert_eq!(
                m,
                rlwe.decode(rlwe.decrypt(sk.as_view(), ct_sk.as_view()).as_view())
            );
            assert_eq!(
                m,
                rlwe.decode(rlwe.decrypt(sk.as_view(), ct_pk.as_view()).as_view())
            );
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
        let ks_key = rlwe.ks_key_gen(sk_from.as_view(), sk_to.as_view(), &mut rng);
        for _ in 0..100 {
            let m = Uniform::new(0, param.message_modulus).sample_vec(param.ring_size, &mut rng);
            let pt = rlwe.encode(&m);
            let ct_from = rlwe.sk_encrypt(sk_from.as_view(), pt.as_view(), &mut rng);
            let ct_to = rlwe.key_switch(ks_key.as_view(), ct_from.as_view());
            assert_eq!(
                m,
                rlwe.decode(rlwe.decrypt(sk_to.as_view(), ct_to.as_view()).as_view())
            );
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
        let ct_rlwe = rlwe.sk_encrypt(sk.as_view(), pt.as_view(), &mut rng);
        for (idx, m) in m.iter().enumerate() {
            let ct_lwe = rlwe.sample_extract(ct_rlwe.as_view(), idx);
            assert_eq!(
                *m,
                rlwe.decode_lwe(rlwe.decrypt_lwe(sk.as_view(), ct_lwe.as_view()))
            );
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
            let auto_key = rlwe.auto_key_gen(sk.as_view(), k as _, &mut rng);
            let m = Uniform::new(0, param.message_modulus).sample_vec(param.ring_size, &mut rng);
            let pt = rlwe.encode(&m);
            let ct = rlwe.sk_encrypt(sk.as_view(), pt.as_view(), &mut rng);
            let ct_auto = rlwe.automorphism(auto_key.as_view(), ct.as_view());
            let m_auto = {
                let neg = |v: &_| neg_mod(*v, param.message_modulus);
                auto_key.map().apply(&m, neg).collect_vec()
            };
            assert_eq!(
                m_auto,
                rlwe.decode(rlwe.decrypt(sk.as_view(), ct_auto.as_view()).as_view())
            );
        }
    }

    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}
