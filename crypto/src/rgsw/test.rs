use crate::{
    rgsw::{self, RgswCiphertext, RgswCiphertextOwned},
    rlwe::{
        self,
        test::{Rlwe, RlweParam},
        RlweCiphertextOwned, RlwePlaintext, RlwePlaintextOwned, RlweSecretKeyOwned,
    },
};
use core::ops::Deref;
use phantom_zone_math::{
    distribution::Sampler,
    izip_eq,
    modulus::{Modulus, PowerOfTwo, Prime},
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
};
use rand::{thread_rng, RngCore};

#[derive(Clone, Copy, Debug)]
struct RgswParam {
    rlwe: RlweParam,
    decomposition_log_base: usize,
    decomposition_level_a: usize,
    decomposition_level_b: usize,
}

impl Deref for RgswParam {
    type Target = RlweParam;

    fn deref(&self) -> &Self::Target {
        &self.rlwe
    }
}

impl RgswParam {
    fn build<R: RingOps>(self) -> Rgsw<R> {
        let rlwe = self.rlwe.build();
        Rgsw { param: self, rlwe }
    }
}

#[derive(Clone, Debug)]
struct Rgsw<R: RingOps> {
    param: RgswParam,
    rlwe: Rlwe<R>,
}

impl<R: RingOps> Rgsw<R> {
    fn ring(&self) -> &R {
        self.rlwe.ring()
    }

    fn ring_size(&self) -> usize {
        self.ring().ring_size()
    }

    fn message_ring(&self) -> &NonNativePowerOfTwoRing {
        self.rlwe.message_ring()
    }

    fn encode(&self, m: &[u64]) -> RlwePlaintextOwned<R::Elem> {
        let encode = |m: &_| self.ring().elem_from(*m);
        let mut pt = RlwePlaintext::allocate(self.ring_size());
        izip_eq!(pt.as_mut(), m).for_each(|(pt, m)| *pt = encode(m));
        pt
    }

    fn message_poly_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut scratch = self.message_ring().allocate_scratch(0, 2);
        let mut c = self.message_ring().allocate_poly();
        self.message_ring()
            .poly_mul(&mut c, a, b, scratch.borrow_mut());
        c
    }

    fn sk_encrypt(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        pt: &RlwePlaintextOwned<R::Elem>,
        rng: impl RngCore,
    ) -> RgswCiphertextOwned<R::Elem> {
        let mut ct = RgswCiphertext::allocate(
            self.ring_size(),
            self.param.decomposition_log_base,
            self.param.decomposition_level_a,
            self.param.decomposition_level_b,
        );
        let mut scratch = self.ring().allocate_scratch(0, 2);
        rgsw::sk_encrypt(
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

    fn rlwe_by_rgsw(
        &self,
        ct_rlwe: &RlweCiphertextOwned<R::Elem>,
        ct_rgsw: &RgswCiphertextOwned<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct_rlwe = ct_rlwe.clone();
        let mut scratch = self.ring().allocate_scratch(2, 4);
        rgsw::rlwe_by_rgsw(self.ring(), &mut ct_rlwe, ct_rgsw, scratch.borrow_mut());
        ct_rlwe
    }

    fn prepare_rgsw(&self, ct: &RgswCiphertextOwned<R::Elem>) -> RgswCiphertextOwned<R::EvalPrep> {
        let mut scratch = self.ring().allocate_scratch(0, 1);
        rgsw::prepare_rgsw(self.ring(), ct, scratch.borrow_mut())
    }

    fn rlwe_by_rgsw_prep(
        &self,
        ct_rlwe: &RlweCiphertextOwned<R::Elem>,
        ct_rgsw: &RgswCiphertextOwned<R::EvalPrep>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct_rlwe = ct_rlwe.clone();
        let mut scratch = self.ring().allocate_scratch(2, 3);
        rgsw::rlwe_by_rgsw_prep(self.ring(), &mut ct_rlwe, ct_rgsw, scratch.borrow_mut());
        ct_rlwe
    }
}

fn test_param(ciphertext_modulus: impl Into<Modulus>) -> RgswParam {
    RgswParam {
        rlwe: rlwe::test::test_param(ciphertext_modulus),
        decomposition_log_base: 8,
        decomposition_level_a: 6,
        decomposition_level_b: 5,
    }
}

#[test]
fn rlwe_by_rgsw() {
    fn run<R: RingOps>(param: RgswParam) {
        let mut rng = thread_rng();
        let rgsw = param.build::<R>();
        let rlwe = &rgsw.rlwe;
        let sk = rlwe.sk_gen(&mut rng);
        for _ in 0..100 {
            let m_rlwe = rgsw.message_ring().sample_uniform_poly(&mut rng);
            let m_rgsw = rgsw.message_ring().sample_uniform_poly(&mut rng);
            let m = rgsw.message_poly_mul(&m_rlwe, &m_rgsw);
            let ct_rlwe = rlwe.sk_encrypt(&sk, &rlwe.encode(&m_rlwe), &mut rng);
            let ct_rgsw = rgsw.sk_encrypt(&sk, &rgsw.encode(&m_rgsw), &mut rng);
            let ct_rgsw_prep = rgsw.prepare_rgsw(&ct_rgsw);
            let ct = rgsw.rlwe_by_rgsw(&ct_rlwe, &ct_rgsw);
            assert_eq!(m, rlwe.decode(&rlwe.decrypt(&sk, &ct)));
            let ct = rgsw.rlwe_by_rgsw_prep(&ct_rlwe, &ct_rgsw_prep);
            assert_eq!(m, rlwe.decode(&rlwe.decrypt(&sk, &ct)));
        }
    }

    run::<NoisyNativeRing>(test_param(Modulus::native()));
    run::<NoisyNonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(PowerOfTwo::new(50)));
    run::<NoisyPrimeRing>(test_param(Prime::gen(50, 9)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}
