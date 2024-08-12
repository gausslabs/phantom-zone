use crate::{
    core::{
        rgsw::{self, RgswCiphertext, RgswCiphertextOwned},
        rlwe::{
            self,
            test::{Rlwe, RlweParam},
            RlweCiphertextOwned, RlwePlaintext, RlwePlaintextOwned, RlweSecretKeyOwned,
        },
    },
    util::rng::{test::StdLweRng, LweRng},
};
use core::ops::Deref;
use phantom_zone_math::{
    decomposer::Decomposer,
    distribution::Sampler,
    izip_eq,
    modulus::{ElemFrom, Modulus, NonNativePowerOfTwo, Prime},
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
};
use rand::RngCore;

#[derive(Clone, Copy, Debug)]
pub struct RgswParam {
    pub rlwe: RlweParam,
    pub decomposition_log_base: usize,
    pub decomposition_level_a: usize,
    pub decomposition_level_b: usize,
}

impl Deref for RgswParam {
    type Target = RlweParam;

    fn deref(&self) -> &Self::Target {
        &self.rlwe
    }
}

impl RgswParam {
    pub fn build<R: RingOps>(self) -> Rgsw<R> {
        let rlwe = self.rlwe.build();
        Rgsw { param: self, rlwe }
    }
}

#[derive(Clone, Debug)]
pub struct Rgsw<R: RingOps> {
    param: RgswParam,
    rlwe: Rlwe<R>,
}

impl<R: RingOps> Rgsw<R> {
    pub fn rlwe(&self) -> &Rlwe<R> {
        &self.rlwe
    }

    pub fn ring(&self) -> &R {
        self.rlwe.ring()
    }

    pub fn ring_size(&self) -> usize {
        self.ring().ring_size()
    }

    pub fn message_ring(&self) -> &NonNativePowerOfTwoRing {
        self.rlwe.message_ring()
    }

    pub fn encode(&self, m: &[u64]) -> RlwePlaintextOwned<R::Elem> {
        let mut pt = RlwePlaintext::allocate(self.ring_size());
        self.ring().slice_elem_from(pt.as_mut(), m);
        pt
    }

    pub fn decode(&self, pt: &RlwePlaintextOwned<R::Elem>) -> Vec<u64> {
        let decode = |pt| self.message_ring().elem_from(self.ring().to_u64(pt));
        let mut m = vec![0; self.ring_size()];
        izip_eq!(&mut m, pt.as_ref()).for_each(|(m, pt)| *m = decode(*pt));
        m
    }

    pub fn message_poly_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut scratch = self.message_ring().allocate_scratch(0, 2, 0);
        let mut c = self.message_ring().allocate_poly();
        self.message_ring()
            .poly_mul(&mut c, a, b, scratch.borrow_mut());
        c
    }

    pub fn sk_encrypt(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        pt: &RlwePlaintextOwned<R::Elem>,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> RgswCiphertextOwned<R::Elem> {
        let mut ct = RgswCiphertext::allocate(
            self.ring_size(),
            self.param.decomposition_log_base,
            self.param.decomposition_level_a,
            self.param.decomposition_level_b,
        );
        let mut scratch = self.ring().allocate_scratch(0, 2, 0);
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

    pub fn decrypt(
        &self,
        sk: &RlweSecretKeyOwned<i32>,
        ct: &RgswCiphertextOwned<R::Elem>,
    ) -> RlwePlaintextOwned<R::Elem> {
        let decomposer_b = R::Decomposer::new(self.ring().modulus(), ct.decomposition_param_b());
        let log_beta_last = decomposer_b.log_gadget_iter().last().unwrap();
        let rounding_shr = |pt: u64| pt.wrapping_add((1 << log_beta_last) >> 1) >> log_beta_last;
        let ct = ct.b_ct_iter().last().unwrap().cloned();
        let mut pt = self.rlwe().decrypt(sk, &ct);
        pt.as_mut()
            .iter_mut()
            .for_each(|pt| *pt = self.ring().elem_from(rounding_shr(self.ring().to_u64(*pt))));
        pt
    }

    pub fn rlwe_by_rgsw(
        &self,
        ct_rlwe: &RlweCiphertextOwned<R::Elem>,
        ct_rgsw: &RgswCiphertextOwned<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct_rlwe = ct_rlwe.clone();
        let mut scratch = self.ring().allocate_scratch(2, 4, 0);
        rgsw::rlwe_by_rgsw_in_place(self.ring(), &mut ct_rlwe, ct_rgsw, scratch.borrow_mut());
        ct_rlwe
    }

    pub fn prepare_rgsw(
        &self,
        ct: &RgswCiphertextOwned<R::Elem>,
    ) -> RgswCiphertextOwned<R::EvalPrep> {
        let mut ct_b_prep = RgswCiphertext::allocate_eval(
            self.ring().ring_size(),
            self.ring().eval_size(),
            ct.decomposition_log_base(),
            ct.decomposition_level_a(),
            ct.decomposition_level_b(),
        );
        let mut scratch = self.ring().allocate_scratch(0, 1, 0);
        rgsw::prepare_rgsw(self.ring(), &mut ct_b_prep, ct, scratch.borrow_mut());
        ct_b_prep
    }

    pub fn rlwe_by_rgsw_prep(
        &self,
        ct_rlwe: &RlweCiphertextOwned<R::Elem>,
        ct_rgsw: &RgswCiphertextOwned<R::EvalPrep>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct_rlwe = ct_rlwe.clone();
        let mut scratch = self.ring().allocate_scratch(2, 3, 0);
        rgsw::rlwe_by_rgsw_prep_in_place(self.ring(), &mut ct_rlwe, ct_rgsw, scratch.borrow_mut());
        ct_rlwe
    }

    pub fn rgsw_by_rgsw(
        &self,
        ct_a: &RgswCiphertextOwned<R::Elem>,
        ct_b: &RgswCiphertextOwned<R::Elem>,
    ) -> RgswCiphertextOwned<R::Elem> {
        let mut ct_a = ct_a.clone();
        let mut scratch = self
            .ring()
            .allocate_scratch(2, 3, 2 * ct_b.ct_iter().count());
        rgsw::rgsw_by_rgsw_in_place(self.ring(), &mut ct_a, ct_b, scratch.borrow_mut());
        ct_a
    }
}

pub fn test_param(ciphertext_modulus: impl Into<Modulus>) -> RgswParam {
    RgswParam {
        rlwe: rlwe::test::test_param(ciphertext_modulus),
        decomposition_log_base: 20,
        decomposition_level_a: 2,
        decomposition_level_b: 2,
    }
}

#[test]
fn rlwe_by_rgsw() {
    fn run<R: RingOps>(param: RgswParam) {
        let mut rng = StdLweRng::from_entropy();
        let rgsw = param.build::<R>();
        let rlwe = &rgsw.rlwe;
        let sk = rlwe.sk_gen();
        for _ in 0..100 {
            let m_rlwe = rgsw.message_ring().sample_uniform_poly(rng.noise());
            let m_rgsw = rgsw.message_ring().sample_uniform_poly(rng.noise());
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
    run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NoisyPrimeRing>(test_param(Prime::gen(50, 9)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}

#[test]
fn rgsw_by_rgsw() {
    fn run<R: RingOps>(param: RgswParam) {
        let mut rng = StdLweRng::from_entropy();
        let rgsw = param.build::<R>();
        let rlwe = &rgsw.rlwe;
        let sk = rlwe.sk_gen();
        for _ in 0..100 {
            let m_a = rgsw.message_ring().sample_uniform_poly(rng.noise());
            let m_b = rgsw.message_ring().sample_uniform_poly(rng.noise());
            let m_c = rgsw.message_poly_mul(&m_a, &m_b);
            let ct_a = rgsw.sk_encrypt(&sk, &rgsw.encode(&m_a), &mut rng);
            let ct_b = rgsw.sk_encrypt(&sk, &rgsw.encode(&m_b), &mut rng);
            let ct_c = rgsw.rgsw_by_rgsw(&ct_a, &ct_b);
            assert_eq!(m_c, rgsw.decode(&rgsw.decrypt(&sk, &ct_c)));
        }
    }

    run::<NoisyNativeRing>(test_param(Modulus::native()));
    run::<NoisyNonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NativeRing>(test_param(Modulus::native()));
    run::<NonNativePowerOfTwoRing>(test_param(NonNativePowerOfTwo::new(50)));
    run::<NoisyPrimeRing>(test_param(Prime::gen(50, 9)));
    run::<PrimeRing>(test_param(Prime::gen(50, 9)));
}
