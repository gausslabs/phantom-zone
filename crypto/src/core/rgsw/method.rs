use crate::{
    core::{
        rgsw::structure::{RgswCiphertext, RgswCiphertextMutView, RgswCiphertextView},
        rlwe::{
            decomposed_fma, decomposed_fma_prep, pk_encrypt_zero, sk_encrypt_zero, RlweCiphertext,
            RlweCiphertextMutView, RlwePlaintextView, RlwePublicKeyView, RlweSecretKeyView,
        },
    },
    util::{
        distribution::{NoiseDistribution, SecretKeyDistribution},
        rng::LweRng,
    },
};
use phantom_zone_math::{
    decomposer::Decomposer, izip_eq, modulus::ElemFrom, ring::RingOps, util::scratch::Scratch,
};
use rand::RngCore;

pub fn sk_encrypt<'a, 'b, 'c, R, T>(
    ring: &R,
    ct: impl Into<RgswCiphertextMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    pt: impl Into<RlwePlaintextView<'c, R::Elem>>,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let (mut ct, sk, pt) = (ct.into(), sk.into(), pt.into());
    let decomposer_a = R::Decomposer::new(ring.modulus(), ct.decomposition_param_a());
    let decomposer_b = R::Decomposer::new(ring.modulus(), ct.decomposition_param_b());
    izip_eq!(ct.a_ct_iter_mut(), decomposer_a.gadget_iter()).for_each(|(mut ct, beta_i)| {
        let scratch = scratch.reborrow();
        sk_encrypt_zero(ring, &mut ct, sk, noise_distribution, scratch, rng);
        ring.slice_scalar_fma(ct.a_mut(), pt.as_ref(), &beta_i);
    });
    izip_eq!(ct.b_ct_iter_mut(), decomposer_b.gadget_iter()).for_each(|(mut ct, beta_i)| {
        let scratch = scratch.reborrow();
        sk_encrypt_zero(ring, &mut ct, sk, noise_distribution, scratch, rng);
        ring.slice_scalar_fma(ct.b_mut(), pt.as_ref(), &beta_i);
    });
}

pub fn pk_encrypt<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    ct: impl Into<RgswCiphertextMutView<'a, R::Elem>>,
    pk: impl Into<RlwePublicKeyView<'b, R::Elem>>,
    pt: impl Into<RlwePlaintextView<'c, R::Elem>>,
    u_distribution: SecretKeyDistribution,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) {
    let (mut ct, pk, pt) = (ct.into(), pk.into(), pt.into());
    let decomposer_a = R::Decomposer::new(ring.modulus(), ct.decomposition_param_a());
    let decomposer_b = R::Decomposer::new(ring.modulus(), ct.decomposition_param_b());
    izip_eq!(ct.a_ct_iter_mut(), decomposer_a.gadget_iter()).for_each(|(mut ct, beta_i)| {
        let scratch = scratch.reborrow();
        pk_encrypt_zero(
            ring,
            &mut ct,
            pk,
            u_distribution,
            noise_distribution,
            scratch,
            rng,
        );
        ring.slice_scalar_fma(ct.a_mut(), pt.as_ref(), &beta_i);
    });
    izip_eq!(ct.b_ct_iter_mut(), decomposer_b.gadget_iter()).for_each(|(mut ct, beta_i)| {
        let scratch = scratch.reborrow();
        pk_encrypt_zero(
            ring,
            &mut ct,
            pk,
            u_distribution,
            noise_distribution,
            scratch,
            rng,
        );
        ring.slice_scalar_fma(ct.b_mut(), pt.as_ref(), &beta_i);
    });
}

pub fn rlwe_by_rgsw_in_place<'a, 'b, R: RingOps>(
    ring: &R,
    ct_rlwe: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    ct_rgsw: impl Into<RgswCiphertextView<'b, R::Elem>>,
    mut scratch: Scratch,
) {
    let (mut ct_rlwe, ct_rgsw) = (ct_rlwe.into(), ct_rgsw.into());
    let mut ct_eval = RlweCiphertext::scratch(ring.ring_size(), ring.eval_size(), &mut scratch);
    decomposed_fma::<_, true>(
        ring,
        ct_rgsw.decomposition_param_a(),
        &mut ct_eval,
        ct_rlwe.a(),
        ct_rgsw.a_ct_iter(),
        scratch.reborrow(),
    );
    decomposed_fma::<_, false>(
        ring,
        ct_rgsw.decomposition_param_b(),
        &mut ct_eval,
        ct_rlwe.b(),
        ct_rgsw.b_ct_iter(),
        scratch.reborrow(),
    );
    ring.backward_normalized(ct_rlwe.a_mut(), ct_eval.a_mut());
    ring.backward_normalized(ct_rlwe.b_mut(), ct_eval.b_mut());
}

pub fn prepare_rgsw<'a, 'b, R: RingOps>(
    ring: &R,
    ct_prep: impl Into<RgswCiphertextMutView<'a, R::EvalPrep>>,
    ct: impl Into<RgswCiphertextView<'b, R::Elem>>,
    mut scratch: Scratch,
) {
    let (mut ct_prep, ct) = (ct_prep.into(), ct.into());
    let eval = ring.take_eval(&mut scratch);
    izip_eq!(ct_prep.ct_iter_mut(), ct.ct_iter()).for_each(|(mut ct_prep, ct)| {
        ring.forward_normalized(eval, ct.a());
        ring.eval_prepare(ct_prep.a_mut(), eval);
        ring.forward_normalized(eval, ct.b());
        ring.eval_prepare(ct_prep.b_mut(), eval);
    });
}

pub fn rlwe_by_rgsw_prep_in_place<'a, 'b, R: RingOps>(
    ring: &R,
    ct_rlwe: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    ct_rgsw: impl Into<RgswCiphertextView<'b, R::EvalPrep>>,
    mut scratch: Scratch,
) {
    let (mut ct_rlwe, ct_rgsw) = (ct_rlwe.into(), ct_rgsw.into());
    let mut ct_eval = RlweCiphertext::scratch(ring.ring_size(), ring.eval_size(), &mut scratch);
    decomposed_fma_prep::<_, true>(
        ring,
        ct_rgsw.decomposition_param_a(),
        &mut ct_eval,
        ct_rlwe.a(),
        ct_rgsw.a_ct_iter(),
        scratch.reborrow(),
    );
    decomposed_fma_prep::<_, false>(
        ring,
        ct_rgsw.decomposition_param_b(),
        &mut ct_eval,
        ct_rlwe.b(),
        ct_rgsw.b_ct_iter(),
        scratch.reborrow(),
    );
    ring.backward(ct_rlwe.a_mut(), ct_eval.a_mut());
    ring.backward(ct_rlwe.b_mut(), ct_eval.b_mut());
}

pub fn rgsw_by_rgsw_in_place<'a, 'b, R: RingOps>(
    ring: &R,
    ct_a: impl Into<RgswCiphertextMutView<'a, R::Elem>>,
    ct_b: impl Into<RgswCiphertextView<'b, R::Elem>>,
    mut scratch: Scratch,
) {
    let (mut ct_a, ct_b) = (ct_a.into(), ct_b.into());
    let mut ct_b_prep = RgswCiphertext::scratch(
        ring.ring_size(),
        ring.eval_size(),
        ct_b.decomposition_param(),
        &mut scratch,
    );
    prepare_rgsw(ring, &mut ct_b_prep, ct_b, scratch.reborrow());
    ct_a.ct_iter_mut()
        .for_each(|ct| rlwe_by_rgsw_prep_in_place(ring, ct, &ct_b_prep, scratch.reborrow()));
}
