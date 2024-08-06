use crate::{
    distribution::NoiseDistribution,
    rgsw::structure::{
        RgswCiphertext, RgswCiphertextMutView, RgswCiphertextOwned, RgswCiphertextView,
    },
    rlwe::{
        decomposed_fma, decomposed_fma_prep, sk_encrypt_with_pt_in_b, RlweCiphertext,
        RlweCiphertextMutView, RlwePlaintextView, RlweSecretKeyView,
    },
};
use phantom_zone_math::{
    decomposer::Decomposer,
    izip_eq,
    misc::scratch::Scratch,
    ring::{ElemFrom, RingOps},
};
use rand::RngCore;

pub fn sk_encrypt<'a, 'b, 'c, R, T>(
    ring: &R,
    ct: impl Into<RgswCiphertextMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    pt: impl Into<RlwePlaintextView<'c, R::Elem>>,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch,
    mut rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let (mut ct, sk, pt) = (ct.into(), sk.into(), pt.into());
    let decomposer_a = R::Decomposer::new(ring.modulus(), ct.decomposition_param_a());
    let decomposer_b = R::Decomposer::new(ring.modulus(), ct.decomposition_param_b());
    izip_eq!(ct.a_ct_iter_mut(), decomposer_a.gadget_iter()).for_each(|(mut ct, beta_i)| {
        let scratch = scratch.reborrow();
        sk_encrypt_with_pt_in_b(ring, &mut ct, sk, noise_distribution, scratch, &mut rng);
        ring.slice_scalar_fma(ct.a_mut(), pt.as_ref(), &beta_i)
    });
    izip_eq!(ct.b_ct_iter_mut(), decomposer_b.gadget_iter()).for_each(|(mut ct, beta_i)| {
        let scratch = scratch.reborrow();
        sk_encrypt_with_pt_in_b(ring, &mut ct, sk, noise_distribution, scratch, &mut rng);
        ring.slice_scalar_fma(ct.b_mut(), pt.as_ref(), &beta_i)
    });
}

pub fn rlwe_by_rgsw_in_place<'a, 'b, R: RingOps>(
    ring: &R,
    ct_rlwe: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    ct_rgsw: impl Into<RgswCiphertextView<'a, R::Elem>>,
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

pub fn prepare_rgsw<'a, R: RingOps>(
    ring: &R,
    ct: impl Into<RgswCiphertextView<'a, R::Elem>>,
    mut scratch: Scratch,
) -> RgswCiphertextOwned<R::EvalPrep> {
    let ct = ct.into();
    let eval = ring.take_eval(&mut scratch);
    let mut ct_prep = RgswCiphertext::allocate_eval(
        ring.ring_size(),
        ring.eval_size(),
        ct.decomposition_log_base(),
        ct.decomposition_level_a(),
        ct.decomposition_level_b(),
    );
    izip_eq!(ct_prep.ct_iter_mut(), ct.ct_iter()).for_each(|(mut ct_prep, ct)| {
        ring.forward_normalized(eval, ct.a());
        ring.eval_prepare(ct_prep.a_mut(), eval);
        ring.forward_normalized(eval, ct.b());
        ring.eval_prepare(ct_prep.b_mut(), eval);
    });
    ct_prep
}

pub fn rlwe_by_rgsw_prep_in_place<'a, 'b, R: RingOps>(
    ring: &R,
    ct_rlwe: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    ct_rgsw: impl Into<RgswCiphertextView<'a, R::EvalPrep>>,
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
