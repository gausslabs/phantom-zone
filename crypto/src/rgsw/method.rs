use crate::{
    distribution::NoiseDistribution,
    rgsw::structure::{RgswCiphertextMutView, RgswCiphertextView},
    rlwe::{sk_encrypt_with_pt_in_b, RlweCiphertextMutView, RlwePlaintextView, RlweSecretKeyView},
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
    let decomposer_a = R::Decomposer::new(ring.modulus(), ct_rgsw.decomposition_param_a());
    let decomposer_b = R::Decomposer::new(ring.modulus(), ct_rgsw.decomposition_param_b());
    let [ct_rlwe_a_eval, ct_rlwe_b_eval, t0, t1] = ring.take_evals(&mut scratch);
    decomposer_a.slice_decompose_zip_for_each(
        ct_rlwe.a(),
        ct_rgsw.a_ct_iter(),
        scratch.reborrow(),
        |i, a_i, ct_rgsw_a_i| {
            let eval_fma = if i == 0 { R::eval_mul } else { R::eval_fma };
            ring.forward(t0, a_i);
            ring.forward(t1, ct_rgsw_a_i.a());
            eval_fma(ring, ct_rlwe_a_eval, t0, t1);
            ring.forward(t1, ct_rgsw_a_i.b());
            eval_fma(ring, ct_rlwe_b_eval, t0, t1);
        },
    );
    decomposer_b.slice_decompose_zip_for_each(
        ct_rlwe.b(),
        ct_rgsw.b_ct_iter(),
        scratch,
        |_, b_i, ct_rgsw_b_i| {
            ring.forward(t0, b_i);
            ring.forward(t1, ct_rgsw_b_i.a());
            ring.eval_fma(ct_rlwe_a_eval, t0, t1);
            ring.forward(t1, ct_rgsw_b_i.b());
            ring.eval_fma(ct_rlwe_b_eval, t0, t1);
        },
    );
    ring.backward_normalized(ct_rlwe.a_mut(), ct_rlwe_a_eval);
    ring.backward_normalized(ct_rlwe.b_mut(), ct_rlwe_b_eval);
}
