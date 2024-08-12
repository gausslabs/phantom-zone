use crate::{
    core::{
        lwe::LweCiphertextMutView,
        rlwe::structure::{
            RlweAutoKeyMutView, RlweAutoKeyView, RlweCiphertext, RlweCiphertextMutView,
            RlweCiphertextView, RlweKeySwitchKeyMutView, RlweKeySwitchKeyView,
            RlwePlaintextMutView, RlwePlaintextView, RlwePublicKeyMutView, RlwePublicKeyView,
            RlweSecretKeyView,
        },
    },
    util::distribution::NoiseDistribution,
};
use core::{borrow::Borrow, ops::Neg};
use phantom_zone_math::{
    decomposer::{Decomposer, DecompositionParam},
    izip_eq,
    modulus::ElemFrom,
    ring::RingOps,
    util::scratch::Scratch,
};
use rand::RngCore;

pub fn pk_gen<'a, 'b, R, T>(
    ring: &R,
    pk: impl Into<RlwePublicKeyMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    scratch: Scratch,
    rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    sk_encrypt_with_pt_in_b(
        ring,
        pk.into().as_ct_mut(),
        sk,
        noise_distribution,
        scratch,
        rng,
    );
}

pub fn sk_encrypt<'a, 'b, 'c, R, T>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    pt: impl Into<RlwePlaintextView<'c, R::Elem>>,
    noise_distribution: NoiseDistribution,
    scratch: Scratch,
    rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let mut ct = ct.into();
    ct.b_mut().copy_from_slice(pt.into().as_ref());
    sk_encrypt_with_pt_in_b(ring, ct, sk, noise_distribution, scratch, rng);
}

pub fn sk_encrypt_with_pt_in_b<'a, 'b, R, T>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    scratch: Scratch,
    mut rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let mut ct = ct.into();
    let (a, b) = ct.a_b_mut();
    ring.sample_uniform_into(a, &mut rng);
    ring.poly_fma_elem_from(b, a, sk.into().as_ref(), scratch);
    ring.slice_add_assign_iter(b, ring.sample_iter::<i64>(noise_distribution, &mut rng));
}

pub fn pk_encrypt<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    pk: impl Into<RlwePublicKeyView<'b, R::Elem>>,
    pt: impl Into<RlwePlaintextView<'c, R::Elem>>,
    u_distribution: NoiseDistribution,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch,
    mut rng: impl RngCore,
) {
    let (mut ct, pk) = (ct.into(), pk.into());

    let t0 = ring.take_eval(&mut scratch);
    let u = ring.take_poly(&mut scratch.reborrow());
    ring.sample_into::<i64>(u, u_distribution, &mut rng);
    ring.forward(t0, u);

    let t1 = ring.take_eval(&mut scratch);
    ring.forward(t1, pk.a());
    ring.eval_mul_assign(t1, t0);
    ring.sample_into::<i64>(ct.a_mut(), noise_distribution, &mut rng);
    ring.add_backward_normalized(ct.a_mut(), t1);

    ring.forward(t1, pk.b());
    ring.eval_mul_assign(t1, t0);
    ring.sample_into::<i64>(ct.b_mut(), noise_distribution, &mut rng);
    ring.add_backward_normalized(ct.b_mut(), t1);
    ring.slice_add_assign(ct.b_mut(), pt.into().as_ref());
}

pub fn decrypt<'a, 'b, 'c, R, T>(
    ring: &R,
    pt: impl Into<RlwePlaintextMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    ct: impl Into<RlweCiphertextView<'c, R::Elem>>,
    scratch: Scratch,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let (mut pt, ct) = (pt.into(), ct.into());
    ring.poly_mul_elem_from(pt.as_mut(), ct.a(), sk.into().as_ref(), scratch);
    ring.slice_neg_assign(pt.as_mut());
    ring.slice_add_assign(pt.as_mut(), ct.b().as_ref());
}

pub fn sample_extract<'a, 'b, R: RingOps>(
    ring: &R,
    ct_lwe: impl Into<LweCiphertextMutView<'a, R::Elem>>,
    ct_rlwe: impl Into<RlweCiphertextView<'b, R::Elem>>,
    idx: usize,
) {
    assert!(idx < ring.ring_size());
    let (mut ct_lwe, ct_rlwe) = (ct_lwe.into(), ct_rlwe.into());
    ct_lwe.a_mut().copy_from_slice(ct_rlwe.a());
    ct_lwe.a_mut().reverse();
    ring.slice_neg_assign(&mut ct_lwe.a_mut()[..ring.ring_size() - idx - 1]);
    ct_lwe.a_mut().rotate_left(ring.ring_size() - idx - 1);
    *ct_lwe.b_mut() = ct_rlwe.b()[idx];
}

pub fn ks_key_gen<'a, 'b, 'c, R, T>(
    ring: &R,
    ks_key: impl Into<RlweKeySwitchKeyMutView<'a, R::Elem>>,
    sk_from: impl Into<RlweSecretKeyView<'b, T>>,
    sk_to: impl Into<RlweSecretKeyView<'c, T>>,
    noise_distribution: NoiseDistribution,
    scratch: Scratch,
    rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + 'c + Copy,
{
    ks_key_gen_inner(
        ring,
        ks_key.into(),
        sk_from.into().as_ref(),
        sk_to.into(),
        noise_distribution,
        scratch,
        rng,
    )
}

fn ks_key_gen_inner<R, T>(
    ring: &R,
    mut ks_key: RlweKeySwitchKeyMutView<R::Elem>,
    sk_from: impl Clone + IntoIterator<Item: Borrow<T>>,
    sk_to: RlweSecretKeyView<T>,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch,
    mut rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: Copy,
{
    let decomposer = R::Decomposer::new(ring.modulus(), ks_key.decomposition_param());
    izip_eq!(ks_key.ct_iter_mut(), decomposer.gadget_iter()).for_each(|(mut ks_key_i, beta_i)| {
        let scratch = scratch.reborrow();
        ring.slice_elem_from_iter(ks_key_i.b_mut(), sk_from.clone());
        ring.slice_scalar_mul_assign(ks_key_i.b_mut(), &ring.neg(&beta_i));
        sk_encrypt_with_pt_in_b(ring, ks_key_i, sk_to, noise_distribution, scratch, &mut rng);
    });
}

pub fn key_switch_in_place<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    ks_key: impl Into<RlweKeySwitchKeyView<'b, R::Elem>>,
    mut scratch: Scratch,
) {
    let (mut ct, ks_key) = (ct.into(), ks_key.into());
    let mut ct_eval = RlweCiphertext::scratch(ring.ring_size(), ring.eval_size(), &mut scratch);
    decomposed_fma::<_, true>(
        ring,
        ks_key.decomposition_param(),
        &mut ct_eval,
        ct.a(),
        ks_key.ct_iter(),
        scratch.reborrow(),
    );
    ring.backward_normalized(ct.a_mut(), ct_eval.a_mut());
    ring.add_backward_normalized(ct.b_mut(), ct_eval.b_mut());
}

pub fn auto_key_gen<'a, 'b, R, T>(
    ring: &R,
    auto_key: impl Into<RlweAutoKeyMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    scratch: Scratch,
    rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy + Neg<Output = T>,
{
    let (mut auto_key, sk) = (auto_key.into(), sk.into());
    let (auto_map, ks_key) = auto_key.auto_map_and_ks_key_mut();
    let sk_auto = auto_map.apply(sk.as_ref(), |&v| -v);
    ks_key_gen_inner(ring, ks_key, sk_auto, sk, noise_distribution, scratch, rng);
}

pub fn automorphism_in_place<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    auto_key: impl Into<RlweAutoKeyView<'b, R::Elem>>,
    mut scratch: Scratch,
) {
    let (mut ct, auto_key) = (ct.into(), auto_key.into());
    let (auto_map, ks_key) = auto_key.auto_map_and_ks_key();
    let mut ct_eval = RlweCiphertext::scratch(ring.ring_size(), ring.eval_size(), &mut scratch);
    decomposed_fma::<_, true>(
        ring,
        ks_key.decomposition_param(),
        &mut ct_eval,
        auto_map.apply(ct.a(), |v| ring.neg(v)),
        ks_key.ct_iter(),
        scratch.reborrow(),
    );
    let b = scratch.copy_slice(ct.b());
    ring.backward_normalized(ct.a_mut(), ct_eval.a_mut());
    ring.backward_normalized(ct.b_mut(), ct_eval.b_mut());
    ring.poly_add_auto(ct.b_mut(), b, auto_map);
}

pub fn prepare_ks_key<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    ks_key_prep: impl Into<RlweKeySwitchKeyMutView<'a, R::EvalPrep>>,
    ks_key: impl Into<RlweKeySwitchKeyView<'a, R::Elem>>,
    mut scratch: Scratch,
) {
    let (mut ks_key_prep, ks_key) = (ks_key_prep.into(), ks_key.into());
    let eval = ring.take_eval(&mut scratch);
    izip_eq!(ks_key_prep.ct_iter_mut(), ks_key.ct_iter()).for_each(|(mut ct_prep, ct)| {
        ring.forward_normalized(eval, ct.a());
        ring.eval_prepare(ct_prep.a_mut(), eval);
        ring.forward_normalized(eval, ct.b());
        ring.eval_prepare(ct_prep.b_mut(), eval);
    });
}

pub fn prepare_auto_key<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    auto_key_prep: impl Into<RlweAutoKeyMutView<'a, R::EvalPrep>>,
    auto_key: impl Into<RlweAutoKeyView<'a, R::Elem>>,
    scratch: Scratch,
) {
    prepare_ks_key(
        ring,
        auto_key_prep.into().as_ks_key_mut(),
        auto_key.into().as_ks_key(),
        scratch,
    );
}

pub fn key_switch_prep_in_place<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    ks_key: impl Into<RlweKeySwitchKeyView<'b, R::EvalPrep>>,
    mut scratch: Scratch,
) {
    let (mut ct, ks_key) = (ct.into(), ks_key.into());
    let mut ct_eval = RlweCiphertext::scratch(ring.ring_size(), ring.eval_size(), &mut scratch);
    decomposed_fma_prep::<_, true>(
        ring,
        ks_key.decomposition_param(),
        &mut ct_eval,
        ct.a(),
        ks_key.ct_iter(),
        scratch.reborrow(),
    );
    ring.backward(ct.a_mut(), ct_eval.a_mut());
    ring.add_backward(ct.b_mut(), ct_eval.b_mut());
}

pub fn automorphism_prep_in_place<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    auto_key: impl Into<RlweAutoKeyView<'b, R::EvalPrep>>,
    mut scratch: Scratch,
) {
    let (mut ct, auto_key) = (ct.into(), auto_key.into());
    let (auto_map, ks_key) = auto_key.auto_map_and_ks_key();
    let mut ct_eval = RlweCiphertext::scratch(ring.ring_size(), ring.eval_size(), &mut scratch);
    decomposed_fma_prep::<_, true>(
        ring,
        ks_key.decomposition_param(),
        &mut ct_eval,
        auto_map.apply(ct.a(), |v| ring.neg(v)),
        ks_key.ct_iter(),
        scratch.reborrow(),
    );
    let b = scratch.copy_slice(ct.b());
    ring.backward(ct.a_mut(), ct_eval.a_mut());
    ring.backward(ct.b_mut(), ct_eval.b_mut());
    ring.poly_add_auto(ct.b_mut(), b, auto_map);
}

pub(crate) fn decomposed_fma<'a, 'b, R: RingOps, const DIRTY: bool>(
    ring: &R,
    decomposition_param: DecompositionParam,
    ct_eval: impl Into<RlweCiphertextMutView<'a, R::Eval>>,
    a: impl IntoIterator<Item: Borrow<R::Elem>>,
    b: impl IntoIterator<Item = RlweCiphertextView<'b, R::Elem>>,
    mut scratch: Scratch,
) {
    let mut ct_eval = ct_eval.into();
    let decomposer = R::Decomposer::new(ring.modulus(), decomposition_param);
    let [t0, t1] = ring.take_evals(&mut scratch);
    decomposer.slice_decompose_zip_for_each(a, b, scratch, |i, a_i, b_i| {
        let eval_fma = if DIRTY && i == 0 {
            R::eval_mul
        } else {
            R::eval_fma
        };
        ring.forward(t0, a_i);
        ring.forward(t1, b_i.a());
        eval_fma(ring, ct_eval.a_mut(), t0, t1);
        ring.forward(t1, b_i.b());
        eval_fma(ring, ct_eval.b_mut(), t0, t1);
    });
}

pub(crate) fn decomposed_fma_prep<'a, 'b, R: RingOps, const DIRTY: bool>(
    ring: &R,
    decomposition_param: DecompositionParam,
    ct_eval: impl Into<RlweCiphertextMutView<'a, R::Eval>>,
    a: impl IntoIterator<Item: Borrow<R::Elem>>,
    b: impl IntoIterator<Item = RlweCiphertextView<'b, R::EvalPrep>>,
    mut scratch: Scratch,
) {
    let mut ct_eval = ct_eval.into();
    let decomposer = R::Decomposer::new(ring.modulus(), decomposition_param);
    let t0 = ring.take_eval(&mut scratch);
    decomposer.slice_decompose_zip_for_each(a, b, scratch, |i, a_i, b_i| {
        let eval_fma_prep = if DIRTY && i == 0 {
            R::eval_mul_prep
        } else {
            R::eval_fma_prep
        };
        ring.forward(t0, a_i);
        eval_fma_prep(ring, ct_eval.a_mut(), t0, b_i.a());
        eval_fma_prep(ring, ct_eval.b_mut(), t0, b_i.b());
    });
}
