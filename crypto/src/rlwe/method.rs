use crate::{
    distribution::NoiseDistribution,
    lwe::LweCiphertextMutView,
    rlwe::structure::{
        RlweAutoKeyMutView, RlweAutoKeyView, RlweCiphertext, RlweCiphertextMutView,
        RlweCiphertextView, RlweKeySwitchKeyMutView, RlweKeySwitchKeyView, RlwePlaintextMutView,
        RlwePlaintextView, RlwePublicKeyMutView, RlwePublicKeyView, RlweSecretKeyView,
    },
};
use core::{borrow::Borrow, ops::Neg};
use phantom_zone_math::{
    decomposer::Decomposer,
    distribution::Ternary,
    izip_eq,
    ring::{ElemFrom, RingOps, SliceOps},
};
use rand::RngCore;

pub fn pk_gen<'a, 'b, R, T>(
    ring: &R,
    pk: impl Into<RlwePublicKeyMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let mut scratch = ring.allocate_scratch();
    sk_encrypt_with_pt_in_b(
        ring,
        pk.into().as_ct_mut(),
        sk,
        noise_distribution,
        rng,
        &mut scratch,
    );
}

pub fn sk_encrypt<'a, 'b, 'c, R, T>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    pt: impl Into<RlwePlaintextView<'c, R::Elem>>,
    noise_distribution: NoiseDistribution,
    rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let mut ct = ct.into();
    let mut scratch = ring.allocate_scratch();
    ct.b_mut().copy_from_slice(pt.into().as_ref());
    sk_encrypt_with_pt_in_b(ring, ct, sk, noise_distribution, rng, &mut scratch);
}

pub fn sk_encrypt_with_pt_in_b<'a, 'b, R, T>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    mut rng: impl RngCore,
    scratch: &mut [R::Eval],
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
    mut rng: impl RngCore,
) {
    let (mut ct, pk) = (ct.into(), pk.into());
    let mut scratch = ring.allocate_scratch();
    let (t0, t1) = scratch.split_at_mut(ring.eval_size());
    ring.sample_into::<i64>(ct.a_mut(), Ternary(ring.ring_size() / 2), &mut rng);
    ring.forward(t1, ct.a());
    ring.forward(t0, pk.a());
    ring.eval().slice_mul_assign(t0, t1);
    ring.backward_normalized(ct.a_mut(), t0);
    ring.forward(t0, pk.b());
    ring.eval().slice_mul_assign(t0, t1);
    ring.backward_normalized(ct.b_mut(), t0);
    ring.slice_add_assign(ct.b_mut(), pt.into().as_ref());
}

pub fn decrypt<'a, 'b, 'c, R, T>(
    ring: &R,
    pt: impl Into<RlwePlaintextMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    ct: impl Into<RlweCiphertextView<'c, R::Elem>>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let (mut pt, ct) = (pt.into(), ct.into());
    ring.poly_mul_elem_from(
        pt.as_mut(),
        ct.a(),
        sk.into().as_ref(),
        &mut ring.allocate_scratch(),
    );
    ring.slice_neg_assign(pt.as_mut());
    ring.slice_add_assign(pt.as_mut(), ct.b().as_ref());
}

pub fn ks_key_gen<'a, 'b, 'c, R, T>(
    ring: &R,
    ks_key: impl Into<RlweKeySwitchKeyMutView<'a, R::Elem>>,
    sk_from: impl Into<RlweSecretKeyView<'b, T>>,
    sk_to: impl Into<RlweSecretKeyView<'c, T>>,
    noise_distribution: NoiseDistribution,
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
        rng,
    )
}

fn ks_key_gen_inner<R, T>(
    ring: &R,
    mut ks_key: RlweKeySwitchKeyMutView<R::Elem>,
    sk_from: impl Clone + IntoIterator<Item: Borrow<T>>,
    sk_to: RlweSecretKeyView<T>,
    noise_distribution: NoiseDistribution,
    mut rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: Copy,
{
    let decomposer = R::Decomposer::new(ring.modulus(), ks_key.decomposition_param());
    let mut scratch = ring.allocate_scratch();
    izip_eq!(ks_key.ct_iter_mut(), decomposer.gadget_iter()).for_each(|(mut ks_key_i, beta_i)| {
        ring.slice_elem_from_iter(ks_key_i.b_mut(), sk_from.clone());
        ring.slice_scalar_mul_assign(ks_key_i.b_mut(), &ring.neg(&beta_i));
        sk_encrypt_with_pt_in_b(
            ring,
            ks_key_i,
            sk_to,
            noise_distribution,
            &mut rng,
            &mut scratch,
        );
    });
}

pub fn key_switch<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    ct_to: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    ks_key: impl Into<RlweKeySwitchKeyView<'b, R::Elem>>,
    ct_from: impl Into<RlweCiphertextView<'c, R::Elem>>,
) {
    let ct_from = ct_from.into();
    key_switch_inner(ring, ct_to.into(), ks_key.into(), ct_from.a(), ct_from.b())
}

fn key_switch_inner<R: RingOps>(
    ring: &R,
    mut ct_to: RlweCiphertextMutView<R::Elem>,
    ks_key: RlweKeySwitchKeyView<R::Elem>,
    ct_from_a: impl IntoIterator<Item: Borrow<R::Elem>>,
    ct_from_b: impl IntoIterator<Item: Borrow<R::Elem>>,
) {
    let decomposer = R::Decomposer::new(ring.modulus(), ks_key.decomposition_param());
    let mut ct_to_eval = RlweCiphertext::allocate(ring.eval_size());
    let mut forward_scratch = ring.allocate_scratch();
    let mut decomposer_scratch = decomposer.allocate_scratch(ct_to.a());
    decomposer.slice_decompose_zip_for_each(
        ct_from_a,
        ks_key.ct_iter(),
        &mut decomposer_scratch,
        |(a_i, ks_key_i)| {
            let (t0, t1) = forward_scratch.split_at_mut(ring.eval_size());
            ring.forward(t0, a_i);
            ring.forward(t1, ks_key_i.a());
            ring.eval().slice_fma(ct_to_eval.a_mut(), t0, t1);
            ring.forward(t1, ks_key_i.b());
            ring.eval().slice_fma(ct_to_eval.b_mut(), t0, t1);
        },
    );
    ring.backward_normalized(ct_to.a_mut(), ct_to_eval.a_mut());
    ring.backward_normalized(ct_to.b_mut(), ct_to_eval.b_mut());
    ring.slice_add_assign_iter(ct_to.b_mut(), ct_from_b);
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

pub fn auto_key_gen<'a, 'b, R, T>(
    ring: &R,
    auto_key: impl Into<RlweAutoKeyMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    rng: impl RngCore,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy + Neg<Output = T>,
{
    let (mut auto_key, sk) = (auto_key.into(), sk.into());
    let (auto_map, ks_key) = auto_key.split_into_map_and_ks_key_mut();
    let sk_auto = auto_map.apply(sk.as_ref(), |&v| -v);
    ks_key_gen_inner(ring, ks_key, sk_auto, sk, noise_distribution, rng);
}

pub fn automorphism<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    ct_auto: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    auto_key: impl Into<RlweAutoKeyView<'b, R::Elem>>,
    ct: impl Into<RlweCiphertextView<'c, R::Elem>>,
) {
    let (auto_key, ct) = (auto_key.into(), ct.into());
    let map = auto_key.map();
    let ct_a = map.apply(ct.a(), |v| ring.neg(v));
    let ct_b = map.apply(ct.b(), |v| ring.neg(v));
    key_switch_inner(ring, ct_auto.into(), auto_key.as_ks_key(), ct_a, ct_b)
}
