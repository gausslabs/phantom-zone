use crate::{
    distribution::{NoiseDistribution, SecretKeyDistribution},
    lwe::LweCiphertextMutView,
    rlwe::structure::{
        RlweAutoKeyMutView, RlweAutoKeyView, RlweCiphertext, RlweCiphertextMutView,
        RlweCiphertextView, RlweKeySwitchKeyMutView, RlweKeySwitchKeyView, RlwePlaintextMutView,
        RlwePlaintextView, RlweSecretKeyMutView, RlweSecretKeyView,
    },
};
use core::{borrow::Borrow, ops::Neg};
use num_traits::{FromPrimitive, Signed};
use phantom_zone_math::{
    decomposer::Decomposer,
    distribution::DistributionSized,
    izip_eq,
    ring::{ElemFrom, RingOps, SliceOps},
};
use rand::RngCore;

pub fn sk_gen<T: Signed + FromPrimitive>(
    mut sk: RlweSecretKeyMutView<T>,
    sk_distribution: SecretKeyDistribution,
    rng: impl RngCore,
) {
    sk_distribution.sample_into(sk.as_mut(), rng)
}

pub fn sk_encrypt<R: RingOps + ElemFrom<T>, T: Copy>(
    ring: &R,
    mut ct: RlweCiphertextMutView<R::Elem>,
    sk: RlweSecretKeyView<T>,
    pt: RlwePlaintextView<R::Elem>,
    noise_distribution: NoiseDistribution,
    rng: impl RngCore,
) {
    let mut scratch = ring.allocate_scratch();
    ct.b_mut().copy_from_slice(pt.as_ref());
    sk_encrypt_with_pt_in_b(ring, ct, sk, noise_distribution, rng, &mut scratch);
}

pub fn sk_encrypt_with_pt_in_b<R: RingOps + ElemFrom<T>, T: Copy>(
    ring: &R,
    mut ct: RlweCiphertextMutView<R::Elem>,
    sk: RlweSecretKeyView<T>,
    noise_distribution: NoiseDistribution,
    mut rng: impl RngCore,
    scratch: &mut [R::Eval],
) {
    let (a, b) = ct.a_b_mut();
    ring.sample_uniform_into(a, &mut rng);
    ring.poly_fma_elem_from(b, a, sk.as_ref(), scratch);
    ring.slice_add_assign_iter(b, ring.sample_iter::<i64>(noise_distribution, &mut rng));
}

pub fn decrypt<R: RingOps + ElemFrom<T>, T: Copy>(
    ring: &R,
    mut pt: RlwePlaintextMutView<R::Elem>,
    sk: RlweSecretKeyView<T>,
    ct: RlweCiphertextView<R::Elem>,
) {
    let (a, b) = ct.a_b();
    ring.poly_mul_elem_from(pt.as_mut(), a, sk.as_ref(), &mut ring.allocate_scratch());
    ring.slice_neg_assign(pt.as_mut());
    ring.slice_add_assign(pt.as_mut(), b.as_ref());
}

pub fn ks_key_gen<R: RingOps + ElemFrom<T>, T: Copy>(
    ring: &R,
    ks_key: RlweKeySwitchKeyMutView<R::Elem>,
    sk_from: RlweSecretKeyView<T>,
    sk_to: RlweSecretKeyView<T>,
    noise_distribution: NoiseDistribution,
    rng: impl RngCore,
) {
    ks_key_gen_inner(
        ring,
        ks_key,
        sk_from.as_ref(),
        sk_to,
        noise_distribution,
        rng,
    )
}

fn ks_key_gen_inner<R: RingOps + ElemFrom<T>, T: Copy>(
    ring: &R,
    mut ks_key: RlweKeySwitchKeyMutView<R::Elem>,
    sk_from: impl Clone + IntoIterator<Item: Borrow<T>>,
    sk_to: RlweSecretKeyView<T>,
    noise_distribution: NoiseDistribution,
    mut rng: impl RngCore,
) {
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

pub fn key_switch<R: RingOps>(
    ring: &R,
    ct_to: RlweCiphertextMutView<R::Elem>,
    ks_key: RlweKeySwitchKeyView<R::Elem>,
    ct_from: RlweCiphertextView<R::Elem>,
) {
    key_switch_inner(ring, ct_to, ks_key, ct_from.a(), ct_from.b())
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

pub fn sample_extract<R: RingOps>(
    ring: &R,
    mut ct_lwe: LweCiphertextMutView<R::Elem>,
    ct_rlwe: RlweCiphertextView<R::Elem>,
    idx: usize,
) {
    assert!(idx < ring.ring_size());
    ct_lwe.a_mut().copy_from_slice(ct_rlwe.a());
    ct_lwe.a_mut().reverse();
    ring.slice_neg_assign(&mut ct_lwe.a_mut()[..ring.ring_size() - idx - 1]);
    ct_lwe.a_mut().rotate_left(ring.ring_size() - idx - 1);
    *ct_lwe.b_mut() = ct_rlwe.b()[idx];
}

pub fn auto_key_gen<R: RingOps + ElemFrom<T>, T: Copy + Neg<Output = T>>(
    ring: &R,
    mut auto_key: RlweAutoKeyMutView<R::Elem>,
    sk: RlweSecretKeyView<T>,
    noise_distribution: NoiseDistribution,
    rng: impl RngCore,
) {
    let (auto_map, ks_key) = auto_key.split_into_map_and_ks_key_mut();
    let sk_auto = auto_map.apply(sk.as_ref(), |&v| -v);
    ks_key_gen_inner(ring, ks_key, sk_auto, sk, noise_distribution, rng);
}

pub fn automorphism<R: RingOps>(
    ring: &R,
    ct_auto: RlweCiphertextMutView<R::Elem>,
    auto_key: RlweAutoKeyView<R::Elem>,
    ct: RlweCiphertextView<R::Elem>,
) {
    let map = auto_key.map();
    let ct_a = map.apply(ct.a(), |v| ring.neg(v));
    let ct_b = map.apply(ct.b(), |v| ring.neg(v));
    key_switch_inner(ring, ct_auto, auto_key.as_ks_key(), ct_a, ct_b)
}
