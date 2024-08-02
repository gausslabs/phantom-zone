use crate::{
    lwe::LweCiphertextMutView,
    misc::{NoiseDistribution, SecretKeyDistribution},
    rlwe::structure::{
        RlweCiphertext, RlweCiphertextMutView, RlweCiphertextView, RlweKeySwitchKeyMutView,
        RlweKeySwitchKeyView, RlwePlaintextMutView, RlwePlaintextView, RlweSecretKeyMutView,
        RlweSecretKeyView,
    },
};
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

pub fn ksk_gen<R: RingOps + ElemFrom<T>, T: Copy>(
    ring: &R,
    mut ksk: RlweKeySwitchKeyMutView<R::Elem>,
    sk_from: RlweSecretKeyView<T>,
    sk_to: RlweSecretKeyView<T>,
    noise_distribution: NoiseDistribution,
    mut rng: impl RngCore,
) {
    let decomposer = R::Decomposer::new(ring.modulus(), ksk.decomposition_param());
    let mut scratch = ring.allocate_scratch();
    izip_eq!(ksk.ct_iter_mut(), decomposer.gadget_iter()).for_each(|(mut ksk_i, beta_i)| {
        ring.slice_elem_from(ksk_i.b_mut(), sk_from.as_ref());
        ring.slice_scalar_mul_assign(ksk_i.b_mut(), &ring.neg(&beta_i));
        sk_encrypt_with_pt_in_b(
            ring,
            ksk_i,
            sk_to,
            noise_distribution,
            &mut rng,
            &mut scratch,
        );
    });
}

pub fn key_switch<R: RingOps>(
    ring: &R,
    mut ct_to: RlweCiphertextMutView<R::Elem>,
    ksk: RlweKeySwitchKeyView<R::Elem>,
    ct_from: RlweCiphertextView<R::Elem>,
) {
    let decomposer = R::Decomposer::new(ring.modulus(), ksk.decomposition_param());
    let mut ct_to_eval = RlweCiphertext::allocate(ring.eval_size());
    let mut forward_scratch = ring.allocate_scratch();
    let mut decomposer_scratch = decomposer.allocate_scratch(ct_from.a());
    decomposer.slice_decompose_zip_for_each(
        ct_from.a(),
        ksk.ct_iter(),
        &mut decomposer_scratch,
        |(a_i, ksk_i)| {
            let (t0, t1) = forward_scratch.split_at_mut(ring.eval_size());
            ring.forward(t0, a_i);
            ring.forward(t1, ksk_i.a());
            ring.eval().slice_fma(ct_to_eval.a_mut(), t0, t1);
            ring.forward(t1, ksk_i.b());
            ring.eval().slice_fma(ct_to_eval.b_mut(), t0, t1);
        },
    );
    ring.backward_normalized(ct_to.a_mut(), ct_to_eval.a_mut());
    ring.backward_normalized(ct_to.b_mut(), ct_to_eval.b_mut());
    ring.slice_add_assign(ct_to.b_mut(), ct_from.b());
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
