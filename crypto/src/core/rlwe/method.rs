use crate::{
    core::{
        lwe::LweCiphertextMutView,
        rlwe::structure::{
            RlweAutoKeyMutView, RlweAutoKeyView, RlweCiphertext, RlweCiphertextMutView,
            RlweCiphertextView, RlweDecryptionShareMutView, RlweDecryptionShareView,
            RlweKeySwitchKeyMutView, RlweKeySwitchKeyView, RlwePlaintextMutView, RlwePlaintextView,
            RlwePublicKey, RlwePublicKeyMutView, RlwePublicKeyView, RlweSecretKeyView,
            SeededRlweAutoKeyMutView, SeededRlweAutoKeyView, SeededRlweCiphertextView,
            SeededRlweKeySwitchKeyMutView, SeededRlweKeySwitchKeyView, SeededRlwePublicKeyMutView,
            SeededRlwePublicKeyView,
        },
    },
    util::{
        distribution::{NoiseDistribution, SecretDistribution},
        rng::LweRng,
    },
};
use core::{borrow::Borrow, ops::Neg};
use phantom_zone_math::{
    decomposer::{Decomposer, DecompositionParam},
    izip_eq,
    modulus::ElemFrom,
    poly::automorphism::AutomorphismMap,
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
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    sk_encrypt_zero(
        ring,
        pk.into().ct_mut(),
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
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let mut ct = ct.into();
    sk_encrypt_zero(ring, &mut ct, sk, noise_distribution, scratch, rng);
    ring.slice_add_assign(ct.b_mut(), pt.into().as_ref());
}

pub(crate) fn sk_encrypt_zero<'a, 'b, R, T>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    scratch: Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let mut ct = ct.into();
    let (a, b) = ct.a_b_mut();
    ring.sample_uniform_into(a, rng.seedable());
    ring.sample_into::<i64>(b, noise_distribution, &mut *rng);
    ring.poly_fma_elem_from(b, a, sk.into().as_ref(), scratch);
}

pub fn pk_encrypt<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    pk: impl Into<RlwePublicKeyView<'b, R::Elem>>,
    pt: impl Into<RlwePlaintextView<'c, R::Elem>>,
    u_distribution: SecretDistribution,
    noise_distribution: NoiseDistribution,
    scratch: Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) {
    let mut ct = ct.into();
    pk_encrypt_zero(
        ring,
        &mut ct,
        pk,
        u_distribution,
        noise_distribution,
        scratch,
        rng,
    );
    ring.slice_add_assign(ct.b_mut(), pt.into().as_ref());
}

pub fn pk_encrypt_zero<'a, 'b, R: RingOps>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    pk: impl Into<RlwePublicKeyView<'b, R::Elem>>,
    u_distribution: SecretDistribution,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) {
    let (mut ct, pk) = (ct.into(), pk.into());

    let eval_scratch = ring.take_eval_scratch(&mut scratch);
    let t0 = ring.take_eval(&mut scratch);
    let u = ring.take_poly(&mut scratch.reborrow());
    ring.sample_into::<i64>(u, u_distribution, &mut *rng);
    ring.forward(t0, u, eval_scratch);

    let t1 = ring.take_eval(&mut scratch);
    ring.forward(t1, pk.a(), eval_scratch);
    ring.eval_mul_assign(t1, t0);
    ring.sample_into::<i64>(ct.a_mut(), noise_distribution, &mut *rng);
    ring.add_backward_normalized(ct.a_mut(), t1, eval_scratch);

    ring.forward(t1, pk.b(), eval_scratch);
    ring.eval_mul_assign(t1, t0);
    ring.sample_into::<i64>(ct.b_mut(), noise_distribution, &mut *rng);
    ring.add_backward_normalized(ct.b_mut(), t1, eval_scratch);
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

pub fn decrypt_share<'a, 'b, 'c, R, T>(
    ring: &R,
    dec_share: impl Into<RlweDecryptionShareMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    ct: impl Into<RlweCiphertextView<'c, R::Elem>>,
    noise_distribution: NoiseDistribution,
    scratch: Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let (mut dec_share, ct) = (dec_share.into(), ct.into());
    ring.sample_into::<i64>(dec_share.as_mut(), noise_distribution, &mut *rng);
    ring.poly_fma_elem_from(dec_share.as_mut(), ct.a(), sk.into().as_ref(), scratch);
}

pub fn aggregate_decryption_shares<'a, 'b, 'c, R: RingOps>(
    ring: &R,
    pt: impl Into<RlwePlaintextMutView<'a, R::Elem>>,
    ct: impl Into<RlweCiphertextView<'a, R::Elem>>,
    dec_shares: impl IntoIterator<Item: Into<RlweDecryptionShareView<'c, R::Elem>>>,
) {
    let (mut pt, ct) = (pt.into(), ct.into());
    pt.as_mut().copy_from_slice(ct.b());
    dec_shares
        .into_iter()
        .for_each(|dec_share| ring.slice_sub_assign(pt.as_mut(), dec_share.into().as_ref()));
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
    rng: &mut LweRng<impl RngCore, impl RngCore>,
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
    sk_from: &[T],
    sk_to: RlweSecretKeyView<T>,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: Copy,
{
    let decomposer = R::Decomposer::new(ring.modulus(), ks_key.decomposition_param());
    izip_eq!(ks_key.ct_iter_mut(), decomposer.gadget_iter()).for_each(|(mut ks_key_i, beta_i)| {
        let scratch = scratch.reborrow();
        sk_encrypt_zero(ring, &mut ks_key_i, sk_to, noise_distribution, scratch, rng);
        ring.slice_scalar_fma_elem_from(ks_key_i.b_mut(), sk_from, &ring.neg(&beta_i));
    });
}

pub fn key_switch_in_place<'a, 'b, R: RingOps>(
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
    let eval_scratch = ring.take_eval_scratch(&mut scratch);
    ring.backward_normalized(ct.a_mut(), ct_eval.a_mut(), eval_scratch);
    ring.add_backward_normalized(ct.b_mut(), ct_eval.b_mut(), eval_scratch);
}

pub fn auto_key_gen<'a, 'b, R, T>(
    ring: &R,
    auto_key: impl Into<RlweAutoKeyMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch<'b>,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy + Neg<Output = T>,
{
    let (mut auto_key, sk) = (auto_key.into(), sk.into());
    let (auto_map, ks_key) = auto_key.auto_map_and_ks_key_mut();
    let sk_auto = scratch.copy_iter(auto_map.apply(sk.as_ref(), |&v| -v));
    ks_key_gen_inner(ring, ks_key, sk_auto, sk, noise_distribution, scratch, rng);
}

pub fn automorphism_in_place<'a, 'b, R: RingOps>(
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
    let eval_scratch = ring.take_eval_scratch(&mut scratch);
    ring.backward_normalized(ct.a_mut(), ct_eval.a_mut(), eval_scratch);
    ring.backward_normalized(ct.b_mut(), ct_eval.b_mut(), eval_scratch);
    ring.poly_add_auto(ct.b_mut(), b, auto_map);
}

pub fn prepare_ks_key<'a, 'b, R: RingOps>(
    ring: &R,
    ks_key_prep: impl Into<RlweKeySwitchKeyMutView<'a, R::EvalPrep>>,
    ks_key: impl Into<RlweKeySwitchKeyView<'b, R::Elem>>,
    mut scratch: Scratch,
) {
    let (mut ks_key_prep, ks_key) = (ks_key_prep.into(), ks_key.into());
    let eval = ring.take_eval(&mut scratch);
    let eval_scratch = ring.take_eval_scratch(&mut scratch);
    izip_eq!(ks_key_prep.ct_iter_mut(), ks_key.ct_iter()).for_each(|(mut ct_prep, ct)| {
        ring.forward_normalized(eval, ct.a(), eval_scratch);
        ring.eval_prepare(ct_prep.a_mut(), eval);
        ring.forward_normalized(eval, ct.b(), eval_scratch);
        ring.eval_prepare(ct_prep.b_mut(), eval);
    });
}

pub fn prepare_auto_key<'a, 'b, R: RingOps>(
    ring: &R,
    auto_key_prep: impl Into<RlweAutoKeyMutView<'a, R::EvalPrep>>,
    auto_key: impl Into<RlweAutoKeyView<'b, R::Elem>>,
    scratch: Scratch,
) {
    prepare_ks_key(
        ring,
        auto_key_prep.into().ks_key_mut(),
        auto_key.into().ks_key(),
        scratch,
    );
}

pub fn key_switch_prep_in_place<'a, 'b, R: RingOps>(
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
    let eval_scratch = ring.take_eval_scratch(&mut scratch);
    ring.backward(ct.a_mut(), ct_eval.a_mut(), eval_scratch);
    ring.add_backward(ct.b_mut(), ct_eval.b_mut(), eval_scratch);
}

pub fn automorphism_prep_in_place<'a, 'b, R: RingOps>(
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
    let eval_scratch = ring.take_eval_scratch(&mut scratch);
    ring.backward(ct.a_mut(), ct_eval.a_mut(), eval_scratch);
    ring.backward(ct.b_mut(), ct_eval.b_mut(), eval_scratch);
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
    let eval_scratch = ring.take_eval_scratch(&mut scratch);
    decomposer.slice_decompose_zip_for_each(a, b, scratch, |i, a_i, b_i| {
        let eval_fma = if DIRTY && i == 0 {
            R::eval_mul
        } else {
            R::eval_fma
        };
        ring.forward(t0, a_i, eval_scratch);
        ring.forward(t1, b_i.a(), eval_scratch);
        eval_fma(ring, ct_eval.a_mut(), t0, t1);
        ring.forward(t1, b_i.b(), eval_scratch);
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
    let eval_scratch = ring.take_eval_scratch(&mut scratch);
    decomposer.slice_decompose_zip_for_each(a, b, scratch, |i, a_i, b_i| {
        let eval_fma_prep = if DIRTY && i == 0 {
            R::eval_mul_prep
        } else {
            R::eval_fma_prep
        };
        ring.forward_lazy(t0, a_i, eval_scratch);
        eval_fma_prep(ring, ct_eval.a_mut(), t0, b_i.a());
        eval_fma_prep(ring, ct_eval.b_mut(), t0, b_i.b());
    });
}

pub fn seeded_pk_gen<'a, 'b, R, T>(
    ring: &R,
    pk: impl Into<SeededRlwePublicKeyMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let mut t = RlwePublicKey::scratch(ring.ring_size(), ring.ring_size(), &mut scratch);
    pk_gen(ring, &mut t, sk, noise_distribution, scratch, rng);
    pk.into().ct_mut().b_mut().copy_from_slice(t.b());
}

fn seeded_ks_key_gen_inner<R, T>(
    ring: &R,
    mut ks_key: SeededRlweKeySwitchKeyMutView<R::Elem>,
    sk_from: &[T],
    sk_to: RlweSecretKeyView<T>,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: Copy,
{
    let decomposer = R::Decomposer::new(ring.modulus(), ks_key.decomposition_param());
    let mut t = RlweCiphertext::scratch(ring.ring_size(), ring.ring_size(), &mut scratch);
    izip_eq!(ks_key.ct_iter_mut(), decomposer.gadget_iter()).for_each(|(mut ks_key_i, beta_i)| {
        let scratch = scratch.reborrow();
        sk_encrypt_zero(ring, &mut t, sk_to, noise_distribution, scratch, rng);
        ks_key_i.b_mut().copy_from_slice(t.b());
        ring.slice_scalar_fma_elem_from(ks_key_i.b_mut(), sk_from, &ring.neg(&beta_i));
    });
}

pub fn seeded_auto_key_gen<'a, 'b, R, T>(
    ring: &R,
    auto_key_seeded: impl Into<SeededRlweAutoKeyMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    mut scratch: Scratch<'b>,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'b + Copy + Neg<Output = T>,
{
    let (mut auto_key_seeded, sk) = (auto_key_seeded.into(), sk.into());
    let auto_map = AutomorphismMap::new(ring.ring_size(), auto_key_seeded.k());
    let ks_key = auto_key_seeded.ks_key_mut();
    let sk_auto = scratch.copy_iter(auto_map.apply(sk.as_ref(), |&v| -v));
    seeded_ks_key_gen_inner(ring, ks_key, sk_auto, sk, noise_distribution, scratch, rng);
}

pub fn unseed_ct<'a, 'b, R: RingOps>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    ct_seeded: impl Into<SeededRlweCiphertextView<'b, R::Elem>>,
    rng: &mut LweRng<(), impl RngCore>,
) {
    let mut ct = ct.into();
    ring.sample_uniform_into(ct.a_mut(), rng.seedable());
    ct.b_mut().copy_from_slice(ct_seeded.into().b());
}

pub fn unseed_pk<'a, 'b, R: RingOps>(
    ring: &R,
    pk: impl Into<RlwePublicKeyMutView<'a, R::Elem>>,
    pk_seeded: impl Into<SeededRlwePublicKeyView<'b, R::Elem>>,
    rng: &mut LweRng<(), impl RngCore>,
) {
    unseed_ct(ring, pk.into().ct_mut(), pk_seeded.into().ct(), rng);
}

pub fn unseed_ks_key<'a, 'b, R: RingOps>(
    ring: &R,
    ks_key: impl Into<RlweKeySwitchKeyMutView<'a, R::Elem>>,
    ks_key_seeded: impl Into<SeededRlweKeySwitchKeyView<'b, R::Elem>>,
    rng: &mut LweRng<(), impl RngCore>,
) {
    izip_eq!(ks_key.into().ct_iter_mut(), ks_key_seeded.into().ct_iter())
        .for_each(|(ct, ct_seeded)| unseed_ct(ring, ct, ct_seeded, rng))
}

pub fn unseed_auto_key<'a, 'b, R: RingOps>(
    ring: &R,
    auto_key: impl Into<RlweAutoKeyMutView<'a, R::Elem>>,
    auto_key_seeded: impl Into<SeededRlweAutoKeyView<'b, R::Elem>>,
    rng: &mut LweRng<(), impl RngCore>,
) {
    unseed_ks_key(
        ring,
        auto_key.into().ks_key_mut(),
        auto_key_seeded.into().ks_key(),
        rng,
    )
}
