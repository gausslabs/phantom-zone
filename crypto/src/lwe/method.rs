use crate::{
    distribution::{NoiseDistribution, SecretKeyDistribution},
    lwe::structure::{
        LweCiphertextMutView, LweCiphertextView, LweKeySwitchKeyMutView, LweKeySwitchKeyView,
        LwePlaintext, LweSecretKeyMutView, LweSecretKeyView,
    },
};
use itertools::Itertools;
use num_traits::{FromPrimitive, Signed};
use phantom_zone_math::{
    decomposer::Decomposer,
    distribution::DistributionSized,
    izip_eq,
    ring::{ElemFrom, RingOps},
};
use rand::RngCore;

pub fn sk_gen<T: Signed + FromPrimitive>(
    mut sk: LweSecretKeyMutView<T>,
    sk_distribution: SecretKeyDistribution,
    rng: impl RngCore,
) {
    sk_distribution.sample_into(sk.as_mut(), rng)
}

pub fn sk_encrypt<R: RingOps + ElemFrom<T>, T: Copy>(
    ring: &R,
    mut ct: LweCiphertextMutView<R::Elem>,
    sk: LweSecretKeyView<T>,
    pt: LwePlaintext<R::Elem>,
    noise_distribution: NoiseDistribution,
    mut rng: impl RngCore,
) {
    ring.sample_uniform_into(ct.a_mut(), &mut rng);
    let a_sk = ring.slice_dot_elem_from(ct.a(), sk.as_ref());
    let e = ring.sample::<i64>(&noise_distribution, &mut rng);
    *ct.b_mut() = ring.add(&ring.add(&a_sk, &e), &pt.0);
}

pub fn decrypt<R: RingOps + ElemFrom<T>, T: Copy>(
    ring: &R,
    sk: LweSecretKeyView<T>,
    ct: LweCiphertextView<R::Elem>,
) -> LwePlaintext<R::Elem> {
    let a_sk = ring.slice_dot_elem_from(ct.a(), sk.as_ref());
    LwePlaintext(ring.sub(ct.b(), &a_sk))
}

pub fn ks_key_gen<R: RingOps + ElemFrom<T>, T: Copy>(
    ring: &R,
    mut ks_key: LweKeySwitchKeyMutView<R::Elem>,
    sk_from: LweSecretKeyView<T>,
    sk_to: LweSecretKeyView<T>,
    noise_distribution: NoiseDistribution,
    mut rng: impl RngCore,
) {
    let decomposer = R::Decomposer::new(ring.modulus(), ks_key.decomposition_param());
    izip_eq!(
        &ks_key.ct_iter_mut().chunks(decomposer.level()),
        sk_from.as_ref(),
    )
    .for_each(|(ks_key_i, sk_from_i)| {
        izip_eq!(ks_key_i, decomposer.gadget_iter()).for_each(|(ks_key_i_j, beta_j)| {
            let pt = LwePlaintext(ring.mul_elem_from(&ring.neg(&beta_j), sk_from_i));
            sk_encrypt(ring, ks_key_i_j, sk_to, pt, noise_distribution, &mut rng)
        })
    });
}

pub fn key_switch<R: RingOps>(
    ring: &R,
    mut ct_to: LweCiphertextMutView<R::Elem>,
    ks_key: LweKeySwitchKeyView<R::Elem>,
    ct_from: LweCiphertextView<R::Elem>,
) {
    let decomposer = R::Decomposer::new(ring.modulus(), ks_key.decomposition_param());
    izip_eq!(&ks_key.ct_iter().chunks(decomposer.level()), ct_from.a()).for_each(
        |(ks_key_i, a_i)| {
            izip_eq!(ks_key_i, decomposer.decompose_iter(a_i)).for_each(|(ks_key_i_j, a_i_j)| {
                ring.slice_scalar_fma(ct_to.as_mut(), ks_key_i_j.as_ref(), &a_i_j)
            })
        },
    );
    *ct_to.b_mut() = ring.add(ct_to.b(), ct_from.b());
}
