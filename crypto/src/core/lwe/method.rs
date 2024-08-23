use crate::{
    core::lwe::structure::{
        LweCiphertextMutView, LweCiphertextView, LweKeySwitchKeyMutView, LweKeySwitchKeyView,
        LwePlaintext, LweSecretKeyView,
    },
    util::{distribution::NoiseDistribution, rng::LweRng},
};
use phantom_zone_math::{
    decomposer::Decomposer,
    izip_eq,
    modulus::{ElemFrom, ModulusOps},
};
use rand::RngCore;

pub fn sk_encrypt<'a, 'b, M, T>(
    modulus: &M,
    ct: impl Into<LweCiphertextMutView<'a, M::Elem>>,
    sk: impl Into<LweSecretKeyView<'b, T>>,
    pt: LwePlaintext<M::Elem>,
    noise_distribution: NoiseDistribution,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    M: ModulusOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let mut ct = ct.into();
    sk_encrypt_zero(modulus, &mut ct, sk, noise_distribution, rng);
    modulus.add_assign(ct.b_mut(), &pt.0);
}

fn sk_encrypt_zero<'a, 'b, M, T>(
    modulus: &M,
    ct: impl Into<LweCiphertextMutView<'a, M::Elem>>,
    sk: impl Into<LweSecretKeyView<'b, T>>,
    noise_distribution: NoiseDistribution,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    M: ModulusOps + ElemFrom<T>,
    T: 'b + Copy,
{
    let mut ct = ct.into();
    modulus.sample_uniform_into(ct.a_mut(), rng.seedable());
    let a_sk = modulus.slice_dot_elem_from(ct.a(), sk.into().as_ref());
    let e = modulus.sample::<i64>(&noise_distribution, rng);
    *ct.b_mut() = modulus.add(&a_sk, &e);
}

pub fn decrypt<'a, 'b, M, T>(
    modulus: &M,
    sk: impl Into<LweSecretKeyView<'a, T>>,
    ct: impl Into<LweCiphertextView<'b, M::Elem>>,
) -> LwePlaintext<M::Elem>
where
    M: ModulusOps + ElemFrom<T>,
    T: 'a + Copy,
{
    let ct = ct.into();
    let a_sk = modulus.slice_dot_elem_from(ct.a(), sk.into().as_ref());
    LwePlaintext(modulus.sub(ct.b(), &a_sk))
}

pub fn ks_key_gen<'a, 'b, 'c, M, T>(
    modulus: &M,
    ks_key: impl Into<LweKeySwitchKeyMutView<'a, M::Elem>>,
    sk_from: impl Into<LweSecretKeyView<'b, T>>,
    sk_to: impl Into<LweSecretKeyView<'c, T>>,
    noise_distribution: NoiseDistribution,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    M: ModulusOps + ElemFrom<T>,
    T: 'b + 'c + Copy,
{
    let (mut ks_key, sk_to) = (ks_key.into(), sk_to.into());
    let decomposer = M::Decomposer::new(modulus.modulus(), ks_key.decomposition_param());
    izip_eq!(ks_key.cts_iter_mut(), sk_from.into().as_ref()).for_each(
        |(mut ks_key_i, sk_from_i)| {
            izip_eq!(ks_key_i.iter_mut(), decomposer.gadget_iter()).for_each(
                |(mut ks_key_i_j, beta_j)| {
                    sk_encrypt_zero(modulus, &mut ks_key_i_j, sk_to, noise_distribution, rng);
                    modulus.sub_assign(
                        ks_key_i_j.b_mut(),
                        &modulus.mul_elem_from(&beta_j, sk_from_i),
                    );
                },
            )
        },
    );
}

pub fn key_switch<'a, 'b, 'c, M: ModulusOps>(
    modulus: &M,
    ct_to: impl Into<LweCiphertextMutView<'a, M::Elem>>,
    ks_key: impl Into<LweKeySwitchKeyView<'b, M::Elem>>,
    ct_from: impl Into<LweCiphertextView<'c, M::Elem>>,
) {
    let (mut ct_to, ks_key, ct_from) = (ct_to.into(), ks_key.into(), ct_from.into());
    let decomposer = M::Decomposer::new(modulus.modulus(), ks_key.decomposition_param());
    izip_eq!(ks_key.cts_iter(), ct_from.a())
        .enumerate()
        .for_each(|(i, (ks_key_i, a_i))| {
            izip_eq!(ks_key_i.iter(), decomposer.decompose_iter(a_i))
                .enumerate()
                .for_each(|(j, (ks_key_i_j, a_i_j))| {
                    let slice_scalar_fma = if i == 0 && j == 0 {
                        M::slice_scalar_mul
                    } else {
                        M::slice_scalar_fma
                    };
                    slice_scalar_fma(modulus, ct_to.as_mut(), ks_key_i_j.as_ref(), &a_i_j)
                })
        });
    *ct_to.b_mut() = modulus.add(ct_to.b(), ct_from.b());
}
