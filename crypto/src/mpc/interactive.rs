use std::ops::Neg;

use itertools::Itertools;
use num_traits::AsPrimitive;
use phantom_zone_math::{
    modulus::ElemFrom,
    ring::RingOps,
    util::{
        as_slice::{AsMutSlice, AsSlice},
        scratch::Scratch,
    },
};
use rand::{RngCore, SeedableRng};
use structure::{
    split_lwe_dimension_for_rgsws, CommonReferenceSeededInteractiveServerKeyShare,
    InteractiveMpcParameters, RlwePublicKeyShareMutView,
};

use crate::{
    core::{
        lwe::{seeded_ks_key_gen, LweSecretKey, LweSecretKeyView, SeededLweKeySwitchKey},
        rgsw::pk_encrypt,
        rlwe::{
            seeded_auto_key_gen, RlwePlaintext, RlwePublicKeyView, RlweSecretKeyView,
            SeededRlweAutoKey,
        },
    },
    scheme::blind_rotation::lmkcdey::power_g_mod_q,
    util::{distribution::NoiseDistribution, rng::LweRng},
};

mod structure;

fn collective_rlwe_public_key_share<'a, R, T>(
    ring: &R,
    pk_share: impl Into<RlwePublicKeyShareMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'a, T>>,
    mut scratch: Scratch,
    noise_distribution: NoiseDistribution,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'a + Copy,
{
    let (mut pk_share, sk) = (pk_share.into(), sk.into());

    let t0 = ring.take_poly(&mut scratch);
    ring.sample_uniform_into(t0, rng.a());
    ring.poly_mul_assign_elem_from(t0, sk.as_ref(), scratch.reborrow());

    ring.sample_into::<f64>(pk_share.as_mut(), noise_distribution, rng.noise());
    ring.slice_add_assign(pk_share.as_mut(), t0);
}

pub fn crs_server_key_share<
    'a,
    'b,
    R,
    T,
    S1: AsMutSlice<Elem = R::Elem>,
    S2: AsMutSlice<Elem = usize>,
    Seed,
    Rng,
>(
    parameters: InteractiveMpcParameters,
    lwe_sk: impl Into<LweSecretKeyView<'a, T>>,
    rlwe_sk: impl Into<RlweSecretKeyView<'b, T>> + Clone,
    rlwe_pk: impl Into<RlwePublicKeyView<'b, R::Elem>>,
    // Random number generator to sample secret random / noise polynomials
    secret_noise_rng: &mut Rng,
    user_id: usize,
    total_users: usize,
    key: &mut CommonReferenceSeededInteractiveServerKeyShare<S1, S2, Seed>,
    mut scratch: Scratch,
) where
    R: RingOps + ElemFrom<T>,
    T: 'a + 'b + Copy + Neg<Output = T> + AsPrimitive<i64>,
    S1: Copy,
    Seed: Copy + Default + AsMut<[u8]>,
    Rng: RngCore + SeedableRng<Seed = Seed>,
{
    let (lwe_sk, rlwe_sk) = (lwe_sk.into(), rlwe_sk.into());
    let ring_rlwe = R::new(parameters.rlwe_q(), parameters.rlwe_n());
    let ring_lwe = R::new(parameters.lwe_q(), parameters.lwe_n());
    let crs = key.crs();

    // Seeded LWE key switching key
    let mut lwe_ksk_rng = crs.lwe_ksk_rng::<Rng>(secret_noise_rng);
    seeded_ks_key_gen(
        &ring_lwe,
        key.as_mut_lwe_ksk(),
        LweSecretKey::from(rlwe_sk).as_view(),
        lwe_sk,
        parameters.noise_distribution(),
        scratch.reborrow(),
        &mut lwe_ksk_rng,
    );

    // Seeded auto keys
    let mut auto_keys_rng = crs.auto_keys_rng::<Rng>(secret_noise_rng);
    key.auto_keys_mut_iter()
        .zip(power_g_mod_q(parameters.g(), parameters.br_q()).skip(1))
        .for_each(|(mut auto_key, k)| {
            assert_eq!(k, auto_key.auto_map().k());
            seeded_auto_key_gen(
                &ring_rlwe,
                auto_key.as_mut_view(),
                rlwe_sk.clone(),
                parameters.noise_distribution(),
                scratch.reborrow(),
                &mut auto_keys_rng,
            );
        });

    // Self RGSW ciphertexts
    let rlwe_pk = rlwe_pk.into();
    let (self_indices, not_self_indices) =
        split_lwe_dimension_for_rgsws(parameters.lwe_n(), user_id, total_users);
    let mut rgsw_cts_rng = crs.rgsw_cts_rng::<Rng>(secret_noise_rng);

    key.self_rgsw_cts_mut_iter()
        .zip_eq(self_indices)
        .for_each(|(mut rgsw_ct, index)| {
            // m = X^{s[i]}
            let lwe_sk_i = (parameters.embedding_factor() as i64) * (lwe_sk.as_ref()[index].as_());
            let mut m = RlwePlaintext::allocate(parameters.rlwe_n());
            ring_rlwe.poly_set_monomial(m.as_mut(), lwe_sk_i);

            pk_encrypt(
                &ring_rlwe,
                rgsw_ct.as_mut_view(),
                rlwe_pk.as_view(),
                m.as_view(),
                parameters.rlwe_secret_key_distribution(),
                parameters.noise_distribution(),
                scratch.reborrow(),
                &mut rgsw_cts_rng,
            );
        });

    key.not_self_rgsw_cts_mut_iter()
        .zip_eq(not_self_indices)
        .for_each(|(mut rgsw_ct, index)| {
            // m = X^{s[i]}
            let lwe_sk_i = (parameters.embedding_factor() as i64) * (lwe_sk.as_ref()[index].as_());
            let mut m = RlwePlaintext::allocate(parameters.rlwe_n());
            ring_rlwe.poly_set_monomial(m.as_mut(), lwe_sk_i);

            pk_encrypt(
                &ring_rlwe,
                rgsw_ct.as_mut_view(),
                rlwe_pk.as_view(),
                m.as_view(),
                parameters.rlwe_secret_key_distribution(),
                parameters.noise_distribution(),
                scratch.reborrow(),
                &mut rgsw_cts_rng,
            );
        });
}

fn aggregate_common_reference_seeded_server_key_shares<
    R,
    S1: AsMutSlice + Clone,
    S2: AsSlice<Elem = usize> + Clone,
    Seed: Clone,
>(
    parameters: InteractiveMpcParameters,
    shares: &[CommonReferenceSeededInteractiveServerKeyShare<S1, S2, Seed>],
    total_users: usize,
) where
    R: RingOps<Elem = S1::Elem>,
{
    // TODO(Jay): Make sure passed in shares are in order

    let rlwe_ring = R::new(parameters.rlwe_q(), parameters.rlwe_n());
    let lwe_ring = R::new(parameters.lwe_q(), parameters.lwe_n());

    let agg_seeded_lwe_ksk = SeededLweKeySwitchKey::aggregate(
        &lwe_ring,
        shares
            .iter()
            .map(|k| k.seeded_lwe_ksk())
            .collect_vec()
            .as_slice(),
    );

    let agg_auto_keys = ([(parameters.br_q() - parameters.g())].into_iter())
        .chain(
            power_g_mod_q(parameters.g(), parameters.br_q())
                .take(parameters.w() + 1)
                .skip(1)
                .map(|k| k),
        )
        .enumerate()
        .map(|(index, k)| {
            let auto_k_shares = shares
                .iter()
                .map(|s| {
                    let key = &s.seeded_auto_keys()[index];
                    assert!(key.auto_map().k() == k as usize);
                    key
                })
                .collect_vec();
            SeededRlweAutoKey::aggregate(&rlwe_ring, auto_k_shares.as_ref())
        })
        .collect_vec();

    for user_id in 0..total_users {
        let (self_indices, _) =
            split_lwe_dimension_for_rgsws(parameters.lwe_n(), user_id, total_users);

        self_indices.iter().for_each(|lwe_index| {
            let rgsw_acc = shares[user_id].rgsw_ciphertext_at_index(*lwe_index).clone();
            assert!(rgsw_acc.decomposition_param() == parameters.rlwe_x_rgsw_decomposer());

            for other_user_id in 0..total_users {
                if other_user_id == user_id {
                    continue;
                }

                let other_rgsw_ct = shares[other_user_id].rgsw_ciphertext_at_index(*lwe_index);
                assert!(other_rgsw_ct.decomposition_param() == parameters.rgsw_x_rgsw_decomposer());

                // multiply and accumulate
            }
        });
    }
}

pub trait KeyShareAggregator {
    type Key;
    type Elem;
    fn aggregate<R: RingOps<Elem = Self::Elem>>(ring: &R, values: &[&Self::Key]) -> Self::Key;
}
