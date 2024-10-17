use crate::{
    core::{
        lwe::{self, LweSecretKeyView},
        rgsw::{self},
        rlwe::{
            self, RlwePlaintext, RlwePublicKeyMutView, RlwePublicKeyView, RlweSecretKeyView,
            SeededRlwePublicKeyMutView, SeededRlwePublicKeyView,
        },
    },
    scheme::blind_rotation::lmkcdey::{
        interactive::structure::{LmkcdeyMpiCrs, LmkcdeyMpiKeyShareOwned, LmkcdeyMpiParam},
        structure::LmkcdeyKeyOwned,
    },
    util::rng::HierarchicalSeedableRng,
};
use core::ops::{Deref, Neg};
use itertools::{izip, Itertools};
use num_traits::AsPrimitive;
use phantom_zone_math::{
    izip_eq,
    modulus::{ElemFrom, ModulusOps},
    ring::RingOps,
};
use rand::{RngCore, SeedableRng};

pub fn pk_share_gen<'a, 'b, R, S, T>(
    ring: &R,
    pk: impl Into<SeededRlwePublicKeyMutView<'a, R::Elem>>,
    param: &LmkcdeyMpiParam,
    crs: &LmkcdeyMpiCrs<S>,
    sk: impl Into<RlweSecretKeyView<'b, T>>,
    rng: &mut (impl RngCore + SeedableRng),
) where
    R: RingOps + ElemFrom<T>,
    S: HierarchicalSeedableRng,
    T: 'b + Copy,
{
    let mut scratch = ring.allocate_scratch(2, 2, 0);
    let mut pk_rng = crs.pk_rng(rng);
    rlwe::seeded_pk_gen(
        ring,
        pk,
        sk,
        param.noise_distribution,
        scratch.borrow_mut(),
        &mut pk_rng,
    )
}

pub fn bs_key_share_gen<'a, 'b, 'c, R, M, S, T>(
    ring: &R,
    mod_ks: &M,
    bs_key_share: &mut LmkcdeyMpiKeyShareOwned<R::Elem, M::Elem>,
    crs: &LmkcdeyMpiCrs<S>,
    sk: impl Into<RlweSecretKeyView<'a, T>>,
    pk: impl Into<RlwePublicKeyView<'b, R::Elem>>,
    sk_ks: impl Into<LweSecretKeyView<'c, T>>,
    rng: &mut (impl RngCore + SeedableRng),
) where
    R: RingOps + ElemFrom<T>,
    M: ModulusOps + ElemFrom<T>,
    S: HierarchicalSeedableRng,
    T: 'a + 'c + Copy + AsPrimitive<i64> + Neg<Output = T>,
{
    let (sk, pk, sk_ks) = (sk.into(), pk.into(), sk_ks.into());
    let embedding_factor = bs_key_share.param().embedding_factor();
    let u_distribution = bs_key_share.param().u_distribution;
    let lwe_noise_distribution = bs_key_share.param().lwe_noise_distribution;
    let noise_distribution = bs_key_share.param().noise_distribution;
    let mut scratch = ring.allocate_scratch(3, 2, 0);

    let mut ks_key_rng = crs.ks_key_rng(rng);
    lwe::seeded_ks_key_gen(
        mod_ks,
        bs_key_share.ks_key_mut(),
        sk,
        sk_ks,
        lwe_noise_distribution,
        scratch.borrow_mut(),
        &mut ks_key_rng,
    );

    let mut ak_rng = crs.ak_rng(rng);
    bs_key_share.aks_mut().for_each(|ak| {
        rlwe::seeded_auto_key_gen(
            ring,
            ak,
            sk,
            noise_distribution,
            scratch.borrow_mut(),
            &mut ak_rng,
        );
    });

    let mut brk_rng = crs.brk_rng(rng);
    izip!(bs_key_share.brks_mut(), sk_ks.as_ref()).for_each(|(brk, sk_ks_i)| {
        let mut scratch = scratch.borrow_mut();
        let mut pt = RlwePlaintext::scratch(ring.ring_size(), &mut scratch);
        ring.poly_set_monomial(pt.as_mut(), embedding_factor as i64 * sk_ks_i.as_());
        rgsw::pk_encrypt(
            ring,
            brk,
            pk,
            &pt,
            u_distribution,
            noise_distribution,
            scratch,
            &mut brk_rng,
        );
    });
}

pub fn aggregate_bs_key_shares<'a, R, M, S>(
    ring: &R,
    mod_ks: &M,
    bs_key: &mut LmkcdeyKeyOwned<R::Elem, M::Elem>,
    crs: &LmkcdeyMpiCrs<S>,
    bs_key_shares: impl IntoIterator<Item = &'a LmkcdeyMpiKeyShareOwned<R::Elem, M::Elem>>,
) where
    R: RingOps,
    M: ModulusOps,
    S: HierarchicalSeedableRng,
{
    let bs_key_shares = bs_key_shares.into_iter().collect_vec();
    debug_assert!(!bs_key_shares.is_empty());
    debug_assert_eq!(bs_key_shares.len(), bs_key_shares[0].param().total_shares);
    izip!(0.., &bs_key_shares).for_each(|(share_idx, bs_key_share)| {
        debug_assert_eq!(bs_key_share.param().deref(), bs_key.param());
        debug_assert_eq!(share_idx, bs_key_share.share_idx());
    });

    let mut ks_key_rng = crs.unseed_ks_key_rng();
    lwe::unseed_ks_key(
        mod_ks,
        bs_key.ks_key_mut(),
        bs_key_shares[0].ks_key(),
        &mut ks_key_rng,
    );
    bs_key_shares[1..].iter().for_each(|bs_key_share| {
        izip_eq!(
            bs_key.ks_key_mut().cts_iter_mut(),
            bs_key_share.ks_key().cts_iter()
        )
        .for_each(|(mut cts, cts_share)| {
            izip_eq!(cts.iter_mut(), cts_share.iter())
                .for_each(|(mut ct, ct_share)| mod_ks.add_assign(ct.b_mut(), ct_share.b()));
        });
    });

    let mut ak_rng = crs.unseed_ak_rng();
    izip_eq!(bs_key.aks_mut(), bs_key_shares[0].aks())
        .for_each(|(ak, ak_share)| rlwe::unseed_auto_key(ring, ak, ak_share, &mut ak_rng));
    bs_key_shares[1..].iter().for_each(|bs_key_share| {
        izip_eq!(bs_key.aks_mut(), bs_key_share.aks()).for_each(|(mut ak, ak_share)| {
            izip_eq!(ak.ks_key_mut().ct_iter_mut(), ak_share.ks_key().ct_iter())
                .for_each(|(mut ct, ct_share)| ring.slice_add_assign(ct.b_mut(), ct_share.b()));
        });
    });

    let mut scratch = {
        let decomposition_param = bs_key_shares[0].param().rgsw_by_rgsw_decomposition_param;
        let level = decomposition_param.level_a + decomposition_param.level_b;
        ring.allocate_scratch(2, 3, 2 * level)
    };
    let chunk_size = bs_key.param().lwe_dimension.div_ceil(bs_key_shares.len());
    for bs_key_share_init in &bs_key_shares {
        let start = chunk_size * bs_key_share_init.share_idx();
        izip_eq!(bs_key.brks_mut(), bs_key_share_init.brks())
            .skip(start)
            .take(chunk_size)
            .for_each(|(mut ct, ct_init)| {
                izip_eq!(ct.ct_iter_mut(), ct_init.ct_iter())
                    .for_each(|(mut ct, ct_init)| ct.as_mut().copy_from_slice(ct_init.as_ref()));
            });
        bs_key_shares
            .iter()
            .filter(|bs_key_share| bs_key_share.share_idx() != bs_key_share_init.share_idx())
            .for_each(|bs_key_share| {
                izip_eq!(bs_key.brks_mut(), bs_key_share.brks())
                    .skip(start)
                    .take(chunk_size)
                    .for_each(|(ct, ct_share)| {
                        rgsw::rgsw_by_rgsw_in_place(ring, ct, ct_share, scratch.borrow_mut());
                    });
            });
    }
}

pub fn aggregate_pk_shares<'a, 'b, R, S>(
    ring: &R,
    pk: impl Into<RlwePublicKeyMutView<'a, R::Elem>>,
    crs: &LmkcdeyMpiCrs<S>,
    pk_shares: impl IntoIterator<Item: Into<SeededRlwePublicKeyView<'b, R::Elem>>>,
) where
    R: RingOps,
    S: HierarchicalSeedableRng,
{
    let pk_shares = pk_shares.into_iter().map_into().collect_vec();
    debug_assert!(!pk_shares.is_empty());

    let mut pk = pk.into();
    rlwe::unseed_pk(ring, &mut pk, pk_shares[0], &mut crs.unseed_pk_rng());
    pk_shares[1..]
        .iter()
        .for_each(|pk_share| ring.slice_add_assign(pk.b_mut(), pk_share.b()));
}
