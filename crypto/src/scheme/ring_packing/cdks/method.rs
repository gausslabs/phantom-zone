use crate::{
    core::{
        lwe::LweCiphertextView,
        rlwe::{
            self, auto_key_gen, seeded_auto_key_gen, RlweCiphertext, RlweCiphertextMutView,
            RlweSecretKeyView,
        },
    },
    scheme::ring_packing::cdks::structure::{CdksCrs, CdksKeyOwned, CdksKeyShareOwned},
    util::rng::{HierarchicalSeedableRng, LweRng},
};
use core::ops::Neg;
use itertools::Itertools;
use phantom_zone_math::{
    izip_eq,
    prelude::{ElemFrom, ModulusOps},
    ring::RingOps,
    util::scratch::Scratch,
};
use rand::{RngCore, SeedableRng};

pub fn rp_key_gen<'a, R, T>(
    ring: &R,
    rp_key: &mut CdksKeyOwned<R::Elem>,
    sk: impl Into<RlweSecretKeyView<'a, T>>,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'a + Copy + Neg<Output = T>,
{
    let sk_auto_neg_one = sk.into().automorphism(2 * ring.ring_size() - 1);
    let noise_distribution = rp_key.param().noise_distribution;
    let mut scratch = ring.allocate_scratch(0, 3, 0);
    rp_key.aks_mut().for_each(|ak| {
        let scratch = scratch.borrow_mut();
        auto_key_gen(ring, ak, &sk_auto_neg_one, noise_distribution, scratch, rng);
    });
}

pub fn prepare_rp_key<R: RingOps>(
    ring: &R,
    rp_key_prep: &mut CdksKeyOwned<R::EvalPrep>,
    rp_key: &CdksKeyOwned<R::Elem>,
) {
    debug_assert_eq!(rp_key_prep.param(), rp_key.param());
    let mut scratch = ring.allocate_scratch(0, 1, 0);
    izip_eq!(rp_key_prep.aks_mut(), rp_key.aks())
        .for_each(|(ak_prep, ak)| rlwe::prepare_auto_key(ring, ak_prep, ak, scratch.borrow_mut()));
}

pub fn pack_lwes<'a, 'b, R: RingOps>(
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    rp_key: &CdksKeyOwned<R::EvalPrep>,
    cts: impl IntoIterator<Item: Into<LweCiphertextView<'b, R::Elem>>>,
) {
    let cts = cts.into_iter().map_into().collect_vec();
    let n = cts.len().next_power_of_two();
    let n_inv = &ring.inv(&ring.elem_from(n as u64)).unwrap();
    pack_lwes_inner(ring, ct.into(), rp_key, &cts, &|mut ct, ct_0| {
        ring.slice_scalar_mul(ct.a_mut(), ct_0.a(), n_inv);
        ct.b_mut()[0] = ring.mul(ct_0.b(), n_inv);
        ct.b_mut()[1..].fill(ring.zero());
    })
}

pub fn pack_lwes_ms<'a, 'b, M: ModulusOps, R: RingOps>(
    mod_lwe: &M,
    ring: &R,
    ct: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    rp_key: &CdksKeyOwned<R::EvalPrep>,
    cts: impl IntoIterator<Item: Into<LweCiphertextView<'b, M::Elem>>>,
) {
    let cts = cts.into_iter().map_into().collect_vec();
    let n = cts.len().next_power_of_two();
    let n_inv = &ring.inv(&ring.elem_from(n as u64)).unwrap();
    pack_lwes_inner(ring, ct.into(), rp_key, &cts, &|mut ct, ct_0| {
        mod_lwe.slice_mod_switch(ct.a_mut(), ct_0.a(), ring);
        ring.slice_scalar_mul_assign(ct.a_mut(), n_inv);
        ct.b_mut()[0] = ring.mul(&mod_lwe.mod_switch(ct_0.b(), ring), n_inv);
        ct.b_mut()[1..].fill(ring.zero());
    })
}

fn pack_lwes_inner<R: RingOps, T>(
    ring: &R,
    mut ct: RlweCiphertextMutView<R::Elem>,
    rp_key: &CdksKeyOwned<R::EvalPrep>,
    cts: &[LweCiphertextView<T>],
    iota: &impl Fn(RlweCiphertextMutView<R::Elem>, LweCiphertextView<T>),
) {
    assert!(cts.len() <= ring.ring_size());
    let ell = cts.len().next_power_of_two().ilog2() as usize;
    let mut scratch = ring.allocate_scratch(2 + 2 * (ell + 2), 3, 0);
    if let (Some(ct_packed), _) = recurse(ring, rp_key, cts, iota, 1, 0, scratch.borrow_mut()) {
        ct.as_mut().copy_from_slice(ct_packed.as_ref());
    }

    fn recurse<'a, R: RingOps, T>(
        ring: &R,
        rp_key: &CdksKeyOwned<R::EvalPrep>,
        cts: &[LweCiphertextView<T>],
        iota: &impl Fn(RlweCiphertextMutView<R::Elem>, LweCiphertextView<T>),
        step: usize,
        skip: usize,
        mut scratch: Scratch<'a>,
    ) -> (Option<RlweCiphertextMutView<'a, R::Elem>>, Scratch<'a>) {
        let ring_size = ring.ring_size();
        let ell = (cts.len().next_power_of_two() / step).ilog2() as usize;

        if ell == 0 {
            return (
                cts.get(skip).map(|ct_0| {
                    let mut ct = RlweCiphertext::scratch(ring_size, ring_size, &mut scratch);
                    iota(ct.as_mut_view(), ct_0.as_view());
                    ct
                }),
                scratch,
            );
        }

        let (ct_even, mut scratch) = recurse(ring, rp_key, cts, iota, 2 * step, skip, scratch);
        let (ct_odd, mut tmp) = recurse(
            ring,
            rp_key,
            cts,
            iota,
            2 * step,
            skip + step,
            scratch.reborrow(),
        );

        let ct = match (ct_even, ct_odd) {
            (Some(mut ct_even), Some(mut ct_odd)) => {
                let ct_odd_rot = {
                    ring.poly_mul_assign_monomial(ct_odd.a_mut(), (ring_size >> ell) as _);
                    ring.poly_mul_assign_monomial(ct_odd.b_mut(), (ring_size >> ell) as _);
                    ct_odd
                };
                let ct_auto = {
                    let mut ct_auto = RlweCiphertext::scratch(ring_size, ring_size, &mut tmp);
                    ring.slice_sub(ct_auto.as_mut(), ct_even.as_ref(), ct_odd_rot.as_ref());
                    rlwe::automorphism_prep_in_place(ring, &mut ct_auto, rp_key.ak(ell), tmp);
                    ct_auto
                };
                ring.slice_add_assign(ct_even.as_mut(), ct_odd_rot.as_ref());
                ring.slice_add_assign(ct_even.as_mut(), ct_auto.as_ref());
                Some(ct_even)
            }
            (Some(mut ct_even), None) => {
                let ct_auto = {
                    let mut ct_auto = RlweCiphertext::scratch(ring_size, ring_size, &mut tmp);
                    ct_auto.as_mut().copy_from_slice(ct_even.as_ref());
                    rlwe::automorphism_prep_in_place(ring, &mut ct_auto, rp_key.ak(ell), tmp);
                    ct_auto
                };
                ring.slice_add_assign(ct_even.as_mut(), ct_auto.as_ref());
                Some(ct_even)
            }
            (None, None) => None,
            _ => unreachable!(),
        };
        (ct, scratch)
    }
}

pub fn rp_key_share_gen<'a, R, S, T>(
    ring: &R,
    rp_key_share: &mut CdksKeyShareOwned<R::Elem>,
    crs: &CdksCrs<S>,
    sk: impl Into<RlweSecretKeyView<'a, T>>,
    rng: &mut (impl RngCore + SeedableRng),
) where
    R: RingOps + ElemFrom<T>,
    S: HierarchicalSeedableRng,
    T: 'a + Copy + Neg<Output = T>,
{
    let sk_auto_neg_one = sk.into().automorphism(2 * ring.ring_size() - 1);
    let noise_distribution = rp_key_share.param().noise_distribution;
    let mut scratch = ring.allocate_scratch(0, 3, 0);

    let mut ak_rng = crs.ak_rng(rng);
    rp_key_share.aks_mut().for_each(|ak| {
        let scratch = scratch.borrow_mut();
        seeded_auto_key_gen(
            ring,
            ak,
            &sk_auto_neg_one,
            noise_distribution,
            scratch,
            &mut ak_rng,
        );
    });
}

pub fn aggregate_rp_key_shares<R, S>(
    ring: &R,
    rp_key: &mut CdksKeyOwned<R::Elem>,
    crs: &CdksCrs<S>,
    rp_key_shares: &[CdksKeyShareOwned<R::Elem>],
) where
    R: RingOps,
    S: HierarchicalSeedableRng,
{
    debug_assert!(!rp_key_shares.is_empty());
    rp_key_shares
        .iter()
        .for_each(|rp_key_share| debug_assert_eq!(rp_key_share.param(), rp_key.param()));

    let mut ak_rng = crs.unseed_ak_rng();
    izip_eq!(rp_key.aks_mut(), rp_key_shares[0].aks())
        .for_each(|(ak, ak_share)| rlwe::unseed_auto_key(ring, ak, ak_share, &mut ak_rng));
    rp_key_shares[1..].iter().for_each(|rp_key_share| {
        izip_eq!(rp_key.aks_mut(), rp_key_share.aks()).for_each(|(mut ak, ak_share)| {
            izip_eq!(ak.ks_key_mut().ct_iter_mut(), ak_share.ks_key().ct_iter())
                .for_each(|(mut ct, ct_share)| ring.slice_add_assign(ct.b_mut(), ct_share.b()));
        });
    });
}
