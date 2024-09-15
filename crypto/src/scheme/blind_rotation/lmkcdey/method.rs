use crate::{
    core::{
        lwe::{self, LweCiphertext, LweCiphertextMutView, LweCiphertextView, LweSecretKeyView},
        rgsw::{self},
        rlwe::{
            self, RlweCiphertext, RlweCiphertextMutView, RlwePlaintext, RlwePlaintextView,
            RlweSecretKeyView,
        },
    },
    scheme::blind_rotation::lmkcdey::structure::{LmkcdeyKeyOwned, LogGMap},
    util::rng::LweRng,
};
use core::{cmp::Reverse, ops::Neg};
use itertools::{izip, Itertools};
use num_traits::AsPrimitive;
use phantom_zone_math::{
    izip_eq,
    modulus::{ElemFrom, ModulusOps, NonNativePowerOfTwo},
    ring::RingOps,
    util::scratch::Scratch,
};
use rand::RngCore;

pub fn bs_key_gen<'a, 'b, R, M, T>(
    ring: &R,
    mod_ks: &M,
    bs_key: &mut LmkcdeyKeyOwned<R::Elem, M::Elem>,
    sk: impl Into<RlweSecretKeyView<'a, T>>,
    sk_ks: impl Into<LweSecretKeyView<'b, T>>,
    mut scratch: Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    M: ModulusOps + ElemFrom<T>,
    T: 'a + 'b + Copy + AsPrimitive<i64> + Neg<Output = T>,
{
    let (sk, sk_ks) = (sk.into(), sk_ks.into());
    let embedding_factor = bs_key.param().embedding_factor();
    let lwe_noise_distribution = bs_key.param().lwe_noise_distribution;
    let noise_distribution = bs_key.param().noise_distribution;

    lwe::ks_key_gen(
        mod_ks,
        bs_key.ks_key_mut(),
        sk,
        sk_ks,
        lwe_noise_distribution,
        rng,
    );

    bs_key.aks_mut().for_each(|ak| {
        rlwe::auto_key_gen(ring, ak, sk, noise_distribution, scratch.reborrow(), rng);
    });

    izip!(bs_key.brks_mut(), sk_ks.as_ref()).for_each(|(brk, sk_ks_i)| {
        let mut scratch = scratch.reborrow();
        let mut pt = RlwePlaintext::scratch(brk.ring_size(), &mut scratch);
        ring.poly_set_monomial(pt.as_mut(), embedding_factor as i64 * sk_ks_i.as_());
        rgsw::sk_encrypt(ring, brk, sk, &pt, noise_distribution, scratch, rng);
    });
}

pub fn prepare_bs_key<R: RingOps, T: Copy>(
    ring: &R,
    key_prep: &mut LmkcdeyKeyOwned<R::EvalPrep, T>,
    bs_key: &LmkcdeyKeyOwned<R::Elem, T>,
    mut scratch: Scratch,
) {
    debug_assert_eq!(key_prep.param(), bs_key.param());
    izip_eq!(
        key_prep.ks_key_mut().cts_iter_mut(),
        bs_key.ks_key().cts_iter()
    )
    .for_each(|(mut ct_prep, ct)| ct_prep.as_mut().copy_from_slice(ct.as_ref()));
    izip_eq!(key_prep.aks_mut(), bs_key.aks())
        .for_each(|(ak_prep, ak)| rlwe::prepare_auto_key(ring, ak_prep, ak, scratch.reborrow()));
    izip_eq!(key_prep.brks_mut(), bs_key.brks())
        .for_each(|(brk_prep, brk)| rgsw::prepare_rgsw(ring, brk_prep, brk, scratch.reborrow()));
}

/// Implementation of Figure 2 + Algorithm 7 in 2022/198.
pub fn bootstrap<'a, 'b, 'c, R: RingOps, M: ModulusOps>(
    ring: &R,
    mod_ks: &M,
    ct: impl Into<LweCiphertextMutView<'a, R::Elem>>,
    bs_key: &LmkcdeyKeyOwned<R::EvalPrep, M::Elem>,
    f_auto_neg_g: impl Into<RlwePlaintextView<'c, R::Elem>>,
    mut scratch: Scratch,
) {
    debug_assert_eq!((2 * ring.ring_size()) % bs_key.q(), 0);
    let (ct, f_auto_neg_g) = (ct.into(), f_auto_neg_g.into());

    let mut ct_ks_mod_switch = LweCiphertext::scratch(bs_key.ks_key().to_dimension(), &mut scratch);
    key_switch_mod_switch_odd(
        ring,
        mod_ks,
        ct_ks_mod_switch.as_mut_view(),
        bs_key,
        ct.as_view(),
        scratch.reborrow(),
    );

    let mut acc = RlweCiphertext::scratch(ring.ring_size(), ring.ring_size(), &mut scratch);
    acc.a_mut().fill(ring.zero());
    let embedding_factor = bs_key.embedding_factor();
    if embedding_factor == 1 {
        acc.b_mut().copy_from_slice(f_auto_neg_g.as_ref());
    } else {
        let acc_b = acc.b_mut().iter_mut().step_by(embedding_factor);
        izip_eq!(acc_b, f_auto_neg_g.as_ref()).for_each(|(b, a)| *b = *a);
    }
    let gb = bs_key.g() * *ct_ks_mod_switch.b() as usize;
    ring.poly_mul_assign_monomial(acc.b_mut(), (embedding_factor * gb) as _);

    blind_rotate_core(ring, &mut acc, bs_key, ct_ks_mod_switch.a(), scratch);

    rlwe::sample_extract(ring, ct, &acc, 0);
}

fn key_switch_mod_switch_odd<R: RingOps, M: ModulusOps>(
    ring: &R,
    mod_ks: &M,
    mut ct_ks_mod_switch: LweCiphertextMutView<u64>,
    bs_key: &LmkcdeyKeyOwned<R::EvalPrep, M::Elem>,
    ct: LweCiphertextView<R::Elem>,
    mut scratch: Scratch,
) {
    let mut ct_mod_switch = LweCiphertext::scratch(bs_key.ks_key().from_dimension(), &mut scratch);
    ring.slice_mod_switch(ct_mod_switch.as_mut(), ct.as_ref(), mod_ks);

    let mut ct_ks = LweCiphertext::scratch(bs_key.ks_key().to_dimension(), &mut scratch);
    lwe::key_switch(mod_ks, &mut ct_ks, bs_key.ks_key(), &ct_mod_switch);

    let mod_q = NonNativePowerOfTwo::new(bs_key.q().ilog2() as _);
    mod_ks.slice_mod_switch_odd(ct_ks_mod_switch.as_mut(), ct_ks.as_ref(), &mod_q);
}

/// Implementation of Algorithm 3 in 2022/198.
fn blind_rotate_core<'a, R: RingOps, T>(
    ring: &R,
    acc: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    bs_key: &LmkcdeyKeyOwned<R::EvalPrep, T>,
    a: &[u64],
    mut scratch: Scratch,
) {
    let [i_n, i_p] = &mut i_n_i_p(bs_key.log_g_map(), a, &mut scratch).map(|i| i.iter().peekable());
    let mut acc = acc.into();
    let mut v = 0;
    for l in (1..bs_key.q() / 4).rev() {
        for (_, j) in i_n.take_while_ref(|(log, _)| *log == l) {
            rgsw::rlwe_by_rgsw_prep_in_place(ring, &mut acc, bs_key.brk(*j), scratch.reborrow());
        }
        v += 1;
        let has_adj = i_n.peek().filter(|(log, _)| (*log == l - 1)).is_some();
        if has_adj || v == bs_key.w() || l == 1 {
            rlwe::automorphism_prep_in_place(ring, &mut acc, bs_key.ak(v), scratch.reborrow());
            v = 0
        }
    }
    for (_, j) in i_n {
        rgsw::rlwe_by_rgsw_prep_in_place(ring, &mut acc, bs_key.brk(*j), scratch.reborrow());
    }
    rlwe::automorphism_prep_in_place(ring, &mut acc, bs_key.ak_neg_g(), scratch.reborrow());
    for l in (1..bs_key.q() / 4).rev() {
        for (_, j) in i_p.take_while_ref(|(log, _)| *log == l) {
            rgsw::rlwe_by_rgsw_prep_in_place(ring, &mut acc, bs_key.brk(*j), scratch.reborrow());
        }
        v += 1;
        let has_adj = i_p.peek().filter(|(log, _)| (*log == l - 1)).is_some();
        if has_adj || v == bs_key.w() || l == 1 {
            rlwe::automorphism_prep_in_place(ring, &mut acc, bs_key.ak(v), scratch.reborrow());
            v = 0
        }
    }
    for (_, j) in i_p {
        rgsw::rlwe_by_rgsw_prep_in_place(ring, &mut acc, bs_key.brk(*j), scratch.reborrow());
    }
}

/// Returns negative and positive sets of indices `j` (of `a_j`) where
/// `a_j = -g^log` and `a_j = g^log`, and sets are sorted by `log` descendingly.
fn i_n_i_p<'a>(
    log_g_map: &LogGMap,
    a: &[u64],
    scratch: &mut Scratch<'a>,
) -> [&'a [(usize, usize)]; 2] {
    let [i_n, i_p] = scratch.take_slice_array::<(usize, usize), 2>(a.len());
    let mut i_n_count = 0;
    let mut i_p_count = 0;
    izip!(0.., a).for_each(|(j, a_j)| {
        if *a_j != 0 {
            let (sign, log) = log_g_map.index(*a_j as usize);
            if sign {
                i_n[i_n_count] = (log, j);
                i_n_count += 1
            } else {
                i_p[i_p_count] = (log, j);
                i_p_count += 1
            }
        }
    });
    i_n[..i_n_count].sort_by_key(|(log, _)| Reverse(*log));
    i_p[..i_p_count].sort_by_key(|(log, _)| Reverse(*log));
    [&i_n[..i_n_count], &i_p[..i_p_count]]
}
