use std::{fmt::Display, marker::PhantomData};

use num_traits::{FromPrimitive, One, PrimInt, ToPrimitive, Zero};

use crate::{
    backend::{ArithmeticOps, Modulus, VectorOps},
    decomposer::Decomposer,
    lwe::lwe_key_switch,
    ntt::Ntt,
    random::DefaultSecureRng,
    rgsw::{galois_auto, rlwe_by_rgsw, IsTrivial, RlweCiphertext},
    Matrix, MatrixEntity, MatrixMut, RowMut,
};
pub(crate) trait PbsKey {
    type M: Matrix;

    /// RGSW ciphertext of LWE secret elements
    fn rgsw_ct_lwe_si(&self, si: usize) -> &Self::M;
    /// Key for automorphism with g^k. For -g use k = 0
    fn galois_key_for_auto(&self, k: usize) -> &Self::M;
    /// LWE ksk to key switch from RLWE secret to LWE secret
    fn lwe_ksk(&self) -> &Self::M;
}

pub(crate) trait PbsInfo {
    type Element;
    type Modulus: Modulus<Element = Self::Element>;
    type NttOp: Ntt<Element = Self::Element>;
    type D: Decomposer<Element = Self::Element>;

    // Although both types have same bounds, they can be different types. For ex,
    // type RlweModOp may only support native modulus, where LweModOp may only
    // support prime modulus, etc.
    type RlweModOp: VectorOps<Element = Self::Element> + ArithmeticOps<Element = Self::Element>;
    type LweModOp: VectorOps<Element = Self::Element> + ArithmeticOps<Element = Self::Element>;

    fn rlwe_q(&self) -> &Self::Modulus;
    fn lwe_q(&self) -> &Self::Modulus;
    fn br_q(&self) -> usize;
    fn rlwe_n(&self) -> usize;
    fn lwe_n(&self) -> usize;
    /// Embedding fator for ring X^{q}+1 inside
    fn embedding_factor(&self) -> usize;
    /// Window size
    fn w(&self) -> usize;
    /// generator g
    fn g(&self) -> isize;
    /// Decomposers
    fn lwe_decomposer(&self) -> &Self::D;
    fn rlwe_rgsw_decomposer(&self) -> &(Self::D, Self::D);
    fn auto_decomposer(&self) -> &Self::D;

    /// Modulus operators
    fn modop_lweq(&self) -> &Self::LweModOp;
    fn modop_rlweq(&self) -> &Self::RlweModOp;

    /// Ntt operators
    fn nttop_rlweq(&self) -> &Self::NttOp;

    /// Maps a \in Z^*_{q} to discrete log k, with generator g (i.e. g^k =
    /// a). Returned vector is of size q that stores dlog of a at `vec[a]`.
    /// For any a, if k is s.t. a = g^{k}, then k is expressed as k. If k is s.t
    /// a = -g^{k}, then k is expressed as k=k+q/4
    fn g_k_dlog_map(&self) -> &[usize];
    /// Returns auto map and index vector for g^k. For -g use k == 0.
    fn rlwe_auto_map(&self, k: usize) -> &(Vec<usize>, Vec<bool>);
}

/// - Mod down
/// - key switching
/// - mod down
/// - blind rotate
pub(crate) fn pbs<
    M: MatrixMut + MatrixEntity,
    P: PbsInfo<Element = M::MatElement>,
    K: PbsKey<M = M>,
>(
    pbs_info: &P,
    test_vec: &M::R,
    lwe_in: &mut M::R,
    pbs_key: &K,
    scratch_lwe_vec: &mut M::R,
    scratch_blind_rotate_matrix: &mut M,
) where
    <M as Matrix>::R: RowMut,
    M::MatElement: PrimInt + FromPrimitive + One + Copy + Zero + Display,
{
    let rlwe_q = pbs_info.rlwe_q();
    let lwe_q = pbs_info.lwe_q();
    let br_q = pbs_info.br_q();
    let rlwe_qf64 = rlwe_q.q_as_f64().unwrap();
    let lwe_qf64 = lwe_q.q_as_f64().unwrap();
    let br_qf64 = br_q.to_f64().unwrap();
    let rlwe_n = pbs_info.rlwe_n();

    // moddown Q -> Q_ks
    lwe_in.as_mut().iter_mut().for_each(|v| {
        *v =
            M::MatElement::from_f64(((v.to_f64().unwrap() * lwe_qf64) / rlwe_qf64).round()).unwrap()
    });

    // key switch RLWE secret to LWE secret
    scratch_lwe_vec.as_mut().fill(M::MatElement::zero());
    lwe_key_switch(
        scratch_lwe_vec,
        lwe_in,
        pbs_key.lwe_ksk(),
        pbs_info.modop_lweq(),
        pbs_info.lwe_decomposer(),
    );

    // odd mowdown Q_ks -> q
    let g_k_dlog_map = pbs_info.g_k_dlog_map();
    let mut g_k_si = vec![vec![]; br_q >> 1];
    scratch_lwe_vec
        .as_ref()
        .iter()
        .skip(1)
        .enumerate()
        .for_each(|(index, v)| {
            let odd_v = mod_switch_odd(v.to_f64().unwrap(), lwe_qf64, br_qf64);
            // dlog `k` for `odd_v` is stored as `k` if odd_v = +g^{k}. If odd_v = -g^{k},
            // then `k` is stored as `q/4 + k`.
            let k = g_k_dlog_map[odd_v];
            // assert!(k != 0);
            g_k_si[k].push(index);
        });

    // handle b and set trivial test RLWE
    let g = pbs_info.g() as usize;
    let g_times_b = (g * mod_switch_odd(
        scratch_lwe_vec.as_ref()[0].to_f64().unwrap(),
        lwe_qf64,
        br_qf64,
    )) % (br_q);
    // v = (v(X) * X^{g*b}) mod X^{q/2}+1
    let br_qby2 = br_q >> 1;
    let mut gb_monomial_sign = true;
    let mut gb_monomial_exp = g_times_b;
    // X^{g*b} mod X^{q/2}+1
    if gb_monomial_exp > br_qby2 {
        gb_monomial_exp -= br_qby2;
        gb_monomial_sign = false
    }
    // monomial mul
    let mut trivial_rlwe_test_poly = RlweCiphertext::<_, DefaultSecureRng> {
        data: M::zeros(2, rlwe_n),
        is_trivial: true,
        _phatom: PhantomData,
    };
    if pbs_info.embedding_factor() == 1 {
        monomial_mul(
            test_vec.as_ref(),
            trivial_rlwe_test_poly.get_row_mut(1).as_mut(),
            gb_monomial_exp,
            gb_monomial_sign,
            br_qby2,
            pbs_info.modop_rlweq(),
        );
    } else {
        // use lwe_in to store the `t = v(X) * X^{g*2} mod X^{q/2}+1` temporarily. This
        // works because q/2 <= N (where N is lwe_in LWE dimension) always.
        monomial_mul(
            test_vec.as_ref(),
            &mut lwe_in.as_mut()[..br_qby2],
            gb_monomial_exp,
            gb_monomial_sign,
            br_qby2,
            pbs_info.modop_rlweq(),
        );

        // emebed poly `t` in ring X^{q/2}+1 inside the bigger ring X^{N}+1
        let embed_factor = pbs_info.embedding_factor();
        let partb_trivial_rlwe = trivial_rlwe_test_poly.get_row_mut(1);
        lwe_in.as_ref()[..br_qby2]
            .iter()
            .enumerate()
            .for_each(|(index, v)| {
                partb_trivial_rlwe[embed_factor * index] = *v;
            });
    }

    // let now = std::time::Instant::now();
    // blind rotate
    blind_rotation(
        &mut trivial_rlwe_test_poly,
        scratch_blind_rotate_matrix,
        pbs_info.g(),
        pbs_info.w(),
        br_q,
        &g_k_si,
        pbs_info.rlwe_rgsw_decomposer(),
        pbs_info.auto_decomposer(),
        pbs_info.nttop_rlweq(),
        pbs_info.modop_rlweq(),
        pbs_info,
        pbs_key,
    );
    // println!("Blind rotation time: {:?}", now.elapsed());

    // sample extract
    sample_extract(lwe_in, &trivial_rlwe_test_poly, pbs_info.modop_rlweq(), 0);
}

/// LMKCY+ Blind rotation
///
/// gk_to_si: [g^0, ..., g^{q/2-1}, -g^0, -g^1, .., -g^{q/2-1}]
fn blind_rotation<
    MT: IsTrivial + MatrixMut,
    Mmut: MatrixMut<MatElement = MT::MatElement>,
    D: Decomposer<Element = MT::MatElement>,
    NttOp: Ntt<Element = MT::MatElement>,
    ModOp: ArithmeticOps<Element = MT::MatElement> + VectorOps<Element = MT::MatElement>,
    K: PbsKey<M = Mmut>,
    P: PbsInfo<Element = MT::MatElement>,
>(
    trivial_rlwe_test_poly: &mut MT,
    scratch_matrix: &mut Mmut,
    _g: isize,
    w: usize,
    q: usize,
    gk_to_si: &[Vec<usize>],
    rlwe_rgsw_decomposer: &(D, D),
    auto_decomposer: &D,
    ntt_op: &NttOp,
    mod_op: &ModOp,
    parameters: &P,
    pbs_key: &K,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut::MatElement: Copy + Zero,
    <MT as Matrix>::R: RowMut,
{
    let q_by_4 = q >> 2;
    let mut count = 0;
    // -(g^k)
    let mut v = 0;
    for i in (1..q_by_4).rev() {
        // dbg!(q_by_4 + i);
        let s_indices = &gk_to_si[q_by_4 + i];

        s_indices.iter().for_each(|s_index| {
            // let new = std::time::Instant::now();
            rlwe_by_rgsw(
                trivial_rlwe_test_poly,
                pbs_key.rgsw_ct_lwe_si(*s_index),
                scratch_matrix,
                rlwe_rgsw_decomposer,
                ntt_op,
                mod_op,
            );
            // println!("Rlwe x Rgsw time: {:?}", new.elapsed());
        });
        v += 1;

        if gk_to_si[q_by_4 + i - 1].len() != 0 || v == w || i == 1 {
            let (auto_map_index, auto_map_sign) = parameters.rlwe_auto_map(v);

            // let now = std::time::Instant::now();
            galois_auto(
                trivial_rlwe_test_poly,
                pbs_key.galois_key_for_auto(v),
                scratch_matrix,
                &auto_map_index,
                &auto_map_sign,
                mod_op,
                ntt_op,
                auto_decomposer,
            );
            // println!("Auto time: {:?}", now.elapsed());

            count += 1;
            v = 0;
        }
    }

    // -(g^0)
    gk_to_si[q_by_4].iter().for_each(|s_index| {
        rlwe_by_rgsw(
            trivial_rlwe_test_poly,
            pbs_key.rgsw_ct_lwe_si(*s_index),
            scratch_matrix,
            rlwe_rgsw_decomposer,
            ntt_op,
            mod_op,
        );
    });
    let (auto_map_index, auto_map_sign) = parameters.rlwe_auto_map(0);
    galois_auto(
        trivial_rlwe_test_poly,
        pbs_key.galois_key_for_auto(0),
        scratch_matrix,
        &auto_map_index,
        &auto_map_sign,
        mod_op,
        ntt_op,
        auto_decomposer,
    );
    count += 1;

    // +(g^k)
    let mut v = 0;
    for i in (1..q_by_4).rev() {
        let s_indices = &gk_to_si[i];
        s_indices.iter().for_each(|s_index| {
            rlwe_by_rgsw(
                trivial_rlwe_test_poly,
                pbs_key.rgsw_ct_lwe_si(*s_index),
                scratch_matrix,
                rlwe_rgsw_decomposer,
                ntt_op,
                mod_op,
            );
        });
        v += 1;

        if gk_to_si[i - 1].len() != 0 || v == w || i == 1 {
            let (auto_map_index, auto_map_sign) = parameters.rlwe_auto_map(v);
            galois_auto(
                trivial_rlwe_test_poly,
                pbs_key.galois_key_for_auto(v),
                scratch_matrix,
                &auto_map_index,
                &auto_map_sign,
                mod_op,
                ntt_op,
                auto_decomposer,
            );
            v = 0;
            count += 1;
        }
    }

    // +(g^0)
    gk_to_si[0].iter().for_each(|s_index| {
        rlwe_by_rgsw(
            trivial_rlwe_test_poly,
            pbs_key.rgsw_ct_lwe_si(*s_index),
            scratch_matrix,
            rlwe_rgsw_decomposer,
            ntt_op,
            mod_op,
        );
    });
    println!("Auto count: {count}");
}

fn mod_switch_odd(v: f64, from_q: f64, to_q: f64) -> usize {
    let odd_v = (((v * to_q) / (from_q)).floor()).to_usize().unwrap();
    //TODO(Jay): check correctness of this
    odd_v + ((odd_v & 1) ^ 1)
}

// TODO(Jay): Add tests for sample extract
pub(crate) fn sample_extract<M: Matrix + MatrixMut, ModOp: ArithmeticOps<Element = M::MatElement>>(
    lwe_out: &mut M::R,
    rlwe_in: &M,
    mod_op: &ModOp,
    index: usize,
) where
    <M as Matrix>::R: RowMut,
    M::MatElement: Copy,
{
    let ring_size = rlwe_in.dimension().1;

    // index..=0
    let to = &mut lwe_out.as_mut()[1..];
    let from = rlwe_in.get_row_slice(0);
    for i in 0..index + 1 {
        to[i] = from[index - i];
    }

    // -(N..index)
    for i in index + 1..ring_size {
        to[i] = mod_op.neg(&from[ring_size + index - i]);
    }

    // set b
    lwe_out.as_mut()[0] = *rlwe_in.get(1, index);
}

/// Monomial multiplication (p(X)*X^{mon_exp})
///
/// - p_out: Output is written to p_out and independent of values in p_out
fn monomial_mul<El, ModOp: ArithmeticOps<Element = El>>(
    p_in: &[El],
    p_out: &mut [El],
    mon_exp: usize,
    mon_sign: bool,
    ring_size: usize,
    mod_op: &ModOp,
) where
    El: Copy,
{
    debug_assert!(p_in.as_ref().len() == ring_size);
    debug_assert!(p_in.as_ref().len() == p_out.as_ref().len());
    debug_assert!(mon_exp < ring_size);

    p_in.as_ref().iter().enumerate().for_each(|(index, v)| {
        let mut to_index = index + mon_exp;
        let mut to_sign = mon_sign;
        if to_index >= ring_size {
            to_index = to_index - ring_size;
            to_sign = !to_sign;
        }

        if !to_sign {
            p_out.as_mut()[to_index] = mod_op.neg(v);
        } else {
            p_out.as_mut()[to_index] = *v;
        }
    });
}
