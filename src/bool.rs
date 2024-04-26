use std::collections::HashMap;

use num_traits::{FromPrimitive, One, PrimInt, ToPrimitive, Zero};

use crate::{
    backend::{ArithmeticOps, VectorOps},
    decomposer::Decomposer,
    lwe::lwe_key_switch,
    ntt::Ntt,
    rgsw::{galois_auto, rlwe_by_rgsw, IsTrivial},
    Matrix, MatrixEntity, MatrixMut, Row, RowMut,
};

struct BoolEvaluator {}

impl BoolEvaluator {}

trait PbsKey {
    type M: Matrix;

    fn rgsw_ct_secret_el(&self, si: usize) -> &Self::M;
    fn galois_key_for_auto(&self, k: isize) -> &Self::M;
    fn auto_map_index(&self, k: isize) -> &[usize];
    fn auto_map_sign(&self, k: isize) -> &[bool];
}

/// LMKCY+ Blind rotation
///
/// gk_to_si: [-g^0, -g^1, .., -g^{q/2-1}, g^0, ..., g^{q/2-1}]
fn blind_rotation<
    MT: IsTrivial + MatrixMut,
    Mmut: MatrixMut<MatElement = MT::MatElement> + Matrix,
    D: Decomposer<Element = MT::MatElement>,
    NttOp: Ntt<Element = MT::MatElement>,
    ModOp: ArithmeticOps<Element = MT::MatElement> + VectorOps<Element = MT::MatElement>,
    K: PbsKey<M = Mmut>,
>(
    trivial_rlwe_test_poly: &mut MT,
    scratch_matrix_dplus2_ring: &mut Mmut,
    g: isize,
    w: usize,
    q: usize,
    gk_to_si: &[Vec<usize>],
    decomposer: &D,
    ntt_op: &NttOp,
    mod_op: &ModOp,
    pbs_key: &K,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut::MatElement: Copy + Zero,
    <MT as Matrix>::R: RowMut,
{
    let q_by_2 = q / 2;

    // -(g^k)
    for i in 1..q_by_2 {
        gk_to_si[q_by_2 + i].iter().for_each(|s_index| {
            rlwe_by_rgsw(
                trivial_rlwe_test_poly,
                pbs_key.rgsw_ct_secret_el(*s_index),
                scratch_matrix_dplus2_ring,
                decomposer,
                ntt_op,
                mod_op,
            );
        });

        galois_auto(
            trivial_rlwe_test_poly,
            pbs_key.galois_key_for_auto(g),
            scratch_matrix_dplus2_ring,
            pbs_key.auto_map_index(g),
            pbs_key.auto_map_sign(g),
            mod_op,
            ntt_op,
            decomposer,
        );
    }

    // -(g^0)
    gk_to_si[q_by_2].iter().for_each(|s_index| {
        rlwe_by_rgsw(
            trivial_rlwe_test_poly,
            pbs_key.rgsw_ct_secret_el(*s_index),
            scratch_matrix_dplus2_ring,
            decomposer,
            ntt_op,
            mod_op,
        );
    });
    galois_auto(
        trivial_rlwe_test_poly,
        pbs_key.galois_key_for_auto(-g),
        scratch_matrix_dplus2_ring,
        pbs_key.auto_map_index(-g),
        pbs_key.auto_map_sign(-g),
        mod_op,
        ntt_op,
        decomposer,
    );

    // +(g^k)
    for i in 1..q_by_2 {
        gk_to_si[i].iter().for_each(|s_index| {
            rlwe_by_rgsw(
                trivial_rlwe_test_poly,
                pbs_key.rgsw_ct_secret_el(*s_index),
                scratch_matrix_dplus2_ring,
                decomposer,
                ntt_op,
                mod_op,
            );
        });

        galois_auto(
            trivial_rlwe_test_poly,
            pbs_key.galois_key_for_auto(g),
            scratch_matrix_dplus2_ring,
            pbs_key.auto_map_index(g),
            pbs_key.auto_map_sign(g),
            mod_op,
            ntt_op,
            decomposer,
        );
    }

    // +(g^0)
    gk_to_si[0].iter().for_each(|s_index| {
        rlwe_by_rgsw(
            trivial_rlwe_test_poly,
            pbs_key.rgsw_ct_secret_el(gk_to_si[q_by_2][*s_index]),
            scratch_matrix_dplus2_ring,
            decomposer,
            ntt_op,
            mod_op,
        );
    });
}

trait Parameters {
    type Element;
    type D: Decomposer<Element = Self::Element>;
    fn rlwe_q(&self) -> Self::Element;
    fn lwe_q(&self) -> Self::Element;
    fn br_q(&self) -> usize;
    fn d_rgsw(&self) -> usize;
    fn d_lwe(&self) -> usize;
    fn rlwe_n(&self) -> usize;
    fn lwe_n(&self) -> usize;
    // Embedding fator for ring X^{q}+1 inside
    fn embedding_factor(&self) -> usize;
    // generator g
    fn g(&self) -> isize;
    fn decomoposer_lwe(&self) -> &Self::D;
    fn decomoposer_rlwe(&self) -> &Self::D;
    /// Maps a \in Z^*_{2q} to discrete log k, with generator g (i.e. g^k =
    /// a). Returned vector is of size q that stores dlog of a at `vec[a]`.
    /// For any a, k is s.t. a = g^{k}, then k is expressed as k. If k is s.t a
    /// = -g^{k/2}, then k is expressed as k=k+q/2
    fn g_k_dlog_map(&self) -> &[usize];
}

/// - Mod down
/// - key switching
/// - mod down
/// - blind rotate
fn pbs<
    M: Matrix + MatrixMut + MatrixEntity,
    MT: MatrixMut<MatElement = M::MatElement, R = M::R> + IsTrivial + MatrixEntity,
    P: Parameters<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement>,
    ModOp: ArithmeticOps<Element = M::MatElement> + VectorOps<Element = M::MatElement>,
    K: PbsKey<M = M>,
>(
    parameters: &P,
    test_vec: &M::R,
    lwe_in: &mut M::R,
    lwe_ksk: &M,
    scratch_lwen_plus1: &mut M::R,
    scratch_matrix_dplus2_ring: &mut M,
    modop_lweq: &ModOp,
    modop_rlweq: &ModOp,
    nttop_rlweq: &NttOp,
    pbs_key: K,
) where
    <M as Matrix>::R: RowMut,
    <MT as Matrix>::R: RowMut,
    M::MatElement: PrimInt + ToPrimitive + FromPrimitive + One + Copy + Zero,
{
    let rlwe_q = parameters.rlwe_q();
    let lwe_q = parameters.lwe_q();
    let br_q = parameters.br_q();
    let rlwe_qf64 = rlwe_q.to_f64().unwrap();
    let lwe_qf64 = lwe_q.to_f64().unwrap();
    let br_qf64 = br_q.to_f64().unwrap();
    let rlwe_n = parameters.rlwe_n();

    // moddown Q -> Q_ks
    lwe_in.as_mut().iter_mut().for_each(|v| {
        *v =
            M::MatElement::from_f64(((v.to_f64().unwrap() * lwe_qf64) / rlwe_qf64).round()).unwrap()
    });

    // key switch
    // let mut lwe_out = M::zeros(1, parameters.lwe_n() + 1);
    scratch_lwen_plus1.as_mut().fill(M::MatElement::zero());
    lwe_key_switch(
        scratch_lwen_plus1,
        lwe_in,
        lwe_ksk,
        modop_lweq,
        parameters.decomoposer_lwe(),
    );

    // odd mowdown Q_ks -> q
    let g_k_dlog_map = parameters.g_k_dlog_map();
    let mut g_k_si = vec![vec![]; br_q];
    scratch_lwen_plus1
        .as_ref()
        .iter()
        .skip(1)
        .enumerate()
        .for_each(|(index, v)| {
            let odd_v = mod_switch_odd(v.to_f64().unwrap(), lwe_qf64, br_qf64);
            let k = g_k_dlog_map[odd_v];
            g_k_si[k].push(index);
        });

    // handle b and set trivial test RLWE
    let g = parameters.g() as usize;
    let g_times_b = (g * mod_switch_odd(
        scratch_lwen_plus1.as_ref()[0].to_f64().unwrap(),
        lwe_qf64,
        br_qf64,
    )) % (br_q);
    // v = (v(X) * X^{g*b}) mod X^{q/2}+1
    let br_qby2 = br_q / 2;
    let mut gb_monomial_sign = true;
    let mut gb_monomial_exp = g_times_b;
    // X^{g*b} mod X^{q}+1
    if gb_monomial_exp > br_qby2 {
        gb_monomial_exp -= br_qby2;
        gb_monomial_sign = false
    }
    // monomial mul
    let mut trivial_rlwe_test_poly = MT::zeros(2, rlwe_n);
    if parameters.embedding_factor() == 1 {
        monomial_mul(
            test_vec.as_ref(),
            trivial_rlwe_test_poly.get_row_mut(1).as_mut(),
            gb_monomial_exp,
            gb_monomial_sign,
            br_q,
            modop_rlweq,
        );
    } else {
        // use lwe_in to store the `t = v(X) * X^{g*2} mod X^{q/2}+1` temporarily. This
        // works because q/2 < N (where N is lwe_in LWE dimension) always.
        monomial_mul(
            test_vec.as_ref(),
            &mut lwe_in.as_mut()[..br_qby2],
            gb_monomial_exp,
            gb_monomial_sign,
            br_q,
            modop_rlweq,
        );

        // emebed poly `t` in ring X^{q/2}+1 inside the bigger ring X^{N}+1
        let partb_trivial_rlwe = trivial_rlwe_test_poly.get_row_mut(1);
        lwe_in.as_ref()[..br_qby2]
            .iter()
            .enumerate()
            .for_each(|(index, v)| {
                partb_trivial_rlwe[2 * index] = *v;
            });
    }
    // TODO Rotate test input

    // blind rotate
    blind_rotation(
        &mut trivial_rlwe_test_poly,
        scratch_matrix_dplus2_ring,
        parameters.g(),
        1,
        br_q,
        &g_k_si,
        parameters.decomoposer_rlwe(),
        nttop_rlweq,
        modop_rlweq,
        &pbs_key,
    );

    // sample extract
    sample_extract(lwe_in, &trivial_rlwe_test_poly, modop_rlweq, 0);
}

fn mod_switch_odd(v: f64, from_q: f64, to_q: f64) -> usize {
    let odd_v = (((v.to_f64().unwrap() * to_q) / (from_q)).floor())
        .to_usize()
        .unwrap();
    //TODO(Jay): check correctness of this
    odd_v + (odd_v ^ (usize::one()))
}

fn sample_extract<M: Matrix + MatrixMut, ModOp: ArithmeticOps<Element = M::MatElement>>(
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
