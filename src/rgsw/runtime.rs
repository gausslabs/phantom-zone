use itertools::izip;
use num_traits::Zero;

use crate::{
    backend::{ArithmeticOps, GetModulus, Modulus, ShoupMatrixFMA, VectorOps},
    decomposer::{Decomposer, RlweDecomposer},
    ntt::Ntt,
    Matrix, MatrixEntity, MatrixMut, Row, RowEntity, RowMut, Secret,
};

use super::IsTrivial;

pub(crate) fn routine<R: RowMut, ModOp: VectorOps<Element = R::Element>>(
    write_to_row: &mut [R::Element],
    matrix_a: &[R],
    matrix_b: &[R],
    mod_op: &ModOp,
) {
    izip!(matrix_a.iter(), matrix_b.iter()).for_each(|(a, b)| {
        mod_op.elwise_fma_mut(write_to_row, a.as_ref(), b.as_ref());
    });
}

/// Decomposes ring polynomial r(X) into d polynomials using decomposer into
/// output matrix decomp_r
///
/// Note that decomposition of r(X) requires decomposition of each of
/// coefficients.
///
/// - decomp_r: must have dimensions d x ring_size. i^th decomposed polynomial
///   will be stored at i^th row.
pub(crate) fn decompose_r<R: RowMut, D: Decomposer<Element = R::Element>>(
    r: &[R::Element],
    decomp_r: &mut [R],
    decomposer: &D,
) where
    R::Element: Copy,
{
    let ring_size = r.len();

    for ri in 0..ring_size {
        decomposer
            .decompose_iter(&r[ri])
            .enumerate()
            .for_each(|(index, el)| {
                decomp_r[index].as_mut()[ri] = el;
            });
    }
}

/// Sends RLWE_{s}(X) -> RLWE_{s}(X^k) where k is some galois element
///
/// - scratch_matrix: must have dimension at-least d+2 x ring_size. d rows to
///   store decomposed polynomials and 2 for rlwe
pub(crate) fn galois_auto<
    MT: Matrix + IsTrivial + MatrixMut,
    Mmut: MatrixMut<MatElement = MT::MatElement>,
    ModOp: ArithmeticOps<Element = MT::MatElement> + VectorOps<Element = MT::MatElement>,
    NttOp: Ntt<Element = MT::MatElement>,
    D: Decomposer<Element = MT::MatElement>,
>(
    rlwe_in: &mut MT,
    ksk: &Mmut,
    scratch_matrix: &mut Mmut,
    auto_map_index: &[usize],
    auto_map_sign: &[bool],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    decomposer: &D,
) where
    <Mmut as Matrix>::R: RowMut,
    <MT as Matrix>::R: RowMut,
    MT::MatElement: Copy + Zero,
{
    let d = decomposer.decomposition_count();
    let ring_size = rlwe_in.dimension().1;
    assert!(rlwe_in.dimension().0 == 2);
    assert!(scratch_matrix.fits(d + 2, ring_size));

    // scratch matrix is guaranteed to have at-least d+2 rows but can have more than
    // d+2 rows. We require to split them into sub-matrices of exact sizes one with
    // d rows for storing decomposed polynomial and second with 2 rows to act
    // tomperary space for RLWE ciphertext. Exact sizes is necessary to avoid any
    // irrelevant extra FMA or NTT ops.
    let (scratch_matrix_d_ring, other_half) = scratch_matrix.split_at_row_mut(d);
    let (tmp_rlwe_out, _) = other_half.split_at_mut(2);

    debug_assert!(tmp_rlwe_out.len() == 2);
    debug_assert!(scratch_matrix_d_ring.len() == d);

    if !rlwe_in.is_trivial() {
        tmp_rlwe_out.iter_mut().for_each(|r| {
            r.as_mut().fill(Mmut::MatElement::zero());
        });

        // send a(X) -> a(X^k) and decompose a(X^k)
        izip!(
            rlwe_in.get_row(0),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            let el_out = if !*sign { mod_op.neg(el_in) } else { *el_in };

            decomposer
                .decompose_iter(&el_out)
                .enumerate()
                .for_each(|(index, el)| {
                    scratch_matrix_d_ring[index].as_mut()[*to_index] = el;
                });
        });

        // transform decomposed a(X^k) to evaluation domain
        scratch_matrix_d_ring.iter_mut().for_each(|r| {
            ntt_op.forward(r.as_mut());
        });

        // RLWE(m^k) = a', b'; RLWE(m) = a, b
        // key switch: (a * RLWE'(s(X^k)))
        let (ksk_a, ksk_b) = ksk.split_at_row(d);
        // a' = decomp<a> * RLWE'_A(s(X^k))
        routine(
            tmp_rlwe_out[0].as_mut(),
            scratch_matrix_d_ring,
            ksk_a,
            mod_op,
        );

        // b' += decomp<a(X^k)> * RLWE'_B(s(X^k))
        routine(
            tmp_rlwe_out[1].as_mut(),
            scratch_matrix_d_ring,
            ksk_b,
            mod_op,
        );

        // transform RLWE(m^k) to coefficient domain
        tmp_rlwe_out
            .iter_mut()
            .for_each(|r| ntt_op.backward(r.as_mut()));

        // send b(X) -> b(X^k) and then b'(X) += b(X^k)
        izip!(
            rlwe_in.get_row(1),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            let row = tmp_rlwe_out[1].as_mut();
            if !*sign {
                row[*to_index] = mod_op.sub(&row[*to_index], el_in);
            } else {
                row[*to_index] = mod_op.add(&row[*to_index], el_in);
            }
        });

        // copy over A; Leave B for later
        rlwe_in
            .get_row_mut(0)
            .copy_from_slice(tmp_rlwe_out[0].as_ref());
    } else {
        // RLWE is trivial, a(X) is 0.
        // send b(X) -> b(X^k)
        izip!(
            rlwe_in.get_row(1),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            if !*sign {
                tmp_rlwe_out[1].as_mut()[*to_index] = mod_op.neg(el_in);
            } else {
                tmp_rlwe_out[1].as_mut()[*to_index] = *el_in;
            }
        });
    }

    // Copy over B
    rlwe_in
        .get_row_mut(1)
        .copy_from_slice(tmp_rlwe_out[1].as_ref());
}

pub(crate) fn galois_auto_shoup<
    MT: Matrix + IsTrivial + MatrixMut,
    Mmut: MatrixMut<MatElement = MT::MatElement>,
    ModOp: ArithmeticOps<Element = MT::MatElement>
        + VectorOps<Element = MT::MatElement>
        + ShoupMatrixFMA<Mmut::R>,
    NttOp: Ntt<Element = MT::MatElement>,
    D: Decomposer<Element = MT::MatElement>,
>(
    rlwe_in: &mut MT,
    ksk: &Mmut,
    ksk_shoup: &Mmut,
    scratch_matrix: &mut Mmut,
    auto_map_index: &[usize],
    auto_map_sign: &[bool],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    decomposer: &D,
) where
    <Mmut as Matrix>::R: RowMut,
    <MT as Matrix>::R: RowMut,
    MT::MatElement: Copy + Zero,
{
    let d = decomposer.decomposition_count();
    let ring_size = rlwe_in.dimension().1;
    assert!(rlwe_in.dimension().0 == 2);
    assert!(scratch_matrix.fits(d + 2, ring_size));

    let (scratch_matrix_d_ring, other_half) = scratch_matrix.split_at_row_mut(d);
    let (tmp_rlwe_out, _) = other_half.split_at_mut(2);

    debug_assert!(tmp_rlwe_out.len() == 2);
    debug_assert!(scratch_matrix_d_ring.len() == d);

    if !rlwe_in.is_trivial() {
        tmp_rlwe_out.iter_mut().for_each(|r| {
            r.as_mut().fill(Mmut::MatElement::zero());
        });

        // send a(X) -> a(X^k) and decompose a(X^k)
        izip!(
            rlwe_in.get_row(0),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            let el_out = if !*sign { mod_op.neg(el_in) } else { *el_in };

            decomposer
                .decompose_iter(&el_out)
                .enumerate()
                .for_each(|(index, el)| {
                    scratch_matrix_d_ring[index].as_mut()[*to_index] = el;
                });
        });

        // transform decomposed a(X^k) to evaluation domain
        scratch_matrix_d_ring.iter_mut().for_each(|r| {
            ntt_op.forward_lazy(r.as_mut());
        });

        // RLWE(m^k) = a', b'; RLWE(m) = a, b
        // key switch: (a * RLWE'(s(X^k)))
        let (ksk_a, ksk_b) = ksk.split_at_row(d);
        let (ksk_a_shoup, ksk_b_shoup) = ksk_shoup.split_at_row(d);
        // a' = decomp<a> * RLWE'_A(s(X^k))
        mod_op.shoup_matrix_fma(
            tmp_rlwe_out[0].as_mut(),
            ksk_a,
            ksk_a_shoup,
            scratch_matrix_d_ring,
        );

        // b'= decomp<a(X^k)> * RLWE'_B(s(X^k))
        mod_op.shoup_matrix_fma(
            tmp_rlwe_out[1].as_mut(),
            ksk_b,
            ksk_b_shoup,
            scratch_matrix_d_ring,
        );

        // transform RLWE(m^k) to coefficient domain
        tmp_rlwe_out
            .iter_mut()
            .for_each(|r| ntt_op.backward(r.as_mut()));

        // send b(X) -> b(X^k) and then b'(X) += b(X^k)
        izip!(
            rlwe_in.get_row(1),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            let row = tmp_rlwe_out[1].as_mut();
            if !*sign {
                row[*to_index] = mod_op.sub(&row[*to_index], el_in);
            } else {
                row[*to_index] = mod_op.add(&row[*to_index], el_in);
            }
        });

        // copy over A; Leave B for later
        rlwe_in
            .get_row_mut(0)
            .copy_from_slice(tmp_rlwe_out[0].as_ref());
    } else {
        // RLWE is trivial, a(X) is 0.
        // send b(X) -> b(X^k)
        izip!(
            rlwe_in.get_row(1),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            if !*sign {
                tmp_rlwe_out[1].as_mut()[*to_index] = mod_op.neg(el_in);
            } else {
                tmp_rlwe_out[1].as_mut()[*to_index] = *el_in;
            }
        });
    }

    // Copy over B
    rlwe_in
        .get_row_mut(1)
        .copy_from_slice(tmp_rlwe_out[1].as_ref());
}

/// Returns RLWE(m0m1) = RLWE(m0) x RGSW(m1). Mutates rlwe_in inplace to equal
/// RLWE(m0m1)
///
/// - rlwe_in: is RLWE(m0) with polynomials in coefficient domain
/// - rgsw_in: is RGSW(m1) with polynomials in evaluation domain
/// - scratch_matrix_d_ring: is a matrix with atleast max(d_a, d_b) rows and
///   ring_size columns. It's used to store decomposed polynomials and out RLWE
///   temoporarily
pub(crate) fn rlwe_by_rgsw<
    Mmut: MatrixMut,
    MT: Matrix<MatElement = Mmut::MatElement> + MatrixMut<MatElement = Mmut::MatElement> + IsTrivial,
    D: RlweDecomposer<Element = Mmut::MatElement>,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
>(
    rlwe_in: &mut MT,
    rgsw_in: &Mmut,
    scratch_matrix: &mut Mmut,
    decomposer: &D,
    ntt_op: &NttOp,
    mod_op: &ModOp,
) where
    Mmut::MatElement: Copy + Zero,
    <Mmut as Matrix>::R: RowMut,
    <MT as Matrix>::R: RowMut,
{
    let decomposer_a = decomposer.a();
    let decomposer_b = decomposer.b();
    let d_a = decomposer_a.decomposition_count();
    let d_b = decomposer_b.decomposition_count();
    let max_d = std::cmp::max(d_a, d_b);
    assert!(scratch_matrix.fits(max_d + 2, rlwe_in.dimension().1));
    assert!(rgsw_in.dimension() == (d_a * 2 + d_b * 2, rlwe_in.dimension().1));

    // decomposed RLWE x RGSW
    let (rlwe_dash_nsm, rlwe_dash_m) = rgsw_in.split_at_row(d_a * 2);
    let (scratch_matrix_d_ring, scratch_rlwe_out) = scratch_matrix.split_at_row_mut(max_d);
    scratch_rlwe_out[0].as_mut().fill(Mmut::MatElement::zero());
    scratch_rlwe_out[1].as_mut().fill(Mmut::MatElement::zero());
    // RLWE_in = a_in, b_in; RLWE_out = a_out, b_out
    if !rlwe_in.is_trivial() {
        // a_in = 0 when RLWE_in is trivial RLWE ciphertext
        // decomp<a_in>
        decompose_r(
            rlwe_in.get_row_slice(0),
            &mut scratch_matrix_d_ring[..d_a],
            decomposer_a,
        );
        scratch_matrix_d_ring
            .iter_mut()
            .take(d_a)
            .for_each(|r| ntt_op.forward(r.as_mut()));
        // a_out += decomp<a_in> \cdot RLWE_A'(-sm)
        routine(
            scratch_rlwe_out[0].as_mut(),
            scratch_matrix_d_ring.as_ref(),
            &rlwe_dash_nsm[..d_a],
            mod_op,
        );
        // b_out += decomp<a_in> \cdot RLWE_B'(-sm)
        routine(
            scratch_rlwe_out[1].as_mut(),
            scratch_matrix_d_ring.as_ref(),
            &rlwe_dash_nsm[d_a..],
            mod_op,
        );
    }
    // decomp<b_in>
    decompose_r(
        rlwe_in.get_row_slice(1),
        &mut scratch_matrix_d_ring[..d_b],
        decomposer_b,
    );
    scratch_matrix_d_ring
        .iter_mut()
        .take(d_b)
        .for_each(|r| ntt_op.forward(r.as_mut()));
    // a_out += decomp<b_in> \cdot RLWE_A'(m)
    routine(
        scratch_rlwe_out[0].as_mut(),
        scratch_matrix_d_ring.as_ref(),
        &rlwe_dash_m[..d_b],
        mod_op,
    );
    // b_out += decomp<b_in> \cdot RLWE_B'(m)
    routine(
        scratch_rlwe_out[1].as_mut(),
        scratch_matrix_d_ring.as_ref(),
        &rlwe_dash_m[d_b..],
        mod_op,
    );

    // transform rlwe_out to coefficient domain
    scratch_rlwe_out
        .iter_mut()
        .for_each(|r| ntt_op.backward(r.as_mut()));

    rlwe_in
        .get_row_mut(0)
        .copy_from_slice(scratch_rlwe_out[0].as_mut());
    rlwe_in
        .get_row_mut(1)
        .copy_from_slice(scratch_rlwe_out[1].as_mut());
    rlwe_in.set_not_trivial();
}

/// Inplace mutates rlwe_0 to equal RGSW(m0m1) = RGSW(m0)xRGSW(m1)
/// in evaluation domain
///
/// Warning -
/// Pass a fresh RGSW ciphertext as the second operand, i.e. as `rgsw_1`.
/// This is to assure minimal error growth in the resulting RGSW ciphertext.
/// RGSWxRGSW boils down to d_rgsw*2 RLWExRGSW multiplications. Hence, the noise
/// growth in resulting ciphertext depends on the norm of second RGSW
/// ciphertext, not the first. This is useful in cases where one is accumulating
/// multiple RGSW ciphertexts into 1. In which case, pass the accumulating RGSW
/// ciphertext as rlwe_0 (the one with higher noise) and subsequent RGSW
/// ciphertexts, with lower noise, to be accumulated as second
/// operand.
///
/// - rgsw_0: RGSW(m0)
/// - rgsw_1_eval: RGSW(m1) in Evaluation domain
/// - scratch_matrix_d_plus_rgsw_by_ring: scratch space matrix with rows
///   (max(d_a, d_b) + d_a*2+d_b*2) and columns ring_size
pub(crate) fn rgsw_by_rgsw_inplace<
    Mmut: MatrixMut,
    D: RlweDecomposer<Element = Mmut::MatElement>,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
>(
    rgsw_0: &mut Mmut,
    rgsw_1_eval: &Mmut,
    decomposer: &D,
    scratch_matrix: &mut Mmut,
    ntt_op: &NttOp,
    mod_op: &ModOp,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut::MatElement: Copy + Zero,
{
    let decomposer_a = decomposer.a();
    let decomposer_b = decomposer.b();
    let d_a = decomposer_a.decomposition_count();
    let d_b = decomposer_b.decomposition_count();
    let max_d = std::cmp::max(d_a, d_b);
    let rgsw_rows = d_a * 2 + d_b * 2;
    assert!(rgsw_0.dimension().0 == rgsw_rows);
    let ring_size = rgsw_0.dimension().1;
    assert!(rgsw_1_eval.dimension() == (rgsw_rows, ring_size));
    assert!(scratch_matrix.fits(max_d + rgsw_rows, ring_size));

    let (decomp_r_space, rgsw_space) = scratch_matrix.split_at_row_mut(max_d);

    // zero rgsw_space
    rgsw_space
        .iter_mut()
        .for_each(|ri| ri.as_mut().fill(Mmut::MatElement::zero()));
    let (rlwe_dash_space_nsm, rlwe_dash_space_m) = rgsw_space.split_at_mut(d_a * 2);
    let (rlwe_dash_space_nsm_parta, rlwe_dash_space_nsm_partb) =
        rlwe_dash_space_nsm.split_at_mut(d_a);
    let (rlwe_dash_space_m_parta, rlwe_dash_space_m_partb) = rlwe_dash_space_m.split_at_mut(d_b);

    let (rgsw0_nsm, rgsw0_m) = rgsw_0.split_at_row(d_a * 2);
    let (rgsw1_nsm, rgsw1_m) = rgsw_1_eval.split_at_row(d_a * 2);

    // RGSW x RGSW
    izip!(
        rgsw0_nsm.iter().take(d_a).chain(rgsw0_m.iter().take(d_b)),
        rgsw0_nsm.iter().skip(d_a).chain(rgsw0_m.iter().skip(d_b)),
        rlwe_dash_space_nsm_parta
            .iter_mut()
            .chain(rlwe_dash_space_m_parta.iter_mut()),
        rlwe_dash_space_nsm_partb
            .iter_mut()
            .chain(rlwe_dash_space_m_partb.iter_mut()),
    )
    .for_each(|(rlwe_a, rlwe_b, rlwe_out_a, rlwe_out_b)| {
        // Part A
        decompose_r(rlwe_a.as_ref(), decomp_r_space.as_mut(), decomposer_a);
        decomp_r_space
            .iter_mut()
            .take(d_a)
            .for_each(|ri| ntt_op.forward(ri.as_mut()));
        routine(
            rlwe_out_a.as_mut(),
            &decomp_r_space[..d_a],
            &rgsw1_nsm[..d_a],
            mod_op,
        );
        routine(
            rlwe_out_b.as_mut(),
            &decomp_r_space[..d_a],
            &rgsw1_nsm[d_a..],
            mod_op,
        );

        // Part B
        decompose_r(rlwe_b.as_ref(), decomp_r_space.as_mut(), decomposer_b);
        decomp_r_space
            .iter_mut()
            .take(d_b)
            .for_each(|ri| ntt_op.forward(ri.as_mut()));
        routine(
            rlwe_out_a.as_mut(),
            &decomp_r_space[..d_b],
            &rgsw1_m[..d_b],
            mod_op,
        );
        routine(
            rlwe_out_b.as_mut(),
            &decomp_r_space[..d_b],
            &rgsw1_m[d_b..],
            mod_op,
        );
    });

    // copy over RGSW(m0m1) into RGSW(m0)
    izip!(rgsw_0.iter_rows_mut(), rgsw_space.iter())
        .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

    // send back to coefficient domain
    rgsw_0
        .iter_rows_mut()
        .for_each(|ri| ntt_op.backward(ri.as_mut()));
}
