use itertools::izip;
use num_traits::Zero;

use crate::{
    backend::{ArithmeticOps, GetModulus, ShoupMatrixFMA, VectorOps},
    decomposer::{Decomposer, RlweDecomposer},
    ntt::Ntt,
    Matrix, MatrixEntity, MatrixMut, RowEntity, RowMut,
};

use super::IsTrivial;

pub(crate) fn poly_fma_routine<R: RowMut, ModOp: VectorOps<Element = R::Element>>(
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

/// Sends RLWE_{s(X)}(m(X)) -> RLWE_{s(X)}(m{X^k}) where k is some galois
/// element
///
/// - rlwe_in: Input ciphertext RLWE_{s(X)}(m(X)).
/// - ksk: Auto key switching key with polynomials in evaluation domain
/// - auto_map_index: If automorphism sends i^th coefficient of m(X) to j^th
///   coefficient of m(X^k) then auto_map_index[i] = j
/// - auto_sign_index: With a = m(X)[i], if m(X^k)[auto_map_index[i]] = -a, then
///   auto_sign_index[i] = false, else auto_sign_index[i] = true
/// - scratch_matrix: must have dimension at-least d+2 x ring_size. `d` rows to
///   store decomposed polynomials nad 2 rows to store out RLWE temporarily.
pub(crate) fn rlwe_auto<
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
        poly_fma_routine(
            tmp_rlwe_out[0].as_mut(),
            scratch_matrix_d_ring,
            ksk_a,
            mod_op,
        );

        // b' += decomp<a(X^k)> * RLWE'_B(s(X^k))
        poly_fma_routine(
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

/// Sends RLWE_{s(X)}(m(X)) -> RLWE_{s(X)}(m{X^k}) where k is some galois
/// element
///
/// This is same as `galois_auto` with the difference that alongside `ksk` with
/// key switching polynomials in evaluation domain, shoup representation,
/// `ksk_shoup`, of the polynomials in evaluation domain is also supplied.
pub(crate) fn galois_auto_shoup<
    Mmut: MatrixMut,
    ModOp: ArithmeticOps<Element = Mmut::MatElement>
        // + VectorOps<Element = MT::MatElement>
        + ShoupMatrixFMA<Mmut::R>,
    NttOp: Ntt<Element = Mmut::MatElement>,
    D: Decomposer<Element = Mmut::MatElement>,
>(
    rlwe_in: &mut Mmut,
    ksk: &Mmut,
    ksk_shoup: &Mmut,
    scratch_matrix: &mut Mmut,
    auto_map_index: &[usize],
    auto_map_sign: &[bool],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    decomposer: &D,
    is_trivial: bool,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut::MatElement: Copy + Zero,
{
    let d = decomposer.decomposition_count();
    let ring_size = rlwe_in.dimension().1;
    assert!(rlwe_in.dimension().0 == 2);
    assert!(scratch_matrix.fits(d + 2, ring_size));

    let (scratch_matrix_d_ring, other_half) = scratch_matrix.split_at_row_mut(d);
    let (tmp_rlwe_out, _) = other_half.split_at_mut(2);

    debug_assert!(tmp_rlwe_out.len() == 2);
    debug_assert!(scratch_matrix_d_ring.len() == d);

    if !is_trivial {
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

/// Inplace mutates RLWE(m0) to equal RLWE(m0m1) = RLWE(m0) x RGSW(m1).
///
/// - rlwe_in: is RLWE(m0) with polynomials in coefficient domain
/// - rgsw_in: is RGSW(m1) with polynomials in evaluation domain
/// - scratch_matrix: with dimension (max(d_a, d_b) + 2) x ring_size columns.
///   It's used to store decomposed polynomials and out RLWE temporarily
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
    let (scratch_matrix_d_ring, rest) = scratch_matrix.split_at_row_mut(max_d);
    let (scratch_rlwe_out, _) = rest.split_at_mut(2);

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
        poly_fma_routine(
            scratch_rlwe_out[0].as_mut(),
            &scratch_matrix_d_ring[..d_a],
            &rlwe_dash_nsm[..d_a],
            mod_op,
        );
        // b_out += decomp<a_in> \cdot RLWE_B'(-sm)
        poly_fma_routine(
            scratch_rlwe_out[1].as_mut(),
            &scratch_matrix_d_ring[..d_a],
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
    poly_fma_routine(
        scratch_rlwe_out[0].as_mut(),
        &scratch_matrix_d_ring[..d_b],
        &rlwe_dash_m[..d_b],
        mod_op,
    );
    // b_out += decomp<b_in> \cdot RLWE_B'(m)
    poly_fma_routine(
        scratch_rlwe_out[1].as_mut(),
        &scratch_matrix_d_ring[..d_b],
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

/// Inplace mutates RLWE(m0) to equal RLWE(m0m1) = RLWE(m0) x RGSW(m1).
///
/// Same as `rlwe_by_rgsw` with the difference that alongside `rgsw_in` with
/// polynomials in evaluation domain, shoup representation of polynomials in
/// evaluation domain, `rgsw_in_shoup`, is also supplied.
pub(crate) fn rlwe_by_rgsw_shoup<
    Mmut: MatrixMut,
    D: RlweDecomposer<Element = Mmut::MatElement>,
    ModOp: ShoupMatrixFMA<Mmut::R>,
    NttOp: Ntt<Element = Mmut::MatElement>,
>(
    rlwe_in: &mut Mmut,
    rgsw_in: &Mmut,
    rgsw_in_shoup: &Mmut,
    scratch_matrix: &mut Mmut,
    decomposer: &D,
    ntt_op: &NttOp,
    mod_op: &ModOp,
    is_trivial: bool,
) where
    Mmut::MatElement: Copy + Zero,
    <Mmut as Matrix>::R: RowMut,
{
    let decomposer_a = decomposer.a();
    let decomposer_b = decomposer.b();
    let d_a = decomposer_a.decomposition_count();
    let d_b = decomposer_b.decomposition_count();
    let max_d = std::cmp::max(d_a, d_b);
    assert!(scratch_matrix.fits(max_d + 2, rlwe_in.dimension().1));
    assert!(rgsw_in.dimension() == (d_a * 2 + d_b * 2, rlwe_in.dimension().1));
    assert!(rgsw_in.dimension() == rgsw_in_shoup.dimension());

    // decomposed RLWE x RGSW
    let (rlwe_dash_nsm, rlwe_dash_m) = rgsw_in.split_at_row(d_a * 2);
    let (rlwe_dash_nsm_shoup, rlwe_dash_m_shoup) = rgsw_in_shoup.split_at_row(d_a * 2);
    let (scratch_matrix_d_ring, rest) = scratch_matrix.split_at_row_mut(max_d);
    let (scratch_rlwe_out, _) = rest.split_at_mut(2);

    scratch_rlwe_out[1].as_mut().fill(Mmut::MatElement::zero());
    scratch_rlwe_out[0].as_mut().fill(Mmut::MatElement::zero());

    // RLWE_in = a_in, b_in; RLWE_out = a_out, b_out
    if !is_trivial {
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
            .for_each(|r| ntt_op.forward_lazy(r.as_mut()));

        // a_out += decomp<a_in> \cdot RLWE_A'(-sm)
        mod_op.shoup_matrix_fma(
            scratch_rlwe_out[0].as_mut(),
            &rlwe_dash_nsm[..d_a],
            &rlwe_dash_nsm_shoup[..d_a],
            &scratch_matrix_d_ring[..d_a],
        );

        // b_out += decomp<a_in> \cdot RLWE_B'(-sm)
        mod_op.shoup_matrix_fma(
            scratch_rlwe_out[1].as_mut(),
            &rlwe_dash_nsm[d_a..],
            &rlwe_dash_nsm_shoup[d_a..],
            &scratch_matrix_d_ring[..d_a],
        );
    }
    {
        // decomp<b_in>
        decompose_r(
            rlwe_in.get_row_slice(1),
            &mut scratch_matrix_d_ring[..d_b],
            decomposer_b,
        );
        scratch_matrix_d_ring
            .iter_mut()
            .take(d_b)
            .for_each(|r| ntt_op.forward_lazy(r.as_mut()));

        // a_out += decomp<b_in> \cdot RLWE_A'(m)
        mod_op.shoup_matrix_fma(
            scratch_rlwe_out[0].as_mut(),
            &rlwe_dash_m[..d_b],
            &rlwe_dash_m_shoup[..d_b],
            &scratch_matrix_d_ring[..d_b],
        );

        // b_out += decomp<b_in> \cdot RLWE_B'(m)
        mod_op.shoup_matrix_fma(
            scratch_rlwe_out[1].as_mut(),
            &rlwe_dash_m[d_b..],
            &rlwe_dash_m_shoup[d_b..],
            &scratch_matrix_d_ring[..d_b],
        );
    }

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
}

/// Inplace mutates RGSW(m0) to equal RGSW(m0m1) = RGSW(m0)xRGSW(m1)
///
/// RGSW x RGSW product requires multiple RLWE x RGSW products. For example,
/// Define
///
///     RGSW(m0) = [RLWE(-sm), RLWE(\beta -sm), ..., RLWE(\beta^{d-1} -sm)
///                 RLWE(m), RLWE(\beta m), ..., RLWE(\beta^{d-1} m)]
///     And RGSW(m1)
///
/// Then RGSW(m0) x RGSW(m1) equals:
///     RGSW(m0m1) = [
///                     rlwe_x_rgsw(RLWE(-sm), RGSW(m1)),
///                     ...,
///                     rlwe_x_rgsw(RLWE(\beta^{d-1} -sm), RGSW(m1)),
///                     rlwe_x_rgsw(RLWE(m), RGSW(m1)),
///                     ...,
///                     rlwe_x_rgsw(RLWE(\beta^{d-1} m), RGSW(m1)),
///                  ]
///
/// Since noise growth in RLWE x RGSW depends on noise in RGSW ciphertext, it is
/// clear to observe from above that noise in resulting RGSW(m0m1) equals noise
/// accumulated in a single RLWE x RGSW and depends on noise in RGSW(m1) (i.e.
/// rgsw_1_eval)
///
/// - rgsw_0: RGSW(m0) in coefficient domain
/// - rgsw_1_eval: RGSW(m1) in evaluation domain
pub(crate) fn rgsw_by_rgsw_inplace<
    Mmut: MatrixMut,
    D: RlweDecomposer<Element = Mmut::MatElement>,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
>(
    rgsw0: &mut Mmut,
    rgsw0_da: usize,
    rgsw0_db: usize,
    rgsw1_eval: &Mmut,
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
    let rgsw1_rows = d_a * 2 + d_b * 2;
    let rgsw0_rows = rgsw0_da * 2 + rgsw0_db * 2;
    let ring_size = rgsw0.dimension().1;
    assert!(rgsw0.dimension().0 == rgsw0_rows);
    assert!(rgsw1_eval.dimension() == (rgsw1_rows, ring_size));
    assert!(scratch_matrix.fits(max_d + rgsw0_rows, ring_size));

    let (decomp_r_space, rgsw_space) = scratch_matrix.split_at_row_mut(max_d);

    // zero rgsw_space
    rgsw_space
        .iter_mut()
        .for_each(|ri| ri.as_mut().fill(Mmut::MatElement::zero()));
    let (rlwe_dash_space_nsm, rlwe_dash_space_m) = rgsw_space.split_at_mut(rgsw0_da * 2);
    let (rlwe_dash_space_nsm_parta, rlwe_dash_space_nsm_partb) =
        rlwe_dash_space_nsm.split_at_mut(rgsw0_da);
    let (rlwe_dash_space_m_parta, rlwe_dash_space_m_partb) =
        rlwe_dash_space_m.split_at_mut(rgsw0_db);

    let (rgsw0_nsm, rgsw0_m) = rgsw0.split_at_row(rgsw0_da * 2);
    let (rgsw1_nsm, rgsw1_m) = rgsw1_eval.split_at_row(d_a * 2);

    // RGSW x RGSW
    izip!(
        rgsw0_nsm
            .iter()
            .take(rgsw0_da)
            .chain(rgsw0_m.iter().take(rgsw0_db)),
        rgsw0_nsm
            .iter()
            .skip(rgsw0_da)
            .chain(rgsw0_m.iter().skip(rgsw0_db)),
        rlwe_dash_space_nsm_parta
            .iter_mut()
            .chain(rlwe_dash_space_m_parta.iter_mut()),
        rlwe_dash_space_nsm_partb
            .iter_mut()
            .chain(rlwe_dash_space_m_partb.iter_mut()),
    )
    .for_each(|(rlwe_a, rlwe_b, rlwe_out_a, rlwe_out_b)| {
        // RLWE(m0) x RGSW(m1)

        // Part A: Decomp<RLWE(m0)[A]> \cdot RLWE'(-sm1)
        decompose_r(rlwe_a.as_ref(), decomp_r_space.as_mut(), decomposer_a);
        decomp_r_space
            .iter_mut()
            .take(d_a)
            .for_each(|ri| ntt_op.forward(ri.as_mut()));
        poly_fma_routine(
            rlwe_out_a.as_mut(),
            &decomp_r_space[..d_a],
            &rgsw1_nsm[..d_a],
            mod_op,
        );
        poly_fma_routine(
            rlwe_out_b.as_mut(),
            &decomp_r_space[..d_a],
            &rgsw1_nsm[d_a..],
            mod_op,
        );

        // Part B: Decompose<RLWE(m0)[B]> \cdot RLWE'(m1)
        decompose_r(rlwe_b.as_ref(), decomp_r_space.as_mut(), decomposer_b);
        decomp_r_space
            .iter_mut()
            .take(d_b)
            .for_each(|ri| ntt_op.forward(ri.as_mut()));
        poly_fma_routine(
            rlwe_out_a.as_mut(),
            &decomp_r_space[..d_b],
            &rgsw1_m[..d_b],
            mod_op,
        );
        poly_fma_routine(
            rlwe_out_b.as_mut(),
            &decomp_r_space[..d_b],
            &rgsw1_m[d_b..],
            mod_op,
        );
    });

    // copy over RGSW(m0m1) into RGSW(m0)
    izip!(rgsw0.iter_rows_mut(), rgsw_space.iter())
        .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

    // send back to coefficient domain
    rgsw0
        .iter_rows_mut()
        .for_each(|ri| ntt_op.backward(ri.as_mut()));
}

/// Key switches input RLWE_{s'}(m) -> RLWE_{s}(m)
///
/// Let RLWE_{s'}(m) = [a, b] s.t. m+e = b - as'
///
/// Given key switchin key Ksk(s' -> s) = RLWE'_{s}(s') = [RLWE_{s}(beta^i s')]
/// = [a, a*s + e + beta^i s'] for i \in [0,d), key switching computes:
/// 1. RLWE_{s}(-s'a) = \sum signed_decompose(-a)[i] RLWE_{s}(beta^i s')
/// 2. RLWE_{s}(m) = (b, 0) + RLWE_{s}(-s'a)
///
/// - rlwe_in: Input rlwe ciphertext
/// - ksk: Key switching key Ksk(s' -> s) with polynomials in evaluation domain
/// - ksk_shoup: Key switching key Ksk(s' -> s) with polynomials in evaluation
///   domain in shoup representation
/// - decomposer: Decomposer used for key switching
pub(crate) fn rlwe_key_switch<
    M: MatrixMut + MatrixEntity,
    ModOp: GetModulus<Element = M::MatElement> + ShoupMatrixFMA<M::R> + VectorOps<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement>,
    D: Decomposer<Element = M::MatElement>,
>(
    rlwe_in: &M,
    ksk: &M,
    ksk_shoup: &M,
    decomposer: &D,
    ntt_op: &NttOp,
    mod_op: &ModOp,
) -> M
where
    <M as Matrix>::R: RowMut + RowEntity,
    M::MatElement: Copy,
{
    let ring_size = rlwe_in.dimension().1;
    assert!(rlwe_in.dimension().0 == 2);
    assert!(ksk.dimension() == (decomposer.decomposition_count() * 2, ring_size));

    let mut rlwe_out = M::zeros(2, ring_size);

    let mut tmp = M::zeros(decomposer.decomposition_count(), ring_size);
    let mut tmp_row = M::R::zeros(ring_size);

    // key switch RLWE part -A
    // negative A
    tmp_row.as_mut().copy_from_slice(rlwe_in.get_row_slice(0));
    mod_op.elwise_neg_mut(tmp_row.as_mut());
    // decompose -A and send to evaluation domain
    decompose_r(tmp_row.as_ref(), tmp.as_mut(), decomposer);
    tmp.iter_rows_mut()
        .for_each(|r| ntt_op.forward_lazy(r.as_mut()));

    // RLWE_s(-A u) = B' + B, A' = (decomp(-A) * Ksk(u -> s)) + (B, 0)
    let (ksk_part_a, ksk_part_b) = ksk.split_at_row(decomposer.decomposition_count());
    let (ksk_part_a_shoup, ksk_part_b_shoup) =
        ksk_shoup.split_at_row(decomposer.decomposition_count());
    // Part A'
    mod_op.shoup_matrix_fma(
        rlwe_out.get_row_mut(0),
        &ksk_part_a,
        &ksk_part_a_shoup,
        tmp.as_ref(),
    );
    // Part B'
    mod_op.shoup_matrix_fma(
        rlwe_out.get_row_mut(1),
        &ksk_part_b,
        &ksk_part_b_shoup,
        tmp.as_ref(),
    );
    // back to coefficient domain
    rlwe_out
        .iter_rows_mut()
        .for_each(|r| ntt_op.backward(r.as_mut()));

    // B' + B
    mod_op.elwise_add_mut(rlwe_out.get_row_mut(1), rlwe_in.get_row_slice(1));

    rlwe_out
}
