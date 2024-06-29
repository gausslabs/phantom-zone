use itertools::izip;
use num_traits::Zero;

use crate::{
    backend::{ArithmeticOps, GetModulus, ShoupMatrixFMA, VectorOps},
    decomposer::{Decomposer, RlweDecomposer},
    ntt::Ntt,
    Matrix, MatrixEntity, MatrixMut, Row, RowEntity, RowMut,
};

/// Degree 1 RLWE ciphertext.
///
/// RLWE(m) = [a, b] s.t. m+e = b - as
pub(crate) trait RlweCiphertext {
    type R: RowMut;
    /// Returns polynomial `a` of RLWE ciphertext as slice of elements
    fn part_a(&self) -> &[<Self::R as Row>::Element];
    /// Returns polynomial `a` of RLWE ciphertext as mutable slice of elements
    fn part_a_mut(&mut self) -> &mut [<Self::R as Row>::Element];
    /// Returns polynomial `b` of RLWE ciphertext as slice of elements
    fn part_b(&self) -> &[<Self::R as Row>::Element];
    /// Returns polynomial `b` of RLWE ciphertext as mut slice of elements
    fn part_b_mut(&mut self) -> &mut [<Self::R as Row>::Element];
    /// Returns ring size of polynomials
    fn ring_size(&self) -> usize;
}

/// RGSW ciphertext
///
/// RGSW is a collection of RLWE' ciphertext which are collection degree 1 of
/// RLWE ciphertexts
///
/// Let
/// RGSW = [RLWE'(-sm) || RLW'(m)] = [RW]
pub(crate) trait RgswCiphertext {
    type R: Row;

    fn split(&self) -> ((&[Self::R], &[Self::R]), (&[Self::R], &[Self::R]));
}

pub(crate) trait RgswCiphertextMut: RgswCiphertext {
    fn split_mut(
        &mut self,
    ) -> (
        (&mut [Self::R], &mut [Self::R]),
        (&mut [Self::R], &mut [Self::R]),
    );
}

pub(crate) struct RlweCiphertextMutRef<'a, R> {
    data: &'a mut [R],
}

impl<'a, R> RlweCiphertextMutRef<'a, R> {
    pub(crate) fn new(data: &'a mut [R]) -> Self {
        Self { data }
    }
}

impl<'a, R: RowMut> RlweCiphertext for RlweCiphertextMutRef<'a, R> {
    type R = R;
    fn part_a(&self) -> &[<Self::R as Row>::Element] {
        self.data[0].as_ref()
    }
    fn part_a_mut(&mut self) -> &mut [<Self::R as Row>::Element] {
        self.data[0].as_mut()
    }
    fn part_b(&self) -> &[<Self::R as Row>::Element] {
        self.data[1].as_ref()
    }
    fn part_b_mut(&mut self) -> &mut [<Self::R as Row>::Element] {
        self.data[1].as_mut()
    }
    fn ring_size(&self) -> usize {
        self.data[0].as_ref().len()
    }
}

pub(crate) struct RgswCiphertextRef<'a, R> {
    data: &'a [R],
    d_a: usize,
    d_b: usize,
}

impl<'a, R> RgswCiphertextRef<'a, R> {
    pub(crate) fn new(data: &'a [R], d_a: usize, d_b: usize) -> Self {
        RgswCiphertextRef { data, d_a, d_b }
    }
}

impl<'a, R> RgswCiphertext for RgswCiphertextRef<'a, R>
where
    R: Row,
{
    type R = R;

    fn split(&self) -> ((&[Self::R], &[Self::R]), (&[Self::R], &[Self::R])) {
        let (rlwe_dash_nsm, rlwe_dash_m) = self.data.split_at(self.d_a * 2);
        (
            rlwe_dash_nsm.split_at(self.d_a),
            rlwe_dash_m.split_at(self.d_b),
        )
    }
}

pub(crate) struct RgswCiphertextMutRef<'a, R> {
    data: &'a mut [R],
    d_a: usize,
    d_b: usize,
}

impl<'a, R> RgswCiphertextMutRef<'a, R> {
    pub(crate) fn new(data: &'a mut [R], d_a: usize, d_b: usize) -> Self {
        RgswCiphertextMutRef { data, d_a, d_b }
    }
}

impl<'a, R: RowMut> AsMut<[R]> for RgswCiphertextMutRef<'a, R> {
    fn as_mut(&mut self) -> &mut [R] {
        &mut self.data
    }
}

impl<'a, R> RgswCiphertext for RgswCiphertextMutRef<'a, R>
where
    R: Row,
{
    type R = R;

    fn split(&self) -> ((&[Self::R], &[Self::R]), (&[Self::R], &[Self::R])) {
        let (rlwe_dash_nsm, rlwe_dash_m) = self.data.split_at(self.d_a * 2);
        (
            rlwe_dash_nsm.split_at(self.d_a),
            rlwe_dash_m.split_at(self.d_b),
        )
    }
}

impl<'a, R> RgswCiphertextMut for RgswCiphertextMutRef<'a, R>
where
    R: RowMut,
{
    fn split_mut(
        &mut self,
    ) -> (
        (&mut [Self::R], &mut [Self::R]),
        (&mut [Self::R], &mut [Self::R]),
    ) {
        let (rlwe_dash_nsm, rlwe_dash_m) = self.data.split_at_mut(self.d_a * 2);
        (
            rlwe_dash_nsm.split_at_mut(self.d_a),
            rlwe_dash_m.split_at_mut(self.d_b),
        )
    }
}

pub(crate) trait RlweKsk {
    type R: Row;
    fn ksk_part_a(&self) -> &[Self::R];
    fn ksk_part_b(&self) -> &[Self::R];
}

pub(crate) struct RlweKskRef<'a, R> {
    data: &'a [R],
    decomposition_count: usize,
}
impl<'a, R: Row> RlweKskRef<'a, R> {
    pub(crate) fn new(ksk: &'a [R], decomposition_count: usize) -> Self {
        Self {
            data: ksk,
            decomposition_count,
        }
    }
}

impl<'a, R: Row> RlweKsk for RlweKskRef<'a, R> {
    type R = R;

    fn ksk_part_a(&self) -> &[Self::R] {
        &self.data[..self.decomposition_count]
    }

    fn ksk_part_b(&self) -> &[Self::R] {
        &self.data[self.decomposition_count..]
    }
}

pub(crate) trait RlweAutoScratch {
    type R: RowMut;
    type Rgsw: RgswCiphertext;

    fn split_for_rlwe_auto_and_zero_rlwe_space(
        &mut self,
        decompostion_count: usize,
    ) -> (&mut [Self::R], &mut [Self::R]);

    fn split_for_rlwe_auto_trivial_case(&mut self) -> &mut Self::R;

    fn split_for_rlwe_x_rgsw_and_zero_rlwe_space<D: RlweDecomposer>(
        &mut self,
        decomposer: &D,
    ) -> (&mut [Self::R], &mut [Self::R]);

    fn split_for_rgsw_x_rgsw_and_zero_rgsw0_space<D: RlweDecomposer>(
        &mut self,
        d0: &D,
        d1: &D,
    ) -> (&mut [Self::R], &mut [Self::R]);
}

pub(crate) struct RuntimeScratchMutRef<'a, R> {
    data: &'a mut [R],
}

impl<'a, R> RuntimeScratchMutRef<'a, R> {
    pub(crate) fn new(data: &'a mut [R]) -> Self {
        Self { data }
    }
}

impl<'a, R: RowMut> RlweAutoScratch for RuntimeScratchMutRef<'a, R>
where
    R::Element: Zero + Clone,
{
    type R = R;
    type Rgsw = RgswCiphertextRef<'a, R>;

    fn split_for_rlwe_auto_and_zero_rlwe_space(
        &mut self,
        decompostion_count: usize,
    ) -> (&mut [Self::R], &mut [Self::R]) {
        let (decomp_poly, other) = self.data.split_at_mut(decompostion_count);
        let (rlwe, _) = other.split_at_mut(2);

        // zero fill rlwe
        rlwe.iter_mut()
            .for_each(|r| r.as_mut().fill(R::Element::zero()));

        (decomp_poly, rlwe)
    }

    fn split_for_rlwe_auto_trivial_case(&mut self) -> &mut Self::R {
        &mut self.data[0]
    }

    fn split_for_rgsw_x_rgsw_and_zero_rgsw0_space<D: RlweDecomposer>(
        &mut self,
        rgsw0_decoposer: &D,
        rgsw1_decoposer: &D,
    ) -> (&mut [Self::R], &mut [Self::R]) {
        let (decomp_poly, other) = self.data.split_at_mut(std::cmp::max(
            rgsw1_decoposer.a().decomposition_count(),
            rgsw1_decoposer.b().decomposition_count(),
        ));
        let (rgsw, _) = other.split_at_mut(
            rgsw0_decoposer.a().decomposition_count() * 2
                + rgsw0_decoposer.b().decomposition_count() * 2,
        );

        // zero fill rgsw0
        rgsw.iter_mut()
            .for_each(|r| r.as_mut().fill(R::Element::zero()));

        (decomp_poly, rgsw)
    }

    fn split_for_rlwe_x_rgsw_and_zero_rlwe_space<D: RlweDecomposer>(
        &mut self,
        decomposer: &D,
    ) -> (&mut [Self::R], &mut [Self::R]) {
        let (decomp_poly, other) = self.data.split_at_mut(std::cmp::max(
            decomposer.a().decomposition_count(),
            decomposer.b().decomposition_count(),
        ));

        let (rlwe, _) = other.split_at_mut(2);

        // zero fill rlwe
        rlwe.iter_mut()
            .for_each(|r| r.as_mut().fill(R::Element::zero()));

        (decomp_poly, rlwe)
    }
}

pub(crate) fn rgsw_x_rgsw_scratch_rows<D: RlweDecomposer>(
    rgsw0_decomposer: &D,
    rgsw1_decomposer: &D,
) -> usize {
    std::cmp::max(
        rgsw1_decomposer.a().decomposition_count(),
        rgsw1_decomposer.b().decomposition_count(),
    ) + rgsw0_decomposer.a().decomposition_count() * 2
        + rgsw0_decomposer.b().decomposition_count() * 2
}

pub(crate) fn rlwe_x_rgsw_scratch_rows<D: RlweDecomposer>(rgsw_decomposer: &D) -> usize {
    std::cmp::max(
        rgsw_decomposer.a().decomposition_count(),
        rgsw_decomposer.b().decomposition_count(),
    ) + 2
}

pub(crate) fn rlwe_auto_scratch_rows<D: Decomposer>(decomposer: &D) -> usize {
    decomposer.decomposition_count() + 2
}

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
    Rlwe: RlweCiphertext,
    Ksk: RlweKsk<R = Rlwe::R>,
    Sc: RlweAutoScratch<R = Rlwe::R>,
    ModOp: ArithmeticOps<Element = <Rlwe::R as Row>::Element>
        + VectorOps<Element = <Rlwe::R as Row>::Element>,
    NttOp: Ntt<Element = <Rlwe::R as Row>::Element>,
    D: Decomposer<Element = <Rlwe::R as Row>::Element>,
>(
    rlwe_in: &mut Rlwe,
    ksk: &Ksk,
    scratch_matrix: &mut Sc,
    auto_map_index: &[usize],
    auto_map_sign: &[bool],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    decomposer: &D,
    is_trivial: bool,
) where
    <Rlwe::R as Row>::Element: Copy + Zero,
{
    // let ring_size = rlwe_in.dimension().1;
    // assert!(rlwe_in.dimension().0 == 2);
    // assert!(scratch_matrix.fits(d + 2, ring_size));

    if !is_trivial {
        let (decomp_poly_scratch, tmp_rlwe) = scratch_matrix
            .split_for_rlwe_auto_and_zero_rlwe_space(decomposer.decomposition_count());
        let mut tmp_rlwe = RlweCiphertextMutRef::new(tmp_rlwe);

        // send a(X) -> a(X^k) and decompose a(X^k)
        izip!(
            rlwe_in.part_a(),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            let el_out = if !*sign { mod_op.neg(el_in) } else { *el_in };

            decomposer
                .decompose_iter(&el_out)
                .enumerate()
                .for_each(|(index, el)| {
                    decomp_poly_scratch[index].as_mut()[*to_index] = el;
                });
        });

        // transform decomposed a(X^k) to evaluation domain
        decomp_poly_scratch.iter_mut().for_each(|r| {
            ntt_op.forward(r.as_mut());
        });

        // RLWE(m^k) = a', b'; RLWE(m) = a, b
        // key switch: (a * RLWE'(s(X^k)))
        // a' = decomp<a> * RLWE'_A(s(X^k))
        poly_fma_routine(
            tmp_rlwe.part_a_mut(),
            decomp_poly_scratch,
            ksk.ksk_part_a(),
            mod_op,
        );

        // b' += decomp<a(X^k)> * RLWE'_B(s(X^k))
        poly_fma_routine(
            tmp_rlwe.part_b_mut(),
            decomp_poly_scratch,
            ksk.ksk_part_b(),
            mod_op,
        );

        // transform RLWE(m^k) to coefficient domain
        ntt_op.backward(tmp_rlwe.part_a_mut());
        ntt_op.backward(tmp_rlwe.part_b_mut());

        // send b(X) -> b(X^k) and then b'(X) += b(X^k)
        izip!(
            rlwe_in.part_b(),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            let row = tmp_rlwe.part_b_mut();
            if !*sign {
                row[*to_index] = mod_op.sub(&row[*to_index], el_in);
            } else {
                row[*to_index] = mod_op.add(&row[*to_index], el_in);
            }
        });

        // copy over A; Leave B for later
        rlwe_in.part_a_mut().copy_from_slice(tmp_rlwe.part_a());
        rlwe_in.part_b_mut().copy_from_slice(tmp_rlwe.part_b());
    } else {
        // RLWE is trivial, a(X) is 0.
        // send b(X) -> b(X^k)
        let tmp_row = scratch_matrix.split_for_rlwe_auto_trivial_case();
        izip!(
            rlwe_in.part_b(),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            if !*sign {
                tmp_row.as_mut()[*to_index] = mod_op.neg(el_in);
            } else {
                tmp_row.as_mut()[*to_index] = *el_in;
            }
        });
        rlwe_in.part_b_mut().copy_from_slice(tmp_row.as_ref());
    }
}

/// Sends RLWE_{s(X)}(m(X)) -> RLWE_{s(X)}(m{X^k}) where k is some galois
/// element
///
/// This is same as `galois_auto` with the difference that alongside `ksk` with
/// key switching polynomials in evaluation domain, shoup representation,
/// `ksk_shoup`, of the polynomials in evaluation domain is also supplied.
pub(crate) fn rlwe_auto_shoup<
    Rlwe: RlweCiphertext,
    Ksk: RlweKsk<R = Rlwe::R>,
    Sc: RlweAutoScratch<R = Rlwe::R>,
    ModOp: ArithmeticOps<Element = <Rlwe::R as Row>::Element>
        // + VectorOps<Element = MT::MatElement>
        + ShoupMatrixFMA<Rlwe::R>,
    NttOp: Ntt<Element = <Rlwe::R as Row>::Element>,
    D: Decomposer<Element = <Rlwe::R as Row>::Element>,
>(
    rlwe_in: &mut Rlwe,
    ksk: &Ksk,
    ksk_shoup: &Ksk,
    scratch_matrix: &mut Sc,
    auto_map_index: &[usize],
    auto_map_sign: &[bool],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    decomposer: &D,
    is_trivial: bool,
) where
    <Rlwe::R as Row>::Element: Copy + Zero,
{
    // let d = decomposer.decomposition_count();
    // let ring_size = rlwe_in.dimension().1;
    // assert!(rlwe_in.dimension().0 == 2);
    // assert!(scratch_matrix.fits(d + 2, ring_size));

    if !is_trivial {
        let (decomp_poly_scratch, tmp_rlwe) = scratch_matrix
            .split_for_rlwe_auto_and_zero_rlwe_space(decomposer.decomposition_count());
        let mut tmp_rlwe = RlweCiphertextMutRef::new(tmp_rlwe);

        // send a(X) -> a(X^k) and decompose a(X^k)
        izip!(
            rlwe_in.part_a(),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            let el_out = if !*sign { mod_op.neg(el_in) } else { *el_in };

            decomposer
                .decompose_iter(&el_out)
                .enumerate()
                .for_each(|(index, el)| {
                    decomp_poly_scratch[index].as_mut()[*to_index] = el;
                });
        });

        // transform decomposed a(X^k) to evaluation domain
        decomp_poly_scratch.iter_mut().for_each(|r| {
            ntt_op.forward_lazy(r.as_mut());
        });

        // RLWE(m^k) = a', b'; RLWE(m) = a, b
        // key switch: (a * RLWE'(s(X^k)))
        // a' = decomp<a> * RLWE'_A(s(X^k))
        mod_op.shoup_matrix_fma(
            tmp_rlwe.part_a_mut(),
            ksk.ksk_part_a(),
            ksk_shoup.ksk_part_a(),
            decomp_poly_scratch,
        );

        // b'= decomp<a(X^k)> * RLWE'_B(s(X^k))
        mod_op.shoup_matrix_fma(
            tmp_rlwe.part_b_mut(),
            ksk.ksk_part_b(),
            ksk_shoup.ksk_part_b(),
            decomp_poly_scratch,
        );

        // transform RLWE(m^k) to coefficient domain
        ntt_op.backward(tmp_rlwe.part_a_mut());
        ntt_op.backward(tmp_rlwe.part_b_mut());

        // send b(X) -> b(X^k) and then b'(X) += b(X^k)
        let row = tmp_rlwe.part_b_mut();
        izip!(
            rlwe_in.part_b(),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            if !*sign {
                row[*to_index] = mod_op.sub(&row[*to_index], el_in);
            } else {
                row[*to_index] = mod_op.add(&row[*to_index], el_in);
            }
        });

        // copy over A, B
        rlwe_in.part_a_mut().copy_from_slice(tmp_rlwe.part_a());
        rlwe_in.part_b_mut().copy_from_slice(tmp_rlwe.part_b());
    } else {
        // RLWE is trivial, a(X) is 0.
        // send b(X) -> b(X^k)
        let row = scratch_matrix.split_for_rlwe_auto_trivial_case();
        izip!(
            rlwe_in.part_b(),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            if !*sign {
                row.as_mut()[*to_index] = mod_op.neg(el_in);
            } else {
                row.as_mut()[*to_index] = *el_in;
            }
        });
        rlwe_in.part_b_mut().copy_from_slice(row.as_ref());
    }
}

/// Inplace mutates RLWE(m0) to equal RLWE(m0m1) = RLWE(m0) x RGSW(m1).
///
/// - rlwe_in: is RLWE(m0) with polynomials in coefficient domain
/// - rgsw_in: is RGSW(m1) with polynomials in evaluation domain
/// - scratch_matrix: with dimension (max(d_a, d_b) + 2) x ring_size columns.
///   It's used to store decomposed polynomials and out RLWE temporarily
pub(crate) fn rlwe_by_rgsw<
    Rlwe: RlweCiphertext,
    Rgsw: RgswCiphertext<R = Rlwe::R>,
    Sc: RlweAutoScratch<R = Rlwe::R>,
    D: RlweDecomposer<Element = <Rlwe::R as Row>::Element>,
    ModOp: VectorOps<Element = <Rlwe::R as Row>::Element>,
    NttOp: Ntt<Element = <Rlwe::R as Row>::Element>,
>(
    rlwe_in: &mut Rlwe,
    rgsw_in: &Rgsw,
    scratch_matrix: &mut Sc,
    decomposer: &D,
    ntt_op: &NttOp,
    mod_op: &ModOp,
    is_trivial: bool,
) where
    <Rlwe::R as Row>::Element: Copy + Zero,
{
    let decomposer_a = decomposer.a();
    let decomposer_b = decomposer.b();
    let d_a = decomposer_a.decomposition_count();
    let d_b = decomposer_b.decomposition_count();

    let ((rlwe_dash_nsm_parta, rlwe_dash_nsm_partb), (rlwe_dash_m_parta, rlwe_dash_m_partb)) =
        rgsw_in.split();

    let (decomposed_poly_scratch, tmp_rlwe) =
        scratch_matrix.split_for_rlwe_x_rgsw_and_zero_rlwe_space(decomposer);

    // RLWE_in = a_in, b_in; RLWE_out = a_out, b_out
    if !is_trivial {
        // a_in = 0 when RLWE_in is trivial RLWE ciphertext
        // decomp<a_in>
        let mut decomposed_polys_of_rlwea = &mut decomposed_poly_scratch[..d_a];
        decompose_r(
            rlwe_in.part_a(),
            &mut decomposed_polys_of_rlwea,
            decomposer_a,
        );

        decomposed_polys_of_rlwea
            .iter_mut()
            .for_each(|r| ntt_op.forward(r.as_mut()));

        // a_out += decomp<a_in> \cdot RLWE_A'(-sm)
        poly_fma_routine(
            tmp_rlwe[0].as_mut(),
            &decomposed_polys_of_rlwea,
            rlwe_dash_nsm_parta,
            mod_op,
        );
        // b_out += decomp<a_in> \cdot RLWE_B'(-sm)
        poly_fma_routine(
            tmp_rlwe[1].as_mut(),
            &decomposed_polys_of_rlwea,
            &rlwe_dash_nsm_partb,
            mod_op,
        );
    }

    {
        // decomp<b_in>
        let mut decomposed_polys_of_rlweb = &mut decomposed_poly_scratch[..d_b];
        decompose_r(
            rlwe_in.part_b(),
            &mut decomposed_polys_of_rlweb,
            decomposer_b,
        );

        decomposed_polys_of_rlweb
            .iter_mut()
            .for_each(|r| ntt_op.forward(r.as_mut()));

        // a_out += decomp<b_in> \cdot RLWE_A'(m)
        poly_fma_routine(
            tmp_rlwe[0].as_mut(),
            &decomposed_polys_of_rlweb,
            &rlwe_dash_m_parta,
            mod_op,
        );
        // b_out += decomp<b_in> \cdot RLWE_B'(m)
        poly_fma_routine(
            tmp_rlwe[1].as_mut(),
            &decomposed_polys_of_rlweb,
            &rlwe_dash_m_partb,
            mod_op,
        );
    }

    // transform rlwe_out to coefficient domain
    tmp_rlwe
        .iter_mut()
        .for_each(|r| ntt_op.backward(r.as_mut()));

    rlwe_in.part_a_mut().copy_from_slice(tmp_rlwe[0].as_mut());
    rlwe_in.part_b_mut().copy_from_slice(tmp_rlwe[1].as_mut());
}

/// Inplace mutates RLWE(m0) to equal RLWE(m0m1) = RLWE(m0) x RGSW(m1).
///
/// Same as `rlwe_by_rgsw` with the difference that alongside `rgsw_in` with
/// polynomials in evaluation domain, shoup representation of polynomials in
/// evaluation domain, `rgsw_in_shoup`, is also supplied.
pub(crate) fn rlwe_by_rgsw_shoup<
    Rlwe: RlweCiphertext,
    Rgsw: RgswCiphertext<R = Rlwe::R>,
    Sc: RlweAutoScratch<R = Rlwe::R>,
    D: RlweDecomposer<Element = <Rlwe::R as Row>::Element>,
    ModOp: ShoupMatrixFMA<Rlwe::R>,
    NttOp: Ntt<Element = <Rlwe::R as Row>::Element>,
>(
    rlwe_in: &mut Rlwe,
    rgsw_in: &Rgsw,
    rgsw_in_shoup: &Rgsw,
    scratch_matrix: &mut Sc,
    decomposer: &D,
    ntt_op: &NttOp,
    mod_op: &ModOp,
    is_trivial: bool,
) where
    <Rlwe::R as Row>::Element: Copy + Zero,
{
    let decomposer_a = decomposer.a();
    let decomposer_b = decomposer.b();
    let d_a = decomposer_a.decomposition_count();
    let d_b = decomposer_b.decomposition_count();

    let ((rlwe_dash_nsm_parta, rlwe_dash_nsm_partb), (rlwe_dash_m_parta, rlwe_dash_m_partb)) =
        rgsw_in.split();

    let (
        (rlwe_dash_nsm_parta_shoup, rlwe_dash_nsm_partb_shoup),
        (rlwe_dash_m_parta_shoup, rlwe_dash_m_partb_shoup),
    ) = rgsw_in_shoup.split();

    let (decomposed_poly_scratch, tmp_rlwe) =
        scratch_matrix.split_for_rlwe_x_rgsw_and_zero_rlwe_space(decomposer);

    // RLWE_in = a_in, b_in; RLWE_out = a_out, b_out
    if !is_trivial {
        // a_in = 0 when RLWE_in is trivial RLWE ciphertext
        // decomp<a_in>
        let mut decomposed_polys_of_rlwea = &mut decomposed_poly_scratch[..d_a];
        decompose_r(
            rlwe_in.part_a(),
            &mut decomposed_polys_of_rlwea,
            decomposer_a,
        );
        decomposed_polys_of_rlwea
            .iter_mut()
            .for_each(|r| ntt_op.forward_lazy(r.as_mut()));

        // a_out += decomp<a_in> \cdot RLWE_A'(-sm)
        mod_op.shoup_matrix_fma(
            tmp_rlwe[0].as_mut(),
            &rlwe_dash_nsm_parta,
            &rlwe_dash_nsm_parta_shoup,
            &decomposed_polys_of_rlwea,
        );

        // b_out += decomp<a_in> \cdot RLWE_B'(-sm)
        mod_op.shoup_matrix_fma(
            tmp_rlwe[1].as_mut(),
            &rlwe_dash_nsm_partb,
            &rlwe_dash_nsm_partb_shoup,
            &decomposed_polys_of_rlwea,
        );
    }
    {
        // decomp<b_in>
        let mut decomposed_polys_of_rlweb = &mut decomposed_poly_scratch[..d_b];
        decompose_r(
            rlwe_in.part_b(),
            &mut decomposed_polys_of_rlweb,
            decomposer_b,
        );
        decomposed_polys_of_rlweb
            .iter_mut()
            .for_each(|r| ntt_op.forward_lazy(r.as_mut()));

        // a_out += decomp<b_in> \cdot RLWE_A'(m)
        mod_op.shoup_matrix_fma(
            tmp_rlwe[0].as_mut(),
            &rlwe_dash_m_parta,
            &rlwe_dash_m_parta_shoup,
            &decomposed_polys_of_rlweb,
        );

        // b_out += decomp<b_in> \cdot RLWE_B'(m)
        mod_op.shoup_matrix_fma(
            tmp_rlwe[1].as_mut(),
            &rlwe_dash_m_partb,
            &rlwe_dash_m_partb_shoup,
            &decomposed_polys_of_rlweb,
        );
    }

    // transform rlwe_out to coefficient domain
    tmp_rlwe
        .iter_mut()
        .for_each(|r| ntt_op.backward(r.as_mut()));

    rlwe_in.part_a_mut().copy_from_slice(tmp_rlwe[0].as_mut());
    rlwe_in.part_b_mut().copy_from_slice(tmp_rlwe[1].as_mut());
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
    Rgsw: RgswCiphertext,
    RgswMut: RgswCiphertextMut<R = Rgsw::R>,
    Sc: RlweAutoScratch<R = Rgsw::R, Rgsw = Rgsw>,
    D: RlweDecomposer<Element = <Rgsw::R as Row>::Element>,
    ModOp: VectorOps<Element = <Rgsw::R as Row>::Element>,
    NttOp: Ntt<Element = <Rgsw::R as Row>::Element>,
>(
    rgsw0: &mut RgswMut,
    rgsw1_eval: &Rgsw,
    rgsw0_decomposer: &D,
    rgsw1_decomposer: &D,
    scratch_matrix: &mut Sc,
    ntt_op: &NttOp,
    mod_op: &ModOp,
) where
    <Rgsw::R as Row>::Element: Copy + Zero,
    RgswMut: AsMut<[Rgsw::R]>,
    RgswMut::R: RowMut,
    // Rgsw: AsRef<[Rgsw::R]>,
{
    // let rgsw0_rows = rgsw0_da * 2 + rgsw0_db * 2;
    // let ring_size = rgsw0.dimension().1;
    // assert!(rgsw0.dimension().0 == rgsw0_rows);
    // assert!(rgsw1_eval.dimension() == (rgsw1_rows, ring_size));
    // assert!(scratch_matrix.fits(max_d + rgsw0_rows, ring_size));

    let (decomp_r_space, rgsw_space) = scratch_matrix
        .split_for_rgsw_x_rgsw_and_zero_rgsw0_space(rgsw0_decomposer, rgsw1_decomposer);

    let mut rgsw_space = RgswCiphertextMutRef::new(
        rgsw_space,
        rgsw0_decomposer.a().decomposition_count(),
        rgsw0_decomposer.b().decomposition_count(),
    );
    let (
        (rlwe_dash_space_nsm_parta, rlwe_dash_space_nsm_partb),
        (rlwe_dash_space_m_parta, rlwe_dash_space_m_partb),
    ) = rgsw_space.split_mut();

    let ((rgsw0_nsm_parta, rgsw0_nsm_partb), (rgsw0_m_parta, rgsw0_m_partb)) = rgsw0.split();
    let ((rgsw1_nsm_parta, rgsw1_nsm_partb), (rgsw1_m_parta, rgsw1_m_partb)) = rgsw1_eval.split();

    // RGSW x RGSW
    izip!(
        rgsw0_nsm_parta.iter().chain(rgsw0_m_parta),
        rgsw0_nsm_partb.iter().chain(rgsw0_m_partb),
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
        {
            let decomp_r_parta = &mut decomp_r_space[..rgsw1_decomposer.a().decomposition_count()];
            decompose_r(
                rlwe_a.as_ref(),
                decomp_r_parta.as_mut(),
                rgsw1_decomposer.a(),
            );
            decomp_r_parta
                .iter_mut()
                .for_each(|ri| ntt_op.forward(ri.as_mut()));
            poly_fma_routine(
                rlwe_out_a.as_mut(),
                &decomp_r_parta,
                &rgsw1_nsm_parta,
                mod_op,
            );
            poly_fma_routine(
                rlwe_out_b.as_mut(),
                &decomp_r_parta,
                &rgsw1_nsm_partb,
                mod_op,
            );
        }

        // Part B: Decompose<RLWE(m0)[B]> \cdot RLWE'(m1)
        {
            let decomp_r_partb = &mut decomp_r_space[..rgsw1_decomposer.b().decomposition_count()];
            decompose_r(
                rlwe_b.as_ref(),
                decomp_r_partb.as_mut(),
                rgsw1_decomposer.b(),
            );
            decomp_r_partb
                .iter_mut()
                .for_each(|ri| ntt_op.forward(ri.as_mut()));
            poly_fma_routine(rlwe_out_a.as_mut(), &decomp_r_partb, &rgsw1_m_parta, mod_op);
            poly_fma_routine(rlwe_out_b.as_mut(), &decomp_r_partb, &rgsw1_m_partb, mod_op);
        }
    });

    // copy over RGSW(m0m1) to RGSW(m0)
    // let d = rgsw0.as_mut();
    izip!(rgsw0.as_mut().iter_mut(), rgsw_space.data.iter())
        .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

    // send back to coefficient domain
    rgsw0
        .as_mut()
        .iter_mut()
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
