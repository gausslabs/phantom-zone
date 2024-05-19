use std::{
    clone,
    fmt::Debug,
    marker::PhantomData,
    ops::{Div, Neg, Sub},
};

use itertools::{izip, Itertools};
use num_traits::{PrimInt, Signed, ToPrimitive, Zero};

use crate::{
    backend::{ArithmeticOps, VectorOps},
    decomposer::{self, Decomposer, RlweDecomposer},
    ntt::{self, Ntt, NttInit},
    random::{DefaultSecureRng, NewWithSeed, RandomGaussianDist, RandomUniformDist},
    utils::{fill_random_ternary_secret_with_hamming_weight, TryConvertFrom, WithLocal},
    Matrix, MatrixEntity, MatrixMut, Row, RowEntity, RowMut, Secret,
};

pub struct SeededAutoKey<M, S>
where
    M: Matrix,
{
    data: M,
    seed: S,
    modulus: M::MatElement,
}

impl<M: Matrix + MatrixEntity, S> SeededAutoKey<M, S> {
    fn empty<D: Decomposer>(
        ring_size: usize,
        auto_decomposer: &D,
        seed: S,
        modulus: M::MatElement,
    ) -> Self {
        SeededAutoKey {
            data: M::zeros(auto_decomposer.decomposition_count(), ring_size),
            seed,
            modulus: modulus,
        }
    }
}

pub struct AutoKeyEvaluationDomain<M, R, N> {
    data: M,
    _phantom: PhantomData<(R, N)>,
}

impl<
        M: MatrixMut + MatrixEntity,
        R: RandomUniformDist<[M::MatElement], Parameters = M::MatElement> + NewWithSeed,
        N: NttInit<Element = M::MatElement> + Ntt<Element = M::MatElement>,
    > From<&SeededAutoKey<M, R::Seed>> for AutoKeyEvaluationDomain<M, R, N>
where
    <M as Matrix>::R: RowMut,
    M::MatElement: Copy,
    R::Seed: Clone,
{
    fn from(value: &SeededAutoKey<M, R::Seed>) -> Self {
        let (d, ring_size) = value.data.dimension();
        let mut data = M::zeros(2 * d, ring_size);

        // sample RLWE'_A(-s(X^k))
        let mut p_rng = R::new_with_seed(value.seed.clone());
        data.iter_rows_mut().take(d).for_each(|r| {
            RandomUniformDist::random_fill(&mut p_rng, &value.modulus, r.as_mut());
        });

        // copy over RLWE'_B(-s(X^k))
        izip!(data.iter_rows_mut().skip(d), value.data.iter_rows()).for_each(|(to_r, from_r)| {
            to_r.as_mut().copy_from_slice(from_r.as_ref());
        });

        // send RLWE'(-s(X^k)) polynomials to evaluation domain
        let ntt_op = N::new(value.modulus, ring_size);
        data.iter_rows_mut()
            .for_each(|r| ntt_op.forward(r.as_mut()));

        AutoKeyEvaluationDomain {
            data,
            _phantom: PhantomData,
        }
    }
}

pub struct RgswCiphertext<M: Matrix> {
    /// Rgsw ciphertext polynomials
    pub(crate) data: M,
    modulus: M::MatElement,
    /// Decomposition for RLWE part A
    d_a: usize,
    /// Decomposition for RLWE part B
    d_b: usize,
}

impl<M: MatrixEntity> RgswCiphertext<M> {
    pub(crate) fn empty<D: RlweDecomposer>(
        ring_size: usize,
        decomposer: &D,
        modulus: M::MatElement,
    ) -> RgswCiphertext<M> {
        RgswCiphertext {
            data: M::zeros(
                decomposer.a().decomposition_count() * 2 + decomposer.b().decomposition_count() * 2,
                ring_size,
            ),
            d_a: decomposer.a().decomposition_count(),
            d_b: decomposer.b().decomposition_count(),
            modulus,
        }
    }
}

pub struct SeededRgswCiphertext<M, S>
where
    M: Matrix,
{
    pub(crate) data: M,
    seed: S,
    modulus: M::MatElement,
    /// Decomposition for RLWE part A
    d_a: usize,
    /// Decomposition for RLWE part B
    d_b: usize,
}

impl<M: Matrix + MatrixEntity, S> SeededRgswCiphertext<M, S> {
    pub(crate) fn empty<D: RlweDecomposer>(
        ring_size: usize,
        decomposer: &D,
        seed: S,
        modulus: M::MatElement,
    ) -> SeededRgswCiphertext<M, S> {
        SeededRgswCiphertext {
            data: M::zeros(
                decomposer.a().decomposition_count() * 2 + decomposer.b().decomposition_count(),
                ring_size,
            ),
            seed,
            modulus: modulus,
            d_a: decomposer.a().decomposition_count(),
            d_b: decomposer.b().decomposition_count(),
        }
    }
}

impl<M: Debug + Matrix, S: Debug> Debug for SeededRgswCiphertext<M, S>
where
    M::MatElement: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeededRgswCiphertext")
            .field("data", &self.data)
            .field("seed", &self.seed)
            .field("modulus", &self.modulus)
            .finish()
    }
}

pub struct RgswCiphertextEvaluationDomain<M, R, N> {
    pub(crate) data: M,
    _phantom: PhantomData<(R, N)>,
}

impl<
        M: MatrixMut + MatrixEntity,
        R: NewWithSeed + RandomUniformDist<[M::MatElement], Parameters = M::MatElement>,
        N: NttInit<Element = M::MatElement> + Ntt<Element = M::MatElement> + Debug,
    > From<&SeededRgswCiphertext<M, R::Seed>> for RgswCiphertextEvaluationDomain<M, R, N>
where
    <M as Matrix>::R: RowMut,
    M::MatElement: Copy,
    R::Seed: Clone,
    M: Debug,
{
    fn from(value: &SeededRgswCiphertext<M, R::Seed>) -> Self {
        let mut data = M::zeros(value.d_a * 2 + value.d_b * 2, value.data.dimension().1);

        // copy RLWE'(-sm)
        izip!(
            data.iter_rows_mut().take(value.d_a * 2),
            value.data.iter_rows().take(value.d_a * 2)
        )
        .for_each(|(to_ri, from_ri)| {
            to_ri.as_mut().copy_from_slice(from_ri.as_ref());
        });

        // sample A polynomials of RLWE'(m) - RLWE'A(m)
        // TODO(Jay): Do we want to be generic over RandomGenerator used here? I think
        // not.
        let mut p_rng = R::new_with_seed(value.seed.clone());
        izip!(data.iter_rows_mut().skip(value.d_a * 2).take(value.d_b * 1))
            .for_each(|ri| p_rng.random_fill(&value.modulus, ri.as_mut()));

        // RLWE'_B(m)
        izip!(
            data.iter_rows_mut().skip(value.d_a * 2 + value.d_b),
            value.data.iter_rows().skip(value.d_a * 2)
        )
        .for_each(|(to_ri, from_ri)| {
            to_ri.as_mut().copy_from_slice(from_ri.as_ref());
        });

        // Send polynomials to evaluation domain
        let ring_size = data.dimension().1;
        let nttop = N::new(value.modulus, ring_size);
        data.iter_rows_mut()
            .for_each(|ri| nttop.forward(ri.as_mut()));

        Self {
            data: data,
            _phantom: PhantomData,
        }
    }
}

impl<
        M: MatrixMut + MatrixEntity,
        R,
        N: NttInit<Element = M::MatElement> + Ntt<Element = M::MatElement>,
    > From<&RgswCiphertext<M>> for RgswCiphertextEvaluationDomain<M, R, N>
where
    <M as Matrix>::R: RowMut,
    M::MatElement: Copy,
    M: Debug,
{
    fn from(value: &RgswCiphertext<M>) -> Self {
        let mut data = M::zeros(value.d_a * 2 + value.d_b * 2, value.data.dimension().1);

        // copy RLWE'(-sm)
        izip!(
            data.iter_rows_mut().take(value.d_a * 2),
            value.data.iter_rows().take(value.d_a * 2)
        )
        .for_each(|(to_ri, from_ri)| {
            to_ri.as_mut().copy_from_slice(from_ri.as_ref());
        });

        // copy RLWE'(m)
        izip!(
            data.iter_rows_mut().skip(value.d_a * 2),
            value.data.iter_rows().skip(value.d_a * 2)
        )
        .for_each(|(to_ri, from_ri)| {
            to_ri.as_mut().copy_from_slice(from_ri.as_ref());
        });

        // Send polynomials to evaluation domain
        let ring_size = data.dimension().1;
        let nttop = N::new(value.modulus, ring_size);
        data.iter_rows_mut()
            .for_each(|ri| nttop.forward(ri.as_mut()));

        Self {
            data: data,
            _phantom: PhantomData,
        }
    }
}

impl<M: Debug, R, N> Debug for RgswCiphertextEvaluationDomain<M, R, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RgswCiphertextEvaluationDomain")
            .field("data", &self.data)
            .field("_phantom", &self._phantom)
            .finish()
    }
}

impl<M: Matrix, R, N> Matrix for RgswCiphertextEvaluationDomain<M, R, N> {
    type MatElement = M::MatElement;
    type R = M::R;

    fn dimension(&self) -> (usize, usize) {
        self.data.dimension()
    }

    fn fits(&self, row: usize, col: usize) -> bool {
        self.data.fits(row, col)
    }
}

impl<M: Matrix, R, N> AsRef<[M::R]> for RgswCiphertextEvaluationDomain<M, R, N> {
    fn as_ref(&self) -> &[M::R] {
        self.data.as_ref()
    }
}

pub struct SeededRlweCiphertext<R, S>
where
    R: Row,
{
    pub(crate) data: R,
    pub(crate) seed: S,
    pub(crate) modulus: R::Element,
}

impl<R: RowEntity, S> SeededRlweCiphertext<R, S> {
    pub(crate) fn empty(ring_size: usize, seed: S, modulus: R::Element) -> Self {
        SeededRlweCiphertext {
            data: R::zeros(ring_size),
            seed,
            modulus,
        }
    }
}

pub struct RlweCiphertext<M, Rng> {
    pub(crate) data: M,
    pub(crate) is_trivial: bool,
    pub(crate) _phatom: PhantomData<Rng>,
}

impl<M, Rng> RlweCiphertext<M, Rng> {
    pub(crate) fn new_trivial(data: M) -> Self {
        RlweCiphertext {
            data,
            is_trivial: true,
            _phatom: PhantomData,
        }
    }
}

impl<M: Matrix, Rng> Matrix for RlweCiphertext<M, Rng> {
    type MatElement = M::MatElement;
    type R = M::R;

    fn dimension(&self) -> (usize, usize) {
        self.data.dimension()
    }

    fn fits(&self, row: usize, col: usize) -> bool {
        self.data.fits(row, col)
    }
}

impl<M: MatrixMut, Rng> MatrixMut for RlweCiphertext<M, Rng> where <M as Matrix>::R: RowMut {}

impl<M: Matrix, Rng> AsRef<[<M as Matrix>::R]> for RlweCiphertext<M, Rng> {
    fn as_ref(&self) -> &[<M as Matrix>::R] {
        self.data.as_ref()
    }
}

impl<M: MatrixMut, Rng> AsMut<[<M as Matrix>::R]> for RlweCiphertext<M, Rng>
where
    <M as Matrix>::R: RowMut,
{
    fn as_mut(&mut self) -> &mut [<M as Matrix>::R] {
        self.data.as_mut()
    }
}

impl<M, Rng> IsTrivial for RlweCiphertext<M, Rng> {
    fn is_trivial(&self) -> bool {
        self.is_trivial
    }
    fn set_not_trivial(&mut self) {
        self.is_trivial = false;
    }
}

impl<R: Row, M: MatrixEntity<R = R, MatElement = R::Element> + MatrixMut, Rng: NewWithSeed>
    From<&SeededRlweCiphertext<R, Rng::Seed>> for RlweCiphertext<M, Rng>
where
    Rng::Seed: Clone,
    Rng: RandomUniformDist<[M::MatElement], Parameters = M::MatElement>,
    <M as Matrix>::R: RowMut,
    R::Element: Copy,
{
    fn from(value: &SeededRlweCiphertext<R, Rng::Seed>) -> Self {
        let mut data = M::zeros(2, value.data.as_ref().len());

        // sample a
        let mut p_rng = Rng::new_with_seed(value.seed.clone());
        RandomUniformDist::random_fill(&mut p_rng, &value.modulus, data.get_row_mut(0));

        data.get_row_mut(1).copy_from_slice(value.data.as_ref());

        RlweCiphertext {
            data,
            is_trivial: false,
            _phatom: PhantomData,
        }
    }
}

pub trait IsTrivial {
    fn is_trivial(&self) -> bool;
    fn set_not_trivial(&mut self);
}

pub struct SeededRlwePublicKey<Ro: Row, S> {
    data: Ro,
    seed: S,
    modulus: Ro::Element,
}

impl<Ro: RowEntity, S> SeededRlwePublicKey<Ro, S> {
    pub(crate) fn empty(ring_size: usize, seed: S, modulus: Ro::Element) -> Self {
        Self {
            data: Ro::zeros(ring_size),
            seed,
            modulus,
        }
    }
}

pub struct RlwePublicKey<M, R> {
    data: M,
    _phantom: PhantomData<R>,
}

impl<
        M: MatrixMut + MatrixEntity,
        Rng: NewWithSeed + RandomUniformDist<[M::MatElement], Parameters = M::MatElement>,
    > From<&SeededRlwePublicKey<M::R, Rng::Seed>> for RlwePublicKey<M, Rng>
where
    <M as Matrix>::R: RowMut,
    M::MatElement: Copy,
    Rng::Seed: Copy,
{
    fn from(value: &SeededRlwePublicKey<M::R, Rng::Seed>) -> Self {
        let mut data = M::zeros(2, value.data.as_ref().len());

        // sample a
        let mut p_rng = Rng::new_with_seed(value.seed);
        RandomUniformDist::random_fill(&mut p_rng, &value.modulus, data.get_row_mut(0));

        // copy over b
        data.get_row_mut(1).copy_from_slice(value.data.as_ref());

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct RlweSecret {
    pub(crate) values: Vec<i32>,
}

impl Secret for RlweSecret {
    type Element = i32;
    fn values(&self) -> &[Self::Element] {
        &self.values
    }
}

impl RlweSecret {
    pub fn random(hw: usize, n: usize) -> RlweSecret {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut out = vec![0i32; n];
            fill_random_ternary_secret_with_hamming_weight(&mut out, hw, rng);

            RlweSecret { values: out }
        })
    }
}

pub(crate) fn generate_auto_map(ring_size: usize, k: isize) -> (Vec<usize>, Vec<bool>) {
    assert!(k & 1 == 1, "Auto {k} must be odd");

    let k = if k < 0 {
        // k is -ve, return k%(2*N)
        (2 * ring_size) - (k.abs() as usize % (2 * ring_size))
    } else {
        k as usize
    };
    let (auto_map_index, auto_sign_index): (Vec<usize>, Vec<bool>) = (0..ring_size)
        .into_iter()
        .map(|i| {
            let mut to_index = (i * k) % (2 * ring_size);
            let mut sign = true;

            // wrap around. false implies negative
            if to_index >= ring_size {
                to_index = to_index - ring_size;
                sign = false;
            }

            (to_index, sign)
        })
        .unzip();
    (auto_map_index, auto_sign_index)
}

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
    let d = decomposer.decomposition_count();

    for ri in 0..ring_size {
        let el_decomposed = decomposer.decompose(&r[ri]);
        for j in 0..d {
            decomp_r[j].as_mut()[ri] = el_decomposed[j];
        }
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

    let (scratch_matrix_d_ring, tmp_rlwe_out) = scratch_matrix.split_at_row_mut(d);

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
            // scratch_matrix_dplus2_ring.set(d + 1, *to_index, *el_in);
        }
    });

    if !rlwe_in.is_trivial() {
        // send a(X) -> a(X^k) and decompose a(X^k)
        izip!(
            rlwe_in.get_row(0),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(el_in, to_index, sign)| {
            let el_out = if !*sign { mod_op.neg(el_in) } else { *el_in };

            let el_out_decomposed = decomposer.decompose(&el_out);
            for j in 0..d {
                scratch_matrix_d_ring[j].as_mut()[*to_index] = el_out_decomposed[j];
            }
        });

        // transform decomposed a(X^k) to evaluation domain
        scratch_matrix_d_ring.iter_mut().for_each(|r| {
            ntt_op.forward(r.as_mut());
        });

        // RLWE(m^k) = a', b'; RLWE(m) = a, b
        // key switch: (a * RLWE'(s(X^k)))
        let (ksk_a, ksk_b) = ksk.split_at_row(d);
        tmp_rlwe_out[0].as_mut().fill(Mmut::MatElement::zero());
        // a' = decomp<a> * RLWE'_A(s(X^k))
        routine(
            tmp_rlwe_out[0].as_mut(),
            scratch_matrix_d_ring,
            ksk_a,
            mod_op,
        );
        // send b(X^k) to evaluation domain
        ntt_op.forward(tmp_rlwe_out[1].as_mut());
        // b' = b(X^k)
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

        rlwe_in
            .get_row_mut(0)
            .copy_from_slice(tmp_rlwe_out[0].as_ref());
    }

    rlwe_in
        .get_row_mut(1)
        .copy_from_slice(tmp_rlwe_out[1].as_ref());
}

/// Returns RLWE(m0m1) = RLWE(m0) x RGSW(m1). Mutates rlwe_in inplace to equal
/// RLWE(m0m1)
///
/// - rlwe_in: is RLWE(m0) with polynomials in coefficient domain
/// - rgsw_in: is RGSW(m1) with polynomials in evaluation domain
/// - scratch_matrix_d_ring: is a matrix of dimension (d_rgsw, ring_size) used
///   as scratch space to store decomposed Ring elements temporarily
pub(crate) fn less1_rlwe_by_rgsw<
    Mmut: MatrixMut,
    MT: Matrix<MatElement = Mmut::MatElement> + MatrixMut<MatElement = Mmut::MatElement> + IsTrivial,
    D: Decomposer<Element = Mmut::MatElement>,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
>(
    rlwe_in: &mut MT,
    rgsw_in: &Mmut,
    scratch_matrix_dplus2_ring: &mut Mmut,
    decomposer: &D,
    ntt_op: &NttOp,
    mod_op: &ModOp,
    skip0: usize,
    skip1: usize,
) where
    Mmut::MatElement: Copy + Zero,
    <Mmut as Matrix>::R: RowMut,
    <MT as Matrix>::R: RowMut,
{
    let d_rgsw = decomposer.decomposition_count();
    assert!(scratch_matrix_dplus2_ring.dimension() == (d_rgsw + 2, rlwe_in.dimension().1));
    assert!(rgsw_in.dimension() == (d_rgsw * 4, rlwe_in.dimension().1));

    // decomposed RLWE x RGSW
    let (rlwe_dash_nsm, rlwe_dash_m) = rgsw_in.split_at_row(d_rgsw * 2);
    let (scratch_matrix_d_ring, scratch_rlwe_out) =
        scratch_matrix_dplus2_ring.split_at_row_mut(d_rgsw);
    scratch_rlwe_out[0].as_mut().fill(Mmut::MatElement::zero());
    scratch_rlwe_out[1].as_mut().fill(Mmut::MatElement::zero());
    // RLWE_in = a_in, b_in; RLWE_out = a_out, b_out
    if !rlwe_in.is_trivial() {
        // a_in = 0 when RLWE_in is trivial RLWE ciphertext
        // decomp<a_in>
        decompose_r(rlwe_in.get_row_slice(0), scratch_matrix_d_ring, decomposer);
        scratch_matrix_d_ring
            .iter_mut()
            .for_each(|r| ntt_op.forward(r.as_mut()));
        // a_out += decomp<a_in> \cdot RLWE_A'(-sm)
        routine(
            scratch_rlwe_out[0].as_mut(),
            scratch_matrix_d_ring[skip0..].as_ref(),
            &rlwe_dash_nsm[skip0..d_rgsw],
            mod_op,
        );
        // b_out += decomp<a_in> \cdot RLWE_B'(-sm)
        routine(
            scratch_rlwe_out[1].as_mut(),
            scratch_matrix_d_ring[skip0..].as_ref(),
            &rlwe_dash_nsm[d_rgsw + skip0..],
            mod_op,
        );
    }
    // decomp<b_in>
    decompose_r(rlwe_in.get_row_slice(1), scratch_matrix_d_ring, decomposer);
    scratch_matrix_d_ring
        .iter_mut()
        .for_each(|r| ntt_op.forward(r.as_mut()));
    // a_out += decomp<b_in> \cdot RLWE_A'(m)
    routine(
        scratch_rlwe_out[0].as_mut(),
        scratch_matrix_d_ring[skip1..].as_ref(),
        &rlwe_dash_m[skip1..d_rgsw],
        mod_op,
    );
    // b_out += decomp<b_in> \cdot RLWE_B'(m)
    routine(
        scratch_rlwe_out[1].as_mut(),
        scratch_matrix_d_ring[skip1..].as_ref(),
        &rlwe_dash_m[d_rgsw + skip1..],
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
            scratch_matrix_d_ring,
            decomposer_a,
        );
        scratch_matrix_d_ring
            .iter_mut()
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
        scratch_matrix_d_ring,
        decomposer_b,
    );
    scratch_matrix_d_ring
        .iter_mut()
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

/// Encrypts message m as a RGSW ciphertext.
///
/// - m_eval: is `m` is evaluation domain
/// - out_rgsw: RGSW(m) is stored as single matrix of dimension (d_rgsw * 3,
///   ring_size). The matrix has the following structure [RLWE'_A(-sm) ||
///   RLWE'_B(-sm) || RLWE'_B(m)]^T and RLWE'_A(m) is generated via seed (where
///   p_rng is assumed to be seeded with seed)
pub(crate) fn secret_key_encrypt_rgsw<
    Mmut: MatrixMut + MatrixEntity,
    S,
    R: RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
    PR: RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
>(
    out_rgsw: &mut Mmut,
    m: &[Mmut::MatElement],
    gadget_a: &[Mmut::MatElement],
    gadget_b: &[Mmut::MatElement],
    s: &[S],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    p_rng: &mut PR,
    rng: &mut R,
) where
    <Mmut as Matrix>::R:
        RowMut + RowEntity + TryConvertFrom<[S], Parameters = Mmut::MatElement> + Debug,
    Mmut::MatElement: Copy + Debug,
{
    let d_a = gadget_a.len();
    let d_b = gadget_b.len();
    let q = mod_op.modulus();
    let ring_size = s.len();
    assert!(out_rgsw.dimension() == (d_a * 2 + d_b, ring_size));
    assert!(m.as_ref().len() == ring_size);

    // RLWE(-sm), RLWE(m)
    let (rlwe_dash_nsm, b_rlwe_dash_m) = out_rgsw.split_at_row_mut(d_a * 2);

    let mut s_eval = Mmut::R::try_convert_from(s, &q);
    ntt_op.forward(s_eval.as_mut());

    let mut scratch_space = Mmut::R::zeros(ring_size);

    // RLWE'(-sm)
    let (a_rlwe_dash_nsm, b_rlwe_dash_nsm) = rlwe_dash_nsm.split_at_mut(d_a);
    izip!(
        a_rlwe_dash_nsm.iter_mut(),
        b_rlwe_dash_nsm.iter_mut(),
        gadget_a.iter()
    )
    .for_each(|(ai, bi, beta_i)| {
        // Sample a_i
        RandomUniformDist::random_fill(rng, &q, ai.as_mut());

        // a_i * s
        scratch_space.as_mut().copy_from_slice(ai.as_ref());
        ntt_op.forward(scratch_space.as_mut());
        mod_op.elwise_mul_mut(scratch_space.as_mut(), s_eval.as_ref());
        ntt_op.backward(scratch_space.as_mut());

        // b_i = e_i + a_i * s
        RandomGaussianDist::random_fill(rng, &q, bi.as_mut());
        mod_op.elwise_add_mut(bi.as_mut(), scratch_space.as_ref());

        // a_i + \beta_i * m
        mod_op.elwise_scalar_mul(scratch_space.as_mut(), m.as_ref(), beta_i);
        mod_op.elwise_add_mut(ai.as_mut(), scratch_space.as_ref());
    });

    // RLWE(m)
    let mut a_rlwe_dash_m = {
        // polynomials of part A of RLWE'(m) are sampled from seed
        let mut a = Mmut::zeros(d_b, ring_size);
        a.iter_rows_mut()
            .for_each(|ai| RandomUniformDist::random_fill(p_rng, &q, ai.as_mut()));
        a
    };

    izip!(
        a_rlwe_dash_m.iter_rows_mut(),
        b_rlwe_dash_m.iter_mut(),
        gadget_b.iter()
    )
    .for_each(|(ai, bi, beta_i)| {
        // ai * s
        ntt_op.forward(ai.as_mut());
        mod_op.elwise_mul_mut(ai.as_mut(), s_eval.as_ref());
        ntt_op.backward(ai.as_mut());

        // beta_i * m
        mod_op.elwise_scalar_mul(scratch_space.as_mut(), m.as_ref(), beta_i);

        // Sample e_i
        RandomGaussianDist::random_fill(rng, &q, bi.as_mut());
        // e_i + beta_i * m + ai*s
        mod_op.elwise_add_mut(bi.as_mut(), scratch_space.as_ref());
        mod_op.elwise_add_mut(bi.as_mut(), ai.as_ref());
    });
}

pub(crate) fn public_key_encrypt_rgsw<
    Mmut: MatrixMut + MatrixEntity,
    M: Matrix<MatElement = Mmut::MatElement>,
    R: RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + RandomUniformDist<[u8], Parameters = u8>
        + RandomUniformDist<usize, Parameters = usize>,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
>(
    out_rgsw: &mut Mmut,
    m: &[M::MatElement],
    public_key: &M,
    gadget_a: &[Mmut::MatElement],
    gadget_b: &[Mmut::MatElement],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    rng: &mut R,
) where
    <Mmut as Matrix>::R: RowMut + RowEntity + TryConvertFrom<[i32], Parameters = Mmut::MatElement>,
    Mmut::MatElement: Copy,
{
    let ring_size = public_key.dimension().1;
    let d_a = gadget_a.len();
    let d_b = gadget_b.len();
    assert!(public_key.dimension().0 == 2);
    assert!(out_rgsw.dimension() == (d_a * 2 + d_b * 2, ring_size));

    let mut pk_eval = Mmut::zeros(2, ring_size);
    izip!(pk_eval.iter_rows_mut(), public_key.iter_rows()).for_each(|(to_i, from_i)| {
        to_i.as_mut().copy_from_slice(from_i.as_ref());
        ntt_op.forward(to_i.as_mut());
    });
    let p0 = pk_eval.get_row_slice(0);
    let p1 = pk_eval.get_row_slice(1);

    let q = mod_op.modulus();

    // RGSW(m) = RLWE'(-sm), RLWE(m)
    let (rlwe_dash_nsm, rlwe_dash_m) = out_rgsw.split_at_row_mut(d_a * 2);

    // RLWE(-sm)
    let (rlwe_dash_nsm_parta, rlwe_dash_nsm_partb) = rlwe_dash_nsm.split_at_mut(d_a);
    izip!(
        rlwe_dash_nsm_parta.iter_mut(),
        rlwe_dash_nsm_partb.iter_mut(),
        gadget_a.iter()
    )
    .for_each(|(ai, bi, beta_i)| {
        // sample ephemeral secret u_i
        let mut u = vec![0i32; ring_size];
        fill_random_ternary_secret_with_hamming_weight(u.as_mut(), ring_size >> 1, rng);
        let mut u_eval = Mmut::R::try_convert_from(u.as_ref(), &q);
        ntt_op.forward(u_eval.as_mut());

        let mut u_eval_copy = Mmut::R::zeros(ring_size);
        u_eval_copy.as_mut().copy_from_slice(u_eval.as_ref());

        // p0 * u
        mod_op.elwise_mul_mut(u_eval.as_mut(), p0.as_ref());
        // p1 * u
        mod_op.elwise_mul_mut(u_eval_copy.as_mut(), p1.as_ref());
        ntt_op.backward(u_eval.as_mut());
        ntt_op.backward(u_eval_copy.as_mut());

        // sample error
        RandomGaussianDist::random_fill(rng, &q, ai.as_mut());
        RandomGaussianDist::random_fill(rng, &q, bi.as_mut());

        // a = p0*u+e0
        mod_op.elwise_add_mut(ai.as_mut(), u_eval.as_ref());
        // b = p1*u+e1
        mod_op.elwise_add_mut(bi.as_mut(), u_eval_copy.as_ref());

        // a = p0*u + e0 + \beta*m
        // use u_eval as scratch
        mod_op.elwise_scalar_mul(u_eval.as_mut(), m.as_ref(), beta_i);
        mod_op.elwise_add_mut(ai.as_mut(), u_eval.as_ref());
    });

    // RLWE(m)
    let (rlwe_dash_m_parta, rlwe_dash_m_partb) = rlwe_dash_m.split_at_mut(d_b);
    izip!(
        rlwe_dash_m_parta.iter_mut(),
        rlwe_dash_m_partb.iter_mut(),
        gadget_b.iter()
    )
    .for_each(|(ai, bi, beta_i)| {
        // sample ephemeral secret u_i
        let mut u = vec![0i32; ring_size];
        fill_random_ternary_secret_with_hamming_weight(u.as_mut(), ring_size >> 1, rng);
        let mut u_eval = Mmut::R::try_convert_from(u.as_ref(), &q);
        ntt_op.forward(u_eval.as_mut());

        let mut u_eval_copy = Mmut::R::zeros(ring_size);
        u_eval_copy.as_mut().copy_from_slice(u_eval.as_ref());

        // p0 * u
        mod_op.elwise_mul_mut(u_eval.as_mut(), p0.as_ref());
        // p1 * u
        mod_op.elwise_mul_mut(u_eval_copy.as_mut(), p1.as_ref());
        ntt_op.backward(u_eval.as_mut());
        ntt_op.backward(u_eval_copy.as_mut());

        // sample error
        RandomGaussianDist::random_fill(rng, &q, ai.as_mut());
        RandomGaussianDist::random_fill(rng, &q, bi.as_mut());

        // a = p0*u+e0
        mod_op.elwise_add_mut(ai.as_mut(), u_eval.as_ref());
        // b = p1*u+e1
        mod_op.elwise_add_mut(bi.as_mut(), u_eval_copy.as_ref());

        // b = p1*u + e0 + \beta*m
        // use u_eval as scratch
        mod_op.elwise_scalar_mul(u_eval.as_mut(), m.as_ref(), beta_i);
        mod_op.elwise_add_mut(bi.as_mut(), u_eval.as_ref());
    });
}

/// Generates RLWE Key switching key to key switch ciphertext RLWE_{from_s}(m)
/// to RLWE_{to_s}(m).
///
/// Key switching equals
///     \sum decompose(c_1)_i * RLWE_{to_s}(\beta^i -from_s)
/// Hence, key switchin key equals RLWE'(-from_s) = RLWE(-from_s), RLWE(beta^1
/// -from_s), ..., RLWE(beta^{d-1} -from_s).
///
/// - ksk_out: Output Key switching key. Key switching key stores only part B
///   polynomials of ksk RLWE ciphertexts (i.e. RLWE'_B(-from_s)) in coefficient
///   domain
/// - neg_from_s: Negative of secret polynomial to key switch from
/// - to_s: secret polynomial to key switch to.
pub(crate) fn rlwe_ksk_gen<
    Mmut: MatrixMut + MatrixEntity,
    ModOp: ArithmeticOps<Element = Mmut::MatElement> + VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
    R: RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
    PR: RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
>(
    ksk_out: &mut Mmut,
    neg_from_s: Mmut::R,
    mut to_s: Mmut::R,
    gadget_vector: &[Mmut::MatElement],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    p_rng: &mut PR,
    rng: &mut R,
) where
    <Mmut as Matrix>::R: RowMut,
{
    let ring_size = neg_from_s.as_ref().len();
    let d = gadget_vector.len();
    assert!(ksk_out.dimension() == (d, ring_size));

    let q = ArithmeticOps::modulus(mod_op);

    ntt_op.forward(to_s.as_mut());

    // RLWE'_{to_s}(-from_s)
    let mut part_a = {
        let mut a = Mmut::zeros(d, ring_size);
        a.iter_rows_mut()
            .for_each(|ai| RandomUniformDist::random_fill(p_rng, &q, ai.as_mut()));
        a
    };
    izip!(
        part_a.iter_rows_mut(),
        ksk_out.iter_rows_mut(),
        gadget_vector.iter(),
    )
    .for_each(|(ai, bi, beta_i)| {
        // si * ai
        ntt_op.forward(ai.as_mut());
        mod_op.elwise_mul_mut(ai.as_mut(), to_s.as_ref());
        ntt_op.backward(ai.as_mut());

        // ei + to_s*ai
        RandomGaussianDist::random_fill(rng, &q, bi.as_mut());
        mod_op.elwise_add_mut(bi.as_mut(), ai.as_ref());

        // beta_i * -from_s
        // use ai as scratch space
        mod_op.elwise_scalar_mul(ai.as_mut(), neg_from_s.as_ref(), beta_i);

        // bi = ei + to_s*ai + beta_i*-from_s
        mod_op.elwise_add_mut(bi.as_mut(), ai.as_ref());
    });
}

pub(crate) fn galois_key_gen<
    Mmut: MatrixMut + MatrixEntity,
    ModOp: ArithmeticOps<Element = Mmut::MatElement> + VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
    S,
    R: RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
    PR: RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
>(
    ksk_out: &mut Mmut,
    s: &[S],
    auto_k: isize,
    gadget_vector: &[Mmut::MatElement],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    p_rng: &mut PR,
    rng: &mut R,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut::R: TryConvertFrom<[S], Parameters = Mmut::MatElement> + RowEntity,
    Mmut::MatElement: Copy + Sub<Output = Mmut::MatElement>,
{
    let ring_size = s.len();
    let (auto_map_index, auto_map_sign) = generate_auto_map(ring_size, auto_k);

    let q = ArithmeticOps::modulus(mod_op);

    // s(X) -> -s(X^k)
    let s = Mmut::R::try_convert_from(s, &q);
    let mut neg_s_auto = Mmut::R::zeros(s.as_ref().len());
    izip!(s.as_ref(), auto_map_index.iter(), auto_map_sign.iter()).for_each(
        |(el, to_index, sign)| {
            // if sign is +ve (true), then negate because we need -s(X) (i.e. do the
            // opposite than the usual case)
            if *sign {
                neg_s_auto.as_mut()[*to_index] = q - *el;
            } else {
                neg_s_auto.as_mut()[*to_index] = *el;
            }
        },
    );

    // Ksk from -s(X^k) to s(X)
    rlwe_ksk_gen(
        ksk_out,
        neg_s_auto,
        s,
        gadget_vector,
        mod_op,
        ntt_op,
        p_rng,
        rng,
    );
}

/// Encrypt polynomial m(X) as RLWE ciphertext.
///
/// - rlwe_out: returned RLWE ciphertext RLWE(m) in coefficient domain. RLWE
///   ciphertext is a matirx with first row consiting polynomial `a` and the
///   second rows consting polynomial `b`
pub(crate) fn secret_key_encrypt_rlwe<
    Ro: Row + RowMut + RowEntity,
    ModOp: VectorOps<Element = Ro::Element>,
    NttOp: Ntt<Element = Ro::Element>,
    S,
    R: RandomGaussianDist<[Ro::Element], Parameters = Ro::Element>,
    PR: RandomUniformDist<[Ro::Element], Parameters = Ro::Element>,
>(
    m: &[Ro::Element],
    b_rlwe_out: &mut Ro,
    s: &[S],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    p_rng: &mut PR,
    rng: &mut R,
) where
    Ro: TryConvertFrom<[S], Parameters = Ro::Element> + Debug,
{
    let ring_size = s.len();
    assert!(m.as_ref().len() == ring_size);
    assert!(b_rlwe_out.as_ref().len() == ring_size);

    let q = mod_op.modulus();

    // sample a
    let mut a = {
        let mut a = Ro::zeros(ring_size);
        RandomUniformDist::random_fill(p_rng, &q, a.as_mut());
        a
    };

    // s * a
    let mut sa = Ro::try_convert_from(s, &q);
    ntt_op.forward(sa.as_mut());
    ntt_op.forward(a.as_mut());
    mod_op.elwise_mul_mut(sa.as_mut(), a.as_ref());
    ntt_op.backward(sa.as_mut());

    // sample e
    RandomGaussianDist::random_fill(rng, &q, b_rlwe_out.as_mut());
    mod_op.elwise_add_mut(b_rlwe_out.as_mut(), m.as_ref());
    mod_op.elwise_add_mut(b_rlwe_out.as_mut(), sa.as_ref());
}

pub(crate) fn public_key_encrypt_rlwe<
    M: Matrix,
    Mmut: MatrixMut<MatElement = M::MatElement>,
    ModOp: VectorOps<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement>,
    S,
    R: RandomGaussianDist<[M::MatElement], Parameters = M::MatElement>
        + RandomUniformDist<[M::MatElement], Parameters = M::MatElement>
        + RandomUniformDist<[u8], Parameters = u8>
        + RandomUniformDist<usize, Parameters = usize>,
>(
    rlwe_out: &mut Mmut,
    pk: &M,
    m: &[M::MatElement],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    rng: &mut R,
) where
    <Mmut as Matrix>::R: RowMut + TryConvertFrom<[S], Parameters = M::MatElement> + RowEntity,
    M::MatElement: Copy,
    S: Zero + Signed + Copy,
{
    let ring_size = m.len();
    assert!(rlwe_out.dimension() == (2, ring_size));

    let q = mod_op.modulus();

    let mut u = vec![S::zero(); ring_size];
    fill_random_ternary_secret_with_hamming_weight(u.as_mut(), ring_size >> 1, rng);
    let mut u = Mmut::R::try_convert_from(&u, &q);
    ntt_op.forward(u.as_mut());

    let mut ua = Mmut::R::zeros(ring_size);
    ua.as_mut().copy_from_slice(pk.get_row_slice(0));
    let mut ub = Mmut::R::zeros(ring_size);
    ub.as_mut().copy_from_slice(pk.get_row_slice(1));

    // a*u
    ntt_op.forward(ua.as_mut());
    mod_op.elwise_mul_mut(ua.as_mut(), u.as_ref());
    ntt_op.backward(ua.as_mut());

    // b*u
    ntt_op.forward(ub.as_mut());
    mod_op.elwise_mul_mut(ub.as_mut(), u.as_ref());
    ntt_op.backward(ub.as_mut());

    // sample error
    rlwe_out.iter_rows_mut().for_each(|ri| {
        RandomGaussianDist::random_fill(rng, &q, ri.as_mut());
    });

    // a*u + e0
    mod_op.elwise_add_mut(rlwe_out.get_row_mut(0), ua.as_ref());
    // b*u + e1
    mod_op.elwise_add_mut(rlwe_out.get_row_mut(1), ub.as_ref());

    // b*u + e1 + m
    mod_op.elwise_add_mut(rlwe_out.get_row_mut(1), m);
}

/// Generates RLWE public key
pub(crate) fn gen_rlwe_public_key<
    Ro: RowMut + RowEntity,
    S,
    ModOp: VectorOps<Element = Ro::Element>,
    NttOp: Ntt<Element = Ro::Element>,
    PRng: RandomUniformDist<[Ro::Element], Parameters = Ro::Element>,
    Rng: RandomGaussianDist<[Ro::Element], Parameters = Ro::Element>,
>(
    part_b_out: &mut Ro,
    s: &[S],
    ntt_op: &NttOp,
    mod_op: &ModOp,
    p_rng: &mut PRng,
    rng: &mut Rng,
) where
    Ro: TryConvertFrom<[S], Parameters = Ro::Element>,
{
    let ring_size = s.len();
    assert!(part_b_out.as_ref().len() == ring_size);

    let q = mod_op.modulus();

    // sample a
    let mut a = {
        let mut tmp = Ro::zeros(ring_size);
        RandomUniformDist::random_fill(p_rng, &q, tmp.as_mut());
        tmp
    };
    ntt_op.forward(a.as_mut());

    // s*a
    let mut sa = Ro::try_convert_from(s, &q);
    ntt_op.forward(sa.as_mut());
    mod_op.elwise_mul_mut(sa.as_mut(), a.as_ref());
    ntt_op.backward(sa.as_mut());

    // s*a + e
    RandomGaussianDist::random_fill(rng, &q, part_b_out.as_mut());
    mod_op.elwise_add_mut(part_b_out.as_mut(), sa.as_ref());
}

/// Decrypts degree 1 RLWE ciphertext RLWE(m) and returns m
///
/// - rlwe_ct: input degree 1 ciphertext RLWE(m).
pub(crate) fn decrypt_rlwe<
    R: RowMut,
    M: Matrix<MatElement = R::Element>,
    ModOp: VectorOps<Element = R::Element>,
    NttOp: Ntt<Element = R::Element>,
    S,
>(
    rlwe_ct: &M,
    s: &[S],
    m_out: &mut R,
    ntt_op: &NttOp,
    mod_op: &ModOp,
) where
    R: TryConvertFrom<[S], Parameters = R::Element>,
    R::Element: Copy,
{
    let ring_size = s.len();
    assert!(rlwe_ct.dimension() == (2, ring_size));
    assert!(m_out.as_ref().len() == ring_size);

    // transform a to evluation form
    m_out.as_mut().copy_from_slice(rlwe_ct.get_row_slice(0));
    ntt_op.forward(m_out.as_mut());

    // -s*a
    let mut s = R::try_convert_from(&s, &mod_op.modulus());
    ntt_op.forward(s.as_mut());
    mod_op.elwise_mul_mut(m_out.as_mut(), s.as_ref());
    mod_op.elwise_neg_mut(m_out.as_mut());
    ntt_op.backward(m_out.as_mut());

    // m+e = b - s*a
    mod_op.elwise_add_mut(m_out.as_mut(), rlwe_ct.get_row_slice(1));
}

// Measures noise in degree 1 RLWE ciphertext against encoded ideal message
// encoded_m
pub(crate) fn measure_noise<
    Mmut: MatrixMut + Matrix,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
    S,
>(
    rlwe_ct: &Mmut,
    encoded_m_ideal: &Mmut::R,
    ntt_op: &NttOp,
    mod_op: &ModOp,
    s: &[S],
) -> f64
where
    <Mmut as Matrix>::R: RowMut,
    Mmut::R: RowEntity + TryConvertFrom<[S], Parameters = Mmut::MatElement>,
    Mmut::MatElement: PrimInt + ToPrimitive + Debug,
{
    let ring_size = s.len();
    assert!(rlwe_ct.dimension() == (2, ring_size));
    assert!(encoded_m_ideal.as_ref().len() == ring_size);

    // -(s * a)
    let q = VectorOps::modulus(mod_op);
    let mut s = Mmut::R::try_convert_from(s, &q);
    ntt_op.forward(s.as_mut());
    let mut a = Mmut::R::zeros(ring_size);
    a.as_mut().copy_from_slice(rlwe_ct.get_row_slice(0));
    ntt_op.forward(a.as_mut());
    mod_op.elwise_mul_mut(s.as_mut(), a.as_ref());
    mod_op.elwise_neg_mut(s.as_mut());
    ntt_op.backward(s.as_mut());

    // m+e = b - s*a
    let mut m_plus_e = s;
    mod_op.elwise_add_mut(m_plus_e.as_mut(), rlwe_ct.get_row_slice(1));

    // difference
    mod_op.elwise_sub_mut(m_plus_e.as_mut(), encoded_m_ideal.as_ref());

    let mut max_diff_bits = f64::MIN;
    m_plus_e.as_ref().iter().for_each(|v| {
        let mut v = *v;
        if v >= (q >> 1) {
            // v is -ve
            v = q - v;
        }

        let bits = (v.to_f64().unwrap()).log2();

        if max_diff_bits < bits {
            max_diff_bits = bits;
        }
    });

    return max_diff_bits;
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{marker::PhantomData, ops::Mul, vec};

    use itertools::{izip, Itertools};
    use rand::{thread_rng, Rng};

    use crate::{
        backend::{ModInit, ModularOpsU64, VectorOps},
        decomposer::{Decomposer, DefaultDecomposer, RlweDecomposer},
        ntt::{self, Ntt, NttBackendU64, NttInit},
        random::{DefaultSecureRng, NewWithSeed, RandomUniformDist},
        rgsw::{
            gen_rlwe_public_key, measure_noise, public_key_encrypt_rgsw, AutoKeyEvaluationDomain,
            RgswCiphertext, RgswCiphertextEvaluationDomain, RlweCiphertext, RlwePublicKey,
            SeededAutoKey, SeededRgswCiphertext, SeededRlweCiphertext, SeededRlwePublicKey,
        },
        utils::{generate_prime, negacyclic_mul, Stats, TryConvertFrom},
        Matrix, Secret,
    };

    use super::{
        decrypt_rlwe, galois_auto, galois_key_gen, generate_auto_map, public_key_encrypt_rlwe,
        rgsw_by_rgsw_inplace, rlwe_by_rgsw, secret_key_encrypt_rgsw, secret_key_encrypt_rlwe,
        RlweSecret,
    };

    pub(crate) fn _sk_encrypt_rlwe(
        m: &[u64],
        s: &[i32],
        ntt_op: &NttBackendU64,
        mod_op: &ModularOpsU64,
    ) -> RlweCiphertext<Vec<Vec<u64>>, DefaultSecureRng> {
        let ring_size = m.len();
        let q = mod_op.modulus();
        assert!(s.len() == ring_size);

        let mut rng = DefaultSecureRng::new();
        let mut rlwe_seed = [0u8; 32];
        rng.fill_bytes(&mut rlwe_seed);
        let mut seeded_rlwe_ct =
            SeededRlweCiphertext::<_, [u8; 32]>::empty(ring_size as usize, rlwe_seed, q);
        let mut p_rng = DefaultSecureRng::new_seeded(rlwe_seed);
        secret_key_encrypt_rlwe(
            &m,
            &mut seeded_rlwe_ct.data,
            s,
            mod_op,
            ntt_op,
            &mut p_rng,
            &mut rng,
        );

        RlweCiphertext::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_rlwe_ct)
    }

    // Encrypt m as RGSW ciphertext RGSW(m) using supplied public key
    pub(crate) fn _pk_encrypt_rgsw(
        m: &[u64],
        public_key: &RlwePublicKey<Vec<Vec<u64>>, DefaultSecureRng>,
        decomposer: &(DefaultDecomposer<u64>, DefaultDecomposer<u64>),
        mod_op: &ModularOpsU64,
        ntt_op: &NttBackendU64,
    ) -> RgswCiphertext<Vec<Vec<u64>>> {
        let (_, ring_size) = Matrix::dimension(&public_key.data);
        let gadget_vector_a = decomposer.a().gadget_vector();
        let gadget_vector_b = decomposer.b().gadget_vector();

        let mut rng = DefaultSecureRng::new();

        assert!(m.len() == ring_size);

        // public key encrypt RGSW(m1)
        let mut rgsw_ct = RgswCiphertext::empty(ring_size, decomposer, mod_op.modulus());
        public_key_encrypt_rgsw(
            &mut rgsw_ct.data,
            m,
            &public_key.data,
            &gadget_vector_a,
            &gadget_vector_b,
            mod_op,
            ntt_op,
            &mut rng,
        );

        rgsw_ct
    }

    /// Encrypts m as RGSW ciphertext RGSW(m) using supplied secret key. Returns
    /// unseeded RGSW ciphertext in coefficient domain
    pub(crate) fn _sk_encrypt_rgsw(
        m: &[u64],
        s: &[i32],
        decomposer: &(DefaultDecomposer<u64>, DefaultDecomposer<u64>),
        mod_op: &ModularOpsU64,
        ntt_op: &NttBackendU64,
    ) -> SeededRgswCiphertext<Vec<Vec<u64>>, [u8; 32]> {
        let ring_size = s.len();
        assert!(m.len() == s.len());

        let q = mod_op.modulus();

        let gadget_vector_a = decomposer.a().gadget_vector();
        let gadget_vector_b = decomposer.b().gadget_vector();

        let mut rng = DefaultSecureRng::new();
        let mut rgsw_seed = [0u8; 32];
        rng.fill_bytes(&mut rgsw_seed);
        let mut seeded_rgsw_ct = SeededRgswCiphertext::<Vec<Vec<u64>>, [u8; 32]>::empty(
            ring_size as usize,
            decomposer,
            rgsw_seed,
            q,
        );
        let mut p_rng = DefaultSecureRng::new_seeded(rgsw_seed);
        secret_key_encrypt_rgsw(
            &mut seeded_rgsw_ct.data,
            m,
            &gadget_vector_a,
            &gadget_vector_b,
            s,
            mod_op,
            ntt_op,
            &mut p_rng,
            &mut rng,
        );
        seeded_rgsw_ct
    }

    /// Prints noise in RGSW ciphertext RGSW(m).
    ///
    /// - rgsw_ct: RGSW ciphertext in coefficient domain
    pub(crate) fn _measure_noise_rgsw(
        rgsw_ct: &[Vec<u64>],
        m: &[u64],
        s: &[i32],
        decomposer: &(DefaultDecomposer<u64>, DefaultDecomposer<u64>),
        q: u64,
    ) {
        let gadget_vector_a = decomposer.a().gadget_vector();
        let gadget_vector_b = decomposer.b().gadget_vector();
        let d_a = gadget_vector_a.len();
        let d_b = gadget_vector_b.len();
        let ring_size = s.len();
        assert!(Matrix::dimension(&rgsw_ct) == (d_a * 2 + d_b * 2, ring_size));
        assert!(m.len() == ring_size);

        let mod_op = ModularOpsU64::new(q);
        let ntt_op = NttBackendU64::new(q, ring_size);

        let mul_mod = |a: &u64, b: &u64| ((*a as u128 * *b as u128) % q as u128) as u64;
        let s_poly = Vec::<u64>::try_convert_from(s, &q);
        let mut neg_s = s_poly.clone();
        mod_op.elwise_neg_mut(neg_s.as_mut());
        let neg_sm0m1 = negacyclic_mul(&neg_s, &m, mul_mod, q);

        // RLWE(\beta^j -s * m)
        for j in 0..d_a {
            let ideal_m = {
                // RLWE(\beta^j -s * m)
                let mut beta_neg_sm0m1 = vec![0u64; ring_size as usize];
                mod_op.elwise_scalar_mul(beta_neg_sm0m1.as_mut(), &neg_sm0m1, &gadget_vector_a[j]);
                beta_neg_sm0m1
            };

            let mut rlwe = vec![vec![0u64; ring_size as usize]; 2];
            rlwe[0].copy_from_slice(rgsw_ct.get_row_slice(j));
            rlwe[1].copy_from_slice(rgsw_ct.get_row_slice(d_a + j));
            let noise = measure_noise(&rlwe, &ideal_m, &ntt_op, &mod_op, s);

            println!(r"Noise RLWE(\beta^{j} -sm0m1): {noise}");
        }

        // RLWE(\beta^j  m)
        for j in 0..d_b {
            let ideal_m = {
                // RLWE(\beta^j  m)
                let mut beta_m0m1 = vec![0u64; ring_size as usize];
                mod_op.elwise_scalar_mul(beta_m0m1.as_mut(), &m, &gadget_vector_b[j]);
                beta_m0m1
            };

            let mut rlwe = vec![vec![0u64; ring_size as usize]; 2];
            rlwe[0].copy_from_slice(rgsw_ct.get_row_slice(d_a * 2 + j));
            rlwe[1].copy_from_slice(rgsw_ct.get_row_slice(d_a * 2 + d_b + j));
            let noise = measure_noise(&rlwe, &ideal_m, &ntt_op, &mod_op, s);

            println!(r"Noise RLWE(\beta^{j} m0m1): {noise}");
        }
    }

    #[test]
    fn rlwe_encrypt_decryption() {
        let logq = 50;
        let logp = 2;
        let ring_size = 1 << 4;
        let q = generate_prime(logq, ring_size, 1u64 << logq).unwrap();
        let p = 1u64 << logp;

        let mut rng = DefaultSecureRng::new();

        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        // sample m0
        let mut m0 = vec![0u64; ring_size as usize];
        RandomUniformDist::<[u64]>::random_fill(&mut rng, &(1u64 << logp), m0.as_mut_slice());

        let ntt_op = NttBackendU64::new(q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);

        // encrypt m0
        let encoded_m = m0
            .iter()
            .map(|v| (((*v as f64) * q as f64) / (p as f64)).round() as u64)
            .collect_vec();
        let rlwe_in_ct = _sk_encrypt_rlwe(&encoded_m, s.values(), &ntt_op, &mod_op);

        let mut encoded_m_back = vec![0u64; ring_size as usize];
        decrypt_rlwe(
            &rlwe_in_ct,
            s.values(),
            &mut encoded_m_back,
            &ntt_op,
            &mod_op,
        );
        let m_back = encoded_m_back
            .iter()
            .map(|v| (((*v as f64 * p as f64) / q as f64).round() as u64) % p)
            .collect_vec();
        assert_eq!(m0, m_back);

        // let noise = measure_noise(&rlwe_in_ct, &encoded_m, &ntt_op, &mod_op,
        // s.values()); println!("Noise: {noise}");
    }

    #[test]
    fn rlwe_by_rgsw_works() {
        let logq = 50;
        let logp = 2;
        let ring_size = 1 << 9;
        let q = generate_prime(logq, ring_size, 1u64 << logq).unwrap();
        let p: u64 = 1u64 << logp;

        let mut rng = DefaultSecureRng::new_seeded([0u8; 32]);

        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        let mut m0 = vec![0u64; ring_size as usize];
        RandomUniformDist::<[u64]>::random_fill(&mut rng, &(1u64 << logp), m0.as_mut_slice());
        let mut m1 = vec![0u64; ring_size as usize];
        m1[thread_rng().gen_range(0..ring_size) as usize] = 1;

        let ntt_op = NttBackendU64::new(q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);
        let d_rgsw = 10;
        let logb = 5;
        let decomposer = (
            DefaultDecomposer::new(q, logb, d_rgsw),
            DefaultDecomposer::new(q, logb, d_rgsw),
        );

        // create public key
        let mut pk_seed = [0u8; 32];
        rng.fill_bytes(&mut pk_seed);
        let mut pk_prng = DefaultSecureRng::new_seeded(pk_seed);
        let mut seeded_pk =
            SeededRlwePublicKey::<Vec<u64>, _>::empty(ring_size as usize, pk_seed, q);
        gen_rlwe_public_key(
            &mut seeded_pk.data,
            s.values(),
            &ntt_op,
            &mod_op,
            &mut pk_prng,
            &mut rng,
        );
        let pk = RlwePublicKey::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_pk);

        // Encrypt m1 as RGSW(m1)
        let rgsw_ct = {
            //TODO(Jay): Figure out better way to test secret key and public key variant of
            // RGSW ciphertext encryption within the same test

            if true {
                // Encryption m1 as RGSW(m1) using secret key
                let seeded_rgsw_ct =
                    _sk_encrypt_rgsw(&m1, s.values(), &decomposer, &mod_op, &ntt_op);
                RgswCiphertextEvaluationDomain::<Vec<Vec<u64>>, DefaultSecureRng, NttBackendU64>::from(&seeded_rgsw_ct)
            } else {
                // Encrypt m1 as RGSW(m1) using public key
                let rgsw_ct = _pk_encrypt_rgsw(&m1, &pk, &decomposer, &mod_op, &ntt_op);
                RgswCiphertextEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(&rgsw_ct)
            }
        };

        // Encrypt m0 as RLWE(m0)
        let mut rlwe_in_ct = {
            let encoded_m = m0
                .iter()
                .map(|v| (((*v as f64) * q as f64) / (p as f64)).round() as u64)
                .collect_vec();

            _sk_encrypt_rlwe(&encoded_m, s.values(), &ntt_op, &mod_op)
        };

        // RLWE(m0m1) = RLWE(m0) x RGSW(m1)
        let mut scratch_space = vec![
            vec![0u64; ring_size as usize];
            std::cmp::max(
                decomposer.a().decomposition_count(),
                decomposer.b().decomposition_count()
            ) + 2
        ];
        rlwe_by_rgsw(
            &mut rlwe_in_ct,
            &rgsw_ct.data,
            &mut scratch_space,
            &decomposer,
            &ntt_op,
            &mod_op,
        );

        // Decrypt RLWE(m0m1)
        let mut encoded_m0m1_back = vec![0u64; ring_size as usize];
        decrypt_rlwe(
            &rlwe_in_ct,
            s.values(),
            &mut encoded_m0m1_back,
            &ntt_op,
            &mod_op,
        );
        let m0m1_back = encoded_m0m1_back
            .iter()
            .map(|v| (((*v as f64 * p as f64) / (q as f64)).round() as u64) % p)
            .collect_vec();

        let mul_mod = |v0: &u64, v1: &u64| (v0 * v1) % p;
        let m0m1 = negacyclic_mul(&m0, &m1, mul_mod, p);

        // {
        //     // measure noise
        //     let encoded_m_ideal = m0m1
        //         .iter()
        //         .map(|v| (((*v as f64) * q as f64) / (p as f64)).round() as u64)
        //         .collect_vec();

        //     let noise = measure_noise(&rlwe_in_ct, &encoded_m_ideal, &ntt_op,
        // &mod_op, s.values());     println!("Noise RLWE(m0m1)(=
        // RLWE(m0)xRGSW(m1)) : {noise}"); }

        assert!(
            m0m1 == m0m1_back,
            "Expected {:?} \n Got {:?}",
            m0m1,
            m0m1_back
        );
    }

    #[test]
    fn galois_auto_works() {
        let logq = 50;
        let ring_size = 1 << 8;
        let q = generate_prime(logq, 2 * ring_size, 1u64 << logq).unwrap();
        let logp = 3;
        let p = 1u64 << logp;
        let d_rgsw = 10;
        let logb = 5;

        let mut rng = DefaultSecureRng::new();
        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        let mut m = vec![0u64; ring_size as usize];
        RandomUniformDist::random_fill(&mut rng, &p, m.as_mut_slice());
        let encoded_m = m
            .iter()
            .map(|v| (((*v as f64 * q as f64) / (p as f64)).round() as u64))
            .collect_vec();

        let ntt_op = NttBackendU64::new(q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);

        // RLWE_{s}(m)
        let mut seed_rlwe = [0u8; 32];
        rng.fill_bytes(&mut seed_rlwe);
        let mut seeded_rlwe_m = SeededRlweCiphertext::empty(ring_size as usize, seed_rlwe, q);
        let mut p_rng = DefaultSecureRng::new_seeded(seed_rlwe);
        secret_key_encrypt_rlwe(
            &encoded_m,
            &mut seeded_rlwe_m.data,
            s.values(),
            &mod_op,
            &ntt_op,
            &mut p_rng,
            &mut rng,
        );
        let mut rlwe_m = RlweCiphertext::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_rlwe_m);

        let auto_k = -5;

        // Generate galois key to key switch from s^k to s
        let decomposer = DefaultDecomposer::new(q, logb, d_rgsw);
        let mut seed_auto = [0u8; 32];
        rng.fill_bytes(&mut seed_auto);
        let mut seeded_auto_key =
            SeededAutoKey::empty(ring_size as usize, &decomposer, seed_auto, q);
        let mut p_rng = DefaultSecureRng::new_seeded(seed_auto);
        let gadget_vector = decomposer.gadget_vector();
        galois_key_gen(
            &mut seeded_auto_key.data,
            s.values(),
            auto_k,
            &gadget_vector,
            &mod_op,
            &ntt_op,
            &mut p_rng,
            &mut rng,
        );
        let auto_key =
            AutoKeyEvaluationDomain::<Vec<Vec<u64>>, DefaultSecureRng, NttBackendU64>::from(
                &seeded_auto_key,
            );

        // Send RLWE_{s}(m) -> RLWE_{s}(m^k)
        let mut scratch_space = vec![vec![0u64; ring_size as usize]; d_rgsw + 2];
        let (auto_map_index, auto_map_sign) = generate_auto_map(ring_size as usize, auto_k);
        galois_auto(
            &mut rlwe_m,
            &auto_key.data,
            &mut scratch_space,
            &auto_map_index,
            &auto_map_sign,
            &mod_op,
            &ntt_op,
            &decomposer,
        );

        let rlwe_m_k = rlwe_m;

        // Decrypt RLWE_{s}(m^k) and check
        let mut encoded_m_k_back = vec![0u64; ring_size as usize];
        decrypt_rlwe(
            &rlwe_m_k,
            s.values(),
            &mut encoded_m_k_back,
            &ntt_op,
            &mod_op,
        );
        let m_k_back = encoded_m_k_back
            .iter()
            .map(|v| (((*v as f64 * p as f64) / q as f64).round() as u64) % p)
            .collect_vec();

        let mut m_k = vec![0u64; ring_size as usize];
        // Send \delta m -> \delta m^k
        izip!(m.iter(), auto_map_index.iter(), auto_map_sign.iter()).for_each(
            |(v, to_index, sign)| {
                if !*sign {
                    m_k[*to_index] = (p - *v) % p;
                } else {
                    m_k[*to_index] = *v;
                }
            },
        );

        {
            let encoded_m_k = m_k
                .iter()
                .map(|v| ((*v as f64 * q as f64) / p as f64).round() as u64)
                .collect_vec();

            let noise = measure_noise(&rlwe_m_k, &encoded_m_k, &ntt_op, &mod_op, s.values());
            println!("Ksk noise: {noise}");
        }

        assert_eq!(m_k_back, m_k);
    }

    #[test]
    fn sk_rgsw_by_rgsw() {
        let logq = 60;
        let logp = 2;
        let ring_size = 1 << 11;
        let q = generate_prime(logq, ring_size, 1u64 << logq).unwrap();
        let p = 1u64 << logp;
        let d_rgsw = 5;
        let logb = 12;

        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        let mut rng = DefaultSecureRng::new();
        let ntt_op = NttBackendU64::new(q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);
        let decomposer = (
            DefaultDecomposer::new(q, logb, d_rgsw),
            DefaultDecomposer::new(q, logb, d_rgsw),
        );

        let mul_mod = |a: &u64, b: &u64| ((*a as u128 * *b as u128) % q as u128) as u64;

        let mut carry_m = vec![0u64; ring_size as usize];
        carry_m[thread_rng().gen_range(0..ring_size) as usize] = 1;

        // RGSW(carry_m)
        let mut rgsw_carrym = {
            let seeded_rgsw = _sk_encrypt_rgsw(&carry_m, s.values(), &decomposer, &mod_op, &ntt_op);
            let mut rgsw_eval =
                RgswCiphertextEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(
                    &seeded_rgsw,
                );
            rgsw_eval
                .data
                .iter_mut()
                .for_each(|ri| ntt_op.backward(ri.as_mut()));
            rgsw_eval.data
        };

        let mut scratch_matrix = vec![
            vec![0u64; ring_size as usize];
            decomposer.a().decomposition_count() * 2
                + decomposer.b().decomposition_count() * 2
                + std::cmp::max(
                    decomposer.a().decomposition_count(),
                    decomposer.b().decomposition_count()
                )
        ];

        _measure_noise_rgsw(&rgsw_carrym, &carry_m, s.values(), &decomposer, q);

        for i in 0..2 {
            let mut m = vec![0u64; ring_size as usize];
            m[thread_rng().gen_range(0..ring_size) as usize] = if (i & 1) == 1 { q - 1 } else { 1 };
            let rgsw_m = RgswCiphertextEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(
                &_sk_encrypt_rgsw(&m, s.values(), &decomposer, &mod_op, &ntt_op),
            );
            rgsw_by_rgsw_inplace(
                &mut rgsw_carrym,
                &rgsw_m.data,
                &decomposer,
                &mut scratch_matrix,
                &ntt_op,
                &mod_op,
            );

            // measure noise
            carry_m = negacyclic_mul(&carry_m, &m, mul_mod, q);
            println!("########### Noise RGSW(carrym) in {i}^th loop ###########");
            _measure_noise_rgsw(&rgsw_carrym, &carry_m, s.values(), &decomposer, q);
        }
        {
            // RLWE(m) x RGSW(carry_m)
            let mut m = vec![0u64; ring_size as usize];
            RandomUniformDist::random_fill(&mut rng, &q, m.as_mut_slice());
            let mut rlwe_ct = _sk_encrypt_rlwe(&m, s.values(), &ntt_op, &mod_op);

            // send rgsw to evaluation domain
            rgsw_carrym
                .iter_mut()
                .for_each(|ri| ntt_op.forward(ri.as_mut_slice()));

            rlwe_by_rgsw(
                &mut rlwe_ct,
                &rgsw_carrym,
                &mut scratch_matrix,
                &decomposer,
                &ntt_op,
                &mod_op,
            );
            let m_expected = negacyclic_mul(&carry_m, &m, mul_mod, q);
            let noise = measure_noise(&rlwe_ct, &m_expected, &ntt_op, &mod_op, s.values());
            println!("RLWE(m) x RGSW(carry_m): {noise}");
        }
    }

    #[test]
    fn some_work() {
        let logq = 50;
        let ring_size = 1 << 10;
        let q = generate_prime(logq, ring_size, 1u64 << logq).unwrap();
        let d_rgsw = 10;
        let logb = 5;
        let decomposer = (
            DefaultDecomposer::new(q, logb, d_rgsw),
            DefaultDecomposer::new(q, logb, d_rgsw),
        );

        let ntt_op = NttBackendU64::new(q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);
        let mut rng = DefaultSecureRng::new_seeded([0u8; 32]);

        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        let mut check = Stats { samples: vec![] };

        for _ in 0..100 {
            let mut m0 = vec![0u64; ring_size as usize];
            m0[thread_rng().gen_range(0..ring_size) as usize] = 1;
            let mut m1 = vec![0u64; ring_size as usize];
            m1[thread_rng().gen_range(0..ring_size) as usize] = 1;

            let mut rgsw_ct0 = {
                let seeded_rgsw_ct =
                    _sk_encrypt_rgsw(&m0, s.values(), &decomposer, &mod_op, &ntt_op);
                RgswCiphertextEvaluationDomain::<Vec<Vec<u64>>, DefaultSecureRng, NttBackendU64>::from(&seeded_rgsw_ct)
            };
            let rgsw_ct1 = {
                let seeded_rgsw_ct =
                    _sk_encrypt_rgsw(&m1, s.values(), &decomposer, &mod_op, &ntt_op);
                RgswCiphertextEvaluationDomain::<Vec<Vec<u64>>, DefaultSecureRng, NttBackendU64>::from(&seeded_rgsw_ct)
            };

            // RGSW x RGSW
            // send RGSW(m0) to coefficient domain
            rgsw_ct0
                .data
                .iter_mut()
                .for_each(|r| ntt_op.backward(r.as_mut_slice()));
            let mut scratch_matrix = vec![
                vec![0u64; ring_size as usize];
                std::cmp::max(
                    decomposer.a().decomposition_count(),
                    decomposer.b().decomposition_count()
                ) + decomposer.a().decomposition_count() * 2
                    + decomposer.b().decomposition_count() * 2
            ];
            rgsw_by_rgsw_inplace(
                &mut rgsw_ct0.data,
                &rgsw_ct1.data,
                &decomposer,
                &mut scratch_matrix,
                &ntt_op,
                &mod_op,
            );
            let mut rgsw_m0m1 = rgsw_ct0;
            // Back to Evaluation for RLWExRGSW
            rgsw_m0m1
                .data
                .iter_mut()
                .for_each(|r| ntt_op.forward(r.as_mut_slice()));

            // Sample m2, encrypt it as RLWE(m2) and multiply RLWE(m2)xRGSW(m0m1)
            let mut m2 = vec![0u64; ring_size as usize];
            RandomUniformDist::random_fill(&mut rng, &q, m2.as_mut_slice());
            let mut rlwe_in_ct = { _sk_encrypt_rlwe(&m2, s.values(), &ntt_op, &mod_op) };
            let mut scratch_space = vec![
                vec![0u64; ring_size as usize];
                std::cmp::max(
                    decomposer.a().decomposition_count(),
                    decomposer.b().decomposition_count()
                ) + 2
            ];
            rlwe_by_rgsw(
                &mut rlwe_in_ct,
                &rgsw_m0m1.data,
                &mut scratch_space,
                &decomposer,
                &ntt_op,
                &mod_op,
            );

            // Decrypt RLWE(m0m1m2)
            let mut m0m1m2_back = vec![0u64; ring_size as usize];
            decrypt_rlwe(&rlwe_in_ct, s.values(), &mut m0m1m2_back, &ntt_op, &mod_op);

            // Calculate m0m1m2
            let mul_mod = |v0: &u64, v1: &u64| ((*v0 as u128 * *v1 as u128) % q as u128) as u64;
            let m0m1 = negacyclic_mul(&m0, &m1, mul_mod, q);
            let m0m1m2 = negacyclic_mul(&m2, &m0m1, mul_mod, q);

            // diff
            mod_op.elwise_sub_mut(m0m1m2_back.as_mut_slice(), m0m1m2.as_ref());

            check.add_more(&Vec::<i64>::try_convert_from(&m0m1m2_back, &q));
        }

        println!("Std: {}", check.std_dev().abs().log2());
    }
}
