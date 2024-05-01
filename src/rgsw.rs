use std::{
    clone,
    fmt::Debug,
    marker::PhantomData,
    ops::{Div, Neg, Sub},
};

use itertools::{izip, Itertools};
use num_traits::{PrimInt, ToPrimitive, Zero};

use crate::{
    backend::{ArithmeticOps, VectorOps},
    decomposer::{self, Decomposer},
    ntt::{self, Ntt, NttInit},
    random::{DefaultSecureRng, NewWithSeed, RandomGaussianDist, RandomUniformDist},
    utils::{fill_random_ternary_secret_with_hamming_weight, TryConvertFrom, WithLocal},
    Matrix, MatrixEntity, MatrixMut, Row, RowEntity, RowMut, Secret,
};

pub struct SeededRgswCiphertext<M, S>
where
    M: Matrix,
{
    data: M,
    seed: S,
    modulus: M::MatElement,
}

impl<M: Matrix + MatrixEntity, S> SeededRgswCiphertext<M, S> {
    fn from_raw(data: M, seed: S, modulus: M::MatElement) -> SeededRgswCiphertext<M, S> {
        assert!(data.dimension().0 % 3 == 0);

        SeededRgswCiphertext {
            data,
            seed,
            modulus,
        }
    }

    fn empty(
        ring_size: usize,
        d_rgsw: usize,
        seed: S,
        modulus: M::MatElement,
    ) -> SeededRgswCiphertext<M, S> {
        SeededRgswCiphertext {
            data: M::zeros(d_rgsw * 3, ring_size),
            seed,
            modulus: modulus,
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
    data: M,
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
        let d = value.data.dimension().0.div(3);

        let mut data = M::zeros(4 * d, value.data.dimension().1);

        // copy RLWE'(-sm)
        izip!(data.iter_rows_mut().take(2 * d), value.data.iter_rows()).for_each(
            |(to_ri, from_ri)| {
                to_ri.as_mut().copy_from_slice(from_ri.as_ref());
            },
        );

        // sample A polynomials of RLWE'(m) - RLWE'A(m)
        // TODO(Jay): Do we want to be generic over RandomGenerator used here? I think
        // not.
        let mut p_rng = R::new_with_seed(value.seed.clone());
        izip!(data.iter_rows_mut().skip(2 * d).take(d))
            .for_each(|ri| p_rng.random_fill(&value.modulus, ri.as_mut()));

        // RLWE'_B(m)
        izip!(
            data.iter_rows_mut().skip(3 * d),
            value.data.iter_rows().skip(2 * d)
        )
        .for_each(|(to_ri, from_ri)| {
            to_ri.as_mut().copy_from_slice(from_ri.as_ref());
        });
        // println!("RGSW key; coefficient {:?}", &data);

        // Send polynomials to evaluation domain
        let ring_size = data.dimension().1;
        let nttop = N::new(value.modulus, ring_size);
        data.iter_rows_mut()
            .for_each(|ri| nttop.forward(ri.as_mut()));
        // println!("RGSW key; {:?}", &data);

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
    data: R,
    seed: S,
    modulus: R::Element,
}

impl<R: RowEntity, S> SeededRlweCiphertext<R, S> {
    fn empty(ring_size: usize, seed: S, modulus: R::Element) -> Self {
        SeededRlweCiphertext {
            data: R::zeros(ring_size),
            seed,
            modulus,
        }
    }
}

pub struct RlweCiphertext<M, Rng> {
    data: M,
    is_trivial: bool,
    _phatom: PhantomData<Rng>,
}

impl<M, Rng> RlweCiphertext<M, Rng> {
    pub(crate) fn from_raw(data: M, is_trivial: bool) -> Self {
        RlweCiphertext {
            data,
            is_trivial,
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

#[derive(Clone)]
pub struct RlweSecret {
    values: Vec<i32>,
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
    R: RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + NewWithSeed,
>(
    ksk_out: &mut Mmut,
    mut neg_from_s: Mmut::R,
    mut to_s: Mmut::R,
    gadget_vector: &[Mmut::MatElement],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    seed: R::Seed,
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
        let mut p_rng = R::new_with_seed(seed);
        a.iter_rows_mut()
            .for_each(|ai| RandomUniformDist::random_fill(&mut p_rng, &q, ai.as_mut()));
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
    R: RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + NewWithSeed,
>(
    ksk_out: &mut Mmut,
    s: &[S],
    auto_k: isize,
    gadget_vector: &[Mmut::MatElement],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    seed: R::Seed,
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
        seed,
        rng,
    );
}

pub(crate) fn routine<M: Matrix + ?Sized, ModOp: VectorOps<Element = M::MatElement>>(
    write_to_row: &mut [M::MatElement],
    matrix_a: &[M::R],
    matrix_b: &[M::R],
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
pub(crate) fn decompose_r<M: Matrix + MatrixMut, D: Decomposer<Element = M::MatElement>>(
    r: &[M::MatElement],
    decomp_r: &mut [M::R],
    decomposer: &D,
) where
    <M as Matrix>::R: RowMut,
    M::MatElement: Copy,
{
    let ring_size = r.len();
    let d = decomposer.d();

    for ri in 0..ring_size {
        let el_decomposed = decomposer.decompose(&r[ri]);
        for j in 0..d {
            decomp_r[j].as_mut()[ri] = el_decomposed[j];
        }
    }
}

/// Sends RLWE_{s}(X) -> RLWE_{s}(X^k) where k is some galois element
pub(crate) fn galois_auto<
    MT: Matrix + IsTrivial + MatrixMut,
    Mmut: MatrixMut<MatElement = MT::MatElement>,
    ModOp: ArithmeticOps<Element = MT::MatElement> + VectorOps<Element = MT::MatElement>,
    NttOp: Ntt<Element = MT::MatElement>,
    D: Decomposer<Element = MT::MatElement>,
>(
    rlwe_in: &mut MT,
    ksk: &Mmut,
    scratch_matrix_dplus2_ring: &mut Mmut,
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
    let d = decomposer.d();

    let (scratch_matrix_d_ring, tmp_rlwe_out) = scratch_matrix_dplus2_ring.split_at_row_mut(d);

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
        routine::<Mmut, ModOp>(
            tmp_rlwe_out[0].as_mut(),
            scratch_matrix_d_ring,
            ksk_a,
            mod_op,
        );
        // send b(X^k) to evaluation domain
        ntt_op.forward(tmp_rlwe_out[1].as_mut());
        // b' = b(X^k)
        // b' += decomp<a(X^k)> * RLWE'_B(s(X^k))
        routine::<Mmut, ModOp>(
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
pub(crate) fn rlwe_by_rgsw<
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
) where
    Mmut::MatElement: Copy + Zero,
    <Mmut as Matrix>::R: RowMut,
    <MT as Matrix>::R: RowMut,
{
    let d_rgsw = decomposer.d();
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
        decompose_r::<Mmut, D>(rlwe_in.get_row_slice(0), scratch_matrix_d_ring, decomposer);
        scratch_matrix_d_ring
            .iter_mut()
            .for_each(|r| ntt_op.forward(r.as_mut()));
        // a_out += decomp<a_in> \cdot RLWE_A'(-sm)
        routine::<Mmut, ModOp>(
            scratch_rlwe_out[0].as_mut(),
            scratch_matrix_d_ring.as_ref(),
            &rlwe_dash_nsm[..d_rgsw],
            mod_op,
        );
        // b_out += decomp<a_in> \cdot RLWE_B'(-sm)
        routine::<Mmut, ModOp>(
            scratch_rlwe_out[1].as_mut(),
            scratch_matrix_d_ring.as_ref(),
            &rlwe_dash_nsm[d_rgsw..],
            mod_op,
        );
    }
    // decomp<b_in>
    decompose_r::<Mmut, D>(rlwe_in.get_row_slice(1), scratch_matrix_d_ring, decomposer);
    scratch_matrix_d_ring
        .iter_mut()
        .for_each(|r| ntt_op.forward(r.as_mut()));
    // a_out += decomp<b_in> \cdot RLWE_A'(m)
    routine::<Mmut, ModOp>(
        scratch_rlwe_out[0].as_mut(),
        scratch_matrix_d_ring.as_ref(),
        &rlwe_dash_m[..d_rgsw],
        mod_op,
    );
    // b_out += decomp<b_in> \cdot RLWE_B'(m)
    routine::<Mmut, ModOp>(
        scratch_rlwe_out[1].as_mut(),
        scratch_matrix_d_ring.as_ref(),
        &rlwe_dash_m[d_rgsw..],
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

/// Encrypts message m as a RGSW ciphertext.
///
/// - m_eval: is `m` is evaluation domain
/// - out_rgsw: RGSW(m) is stored as single matrix of dimension (d_rgsw * 3,
///   ring_size). The matrix has the following structure [RLWE'_A(-sm) ||
///   RLWE'_B(-sm) || RLWE'_B(m)]^T and RLWE'_A(m) is generated via seed
pub(crate) fn secret_key_encrypt_rgsw<
    Mmut: MatrixMut + MatrixEntity,
    S,
    R: RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + NewWithSeed,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
>(
    out_rgsw: &mut Mmut,
    m: &Mmut::R,
    gadget_vector: &[Mmut::MatElement],
    s: &[S],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    seed: R::Seed,
    rng: &mut R,
) where
    <Mmut as Matrix>::R: RowMut + RowEntity + TryConvertFrom<[S], Parameters = Mmut::MatElement>,
    Mmut::MatElement: Copy + Debug,
    Mmut: Debug,
{
    let d = gadget_vector.len();
    let q = mod_op.modulus();
    let ring_size = s.len();
    assert!(out_rgsw.dimension() == (d * 3, ring_size));
    assert!(m.as_ref().len() == ring_size);

    // RLWE(-sm), RLWE(m)
    let (rlwe_dash_nsm, b_rlwe_dash_m) = out_rgsw.split_at_row_mut(d * 2);
    dbg!(rlwe_dash_nsm.len(), b_rlwe_dash_m.len());
    let mut s_eval = Mmut::R::try_convert_from(s, &q);
    ntt_op.forward(s_eval.as_mut());

    let mut scratch_space = Mmut::R::zeros(ring_size);

    // RLWE'(-sm)
    let (a_rlwe_dash_nsm, b_rlwe_dash_nsm) = rlwe_dash_nsm.split_at_mut(d);
    izip!(
        a_rlwe_dash_nsm.iter_mut(),
        b_rlwe_dash_nsm.iter_mut(),
        gadget_vector.iter()
    )
    .for_each(|(ai, bi, beta_i)| {
        // Sample a_i and transform to evaluation domain
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
        let mut p_rng = R::new_with_seed(seed);
        let mut a = Mmut::zeros(d, ring_size);
        a.iter_rows_mut().for_each(|ai| {
            RandomUniformDist::random_fill(&mut p_rng, &q, ai.as_mut());
            ntt_op.forward(ai.as_mut());
        });
        a
    };

    // println!("RGSW sample RLWE'_A(m) {:?}", &a_rlwe_dash_m);

    izip!(
        a_rlwe_dash_m.iter_rows_mut(),
        b_rlwe_dash_m.iter_mut(),
        gadget_vector.iter()
    )
    .for_each(|(ai, bi, beta_i)| {
        dbg!(&beta_i);

        // ai * s
        // ntt_op.forward(ai.as_mut());
        mod_op.elwise_mul_mut(ai.as_mut(), s_eval.as_ref());

        // beta_i * m
        mod_op.elwise_scalar_mul(scratch_space.as_mut(), m.as_ref(), beta_i);

        // Sample e_i
        RandomGaussianDist::random_fill(rng, &q, bi.as_mut());
        // e_i + beta_i * m + ai*s
        mod_op.elwise_add_mut(bi.as_mut(), scratch_space.as_ref());
        mod_op.elwise_add_mut(bi.as_mut(), ai.as_ref());
    });

    out_rgsw
        .iter_rows_mut()
        .for_each(|ri| ntt_op.forward(ri.as_mut()));

    // println!("RGSW key ex {:?}", &out_rgsw);
    out_rgsw
        .iter_rows_mut()
        .for_each(|ri| ntt_op.backward(ri.as_mut()));
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
    R: RandomUniformDist<[Ro::Element], Parameters = Ro::Element>
        + RandomGaussianDist<[Ro::Element], Parameters = Ro::Element>
        + NewWithSeed,
>(
    m: &Ro,
    b_rlwe_out: &mut Ro,
    s: &[S],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    seed: R::Seed,
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
        let mut p_rng = R::new_with_seed(seed);
        RandomUniformDist::random_fill(&mut p_rng, &q, a.as_mut());
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
    Mmut: MatrixMut + Matrix + MatrixEntity,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
    S: Secret,
>(
    rlwe_ct: &Mmut,
    encoded_m_ideal: &Mmut,
    ntt_op: &NttOp,
    mod_op: &ModOp,
    s: &S,
) -> f64
where
    <Mmut as Matrix>::R: RowMut,
    Mmut: TryConvertFrom<[S::Element], Parameters = Mmut::MatElement>,
    Mmut::MatElement: PrimInt + ToPrimitive + Debug,
{
    let ring_size = s.values().len();
    assert!(rlwe_ct.dimension() == (2, ring_size));
    assert!(encoded_m_ideal.dimension() == (1, ring_size));

    // -(s * a)
    let q = VectorOps::modulus(mod_op);
    let mut s = Mmut::try_convert_from(s.values(), &q);
    ntt_op.forward(s.get_row_mut(0));
    let mut a = Mmut::zeros(1, ring_size);
    a.get_row_mut(0).copy_from_slice(rlwe_ct.get_row_slice(0));
    ntt_op.forward(a.get_row_mut(0));
    mod_op.elwise_mul_mut(s.get_row_mut(0), a.get_row_slice(0));
    mod_op.elwise_neg_mut(s.get_row_mut(0));
    ntt_op.backward(s.get_row_mut(0));

    // m+e = b - s*a
    let mut m_plus_e = s;
    mod_op.elwise_add_mut(m_plus_e.get_row_mut(0), rlwe_ct.get_row_slice(1));

    // difference
    mod_op.elwise_sub_mut(m_plus_e.get_row_mut(0), encoded_m_ideal.get_row_slice(0));

    let mut max_diff_bits = f64::MIN;
    m_plus_e.get_row_slice(0).iter().for_each(|v| {
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
mod tests {
    use std::vec;

    use itertools::{izip, Itertools};
    use rand::{thread_rng, Rng};

    use crate::{
        backend::{ModInit, ModularOpsU64},
        decomposer::{gadget_vector, DefaultDecomposer},
        ntt::{self, Ntt, NttBackendU64, NttInit},
        random::{DefaultSecureRng, RandomUniformDist},
        rgsw::{
            measure_noise, RgswCiphertextEvaluationDomain, RlweCiphertext, SeededRgswCiphertext,
            SeededRlweCiphertext,
        },
        utils::{generate_prime, negacyclic_mul},
        Matrix, Secret,
    };

    use super::{
        decrypt_rlwe, galois_auto, galois_key_gen, generate_auto_map, rlwe_by_rgsw,
        secret_key_encrypt_rgsw, secret_key_encrypt_rlwe, RlweSecret,
    };

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
        let mut rlwe_seed = [0u8; 32];
        rng.fill_bytes(&mut rlwe_seed);
        let mut seeded_rlwe_in_ct =
            SeededRlweCiphertext::<_, [u8; 32]>::empty(ring_size as usize, rlwe_seed, q);
        let encoded_m = m0
            .iter()
            .map(|v| (((*v as f64) * q as f64) / (p as f64)).round() as u64)
            .collect_vec();
        secret_key_encrypt_rlwe(
            &encoded_m,
            &mut seeded_rlwe_in_ct.data,
            s.values(),
            &mod_op,
            &ntt_op,
            seeded_rlwe_in_ct.seed,
            &mut rng,
        );
        let rlwe_in_ct =
            RlweCiphertext::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_rlwe_in_ct);

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
            .map(|v| ((*v as f64 * p as f64) / q as f64).round() as u64)
            .collect_vec();
        assert_eq!(m0, m_back);
    }

    #[test]
    fn rlwe_by_rgsw_works() {
        let logq = 50;
        let logp = 2;
        let ring_size = 1 << 4;
        let q = generate_prime(logq, ring_size, 1u64 << logq).unwrap();
        let p = 1u64 << logp;
        let d_rgsw = 10;
        let logb = 5;

        let mut rng = DefaultSecureRng::new_seeded([0u8; 32]);

        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        let mut m0 = vec![0u64; ring_size as usize];
        // RandomUniformDist::<[u64]>::random_fill(&mut rng, &(1u64 << logp),
        // m0.as_mut_slice());
        let mut m1 = vec![0u64; ring_size as usize];
        // m1[thread_rng().gen_range(0..ring_size) as usize] = 1;

        let ntt_op = NttBackendU64::new(q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);

        // Encrypt m1 as RGSW(m1)
        let mut rgsw_seed = [0u8; 32];
        rng.fill_bytes(&mut rgsw_seed);
        let mut seeded_rgsw_ct = SeededRgswCiphertext::<Vec<Vec<u64>>, [u8; 32]>::empty(
            ring_size as usize,
            d_rgsw,
            rgsw_seed,
            q,
        );
        let gadget_vector = gadget_vector(logq, logb, d_rgsw);
        secret_key_encrypt_rgsw(
            &mut seeded_rgsw_ct.data,
            &m1,
            &gadget_vector,
            s.values(),
            &mod_op,
            &ntt_op,
            seeded_rgsw_ct.seed,
            &mut rng,
        );
        let rgsw_ct = RgswCiphertextEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(
            &seeded_rgsw_ct,
        );
        // println!("RGSW(m1) seeded: {:?}", &seeded_rgsw_ct);
        // println!("RGSW(m1): {:?}", &rgsw_ct);

        // Encrypt m0 as RLWE(m0)
        let mut rlwe_seed = [0u8; 32];
        rng.fill_bytes(&mut rlwe_seed);
        let mut seeded_rlwe_in_ct =
            SeededRlweCiphertext::<_, [u8; 32]>::empty(ring_size as usize, rlwe_seed, q);
        let encoded_m = m0
            .iter()
            .map(|v| (((*v as f64) * q as f64) / (p as f64)).round() as u64)
            .collect_vec();
        secret_key_encrypt_rlwe(
            &encoded_m,
            &mut seeded_rlwe_in_ct.data,
            s.values(),
            &mod_op,
            &ntt_op,
            seeded_rlwe_in_ct.seed,
            &mut rng,
        );

        let mut rlwe_in_ct =
            RlweCiphertext::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_rlwe_in_ct);

        // RLWE(m0m1) = RLWE(m0) x RGSW(m1)
        let mut scratch_space = vec![vec![0u64; ring_size as usize]; d_rgsw + 2];
        let decomposer = DefaultDecomposer::new(q, logb, d_rgsw);
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
        assert!(
            m0m1 == m0m1_back,
            "Expected {:?} \n Got {:?}",
            m0m1,
            m0m1_back
        );
        // dbg!(&m0m1_back, m0m1, q);
    }

    // #[test]
    // fn galois_auto_works() {
    //     let logq = 50;
    //     let ring_size = 1 << 5;
    //     let q = generate_prime(logq, 2 * ring_size, 1u64 << logq).unwrap();
    //     let logp = 3;
    //     let p = 1u64 << logp;
    //     let d_rgsw = 10;
    //     let logb = 5;

    //     let mut rng = DefaultSecureRng::new();
    //     let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as
    // usize);

    //     let mut m = vec![0u64; ring_size as usize];
    //     RandomUniformDist::random_fill(&mut rng, &p, m.as_mut_slice());
    //     let encoded_m = m
    //         .iter()
    //         .map(|v| (((*v as f64 * q as f64) / (p as f64)).round() as u64))
    //         .collect_vec();

    //     let ntt_op = NttBackendU64::new(q, ring_size as usize);
    //     let mod_op = ModularOpsU64::new(q);

    //     // RLWE_{s}(m)
    //     let mut rlwe_m = vec![vec![0u64; ring_size as usize]; 2];
    //     secret_key_encrypt_rlwe(
    //         &vec![encoded_m.clone()],
    //         &mut rlwe_m,
    //         &s,
    //         &mod_op,
    //         &ntt_op,
    //         &mut rng,
    //     );

    //     let auto_k = -5;

    //     // Generate galois key to key switch from s^k to s
    //     let mut ksk_out = vec![vec![0u64; ring_size as usize]; d_rgsw * 2];
    //     let gadget_vector = gadget_vector(logq, logb, d_rgsw);
    //     galois_key_gen(
    //         &mut ksk_out,
    //         &s,
    //         auto_k,
    //         &gadget_vector,
    //         &mod_op,
    //         &ntt_op,
    //         &mut rng,
    //     );

    //     // Send RLWE_{s}(m) -> RLWE_{s}(m^k)
    //     let mut rlwe_m = RlweCiphertext(rlwe_m, false);
    //     let mut scratch_space = vec![vec![0u64; ring_size as usize]; d_rgsw +
    // 2];     let (auto_map_index, auto_map_sign) =
    // generate_auto_map(ring_size as usize, auto_k);     let decomposer =
    // DefaultDecomposer::new(q, logb, d_rgsw);     galois_auto(
    //         &mut rlwe_m,
    //         &ksk_out,
    //         &mut scratch_space,
    //         &auto_map_index,
    //         &auto_map_sign,
    //         &mod_op,
    //         &ntt_op,
    //         &decomposer,
    //     );

    //     let rlwe_m_k = rlwe_m;

    //     // Decrypt RLWE_{s}(m^k) and check
    //     let mut encoded_m_k_back = vec![vec![0u64; ring_size as usize]];
    //     decrypt_rlwe(
    //         &rlwe_m_k,
    //         s.values(),
    //         &mut encoded_m_k_back,
    //         &ntt_op,
    //         &mod_op,
    //     );
    //     let m_k_back = encoded_m_k_back[0]
    //         .iter()
    //         .map(|v| (((*v as f64 * p as f64) / q as f64).round() as u64) %
    // p)         .collect_vec();

    //     let mut m_k = vec![0u64; ring_size as usize];
    //     // Send \delta m -> \delta m^k
    //     izip!(m.iter(), auto_map_index.iter(),
    // auto_map_sign.iter()).for_each(         |(v, to_index, sign)| {
    //             if !*sign {
    //                 m_k[*to_index] = (p - *v) % p;
    //             } else {
    //                 m_k[*to_index] = *v;
    //             }
    //         },
    //     );

    //     {
    //         // let encoded_m_k = m_k
    //         //     .iter()
    //         //     .map(|v| ((*v as f64 * q as f64) / p as f64).round() as
    // u64)         //     .collect_vec();

    //         // let noise = measure_noise(&rlwe_m_k, &vec![encoded_m_k],
    // &ntt_op,         // &mod_op, &s); println!("Ksk noise: {noise}");
    //     }

    //     // FIXME(Jay): Galios autormophism will incur high error unless we
    // fix in     // accurate decomoposition of Decomposer when q is prime
    //     assert_eq!(m_k_back, m_k);
    //     // dbg!(m_k_back, m_k, q);
    // }
}
