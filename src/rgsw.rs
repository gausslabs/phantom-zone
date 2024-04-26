use std::{
    clone,
    fmt::Debug,
    ops::{Neg, Sub},
};

use itertools::{izip, Itertools};
use num_traits::{PrimInt, ToPrimitive, Zero};

use crate::{
    backend::{ArithmeticOps, VectorOps},
    decomposer::{self, Decomposer},
    ntt::{self, Ntt},
    random::{DefaultSecureRng, RandomGaussianDist, RandomUniformDist},
    utils::{fill_random_ternary_secret_with_hamming_weight, TryConvertFrom, WithLocal},
    Matrix, MatrixEntity, MatrixMut, RowMut, Secret,
};

struct RlweCiphertext<M>(M, bool);

impl<M: Matrix> Matrix for RlweCiphertext<M> {
    type MatElement = M::MatElement;
    type R = M::R;

    fn dimension(&self) -> (usize, usize) {
        self.0.dimension()
    }
}

impl<M: MatrixMut> MatrixMut for RlweCiphertext<M> where <M as Matrix>::R: RowMut {}

impl<M: Matrix> AsRef<[<M as Matrix>::R]> for RlweCiphertext<M> {
    fn as_ref(&self) -> &[<M as Matrix>::R] {
        self.0.as_ref()
    }
}

impl<M: MatrixMut> AsMut<[<M as Matrix>::R]> for RlweCiphertext<M>
where
    <M as Matrix>::R: RowMut,
{
    fn as_mut(&mut self) -> &mut [<M as Matrix>::R] {
        self.0.as_mut()
    }
}

impl<M> IsTrivial for RlweCiphertext<M> {
    fn is_trivial(&self) -> bool {
        self.1
    }
    fn set_not_trivial(&mut self) {
        self.1 = false;
    }
}

pub trait IsTrivial {
    fn is_trivial(&self) -> bool;
    fn set_not_trivial(&mut self);
}

struct RlweSecret {
    values: Vec<i32>,
}

impl Secret for RlweSecret {
    type Element = i32;
    fn values(&self) -> &[Self::Element] {
        &self.values
    }
}

impl RlweSecret {
    fn random(hw: usize, n: usize) -> RlweSecret {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut out = vec![0i32; n];
            fill_random_ternary_secret_with_hamming_weight(&mut out, hw, rng);

            RlweSecret { values: out }
        })
    }
}

fn generate_auto_map(ring_size: usize, k: usize) -> (Vec<usize>, Vec<bool>) {
    assert!(k & 1 == 1, "Auto {k} must be odd");
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
/// - ksk_out: Output Key switching key. Key switching key stores RLWE
///   ciphertexts as [RLWE'_A(-from_s) || RLWE'_B(-from_s)]
/// - neg_from_s_eval: Negative of secret polynomial to key switch from in
///   evaluation domain
/// - to_s_eval: secret polynomial to key switch to in evalution domain.
pub(crate) fn rlwe_ksk_gen<
    Mmut: MatrixMut + MatrixEntity,
    ModOp: ArithmeticOps<Element = Mmut::MatElement> + VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
    R: RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
>(
    ksk_out: &mut Mmut,
    neg_from_s_eval: &Mmut,
    to_s_eval: &Mmut,
    gadget_vector: &[Mmut::MatElement],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    rng: &mut R,
) where
    <Mmut as Matrix>::R: RowMut,
{
    let ring_size = neg_from_s_eval.dimension().1;
    let d = gadget_vector.len();
    assert!(neg_from_s_eval.dimension().0 == 1);
    assert!(ksk_out.dimension() == (d * 2, ring_size));
    assert!(to_s_eval.dimension() == (1, ring_size));

    let q = ArithmeticOps::modulus(mod_op);

    let mut scratch_space = Mmut::zeros(1, ring_size);

    // RLWE'_{to_s}(-from_s)
    let (part_a, part_b) = ksk_out.split_at_row_mut(d);
    izip!(part_a.iter_mut(), part_b.iter_mut(), gadget_vector.iter()).for_each(
        |(ai, bi, beta_i)| {
            // sample ai and transform to evaluation
            RandomUniformDist::random_fill(rng, &q, ai.as_mut());
            ntt_op.forward(ai.as_mut());

            // to_s * ai
            mod_op.elwise_mul(
                scratch_space.get_row_mut(0),
                ai.as_ref(),
                to_s_eval.get_row_slice(0),
            );

            // ei + to_s*ai
            RandomGaussianDist::random_fill(rng, &q, bi.as_mut());
            ntt_op.forward(bi.as_mut());
            mod_op.elwise_add_mut(bi.as_mut(), scratch_space.get_row_slice(0));

            // beta_i * -from_s
            mod_op.elwise_scalar_mul(
                scratch_space.get_row_mut(0),
                neg_from_s_eval.get_row_slice(0),
                beta_i,
            );

            // bi = ei + to_s*ai + beta_i*-from_s
            mod_op.elwise_add_mut(bi.as_mut(), scratch_space.get_row_slice(0));
        },
    );
}

pub(crate) fn galois_key_gen<
    Mmut: MatrixMut + MatrixEntity,
    ModOp: ArithmeticOps<Element = Mmut::MatElement> + VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
    S: Secret,
    R: RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
>(
    ksk_out: &mut Mmut,
    s: &S,
    auto_k: usize,
    gadget_vector: &[Mmut::MatElement],
    mod_op: &ModOp,
    ntt_op: &NttOp,
    rng: &mut R,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut: TryConvertFrom<[S::Element], Parameters = Mmut::MatElement>,
    Mmut::MatElement: Copy + Sub<Output = Mmut::MatElement>,
{
    let ring_size = s.values().len();
    let (auto_map_index, auto_map_sign) = generate_auto_map(ring_size, auto_k);

    let q = ArithmeticOps::modulus(mod_op);

    // s(X) -> -s(X^k)
    let mut s = Mmut::try_convert_from(s.values(), &q);
    let mut neg_s_auto = Mmut::zeros(1, s.dimension().1);
    izip!(s.get_row(0), auto_map_index.iter(), auto_map_sign.iter()).for_each(
        |(el, to_index, sign)| {
            // if sign is +ve (true), then negate because we need -s(X) (i.e. do the
            // opposite than the usual case)
            if *sign {
                neg_s_auto.set(0, *to_index, q - *el)
            } else {
                neg_s_auto.set(0, *to_index, *el)
            }
        },
    );

    // send both s(X) and -s(X^k) to evaluation domain
    ntt_op.forward(s.get_row_mut(0));
    ntt_op.forward(neg_s_auto.get_row_mut(0));

    // Ksk from -s(X^k) to s(X)
    rlwe_ksk_gen(ksk_out, &neg_s_auto, &s, gadget_vector, mod_op, ntt_op, rng);
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

/// Encrypts message m as a RGSW ciphertext.
///
/// - m_eval: is `m` is evaluation domain
/// - out_rgsw: RGSW(m) is stored as single matrix of dimension (d_rgsw * 4,
///   ring_size). The matrix has the following structure [RLWE'_A(-sm) ||
///   RLWE'_B(-sm) || RLWE'_A(m) || RLWE'_B(m)]^T
pub(crate) fn encrypt_rgsw<
    Mmut: MatrixMut + MatrixEntity,
    M: Matrix<MatElement = Mmut::MatElement> + Clone,
    S: Secret,
    R: RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
>(
    out_rgsw: &mut Mmut,
    m_eval: &M,
    gadget_vector: &[Mmut::MatElement],
    s: &S,
    mod_op: &ModOp,
    ntt_op: &NttOp,
    rng: &mut R,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut: TryConvertFrom<[S::Element], Parameters = Mmut::MatElement>,
{
    let d = gadget_vector.len();
    let q = mod_op.modulus();
    let ring_size = s.values().len();
    assert!(out_rgsw.dimension() == (d * 4, ring_size));
    assert!(m_eval.dimension() == (1, ring_size));

    // RLWE(-sm), RLWE(-sm)
    let (rlwe_dash_nsm, rlwe_dash_m) = out_rgsw.split_at_row_mut(d * 2);

    let mut s_eval = Mmut::try_convert_from(s.values(), &q);
    ntt_op.forward(s_eval.get_row_mut(0).as_mut());

    let mut scratch_space = Mmut::zeros(1, ring_size);

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
        ntt_op.forward(ai.as_mut());

        // a_i * s
        mod_op.elwise_mul(
            scratch_space.get_row_mut(0),
            ai.as_ref(),
            s_eval.get_row_slice(0),
        );
        // b_i = e_i + a_i * s
        RandomGaussianDist::random_fill(rng, &q, bi.as_mut());
        ntt_op.forward(bi.as_mut());
        mod_op.elwise_add_mut(bi.as_mut(), scratch_space.get_row_slice(0));

        // a_i + \beta_i * m
        mod_op.elwise_scalar_mul(
            scratch_space.get_row_mut(0),
            m_eval.get_row_slice(0),
            beta_i,
        );
        mod_op.elwise_add_mut(ai.as_mut(), scratch_space.get_row_slice(0));
    });

    // RLWE(m)
    let (a_rlwe_dash_m, b_rlwe_dash_m) = rlwe_dash_m.split_at_mut(d);
    izip!(
        a_rlwe_dash_m.iter_mut(),
        b_rlwe_dash_m.iter_mut(),
        gadget_vector.iter()
    )
    .for_each(|(ai, bi, beta_i)| {
        // Sample e_i and transform to evaluation domain
        RandomGaussianDist::random_fill(rng, &q, bi.as_mut());
        ntt_op.forward(bi.as_mut());

        // beta_i * m
        mod_op.elwise_scalar_mul(
            scratch_space.get_row_mut(0),
            m_eval.get_row_slice(0),
            beta_i,
        );
        // e_i + beta_i * m
        mod_op.elwise_add_mut(bi.as_mut(), scratch_space.get_row_slice(0));

        // Sample a_i and transform to evaluation domain
        RandomUniformDist::random_fill(rng, &q, ai.as_mut());
        ntt_op.forward(ai.as_mut());

        // ai * s
        mod_op.elwise_mul(
            scratch_space.get_row_mut(0),
            ai.as_ref(),
            s_eval.get_row_slice(0),
        );

        // b_i = a_i*s + e_i + beta_i*m
        mod_op.elwise_add_mut(bi.as_mut(), scratch_space.get_row_slice(0));
    });
}

/// Returns RLWE(mm') =  RLWE(m) x RGSW(m')
///
/// - rgsw_in: RGSW(m') in evaluation domain
/// - rlwe_in_decomposed: decomposed RLWE(m) in evaluation domain
/// - rlwe_out: returned RLWE(mm') in evaluation domain
pub(crate) fn rlwe_in_decomposed_evaluation_domain_mul_rgsw_rlwe_out_evaluation_domain<
    Mmut: MatrixMut + MatrixEntity,
    M: Matrix<MatElement = Mmut::MatElement> + Clone,
    ModOp: VectorOps<Element = Mmut::MatElement>,
>(
    rgsw_in: &M,
    rlwe_in_decomposed_eval: &Mmut,
    rlwe_out_eval: &mut Mmut,
    mod_op: &ModOp,
) where
    <Mmut as Matrix>::R: RowMut,
{
    let ring_size = rgsw_in.dimension().1;
    let d_rgsw = rgsw_in.dimension().0 / 4;
    assert!(rlwe_in_decomposed_eval.dimension() == (2 * d_rgsw, ring_size));
    assert!(rlwe_out_eval.dimension() == (2, ring_size));

    let (a_rlwe_out, b_rlwe_out) = rlwe_out_eval.split_at_row_mut(1);

    // a * RLWE'(-sm)
    let a_rlwe_dash_nsm = rgsw_in.iter_rows().take(d_rgsw);
    let b_rlwe_dash_nsm = rgsw_in.iter_rows().skip(d_rgsw).take(d_rgsw);
    izip!(
        rlwe_in_decomposed_eval.iter_rows().take(d_rgsw),
        a_rlwe_dash_nsm
    )
    .for_each(|(a, b)| {
        mod_op.elwise_fma_mut(a_rlwe_out[0].as_mut(), a.as_ref(), b.as_ref());
    });
    izip!(
        rlwe_in_decomposed_eval.iter_rows().take(d_rgsw),
        b_rlwe_dash_nsm
    )
    .for_each(|(a, b)| {
        mod_op.elwise_fma_mut(b_rlwe_out[0].as_mut(), a.as_ref(), b.as_ref());
    });

    // b * RLWE'(m)
    let a_rlwe_dash_m = rgsw_in.iter_rows().skip(d_rgsw * 2).take(d_rgsw);
    let b_rlwe_dash_m = rgsw_in.iter_rows().skip(d_rgsw * 3);
    izip!(
        rlwe_in_decomposed_eval.iter_rows().skip(d_rgsw),
        a_rlwe_dash_m
    )
    .for_each(|(a, b)| {
        mod_op.elwise_fma_mut(a_rlwe_out[0].as_mut(), a.as_ref(), b.as_ref());
    });
    izip!(
        rlwe_in_decomposed_eval.iter_rows().skip(d_rgsw),
        b_rlwe_dash_m
    )
    .for_each(|(a, b)| {
        mod_op.elwise_fma_mut(b_rlwe_out[0].as_mut(), a.as_ref(), b.as_ref());
    });
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

// pub(crate) fn decompose_rlwe<
//     M: Matrix + Clone,
//     Mmut: MatrixMut<MatElement = M::MatElement> + MatrixEntity,
//     D: Decomposer<Element = M::MatElement>,
// >(
//     rlwe_in: &M,
//     decomposer: &D,
//     rlwe_in_decomposed: &mut Mmut,
// ) where
//     M::MatElement: Copy,
//     <Mmut as Matrix>::R: RowMut,
// {
//     let d_rgsw = decomposer.d();
//     let ring_size = rlwe_in.dimension().1;
//     assert!(rlwe_in_decomposed.dimension() == (2 * d_rgsw, ring_size));

//     // Decompose rlwe_in
//     for ri in 0..ring_size {
//         // ai
//         let ai_decomposed = decomposer.decompose(rlwe_in.get(0, ri));
//         for j in 0..d_rgsw {
//             rlwe_in_decomposed.set(j, ri, ai_decomposed[j]);
//         }

//         // bi
//         let bi_decomposed = decomposer.decompose(rlwe_in.get(1, ri));
//         for j in 0..d_rgsw {
//             rlwe_in_decomposed.set(j + d_rgsw, ri, bi_decomposed[j]);
//         }
//     }
// }

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

/// Encrypt polynomial m(X) as RLWE ciphertext.
///
/// - rlwe_out: returned RLWE ciphertext RLWE(m) in coefficient domain. RLWE
///   ciphertext is a matirx with first row consiting polynomial `a` and the
///   second rows consting polynomial `b`
pub(crate) fn encrypt_rlwe<
    Mmut: Matrix + MatrixMut + Clone,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
    S: Secret,
    R: RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>
        + RandomGaussianDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
>(
    m: &Mmut,
    rlwe_out: &mut Mmut,
    s: &S,
    mod_op: &ModOp,
    ntt_op: &NttOp,
    rng: &mut R,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut: TryConvertFrom<[S::Element], Parameters = Mmut::MatElement>,
{
    let ring_size = s.values().len();
    assert!(rlwe_out.dimension() == (2, ring_size));
    assert!(m.dimension() == (1, ring_size));

    let q = mod_op.modulus();

    // sample a
    RandomUniformDist::random_fill(rng, &q, rlwe_out.get_row_mut(0));

    // s * a
    let mut sa = Mmut::try_convert_from(s.values(), &q);
    ntt_op.forward(sa.get_row_mut(0));
    ntt_op.forward(rlwe_out.get_row_mut(0));
    mod_op.elwise_mul_mut(sa.get_row_mut(0), rlwe_out.get_row_slice(0));
    ntt_op.backward(rlwe_out.get_row_mut(0));
    ntt_op.backward(sa.get_row_mut(0));

    // sample e
    RandomGaussianDist::random_fill(rng, &q, rlwe_out.get_row_mut(1));
    mod_op.elwise_add_mut(rlwe_out.get_row_mut(1), m.get_row_slice(0));
    mod_op.elwise_add_mut(rlwe_out.get_row_mut(1), sa.get_row_slice(0));
}

/// Decrypts degree 1 RLWE ciphertext RLWE(m) and returns m
///
/// - rlwe_ct: input degree 1 ciphertext RLWE(m).
pub(crate) fn decrypt_rlwe<
    Mmut: MatrixMut + Clone,
    M: Matrix<MatElement = Mmut::MatElement>,
    ModOp: VectorOps<Element = Mmut::MatElement>,
    NttOp: Ntt<Element = Mmut::MatElement>,
    S: Secret,
>(
    rlwe_ct: &M,
    s: &S,
    m_out: &mut Mmut,
    ntt_op: &NttOp,
    mod_op: &ModOp,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut: TryConvertFrom<[S::Element], Parameters = Mmut::MatElement>,
    Mmut::MatElement: Copy,
{
    let ring_size = s.values().len();
    assert!(rlwe_ct.dimension() == (2, ring_size));
    assert!(m_out.dimension() == (1, ring_size));

    // transform a to evluation form
    m_out
        .get_row_mut(0)
        .copy_from_slice(rlwe_ct.get_row_slice(0));
    ntt_op.forward(m_out.get_row_mut(0));

    // -s*a
    let mut s = Mmut::try_convert_from(&s.values(), &mod_op.modulus());
    ntt_op.forward(s.get_row_mut(0));
    mod_op.elwise_mul_mut(m_out.get_row_mut(0), s.get_row_slice(0));
    mod_op.elwise_neg_mut(m_out.get_row_mut(0));
    ntt_op.backward(m_out.get_row_mut(0));

    // m+e = b - s*a
    mod_op.elwise_add_mut(m_out.get_row_mut(0), rlwe_ct.get_row_slice(1));
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
        backend::ModularOpsU64,
        decomposer::{gadget_vector, DefaultDecomposer},
        ntt::{self, Ntt, NttBackendU64},
        random::{DefaultSecureRng, RandomUniformDist},
        rgsw::{measure_noise, RlweCiphertext},
        utils::{generate_prime, negacyclic_mul},
        Matrix,
    };

    use super::{
        decrypt_rlwe, encrypt_rgsw, encrypt_rlwe, galois_auto, galois_key_gen, generate_auto_map,
        rlwe_by_rgsw, RlweSecret,
    };

    #[test]
    fn rlwe_by_rgsw_works() {
        let logq = 50;
        let logp = 3;
        let ring_size = 1 << 10;
        let q = generate_prime(logq, ring_size, 1u64 << logq).unwrap();
        let p = 1u64 << logp;
        let d_rgsw = 10;
        let logb = 5;

        let mut rng = DefaultSecureRng::new();

        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        let mut m0 = vec![0u64; ring_size as usize];
        RandomUniformDist::<[u64]>::random_fill(&mut rng, &(1u64 << logp), m0.as_mut_slice());
        let mut m1 = vec![0u64; ring_size as usize];
        m1[thread_rng().gen_range(0..ring_size) as usize] = 1;

        let ntt_op = NttBackendU64::new(q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);

        // Encrypt m1 as RGSW(m1)
        let mut rgsw_ct = vec![vec![0u64; ring_size as usize]; d_rgsw * 4];
        let gadget_vector = gadget_vector(logq, logb, d_rgsw);
        let mut m1_eval = m1.clone();
        ntt_op.forward(&mut m1_eval);
        encrypt_rgsw(
            &mut rgsw_ct,
            &vec![m1_eval],
            &gadget_vector,
            &s,
            &mod_op,
            &ntt_op,
            &mut rng,
        );
        // println!("RGSW(m1): {:?}", &rgsw_ct);

        // Encrypt m0 as RLWE(m0)
        let mut rlwe_in_ct = vec![vec![0u64; ring_size as usize]; 2];
        let encoded_m = m0
            .iter()
            .map(|v| (((*v as f64) * q as f64) / (p as f64)).round() as u64)
            .collect_vec();
        encrypt_rlwe(
            &vec![encoded_m.clone()],
            &mut rlwe_in_ct,
            &s,
            &mod_op,
            &ntt_op,
            &mut rng,
        );
        let mut rlwe_in_ct = RlweCiphertext(rlwe_in_ct, false);

        // RLWE(m0m1) = RLWE(m0) x RGSW(m1)
        let mut scratch_space = vec![vec![0u64; ring_size as usize]; d_rgsw + 2];
        let decomposer = DefaultDecomposer::new(q, logb, d_rgsw);
        rlwe_by_rgsw(
            &mut rlwe_in_ct,
            &rgsw_ct,
            &mut scratch_space,
            &decomposer,
            &ntt_op,
            &mod_op,
        );

        // Decrypt RLWE(m0m1)
        let mut encoded_m0m1_back = vec![vec![0u64; ring_size as usize]];
        decrypt_rlwe(&rlwe_in_ct, &s, &mut encoded_m0m1_back, &ntt_op, &mod_op);
        let m0m1_back = encoded_m0m1_back[0]
            .iter()
            .map(|v| (((*v as f64 * p as f64) / (q as f64)).round() as u64) % p)
            .collect_vec();

        let mul_mod = |v0: &u64, v1: &u64| (v0 * v1) % p;
        let m0m1 = negacyclic_mul(&m0, &m1, mul_mod, p);
        assert_eq!(m0m1, m0m1_back, "Expected {:?} got {:?}", m0m1, m0m1_back);
        // dbg!(&m0m1_back, m0m1, q);
    }

    #[test]
    fn galois_auto_works() {
        let logq = 50;
        let ring_size = 1 << 5;
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
        let mut rlwe_m = vec![vec![0u64; ring_size as usize]; 2];
        encrypt_rlwe(
            &vec![encoded_m.clone()],
            &mut rlwe_m,
            &s,
            &mod_op,
            &ntt_op,
            &mut rng,
        );

        let auto_k = 25;

        // Generate galois key to key switch from s^k to s
        let mut ksk_out = vec![vec![0u64; ring_size as usize]; d_rgsw * 2];
        let gadget_vector = gadget_vector(logq, logb, d_rgsw);
        galois_key_gen(
            &mut ksk_out,
            &s,
            auto_k,
            &gadget_vector,
            &mod_op,
            &ntt_op,
            &mut rng,
        );

        // Send RLWE_{s}(m) -> RLWE_{s}(m^k)
        let mut rlwe_m = RlweCiphertext(rlwe_m, false);
        let mut scratch_space = vec![vec![0u64; ring_size as usize]; d_rgsw + 2];
        let (auto_map_index, auto_map_sign) = generate_auto_map(ring_size as usize, auto_k);
        let decomposer = DefaultDecomposer::new(q, logb, d_rgsw);
        galois_auto(
            &mut rlwe_m,
            &ksk_out,
            &mut scratch_space,
            &auto_map_index,
            &auto_map_sign,
            &mod_op,
            &ntt_op,
            &decomposer,
        );

        let rlwe_m_k = rlwe_m;

        // Decrypt RLWE_{s}(m^k) and check
        let mut encoded_m_k_back = vec![vec![0u64; ring_size as usize]];
        decrypt_rlwe(&rlwe_m_k, &s, &mut encoded_m_k_back, &ntt_op, &mod_op);
        let m_k_back = encoded_m_k_back[0]
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
            // let encoded_m_k = m_k
            //     .iter()
            //     .map(|v| ((*v as f64 * q as f64) / p as f64).round() as u64)
            //     .collect_vec();

            // let noise = measure_noise(&rlwe_m_k, &vec![encoded_m_k], &ntt_op,
            // &mod_op, &s); println!("Ksk noise: {noise}");
        }

        // FIXME(Jay): Galios autormophism will incur high error unless we fix in
        // accurate decomoposition of Decomposer when q is prime
        assert_eq!(m_k_back, m_k);
        // dbg!(m_k_back, m_k, q);
    }
}
