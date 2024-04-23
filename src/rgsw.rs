use itertools::izip;

use crate::{
    backend::VectorOps,
    decomposer::{self, Decomposer},
    ntt::Ntt,
    random::{DefaultSecureRng, RandomGaussianDist, RandomUniformDist},
    utils::{fill_random_ternary_secret_with_hamming_weight, TryConvertFrom, WithLocal},
    Matrix, MatrixEntity, MatrixMut, RowMut, Secret,
};

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

/// Encrypts message m as a RGSW ciphertext.
///
/// - m_eval: is `m` is evaluation domain
/// - out_rgsw: RGSW(m) is stored as single matrix of dimension (d_rgsw * 4,
///   ring_size). The matrix has the following structure [RLWE'_A(-sm) ||
///   RLWE'_B(-sm) || RLWE'_A(m) || RLWE'_B(m)]^T
fn encrypt_rgsw<
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
    let (rlwe_dash_nsm, rlwe_dash_m) = out_rgsw.split_at_row(d * 2);

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
fn rlwe_in_decomposed_evaluation_domain_mul_rgsw_rlwe_out_evaluation_domain<
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

    let (a_rlwe_out, b_rlwe_out) = rlwe_out_eval.split_at_row(1);

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

fn decompose_rlwe<
    M: Matrix + Clone,
    Mmut: MatrixMut<MatElement = M::MatElement> + MatrixEntity,
    D: Decomposer<Element = M::MatElement>,
>(
    rlwe_in: &M,
    decomposer: &D,
    rlwe_in_decomposed: &mut Mmut,
) where
    M::MatElement: Copy,
    <Mmut as Matrix>::R: RowMut,
{
    let d_rgsw = decomposer.d();
    let ring_size = rlwe_in.dimension().1;
    assert!(rlwe_in_decomposed.dimension() == (2 * d_rgsw, ring_size));

    // Decompose rlwe_in
    for ri in 0..ring_size {
        // ai
        let ai_decomposed = decomposer.decompose(rlwe_in.get(0, ri));
        for j in 0..d_rgsw {
            rlwe_in_decomposed.set(j, ri, ai_decomposed[j]);
        }

        // bi
        let bi_decomposed = decomposer.decompose(rlwe_in.get(1, ri));
        for j in 0..d_rgsw {
            rlwe_in_decomposed.set(j + d_rgsw, ri, bi_decomposed[j]);
        }
    }
}

/// Returns RLWE(m0m1) = RLWE(m0) x RGSW(m1)
///
/// - rlwe_in: is RLWE(m0) with polynomials in coefficient domain
/// - rgsw_in: is RGSW(m1) with polynomials in evaluation domain
/// - rlwe_out: is output RLWE(m0m1) with polynomials in coefficient domain
/// - rlwe_in_decomposed: is a matrix of dimension (d_rgsw * 2, ring_size) used
///   as scratch space to store decomposed RLWE(m0)
fn rlwe_by_rgsw<
    M: Matrix + Clone,
    Mmut: MatrixMut<MatElement = M::MatElement> + MatrixEntity,
    D: Decomposer<Element = M::MatElement>,
    ModOp: VectorOps<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement>,
>(
    rlwe_in: &M,
    rgsw_in: &M,
    rlwe_out: &mut Mmut,
    rlwe_in_decomposed: &mut Mmut,
    decomposer: &D,
    ntt_op: &NttOp,
    mod_op: &ModOp,
) where
    M::MatElement: Copy,
    <Mmut as Matrix>::R: RowMut,
{
    decompose_rlwe(rlwe_in, decomposer, rlwe_in_decomposed);

    // transform rlwe_in decomposed to evaluation domain
    rlwe_in_decomposed
        .iter_rows_mut()
        .for_each(|r| ntt_op.forward(r.as_mut()));

    // decomposed RLWE x RGSW
    rlwe_in_decomposed_evaluation_domain_mul_rgsw_rlwe_out_evaluation_domain(
        rgsw_in,
        rlwe_in_decomposed,
        rlwe_out,
        mod_op,
    );

    // transform rlwe_out to coefficient domain
    rlwe_out
        .iter_rows_mut()
        .for_each(|r| ntt_op.backward(r.as_mut()));
}

/// Encrypt polynomial m(X) as RLWE ciphertext.
///
/// - rlwe_out: returned RLWE ciphertext RLWE(m) in coefficient domain. RLWE
///   ciphertext is a matirx with first row consiting polynomial `a` and the
///   second rows consting polynomial `b`
fn encrypt_rlwe<
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
fn decrypt_rlwe<
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

#[cfg(test)]
mod tests {
    use std::vec;

    use itertools::Itertools;
    use rand::{thread_rng, Rng};

    use crate::{
        backend::ModularOpsU64,
        decomposer::{gadget_vector, DefaultDecomposer},
        ntt::{self, Ntt, NttBackendU64},
        random::{DefaultSecureRng, RandomUniformDist},
        utils::{generate_prime, negacyclic_mul},
    };

    use super::{decrypt_rlwe, encrypt_rgsw, encrypt_rlwe, rlwe_by_rgsw, RlweSecret};

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

        // RLWE(m0m1) = RLWE(m0) x RGSW(m1)
        let mut rlwe_out_ct = vec![vec![0u64; ring_size as usize]; 2];
        let mut scratch_space = vec![vec![0u64; ring_size as usize]; d_rgsw * 2];
        let decomposer = DefaultDecomposer::new(q, logb, d_rgsw);
        rlwe_by_rgsw(
            &rlwe_in_ct,
            &rgsw_ct,
            &mut rlwe_out_ct,
            &mut scratch_space,
            &decomposer,
            &ntt_op,
            &mod_op,
        );

        // Decrypt RLWE(m0m1)
        let mut encoded_m0m1_back = vec![vec![0u64; ring_size as usize]];
        decrypt_rlwe(&rlwe_out_ct, &s, &mut encoded_m0m1_back, &ntt_op, &mod_op);
        let m0m1_back = encoded_m0m1_back[0]
            .iter()
            .map(|v| (((*v as f64 * p as f64) / (q as f64)).round() as u64) % p)
            .collect_vec();

        let mul_mod = |v0: &u64, v1: &u64| (v0 * v1) % p;
        let m0m1 = negacyclic_mul(&m0, &m1, mul_mod, p);
        assert_eq!(m0m1, m0m1_back, "Expected {:?} got {:?}", m0m1, m0m1_back);
        // dbg!(&m0m1_back, m0m1, q);
    }
}
