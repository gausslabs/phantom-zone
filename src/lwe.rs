use std::fmt::Debug;

use itertools::{izip, Itertools};
use num_traits::{abs, Zero};

use crate::{
    backend::{ArithmeticOps, VectorOps},
    decomposer::Decomposer,
    lwe,
    num::UnsignedInteger,
    random::{DefaultSecureRng, RandomGaussianDist, RandomUniformDist, DEFAULT_RNG},
    utils::{fill_random_ternary_secret_with_hamming_weight, TryConvertFrom, WithLocal},
    Matrix, MatrixEntity, MatrixMut, Row, RowMut, Secret,
};

trait LweKeySwitchParameters {
    fn n_in(&self) -> usize;
    fn n_out(&self) -> usize;
    fn d_ks(&self) -> usize;
}

trait LweCiphertext<M: Matrix> {}

struct LweSecret {
    values: Vec<i32>,
}

impl Secret for LweSecret {
    type Element = i32;
    fn values(&self) -> &[Self::Element] {
        &self.values
    }
}

impl LweSecret {
    fn random(hw: usize, n: usize) -> LweSecret {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut out = vec![0i32; n];
            fill_random_ternary_secret_with_hamming_weight(&mut out, hw, rng);

            LweSecret { values: out }
        })
    }
}

fn lwe_key_switch<
    M: Matrix,
    Mmut: MatrixMut<MatElement = M::MatElement> + MatrixEntity,
    Op: VectorOps<Element = M::MatElement> + ArithmeticOps<Element = M::MatElement>,
    D: Decomposer<Element = M::MatElement>,
>(
    lwe_out: &mut Mmut,
    lwe_in: &M,
    lwe_ksk: &M,
    operator: &Op,
    decomposer: &D,
) where
    <Mmut as Matrix>::R: RowMut,
{
    assert!(lwe_ksk.dimension().0 == ((lwe_in.dimension().1 - 1) * decomposer.d()));
    assert!(lwe_out.dimension() == (1, lwe_ksk.dimension().1));

    let mut scratch_space = Mmut::zeros(1, lwe_out.dimension().1);

    let lwe_in_a_decomposed = lwe_in
        .get_row(0)
        .skip(1)
        .flat_map(|ai| decomposer.decompose(ai));
    izip!(lwe_in_a_decomposed, lwe_ksk.iter_rows()).for_each(|(ai_j, beta_ij_lwe)| {
        operator.elwise_scalar_mul(scratch_space.get_row_mut(0), beta_ij_lwe.as_ref(), &ai_j);
        operator.elwise_add_mut(lwe_out.get_row_mut(0), scratch_space.get_row_slice(0))
    });

    let out_b = operator.add(lwe_out.get(0, 0), lwe_in.get(0, 0));
    lwe_out.set(0, 0, out_b);
}

fn lwe_ksk_keygen<
    Mmut: MatrixMut,
    S: Secret,
    Op: VectorOps<Element = Mmut::MatElement> + ArithmeticOps<Element = Mmut::MatElement>,
    R: RandomGaussianDist<Mmut::MatElement, Parameters = Mmut::MatElement>
        + RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
>(
    lwe_sk_in: &S,
    lwe_sk_out: &S,
    ksk_out: &mut Mmut,
    gadget: &[Mmut::MatElement],
    operator: &Op,
    rng: &mut R,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut: TryConvertFrom<[S::Element], Parameters = Mmut::MatElement>,
    Mmut::MatElement: Zero + Debug,
{
    assert!(
        ksk_out.dimension()
            == (
                lwe_sk_in.values().len() * gadget.len(),
                lwe_sk_out.values().len() + 1,
            )
    );

    let d = gadget.len();

    let modulus = VectorOps::modulus(operator);
    let mut neg_sk_in_m = Mmut::try_convert_from(lwe_sk_in.values(), &modulus);
    operator.elwise_neg_mut(neg_sk_in_m.get_row_mut(0));
    let sk_out_m = Mmut::try_convert_from(lwe_sk_out.values(), &modulus);

    izip!(
        neg_sk_in_m.get_row(0),
        ksk_out.iter_rows_mut().chunks(d).into_iter()
    )
    .for_each(|(neg_sk_in_si, d_ks_lwes)| {
        izip!(gadget.iter(), d_ks_lwes.into_iter()).for_each(|(f, lwe)| {
            // sample `a`
            RandomUniformDist::random_fill(rng, &modulus, &mut lwe.as_mut()[1..]);

            // a * z
            let mut az = Mmut::MatElement::zero();
            izip!(lwe.as_ref()[1..].iter(), sk_out_m.get_row(0)).for_each(|(ai, si)| {
                let ai_si = operator.mul(ai, si);
                az = operator.add(&az, &ai_si);
            });

            // a*z + (-s_i)*\beta^j + e
            let mut b = operator.add(&az, &operator.mul(f, neg_sk_in_si));
            let mut e = Mmut::MatElement::zero();
            RandomGaussianDist::random_fill(rng, &modulus, &mut e);
            b = operator.add(&b, &e);

            lwe.as_mut()[0] = b;

            // dbg!(&lwe.as_mut(), &f);
        })
    });
}

/// Encrypts encoded message m as LWE ciphertext
fn encrypt_lwe<
    Mmut: MatrixMut + MatrixEntity,
    R: RandomGaussianDist<Mmut::MatElement, Parameters = Mmut::MatElement>
        + RandomUniformDist<[Mmut::MatElement], Parameters = Mmut::MatElement>,
    S: Secret,
    Op: ArithmeticOps<Element = Mmut::MatElement>,
>(
    lwe_out: &mut Mmut,
    m: Mmut::MatElement,
    s: &S,
    operator: &Op,
    rng: &mut R,
) where
    Mmut: TryConvertFrom<[S::Element], Parameters = Mmut::MatElement>,
    Mmut::MatElement: Zero,
    <Mmut as Matrix>::R: RowMut,
{
    let s = Mmut::try_convert_from(s.values(), &operator.modulus());
    assert!(s.dimension().0 == (lwe_out.dimension().0));
    assert!(s.dimension().1 == (lwe_out.dimension().1 - 1));

    // a*s
    RandomUniformDist::random_fill(rng, &operator.modulus(), &mut lwe_out.get_row_mut(0)[1..]);
    let mut sa = Mmut::MatElement::zero();
    izip!(lwe_out.get_row(0).skip(1), s.get_row(0)).for_each(|(ai, si)| {
        let tmp = operator.mul(ai, si);
        sa = operator.add(&tmp, &sa);
    });

    // b = a*s + e + m
    let mut e = Mmut::MatElement::zero();
    RandomGaussianDist::random_fill(rng, &operator.modulus(), &mut e);
    let b = operator.add(&operator.add(&sa, &e), &m);
    lwe_out.set(0, 0, b);
}

fn decrypt_lwe<M: Matrix, Op: ArithmeticOps<Element = M::MatElement>, S: Secret>(
    lwe_ct: &M,
    s: &S,
    operator: &Op,
) -> M::MatElement
where
    M: TryConvertFrom<[S::Element], Parameters = M::MatElement>,
    M::MatElement: Zero,
{
    let s = M::try_convert_from(s.values(), &operator.modulus());

    let mut sa = M::MatElement::zero();
    izip!(lwe_ct.get_row(0).skip(1), s.get_row(0)).for_each(|(ai, si)| {
        let tmp = operator.mul(ai, si);
        sa = operator.add(&tmp, &sa);
    });

    let b = &lwe_ct.get_row_slice(0)[0];
    operator.sub(b, &sa)
}

#[cfg(test)]
mod tests {

    use crate::{
        backend::ModularOpsU64,
        decomposer::{gadget_vector, DefaultDecomposer},
        lwe::lwe_key_switch,
        random::DefaultSecureRng,
    };

    use super::{decrypt_lwe, encrypt_lwe, lwe_ksk_keygen, LweSecret};

    #[test]
    fn encrypt_decrypt_works() {
        let logq = 20;
        let q = 1u64 << logq;
        let lwe_n = 1024;
        let logp = 3;

        let modq_op = ModularOpsU64::new(q);
        let lwe_sk = LweSecret::random(lwe_n >> 1, lwe_n);

        let mut rng = DefaultSecureRng::new();

        // encrypt
        for m in 0..1u64 << logp {
            let encoded_m = m << (logq - logp);
            let mut lwe_ct = vec![vec![0u64; lwe_n + 1]];
            encrypt_lwe(&mut lwe_ct, encoded_m, &lwe_sk, &modq_op, &mut rng);
            let encoded_m_back = decrypt_lwe(&lwe_ct, &lwe_sk, &modq_op);
            let m_back = ((((encoded_m_back as f64) * ((1 << logp) as f64)) / q as f64).round()
                as u64)
                % (1u64 << logp);
            assert_eq!(m, m_back, "Expected {m} but got {m_back}");
        }
    }

    #[test]
    fn key_switch_works() {
        let logq = 16;
        let logp = 3;
        let q = 1u64 << logq;
        let lwe_in_n = 1024;
        let lwe_out_n = 470;
        let d_ks = 3;
        let logb = 4;

        let lwe_sk_in = LweSecret::random(lwe_in_n >> 1, lwe_in_n);
        let lwe_sk_out = LweSecret::random(lwe_out_n >> 1, lwe_out_n);

        let mut rng = DefaultSecureRng::new();
        let modq_op = ModularOpsU64::new(q);

        // genrate ksk
        for _ in 0..10 {
            let mut ksk = vec![vec![0u64; lwe_out_n + 1]; d_ks * lwe_in_n];
            let gadget = gadget_vector(logq, logb, d_ks);
            lwe_ksk_keygen(
                &lwe_sk_in,
                &lwe_sk_out,
                &mut ksk,
                &gadget,
                &modq_op,
                &mut rng,
            );
            // println!("{:?}", ksk);

            for m in 0..(1 << logp) {
                // encrypt using lwe_sk_in
                let encoded_m = m << (logq - logp);
                let mut lwe_in_ct = vec![vec![0u64; lwe_in_n + 1]];
                encrypt_lwe(&mut lwe_in_ct, encoded_m, &lwe_sk_in, &modq_op, &mut rng);

                // key switch from lwe_sk_in to lwe_sk_out
                let decomposer = DefaultDecomposer::new(1u64 << logq, logb, d_ks);
                let mut lwe_out_ct = vec![vec![0u64; lwe_out_n + 1]];
                lwe_key_switch(&mut lwe_out_ct, &lwe_in_ct, &ksk, &modq_op, &decomposer);

                // decrypt lwe_out_ct using lwe_sk_out
                let encoded_m_back = decrypt_lwe(&lwe_out_ct, &lwe_sk_out, &modq_op);
                let m_back = ((((encoded_m_back as f64) * ((1 << logp) as f64)) / q as f64).round()
                    as u64)
                    % (1u64 << logp);
                assert_eq!(m, m_back, "Expected {m} but got {m_back}");
                // dbg!(m, m_back);
                // dbg!(encoded_m, encoded_m_back);
            }
        }
    }
}
