use std::{
    cell::RefCell,
    collections::btree_map::Values,
    fmt::{Debug, Display},
    marker::PhantomData,
};

use itertools::{izip, Itertools};
use num_traits::{abs, PrimInt, ToPrimitive, Zero};

use crate::{
    backend::{ArithmeticOps, VectorOps},
    decomposer::Decomposer,
    lwe,
    num::UnsignedInteger,
    random::{DefaultSecureRng, NewWithSeed, RandomGaussianDist, RandomUniformDist, DEFAULT_RNG},
    utils::{fill_random_ternary_secret_with_hamming_weight, TryConvertFrom, WithLocal},
    Matrix, MatrixEntity, MatrixMut, Row, RowEntity, RowMut, Secret,
};

struct SeededLweKeySwitchingKey<Ro, S>
where
    Ro: Row,
{
    data: Ro,
    seed: S,
    to_lwe_n: usize,
    modulus: Ro::Element,
}

impl<Ro: RowEntity, S> SeededLweKeySwitchingKey<Ro, S> {
    pub(crate) fn empty(
        from_lwe_n: usize,
        to_lwe_n: usize,
        d: usize,
        seed: S,
        modulus: Ro::Element,
    ) -> Self {
        let data = Ro::zeros(from_lwe_n * d);
        SeededLweKeySwitchingKey {
            data,
            to_lwe_n,
            seed,
            modulus,
        }
    }
}

struct LweKeySwitchingKey<M, R> {
    data: M,
    _phantom: PhantomData<R>,
}

impl<
        M: MatrixMut + MatrixEntity,
        R: NewWithSeed + RandomUniformDist<[M::MatElement], Parameters = M::MatElement>,
    > From<&SeededLweKeySwitchingKey<M::R, R::Seed>> for LweKeySwitchingKey<M, R>
where
    M::R: RowMut,
    R::Seed: Clone,
    M::MatElement: Copy,
{
    fn from(value: &SeededLweKeySwitchingKey<M::R, R::Seed>) -> Self {
        let mut p_rng = R::new_with_seed(value.seed.clone());
        let mut data = M::zeros(value.data.as_ref().len(), value.to_lwe_n + 1);
        izip!(value.data.as_ref().iter(), data.iter_rows_mut()).for_each(|(bi, lwe_i)| {
            RandomUniformDist::random_fill(&mut p_rng, &value.modulus, &mut lwe_i.as_mut()[1..]);
            lwe_i.as_mut()[0] = *bi;
        });
        LweKeySwitchingKey {
            data,
            _phantom: PhantomData,
        }
    }
}

trait LweCiphertext<M: Matrix> {}

#[derive(Clone)]
pub struct LweSecret {
    pub(crate) values: Vec<i32>,
}

impl Secret for LweSecret {
    type Element = i32;
    fn values(&self) -> &[Self::Element] {
        &self.values
    }
}

impl LweSecret {
    pub(crate) fn random(hw: usize, n: usize) -> LweSecret {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut out = vec![0i32; n];
            fill_random_ternary_secret_with_hamming_weight(&mut out, hw, rng);

            LweSecret { values: out }
        })
    }
}

pub(crate) fn lwe_key_switch<
    M: Matrix,
    Ro: AsMut<[M::MatElement]> + AsRef<[M::MatElement]>,
    Op: VectorOps<Element = M::MatElement> + ArithmeticOps<Element = M::MatElement>,
    D: Decomposer<Element = M::MatElement>,
>(
    lwe_out: &mut Ro,
    lwe_in: &Ro,
    lwe_ksk: &M,
    operator: &Op,
    decomposer: &D,
) {
    assert!(lwe_ksk.dimension().0 == ((lwe_in.as_ref().len() - 1) * decomposer.d()));
    assert!(lwe_out.as_ref().len() == lwe_ksk.dimension().1);

    let lwe_in_a_decomposed = lwe_in
        .as_ref()
        .iter()
        .skip(1)
        .flat_map(|ai| decomposer.decompose(ai));
    izip!(lwe_in_a_decomposed, lwe_ksk.iter_rows()).for_each(|(ai_j, beta_ij_lwe)| {
        operator.elwise_fma_scalar_mut(lwe_out.as_mut(), beta_ij_lwe.as_ref(), &ai_j);
    });

    let out_b = operator.add(&lwe_out.as_ref()[0], &lwe_in.as_ref()[0]);
    lwe_out.as_mut()[0] = out_b;
}

pub fn lwe_ksk_keygen<
    Ro: Row + RowMut + RowEntity,
    S,
    Op: VectorOps<Element = Ro::Element> + ArithmeticOps<Element = Ro::Element>,
    R: RandomGaussianDist<Ro::Element, Parameters = Ro::Element>,
    PR: RandomUniformDist<[Ro::Element], Parameters = Ro::Element>,
>(
    from_lwe_sk: &[S],
    to_lwe_sk: &[S],
    ksk_out: &mut Ro,
    gadget: &[Ro::Element],
    operator: &Op,
    p_rng: &mut PR,
    rng: &mut R,
) where
    Ro: TryConvertFrom<[S], Parameters = Ro::Element>,
    Ro::Element: Zero + Debug,
{
    assert!(ksk_out.as_ref().len() == (from_lwe_sk.len() * gadget.len()));

    let d = gadget.len();

    let modulus = VectorOps::modulus(operator);
    let mut neg_sk_in_m = Ro::try_convert_from(from_lwe_sk, &modulus);
    operator.elwise_neg_mut(neg_sk_in_m.as_mut());
    let sk_out_m = Ro::try_convert_from(to_lwe_sk, &modulus);

    let mut scratch = Ro::zeros(to_lwe_sk.len());

    izip!(neg_sk_in_m.as_ref(), ksk_out.as_mut().chunks_mut(d)).for_each(
        |(neg_sk_in_si, d_lwes_partb)| {
            izip!(gadget.iter(), d_lwes_partb.into_iter()).for_each(|(f, lwe_b)| {
                // sample `a`
                RandomUniformDist::random_fill(p_rng, &modulus, scratch.as_mut());

                // a * z
                let mut az = Ro::Element::zero();
                izip!(scratch.as_ref().iter(), sk_out_m.as_ref()).for_each(|(ai, si)| {
                    let ai_si = operator.mul(ai, si);
                    az = operator.add(&az, &ai_si);
                });

                // a*z + (-s_i)*\beta^j + e
                let mut b = operator.add(&az, &operator.mul(f, neg_sk_in_si));
                let mut e = Ro::Element::zero();
                RandomGaussianDist::random_fill(rng, &modulus, &mut e);
                b = operator.add(&b, &e);

                *lwe_b = b;
            })
        },
    );
}

/// Encrypts encoded message m as LWE ciphertext
pub fn encrypt_lwe<
    Ro: Row + RowMut,
    R: RandomGaussianDist<Ro::Element, Parameters = Ro::Element>
        + RandomUniformDist<[Ro::Element], Parameters = Ro::Element>,
    S,
    Op: ArithmeticOps<Element = Ro::Element>,
>(
    lwe_out: &mut Ro,
    m: &Ro::Element,
    s: &[S],
    operator: &Op,
    rng: &mut R,
) where
    Ro: TryConvertFrom<[S], Parameters = Ro::Element>,
    Ro::Element: Zero,
{
    let s = Ro::try_convert_from(s, &operator.modulus());
    assert!(s.as_ref().len() == (lwe_out.as_ref().len() - 1));

    // a*s
    RandomUniformDist::random_fill(rng, &operator.modulus(), &mut lwe_out.as_mut()[1..]);
    let mut sa = Ro::Element::zero();
    izip!(lwe_out.as_mut().iter().skip(1), s.as_ref()).for_each(|(ai, si)| {
        let tmp = operator.mul(ai, si);
        sa = operator.add(&tmp, &sa);
    });

    // b = a*s + e + m
    let mut e = Ro::Element::zero();
    RandomGaussianDist::random_fill(rng, &operator.modulus(), &mut e);
    let b = operator.add(&operator.add(&sa, &e), m);
    lwe_out.as_mut()[0] = b;
}

pub fn decrypt_lwe<Ro: Row, Op: ArithmeticOps<Element = Ro::Element>, S>(
    lwe_ct: &Ro,
    s: &[S],
    operator: &Op,
) -> Ro::Element
where
    Ro: TryConvertFrom<[S], Parameters = Ro::Element>,
    Ro::Element: Zero,
{
    let s = Ro::try_convert_from(s, &operator.modulus());

    let mut sa = Ro::Element::zero();
    izip!(lwe_ct.as_ref().iter().skip(1), s.as_ref()).for_each(|(ai, si)| {
        let tmp = operator.mul(ai, si);
        sa = operator.add(&tmp, &sa);
    });

    let b = &lwe_ct.as_ref()[0];
    operator.sub(b, &sa)
}

/// Measures noise in input LWE ciphertext with reference of `ideal_m`
///
/// - ct: Input LWE ciphertext
/// - s: corresponding secret
/// - ideal_m: Ideal `encoded` message
pub(crate) fn measure_noise_lwe<Ro: Row, Op: ArithmeticOps<Element = Ro::Element>, S>(
    ct: &Ro,
    s: &[S],
    operator: &Op,
    ideal_m: &Ro::Element,
) -> f64
where
    Ro: TryConvertFrom<[S], Parameters = Ro::Element>,
    Ro::Element: Zero + ToPrimitive + PrimInt + Display,
{
    assert!(s.len() == ct.as_ref().len() - 1,);

    let s = Ro::try_convert_from(s, &operator.modulus());
    let mut sa = Ro::Element::zero();
    izip!(s.as_ref().iter(), ct.as_ref().iter().skip(1)).for_each(|(si, ai)| {
        sa = operator.add(&sa, &operator.mul(si, ai));
    });
    let m = operator.sub(&ct.as_ref()[0], &sa);

    let mut diff = operator.sub(&m, ideal_m);
    let q = operator.modulus();
    if diff > (q >> 1) {
        diff = q - diff;
    }
    return diff.to_f64().unwrap().log2();
}

#[cfg(test)]
mod tests {

    use crate::{
        backend::{ModInit, ModularOpsU64},
        decomposer::DefaultDecomposer,
        lwe::{lwe_key_switch, measure_noise_lwe},
        random::DefaultSecureRng,
        rgsw::measure_noise,
        Secret,
    };

    use super::{
        decrypt_lwe, encrypt_lwe, lwe_ksk_keygen, LweKeySwitchingKey, LweSecret,
        SeededLweKeySwitchingKey,
    };

    const K: usize = 50;

    #[test]
    fn encrypt_decrypt_works() {
        let logq = 16;
        let q = 1u64 << logq;
        let lwe_n = 1024;
        let logp = 3;

        let modq_op = ModularOpsU64::new(q);
        let lwe_sk = LweSecret::random(lwe_n >> 1, lwe_n);

        let mut rng = DefaultSecureRng::new();

        // encrypt
        for m in 0..1u64 << logp {
            let encoded_m = m << (logq - logp);
            let mut lwe_ct = vec![0u64; lwe_n + 1];
            encrypt_lwe(
                &mut lwe_ct,
                &encoded_m,
                &lwe_sk.values(),
                &modq_op,
                &mut rng,
            );
            let encoded_m_back = decrypt_lwe(&lwe_ct, &lwe_sk.values(), &modq_op);
            let m_back = ((((encoded_m_back as f64) * ((1 << logp) as f64)) / q as f64).round()
                as u64)
                % (1u64 << logp);
            assert_eq!(m, m_back, "Expected {m} but got {m_back}");
        }
    }

    #[test]
    fn key_switch_works() {
        let logq = 18;
        let logp = 2;
        let q = 1u64 << logq;
        let lwe_in_n = 2048;
        let lwe_out_n = 493;
        let d_ks = 3;
        let logb = 6;

        let lwe_sk_in = LweSecret::random(lwe_in_n >> 1, lwe_in_n);
        let lwe_sk_out = LweSecret::random(lwe_out_n >> 1, lwe_out_n);

        let mut rng = DefaultSecureRng::new();
        let modq_op = ModularOpsU64::new(q);

        // genrate ksk
        for _ in 0..K {
            let mut ksk_seed = [0u8; 32];
            rng.fill_bytes(&mut ksk_seed);
            let mut seeded_ksk =
                SeededLweKeySwitchingKey::empty(lwe_in_n, lwe_out_n, d_ks, ksk_seed, q);
            let mut p_rng = DefaultSecureRng::new_seeded(ksk_seed);
            let decomposer = DefaultDecomposer::new(q, logb, d_ks);
            let gadget = decomposer.gadget_vector();
            lwe_ksk_keygen(
                &lwe_sk_in.values(),
                &lwe_sk_out.values(),
                &mut seeded_ksk.data,
                &gadget,
                &modq_op,
                &mut p_rng,
                &mut rng,
            );
            // println!("{:?}", ksk);
            let ksk = LweKeySwitchingKey::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_ksk);

            for m in 0..(1 << logp) {
                // encrypt using lwe_sk_in
                let encoded_m = m << (logq - logp);
                let mut lwe_in_ct = vec![0u64; lwe_in_n + 1];
                encrypt_lwe(
                    &mut lwe_in_ct,
                    &encoded_m,
                    lwe_sk_in.values(),
                    &modq_op,
                    &mut rng,
                );

                // key switch from lwe_sk_in to lwe_sk_out
                let decomposer = DefaultDecomposer::new(1u64 << logq, logb, d_ks);
                let mut lwe_out_ct = vec![0u64; lwe_out_n + 1];
                lwe_key_switch(
                    &mut lwe_out_ct,
                    &lwe_in_ct,
                    &ksk.data,
                    &modq_op,
                    &decomposer,
                );

                // decrypt lwe_out_ct using lwe_sk_out
                let encoded_m_back = decrypt_lwe(&lwe_out_ct, &lwe_sk_out.values(), &modq_op);
                let m_back = ((((encoded_m_back as f64) * ((1 << logp) as f64)) / q as f64).round()
                    as u64)
                    % (1u64 << logp);
                let noise =
                    measure_noise_lwe(&lwe_out_ct, lwe_sk_out.values(), &modq_op, &encoded_m);
                println!("Noise: {noise}");
                assert_eq!(m, m_back, "Expected {m} but got {m_back}");
                // dbg!(m, m_back);
                // dbg!(encoded_m, encoded_m_back);
            }
        }
    }
}
