use std::fmt::Debug;

use itertools::izip;
use num_traits::Zero;

use crate::{
    backend::{ArithmeticOps, GetModulus, VectorOps},
    decomposer::Decomposer,
    random::{RandomFillUniformInModulus, RandomGaussianElementInModulus},
    utils::TryConvertFrom1,
    Matrix, Row, RowEntity, RowMut,
};

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
    assert!(
        lwe_ksk.dimension().0 == ((lwe_in.as_ref().len() - 1) * decomposer.decomposition_count().0)
    );
    assert!(lwe_out.as_ref().len() == lwe_ksk.dimension().1);

    let lwe_in_a_decomposed = lwe_in
        .as_ref()
        .iter()
        .skip(1)
        .flat_map(|ai| decomposer.decompose_iter(ai));
    izip!(lwe_in_a_decomposed, lwe_ksk.iter_rows()).for_each(|(ai_j, beta_ij_lwe)| {
        // let now = std::time::Instant::now();
        operator.elwise_fma_scalar_mut(lwe_out.as_mut(), beta_ij_lwe.as_ref(), &ai_j);
        // println!("Time elwise_fma_scalar_mut: {:?}", now.elapsed());
    });

    let out_b = operator.add(&lwe_out.as_ref()[0], &lwe_in.as_ref()[0]);
    lwe_out.as_mut()[0] = out_b;
}

pub(crate) fn seeded_lwe_ksk_keygen<
    Ro: RowMut + RowEntity,
    S,
    Op: VectorOps<Element = Ro::Element>
        + ArithmeticOps<Element = Ro::Element>
        + GetModulus<Element = Ro::Element>,
    R: RandomGaussianElementInModulus<Ro::Element, Op::M>,
    PR: RandomFillUniformInModulus<[Ro::Element], Op::M>,
>(
    from_lwe_sk: &[S],
    to_lwe_sk: &[S],
    gadget: &[Ro::Element],
    operator: &Op,
    p_rng: &mut PR,
    rng: &mut R,
) -> Ro
where
    Ro: TryConvertFrom1<[S], Op::M>,
    Ro::Element: Zero + Debug,
{
    let mut ksk_out = Ro::zeros(from_lwe_sk.len() * gadget.len());

    let d = gadget.len();

    let modulus = operator.modulus();
    let mut neg_sk_in_m = Ro::try_convert_from(from_lwe_sk, modulus);
    operator.elwise_neg_mut(neg_sk_in_m.as_mut());
    let sk_out_m = Ro::try_convert_from(to_lwe_sk, modulus);

    let mut scratch = Ro::zeros(to_lwe_sk.len());

    izip!(neg_sk_in_m.as_ref(), ksk_out.as_mut().chunks_mut(d)).for_each(
        |(neg_sk_in_si, d_lwes_partb)| {
            izip!(gadget.iter(), d_lwes_partb.into_iter()).for_each(|(beta, lwe_b)| {
                // sample `a`
                RandomFillUniformInModulus::random_fill(p_rng, &modulus, scratch.as_mut());

                // a * z
                let mut az = Ro::Element::zero();
                izip!(scratch.as_ref().iter(), sk_out_m.as_ref()).for_each(|(ai, si)| {
                    let ai_si = operator.mul(ai, si);
                    az = operator.add(&az, &ai_si);
                });

                // a*z + (-s_i)*\beta^j + e
                let mut b = operator.add(&az, &operator.mul(beta, neg_sk_in_si));
                let e = RandomGaussianElementInModulus::random(rng, &modulus);
                b = operator.add(&b, &e);

                *lwe_b = b;
            })
        },
    );

    ksk_out
}

/// Encrypts encoded message m as LWE ciphertext
pub(crate) fn encrypt_lwe<
    Ro: RowMut + RowEntity,
    Op: ArithmeticOps<Element = Ro::Element> + GetModulus<Element = Ro::Element>,
    R: RandomGaussianElementInModulus<Ro::Element, Op::M>
        + RandomFillUniformInModulus<[Ro::Element], Op::M>,
    S,
>(
    m: &Ro::Element,
    s: &[S],
    operator: &Op,
    rng: &mut R,
) -> Ro
where
    Ro: TryConvertFrom1<[S], Op::M>,
    Ro::Element: Zero,
{
    let s = Ro::try_convert_from(s, operator.modulus());
    let mut lwe_out = Ro::zeros(s.as_ref().len() + 1);

    // a*s
    RandomFillUniformInModulus::random_fill(rng, operator.modulus(), &mut lwe_out.as_mut()[1..]);
    let mut sa = Ro::Element::zero();
    izip!(lwe_out.as_mut().iter().skip(1), s.as_ref()).for_each(|(ai, si)| {
        let tmp = operator.mul(ai, si);
        sa = operator.add(&tmp, &sa);
    });

    // b = a*s + e + m
    let e = RandomGaussianElementInModulus::random(rng, operator.modulus());
    let b = operator.add(&operator.add(&sa, &e), m);
    lwe_out.as_mut()[0] = b;

    lwe_out
}

pub(crate) fn decrypt_lwe<
    Ro: Row,
    Op: ArithmeticOps<Element = Ro::Element> + GetModulus<Element = Ro::Element>,
    S,
>(
    lwe_ct: &Ro,
    s: &[S],
    operator: &Op,
) -> Ro::Element
where
    Ro: TryConvertFrom1<[S], Op::M>,
    Ro::Element: Zero,
{
    let s = Ro::try_convert_from(s, operator.modulus());

    let mut sa = Ro::Element::zero();
    izip!(lwe_ct.as_ref().iter().skip(1), s.as_ref()).for_each(|(ai, si)| {
        let tmp = operator.mul(ai, si);
        sa = operator.add(&tmp, &sa);
    });

    let b = &lwe_ct.as_ref()[0];
    operator.sub(b, &sa)
}

#[cfg(test)]
mod tests {

    use std::marker::PhantomData;

    use itertools::izip;

    use crate::{
        backend::{ModInit, ModulusPowerOf2},
        decomposer::DefaultDecomposer,
        random::{DefaultSecureRng, NewWithSeed},
        utils::{fill_random_ternary_secret_with_hamming_weight, WithLocal},
        MatrixEntity, MatrixMut, Secret,
    };

    use super::*;

    const K: usize = 50;

    #[derive(Clone)]
    struct LweSecret {
        pub(crate) values: Vec<i32>,
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

    struct LweKeySwitchingKey<M, R> {
        data: M,
        _phantom: PhantomData<R>,
    }

    impl<
            M: MatrixMut + MatrixEntity,
            R: NewWithSeed + RandomFillUniformInModulus<[M::MatElement], M::MatElement>,
        > From<&(M::R, R::Seed, usize, M::MatElement)> for LweKeySwitchingKey<M, R>
    where
        M::R: RowMut,
        R::Seed: Clone,
        M::MatElement: Copy,
    {
        fn from(value: &(M::R, R::Seed, usize, M::MatElement)) -> Self {
            let data_in = &value.0;
            let seed = &value.1;
            let to_lwe_n = value.2;
            let modulus = value.3;

            let mut p_rng = R::new_with_seed(seed.clone());
            let mut data = M::zeros(data_in.as_ref().len(), to_lwe_n + 1);
            izip!(data_in.as_ref().iter(), data.iter_rows_mut()).for_each(|(bi, lwe_i)| {
                RandomFillUniformInModulus::random_fill(
                    &mut p_rng,
                    &modulus,
                    &mut lwe_i.as_mut()[1..],
                );
                lwe_i.as_mut()[0] = *bi;
            });
            LweKeySwitchingKey {
                data,
                _phantom: PhantomData,
            }
        }
    }

    #[test]
    fn encrypt_decrypt_works() {
        let logq = 16;
        let q = 1u64 << logq;
        let lwe_n = 1024;
        let logp = 3;

        let modq_op = ModulusPowerOf2::new(q);
        let lwe_sk = LweSecret::random(lwe_n >> 1, lwe_n);

        let mut rng = DefaultSecureRng::new();

        // encrypt
        for m in 0..1u64 << logp {
            let encoded_m = m << (logq - logp);
            let lwe_ct =
                encrypt_lwe::<Vec<u64>, _, _, _>(&encoded_m, &lwe_sk.values(), &modq_op, &mut rng);
            let encoded_m_back = decrypt_lwe(&lwe_ct, &lwe_sk.values(), &modq_op);
            let m_back = ((((encoded_m_back as f64) * ((1 << logp) as f64)) / q as f64).round()
                as u64)
                % (1u64 << logp);
            assert_eq!(m, m_back, "Expected {m} but got {m_back}");
        }
    }

    #[test]
    fn key_switch_works() {
        let logq = 20;
        let logp = 2;
        let q = 1u64 << logq;
        let lwe_in_n = 2048;
        let lwe_out_n = 600;
        let d_ks = 5;
        let logb = 4;

        let lwe_sk_in = LweSecret::random(lwe_in_n >> 1, lwe_in_n);
        let lwe_sk_out = LweSecret::random(lwe_out_n >> 1, lwe_out_n);

        let mut rng = DefaultSecureRng::new();
        let modq_op = ModulusPowerOf2::new(q);

        // genrate ksk
        for _ in 0..1 {
            let mut ksk_seed = [0u8; 32];
            rng.fill_bytes(&mut ksk_seed);
            let mut p_rng = DefaultSecureRng::new_seeded(ksk_seed);
            let decomposer = DefaultDecomposer::new(q, logb, d_ks);
            let gadget = decomposer.gadget_vector();
            let seeded_ksk = seeded_lwe_ksk_keygen(
                &lwe_sk_in.values(),
                &lwe_sk_out.values(),
                &gadget,
                &modq_op,
                &mut p_rng,
                &mut rng,
            );
            // println!("{:?}", ksk);
            let ksk = LweKeySwitchingKey::<Vec<Vec<u64>>, DefaultSecureRng>::from(&(
                seeded_ksk, ksk_seed, lwe_out_n, q,
            ));

            for m in 0..(1 << logp) {
                // encrypt using lwe_sk_in
                let encoded_m = m << (logq - logp);
                let lwe_in_ct = encrypt_lwe(&encoded_m, lwe_sk_in.values(), &modq_op, &mut rng);

                // key switch from lwe_sk_in to lwe_sk_out
                let mut lwe_out_ct = vec![0u64; lwe_out_n + 1];
                let now = std::time::Instant::now();
                lwe_key_switch(
                    &mut lwe_out_ct,
                    &lwe_in_ct,
                    &ksk.data,
                    &modq_op,
                    &decomposer,
                );
                println!("Time: {:?}", now.elapsed());

                // decrypt lwe_out_ct using lwe_sk_out
                // TODO(Jay): Fix me
                // let encoded_m_back = decrypt_lwe(&lwe_out_ct,
                // &lwe_sk_out.values(), &modq_op); let m_back =
                // ((((encoded_m_back as f64) * ((1 << logp) as f64)) / q as
                // f64).round()     as u64)
                //     % (1u64 << logp);
                // let noise =
                //     measure_noise_lwe(&lwe_out_ct, lwe_sk_out.values(),
                // &modq_op, &encoded_m); println!("Noise:
                // {noise}"); assert_eq!(m, m_back, "Expected
                // {m} but got {m_back}"); dbg!(m, m_back);
                // dbg!(encoded_m, encoded_m_back);
            }
        }
    }
}
