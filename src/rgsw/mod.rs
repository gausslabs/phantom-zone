mod keygen;
mod runtime;

pub(crate) use keygen::*;
pub(crate) use runtime::*;

#[cfg(test)]
pub(crate) mod tests {
    use std::{fmt::Debug, marker::PhantomData, vec};

    use itertools::{izip, Itertools};
    use rand::{thread_rng, Rng};

    use crate::{
        backend::{GetModulus, ModInit, ModularOpsU64, Modulus, VectorOps},
        decomposer::{Decomposer, DefaultDecomposer, RlweDecomposer},
        ntt::{Ntt, NttBackendU64, NttInit},
        random::{DefaultSecureRng, NewWithSeed, RandomFillUniformInModulus},
        rgsw::{
            rlwe_auto_scratch_rows, rlwe_auto_shoup, rlwe_by_rgsw_shoup, rlwe_x_rgsw_scratch_rows,
            RgswCiphertextRef, RlweCiphertextMutRef, RlweKskRef, RuntimeScratchMutRef,
        },
        utils::{
            fill_random_ternary_secret_with_hamming_weight, generate_prime, negacyclic_mul,
            tests::Stats, ToShoup, TryConvertFrom1, WithLocal,
        },
        Matrix, MatrixEntity, MatrixMut, Row, RowEntity, RowMut, Secret,
    };

    use super::{
        keygen::{
            decrypt_rlwe, generate_auto_map, rlwe_public_key, secret_key_encrypt_rgsw,
            seeded_auto_key_gen, seeded_secret_key_encrypt_rlwe,
        },
        rgsw_x_rgsw_scratch_rows,
        runtime::{rgsw_by_rgsw_inplace, rlwe_auto, rlwe_by_rgsw},
        RgswCiphertextMutRef,
    };

    struct SeededAutoKey<M, S, Mod>
    where
        M: Matrix,
    {
        data: M,
        seed: S,
        modulus: Mod,
    }

    impl<M: Matrix + MatrixEntity, S, Mod: Modulus<Element = M::MatElement>> SeededAutoKey<M, S, Mod> {
        fn empty<D: Decomposer>(
            ring_size: usize,
            auto_decomposer: &D,
            seed: S,
            modulus: Mod,
        ) -> Self {
            SeededAutoKey {
                data: M::zeros(auto_decomposer.decomposition_count(), ring_size),
                seed,
                modulus,
            }
        }
    }

    struct AutoKeyEvaluationDomain<M: Matrix, R, N> {
        data: M,
        _phantom: PhantomData<(R, N)>,
    }

    impl<
            M: MatrixMut + MatrixEntity,
            Mod: Modulus<Element = M::MatElement> + Clone,
            R: RandomFillUniformInModulus<[M::MatElement], Mod> + NewWithSeed,
            N: NttInit<Mod> + Ntt<Element = M::MatElement>,
        > From<&SeededAutoKey<M, R::Seed, Mod>> for AutoKeyEvaluationDomain<M, R, N>
    where
        <M as Matrix>::R: RowMut,
        M::MatElement: Copy,

        R::Seed: Clone,
    {
        fn from(value: &SeededAutoKey<M, R::Seed, Mod>) -> Self {
            let (d, ring_size) = value.data.dimension();
            let mut data = M::zeros(2 * d, ring_size);

            // sample RLWE'_A(-s(X^k))
            let mut p_rng = R::new_with_seed(value.seed.clone());
            data.iter_rows_mut().take(d).for_each(|r| {
                RandomFillUniformInModulus::random_fill(&mut p_rng, &value.modulus, r.as_mut());
            });

            // copy over RLWE'_B(-s(X^k))
            izip!(data.iter_rows_mut().skip(d), value.data.iter_rows()).for_each(
                |(to_r, from_r)| {
                    to_r.as_mut().copy_from_slice(from_r.as_ref());
                },
            );

            // send RLWE'(-s(X^k)) polynomials to evaluation domain
            let ntt_op = N::new(&value.modulus, ring_size);
            data.iter_rows_mut()
                .for_each(|r| ntt_op.forward(r.as_mut()));

            AutoKeyEvaluationDomain {
                data,
                _phantom: PhantomData,
            }
        }
    }

    struct RgswCiphertext<M: Matrix, Mod> {
        /// Rgsw ciphertext polynomials
        data: M,
        modulus: Mod,
        /// Decomposition for RLWE part A
        d_a: usize,
        /// Decomposition for RLWE part B
        d_b: usize,
    }

    impl<M: MatrixEntity, Mod: Modulus<Element = M::MatElement>> RgswCiphertext<M, Mod> {
        pub(crate) fn empty<D: RlweDecomposer>(
            ring_size: usize,
            decomposer: &D,
            modulus: Mod,
        ) -> RgswCiphertext<M, Mod> {
            RgswCiphertext {
                data: M::zeros(
                    decomposer.a().decomposition_count() * 2
                        + decomposer.b().decomposition_count() * 2,
                    ring_size,
                ),
                d_a: decomposer.a().decomposition_count(),
                d_b: decomposer.b().decomposition_count(),
                modulus,
            }
        }
    }

    pub struct SeededRgswCiphertext<M, S, Mod>
    where
        M: Matrix,
    {
        pub(crate) data: M,
        seed: S,
        modulus: Mod,
        /// Decomposition for RLWE part A
        d_a: usize,
        /// Decomposition for RLWE part B
        d_b: usize,
    }

    impl<M: Matrix + MatrixEntity, S, Mod> SeededRgswCiphertext<M, S, Mod> {
        pub(crate) fn empty<D: RlweDecomposer>(
            ring_size: usize,
            decomposer: &D,
            seed: S,
            modulus: Mod,
        ) -> SeededRgswCiphertext<M, S, Mod> {
            SeededRgswCiphertext {
                data: M::zeros(
                    decomposer.a().decomposition_count() * 2 + decomposer.b().decomposition_count(),
                    ring_size,
                ),
                seed,
                modulus,
                d_a: decomposer.a().decomposition_count(),
                d_b: decomposer.b().decomposition_count(),
            }
        }
    }

    impl<M: Debug + Matrix, S: Debug, Mod: Debug> Debug for SeededRgswCiphertext<M, S, Mod>
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

    pub struct RgswCiphertextEvaluationDomain<M, Mod, R, N> {
        pub(crate) data: M,
        modulus: Mod,
        _phantom: PhantomData<(R, N)>,
    }

    impl<
            M: MatrixMut + MatrixEntity,
            Mod: Modulus<Element = M::MatElement> + Clone,
            R: NewWithSeed + RandomFillUniformInModulus<[M::MatElement], Mod>,
            N: NttInit<Mod> + Ntt<Element = M::MatElement> + Debug,
        > From<&SeededRgswCiphertext<M, R::Seed, Mod>>
        for RgswCiphertextEvaluationDomain<M, Mod, R, N>
    where
        <M as Matrix>::R: RowMut,
        M::MatElement: Copy,
        R::Seed: Clone,
        M: Debug,
    {
        fn from(value: &SeededRgswCiphertext<M, R::Seed, Mod>) -> Self {
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
            let nttop = N::new(&value.modulus, ring_size);
            data.iter_rows_mut()
                .for_each(|ri| nttop.forward(ri.as_mut()));

            Self {
                data: data,
                modulus: value.modulus.clone(),
                _phantom: PhantomData,
            }
        }
    }

    impl<
            M: MatrixMut + MatrixEntity,
            Mod: Modulus<Element = M::MatElement> + Clone,
            R,
            N: NttInit<Mod> + Ntt<Element = M::MatElement>,
        > From<&RgswCiphertext<M, Mod>> for RgswCiphertextEvaluationDomain<M, Mod, R, N>
    where
        <M as Matrix>::R: RowMut,
        M::MatElement: Copy,
        M: Debug,
    {
        fn from(value: &RgswCiphertext<M, Mod>) -> Self {
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
            let nttop = N::new(&value.modulus, ring_size);
            data.iter_rows_mut()
                .for_each(|ri| nttop.forward(ri.as_mut()));

            Self {
                data: data,
                modulus: value.modulus.clone(),
                _phantom: PhantomData,
            }
        }
    }

    impl<M: Debug, Mod: Debug, R, N> Debug for RgswCiphertextEvaluationDomain<M, Mod, R, N> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RgswCiphertextEvaluationDomain")
                .field("data", &self.data)
                .field("modulus", &self.modulus)
                .field("_phantom", &self._phantom)
                .finish()
        }
    }

    struct SeededRlweCiphertext<R, S, Mod> {
        data: R,
        seed: S,
        modulus: Mod,
    }

    impl<R: RowEntity, S, Mod> SeededRlweCiphertext<R, S, Mod> {
        fn empty(ring_size: usize, seed: S, modulus: Mod) -> Self {
            SeededRlweCiphertext {
                data: R::zeros(ring_size),
                seed,
                modulus,
            }
        }
    }

    pub struct RlweCiphertext<M, Rng> {
        data: M,
        _phatom: PhantomData<Rng>,
    }

    impl<
            R: Row,
            M: MatrixEntity<R = R, MatElement = R::Element> + MatrixMut,
            Rng: NewWithSeed + RandomFillUniformInModulus<[M::MatElement], Mod>,
            Mod: Modulus<Element = R::Element>,
        > From<&SeededRlweCiphertext<R, Rng::Seed, Mod>> for RlweCiphertext<M, Rng>
    where
        Rng::Seed: Clone,
        <M as Matrix>::R: RowMut,
        R::Element: Copy,
    {
        fn from(value: &SeededRlweCiphertext<R, Rng::Seed, Mod>) -> Self {
            let mut data = M::zeros(2, value.data.as_ref().len());

            // sample a
            let mut p_rng = Rng::new_with_seed(value.seed.clone());
            RandomFillUniformInModulus::random_fill(
                &mut p_rng,
                &value.modulus,
                data.get_row_mut(0),
            );

            data.get_row_mut(1).copy_from_slice(value.data.as_ref());

            RlweCiphertext {
                data,
                _phatom: PhantomData,
            }
        }
    }

    struct SeededRlwePublicKey<Ro: Row, S> {
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

    struct RlwePublicKey<M, R> {
        data: M,
        _phantom: PhantomData<R>,
    }

    impl<
            M: MatrixMut + MatrixEntity,
            Rng: NewWithSeed + RandomFillUniformInModulus<[M::MatElement], M::MatElement>,
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
            RandomFillUniformInModulus::random_fill(
                &mut p_rng,
                &value.modulus,
                data.get_row_mut(0),
            );

            // copy over b
            data.get_row_mut(1).copy_from_slice(value.data.as_ref());

            Self {
                data,
                _phantom: PhantomData,
            }
        }
    }

    #[derive(Clone)]
    struct RlweSecret {
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

    fn random_seed() -> [u8; 32] {
        let mut rng = DefaultSecureRng::new();
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        seed
    }

    /// Encrypts m as RGSW ciphertext RGSW(m) using supplied secret key. Returns
    /// seeded RGSW ciphertext in coefficient domain
    fn sk_encrypt_rgsw<T: Modulus<Element = u64> + Clone>(
        m: &[u64],
        s: &[i32],
        decomposer: &(DefaultDecomposer<u64>, DefaultDecomposer<u64>),
        mod_op: &ModularOpsU64<T>,
        ntt_op: &NttBackendU64,
    ) -> SeededRgswCiphertext<Vec<Vec<u64>>, [u8; 32], T> {
        let ring_size = s.len();
        assert!(m.len() == s.len());

        let mut rng = DefaultSecureRng::new();

        let q = mod_op.modulus();
        let rgsw_seed = random_seed();
        let mut seeded_rgsw_ct = SeededRgswCiphertext::<Vec<Vec<u64>>, [u8; 32], T>::empty(
            ring_size as usize,
            decomposer,
            rgsw_seed,
            q.clone(),
        );
        let mut p_rng = DefaultSecureRng::new_seeded(rgsw_seed);
        secret_key_encrypt_rgsw(
            &mut seeded_rgsw_ct.data,
            m,
            &decomposer.a().gadget_vector(),
            &decomposer.b().gadget_vector(),
            s,
            mod_op,
            ntt_op,
            &mut p_rng,
            &mut rng,
        );
        seeded_rgsw_ct
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
        RandomFillUniformInModulus::<[u64], u64>::random_fill(
            &mut rng,
            &(1u64 << logp),
            m0.as_mut_slice(),
        );

        let ntt_op = NttBackendU64::new(&q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);

        // encrypt m0
        let encoded_m = m0
            .iter()
            .map(|v| (((*v as f64) * q as f64) / (p as f64)).round() as u64)
            .collect_vec();
        let seed = random_seed();
        let mut rlwe_in_ct =
            SeededRlweCiphertext::<Vec<u64>, _, _>::empty(ring_size as usize, seed, q);
        let mut p_rng = DefaultSecureRng::new_seeded(seed);
        seeded_secret_key_encrypt_rlwe(
            &encoded_m,
            &mut rlwe_in_ct.data,
            s.values(),
            &mod_op,
            &ntt_op,
            &mut p_rng,
            &mut rng,
        );
        let rlwe_in_ct = RlweCiphertext::<Vec<Vec<u64>>, DefaultSecureRng>::from(&rlwe_in_ct);

        let mut encoded_m_back = vec![0u64; ring_size as usize];
        decrypt_rlwe(
            &rlwe_in_ct.data,
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
    }

    #[test]
    fn rlwe_by_rgsw_works() {
        let logq = 50;
        let logp = 2;
        let ring_size = 1 << 4;
        let q = generate_prime(logq, ring_size, 1u64 << logq).unwrap();
        let p: u64 = 1u64 << logp;

        let mut rng = DefaultSecureRng::new_seeded([0u8; 32]);

        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        let mut m0 = vec![0u64; ring_size as usize];
        RandomFillUniformInModulus::<[u64], _>::random_fill(
            &mut rng,
            &(1u64 << logp),
            m0.as_mut_slice(),
        );
        let mut m1 = vec![0u64; ring_size as usize];
        m1[thread_rng().gen_range(0..ring_size) as usize] = 1;

        let ntt_op = NttBackendU64::new(&q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);
        let d_rgsw = 10;
        let logb = 5;
        let decomposer = (
            DefaultDecomposer::new(q, logb, d_rgsw),
            DefaultDecomposer::new(q, logb, d_rgsw),
        );

        // create public key
        let pk_seed = random_seed();
        let mut pk_prng = DefaultSecureRng::new_seeded(pk_seed);
        let mut seeded_pk =
            SeededRlwePublicKey::<Vec<u64>, _>::empty(ring_size as usize, pk_seed, q);
        rlwe_public_key(
            &mut seeded_pk.data,
            s.values(),
            &ntt_op,
            &mod_op,
            &mut pk_prng,
            &mut rng,
        );
        // let pk = RlwePublicKey::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_pk);

        // Encrypt m1 as RGSW(m1)
        let rgsw_ct = {
            // Encryption m1 as RGSW(m1) using secret key
            let seeded_rgsw_ct = sk_encrypt_rgsw(&m1, s.values(), &decomposer, &mod_op, &ntt_op);
            RgswCiphertextEvaluationDomain::<Vec<Vec<u64>>, _,DefaultSecureRng, NttBackendU64>::from(&seeded_rgsw_ct)
        };

        // Encrypt m0 as RLWE(m0)
        let mut rlwe_in_ct = {
            let encoded_m = m0
                .iter()
                .map(|v| (((*v as f64) * q as f64) / (p as f64)).round() as u64)
                .collect_vec();

            let seed = random_seed();
            let mut p_rng = DefaultSecureRng::new_seeded(seed);
            let mut seeded_rlwe = SeededRlweCiphertext::empty(ring_size as usize, seed, q);
            seeded_secret_key_encrypt_rlwe(
                &encoded_m,
                &mut seeded_rlwe.data,
                s.values(),
                &mod_op,
                &ntt_op,
                &mut p_rng,
                &mut rng,
            );
            RlweCiphertext::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_rlwe)
        };

        // RLWE(m0m1) = RLWE(m0) x RGSW(m1)
        let mut scratch_space =
            vec![vec![0u64; ring_size as usize]; rlwe_x_rgsw_scratch_rows(&decomposer)];

        // rlwe x rgsw with with soup repr
        let rlwe_in_ct_shoup = {
            let mut rlwe_in_ct_shoup = rlwe_in_ct.data.clone();

            let rgsw_ct_shoup = ToShoup::to_shoup(&rgsw_ct.data, q);

            rlwe_by_rgsw_shoup(
                &mut RlweCiphertextMutRef::new(rlwe_in_ct_shoup.as_mut()),
                &RgswCiphertextRef::new(
                    rgsw_ct.data.as_ref(),
                    decomposer.a().decomposition_count(),
                    decomposer.b().decomposition_count(),
                ),
                &RgswCiphertextRef::new(
                    rgsw_ct_shoup.as_ref(),
                    decomposer.a().decomposition_count(),
                    decomposer.b().decomposition_count(),
                ),
                &mut RuntimeScratchMutRef::new(scratch_space.as_mut()),
                &decomposer,
                &ntt_op,
                &mod_op,
                false,
            );

            rlwe_in_ct_shoup
        };

        // rlwe x rgsw normal
        {
            rlwe_by_rgsw(
                &mut RlweCiphertextMutRef::new(rlwe_in_ct.data.as_mut()),
                &RgswCiphertextRef::new(
                    rgsw_ct.data.as_ref(),
                    decomposer.a().decomposition_count(),
                    decomposer.b().decomposition_count(),
                ),
                &mut RuntimeScratchMutRef::new(scratch_space.as_mut()),
                &decomposer,
                &ntt_op,
                &mod_op,
                false,
            );
        }

        // output from both functions must be equal
        assert_eq!(rlwe_in_ct.data, rlwe_in_ct_shoup);

        // Decrypt RLWE(m0m1)
        let mut encoded_m0m1_back = vec![0u64; ring_size as usize];
        decrypt_rlwe(
            &rlwe_in_ct_shoup,
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
    fn rlwe_auto_works() {
        let logq = 55;
        let ring_size = 1 << 11;
        let q = generate_prime(logq, 2 * ring_size, 1u64 << logq).unwrap();
        let logp = 3;
        let p = 1u64 << logp;
        let d_rgsw = 5;
        let logb = 11;

        let mut rng = DefaultSecureRng::new();
        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        let mut m = vec![0u64; ring_size as usize];
        RandomFillUniformInModulus::random_fill(&mut rng, &p, m.as_mut_slice());
        let encoded_m = m
            .iter()
            .map(|v| (((*v as f64 * q as f64) / (p as f64)).round() as u64))
            .collect_vec();

        let ntt_op = NttBackendU64::new(&q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);

        // RLWE_{s}(m)
        let seed_rlwe = random_seed();
        let mut seeded_rlwe_m = SeededRlweCiphertext::empty(ring_size as usize, seed_rlwe, q);
        let mut p_rng = DefaultSecureRng::new_seeded(seed_rlwe);
        seeded_secret_key_encrypt_rlwe(
            &encoded_m,
            &mut seeded_rlwe_m.data,
            s.values(),
            &mod_op,
            &ntt_op,
            &mut p_rng,
            &mut rng,
        );
        let mut rlwe_m = RlweCiphertext::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_rlwe_m);

        let auto_k = -125;

        // Generate auto key to key switch from s^k to s
        let decomposer = DefaultDecomposer::new(q, logb, d_rgsw);
        let seed_auto = random_seed();
        let mut seeded_auto_key =
            SeededAutoKey::empty(ring_size as usize, &decomposer, seed_auto, q);
        let mut p_rng = DefaultSecureRng::new_seeded(seed_auto);
        let gadget_vector = decomposer.gadget_vector();
        seeded_auto_key_gen(
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
        let mut scratch_space =
            vec![vec![0; ring_size as usize]; rlwe_auto_scratch_rows(&decomposer)];
        let (auto_map_index, auto_map_sign) = generate_auto_map(ring_size as usize, auto_k);

        // galois auto with auto key in shoup repr
        let rlwe_m_shoup = {
            let auto_key_shoup = ToShoup::to_shoup(&auto_key.data, q);
            let mut rlwe_m_shoup = rlwe_m.data.clone();
            rlwe_auto_shoup(
                &mut RlweCiphertextMutRef::new(&mut rlwe_m_shoup),
                &RlweKskRef::new(&auto_key.data, decomposer.decomposition_count()),
                &RlweKskRef::new(&auto_key_shoup, decomposer.decomposition_count()),
                &mut RuntimeScratchMutRef::new(&mut scratch_space),
                &auto_map_index,
                &auto_map_sign,
                &mod_op,
                &ntt_op,
                &decomposer,
                false,
            );
            rlwe_m_shoup
        };

        // normal galois auto
        {
            rlwe_auto(
                &mut RlweCiphertextMutRef::new(rlwe_m.data.as_mut()),
                &RlweKskRef::new(auto_key.data.as_ref(), decomposer.decomposition_count()),
                &mut RuntimeScratchMutRef::new(scratch_space.as_mut()),
                &auto_map_index,
                &auto_map_sign,
                &mod_op,
                &ntt_op,
                &decomposer,
                false,
            );
        }

        // rlwe out from both functions must be same
        assert_eq!(rlwe_m.data, rlwe_m_shoup);

        let rlwe_m_k = rlwe_m;

        // Decrypt RLWE_{s}(m^k) and check
        let mut encoded_m_k_back = vec![0u64; ring_size as usize];
        decrypt_rlwe(
            &rlwe_m_k.data,
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

        // {
        //     let encoded_m_k = m_k
        //         .iter()
        //         .map(|v| ((*v as f64 * q as f64) / p as f64).round() as u64)
        //         .collect_vec();

        //     let noise = measure_noise(&rlwe_m_k, &encoded_m_k, &ntt_op, &mod_op,
        // s.values());     println!("Ksk noise: {noise}");
        // }

        assert_eq!(m_k_back, m_k);
    }

    /// Collect noise stats of RGSW ciphertext
    ///
    /// - rgsw_ct: RGSW ciphertext must be in coefficient domain
    fn rgsw_noise_stats<T: Modulus<Element = u64> + Clone>(
        rgsw_ct: &[Vec<u64>],
        m: &[u64],
        s: &[i32],
        decomposer: &(DefaultDecomposer<u64>, DefaultDecomposer<u64>),
        q: &T,
    ) -> Stats<i64> {
        let gadget_vector_a = decomposer.a().gadget_vector();
        let gadget_vector_b = decomposer.b().gadget_vector();
        let d_a = gadget_vector_a.len();
        let d_b = gadget_vector_b.len();
        let ring_size = s.len();
        assert!(Matrix::dimension(&rgsw_ct) == (d_a * 2 + d_b * 2, ring_size));
        assert!(m.len() == ring_size);

        let mod_op = ModularOpsU64::new(q.clone());
        let ntt_op = NttBackendU64::new(q, ring_size);

        let mul_mod =
            |a: &u64, b: &u64| ((*a as u128 * *b as u128) % q.q().unwrap() as u128) as u64;
        let s_poly = Vec::<u64>::try_convert_from(s, q);
        let mut neg_s = s_poly.clone();
        mod_op.elwise_neg_mut(neg_s.as_mut());
        let neg_sm0m1 = negacyclic_mul(&neg_s, &m, mul_mod, q.q().unwrap());

        let mut stats = Stats::new();

        // RLWE(\beta^j -s * m)
        for j in 0..d_a {
            let want_m = {
                // RLWE(\beta^j -s * m)
                let mut beta_neg_sm0m1 = vec![0u64; ring_size as usize];
                mod_op.elwise_scalar_mul(beta_neg_sm0m1.as_mut(), &neg_sm0m1, &gadget_vector_a[j]);
                beta_neg_sm0m1
            };

            let mut rlwe = vec![vec![0u64; ring_size as usize]; 2];
            rlwe[0].copy_from_slice(rgsw_ct.get_row_slice(j));
            rlwe[1].copy_from_slice(rgsw_ct.get_row_slice(d_a + j));

            let mut got_m = vec![0; ring_size];
            decrypt_rlwe(&rlwe, s, &mut got_m, &ntt_op, &mod_op);

            let mut diff = want_m;
            mod_op.elwise_sub_mut(diff.as_mut(), got_m.as_ref());
            stats.add_more(&Vec::<i64>::try_convert_from(&diff, q));
        }

        // RLWE(\beta^j  m)
        for j in 0..d_b {
            let want_m = {
                // RLWE(\beta^j  m)
                let mut beta_m0m1 = vec![0u64; ring_size as usize];
                mod_op.elwise_scalar_mul(beta_m0m1.as_mut(), &m, &gadget_vector_b[j]);
                beta_m0m1
            };

            let mut rlwe = vec![vec![0u64; ring_size as usize]; 2];
            rlwe[0].copy_from_slice(rgsw_ct.get_row_slice(d_a * 2 + j));
            rlwe[1].copy_from_slice(rgsw_ct.get_row_slice(d_a * 2 + d_b + j));

            let mut got_m = vec![0; ring_size];
            decrypt_rlwe(&rlwe, s, &mut got_m, &ntt_op, &mod_op);

            let mut diff = want_m;
            mod_op.elwise_sub_mut(diff.as_mut(), got_m.as_ref());
            stats.add_more(&Vec::<i64>::try_convert_from(&diff, q));
        }

        stats
    }

    #[test]
    fn print_noise_stats_rgsw_x_rgsw() {
        let logq = 60;
        let logp = 2;
        let ring_size = 1 << 11;
        let q = generate_prime(logq, ring_size, 1u64 << logq).unwrap();
        let d_rgsw = 12;
        let logb = 5;

        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        let ntt_op = NttBackendU64::new(&q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);
        let decomposer = (
            DefaultDecomposer::new(q, logb, d_rgsw),
            DefaultDecomposer::new(q, logb, d_rgsw),
        );

        let d_a = decomposer.a().decomposition_count();
        let d_b = decomposer.b().decomposition_count();

        let mul_mod = |a: &u64, b: &u64| ((*a as u128 * *b as u128) % q as u128) as u64;

        let mut carry_m = vec![0u64; ring_size as usize];
        carry_m[thread_rng().gen_range(0..ring_size) as usize] = 1 << logp;

        // RGSW(carry_m)
        let mut rgsw_carrym = {
            let seeded_rgsw = sk_encrypt_rgsw(&carry_m, s.values(), &decomposer, &mod_op, &ntt_op);
            let mut rgsw_eval =
                RgswCiphertextEvaluationDomain::<_, _, DefaultSecureRng, NttBackendU64>::from(
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
            rgsw_x_rgsw_scratch_rows(&decomposer, &decomposer)
        ];

        rgsw_noise_stats(&rgsw_carrym, &carry_m, s.values(), &decomposer, &q);

        for i in 0..8 {
            let mut m = vec![0u64; ring_size as usize];
            m[thread_rng().gen_range(0..ring_size) as usize] = if (i & 1) == 1 { q - 1 } else { 1 };
            let rgsw_m =
                RgswCiphertextEvaluationDomain::<_, _, DefaultSecureRng, NttBackendU64>::from(
                    &sk_encrypt_rgsw(&m, s.values(), &decomposer, &mod_op, &ntt_op),
                );

            rgsw_by_rgsw_inplace(
                &mut RgswCiphertextMutRef::new(rgsw_carrym.as_mut(), d_a, d_b),
                &RgswCiphertextRef::new(rgsw_m.data.as_ref(), d_a, d_b),
                &decomposer,
                &decomposer,
                &mut RuntimeScratchMutRef::new(scratch_matrix.as_mut()),
                &ntt_op,
                &mod_op,
            );

            // measure noise
            carry_m = negacyclic_mul(&carry_m, &m, mul_mod, q);
            let stats = rgsw_noise_stats(&rgsw_carrym, &carry_m, s.values(), &decomposer, &q);
            println!(
                "Log2 of noise std after {i} RGSW x RGSW: {}",
                stats.std_dev().abs().log2()
            );
        }
    }
}
