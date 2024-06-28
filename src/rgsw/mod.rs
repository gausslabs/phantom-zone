use itertools::{izip, Itertools};
use num_traits::{PrimInt, Signed, ToPrimitive, Zero};
use std::{
    clone,
    fmt::Debug,
    iter,
    marker::PhantomData,
    ops::{Div, Neg, Sub},
};

use crate::{
    backend::Modulus,
    decomposer::{Decomposer, RlweDecomposer},
    ntt::{Ntt, NttInit},
    random::{DefaultSecureRng, NewWithSeed, RandomFillUniformInModulus},
    utils::{fill_random_ternary_secret_with_hamming_weight, ToShoup, WithLocal},
    Matrix, MatrixEntity, MatrixMut, Row, RowEntity, RowMut, Secret,
};

mod keygen;
mod runtime;

pub(crate) use keygen::*;
pub(crate) use runtime::*;

pub struct SeededAutoKey<M, S, Mod>
where
    M: Matrix,
{
    data: M,
    seed: S,
    modulus: Mod,
}

impl<M: Matrix + MatrixEntity, S, Mod: Modulus<Element = M::MatElement>> SeededAutoKey<M, S, Mod> {
    fn empty<D: Decomposer>(ring_size: usize, auto_decomposer: &D, seed: S, modulus: Mod) -> Self {
        SeededAutoKey {
            data: M::zeros(auto_decomposer.decomposition_count(), ring_size),
            seed,
            modulus,
        }
    }
}

pub struct AutoKeyEvaluationDomain<M: Matrix, Mod, R, N> {
    data: M,
    _phantom: PhantomData<(R, N)>,
    modulus: Mod,
}

impl<
        M: MatrixMut + MatrixEntity,
        Mod: Modulus<Element = M::MatElement> + Clone,
        R: RandomFillUniformInModulus<[M::MatElement], Mod> + NewWithSeed,
        N: NttInit<Mod> + Ntt<Element = M::MatElement>,
    > From<&SeededAutoKey<M, R::Seed, Mod>> for AutoKeyEvaluationDomain<M, Mod, R, N>
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
        izip!(data.iter_rows_mut().skip(d), value.data.iter_rows()).for_each(|(to_r, from_r)| {
            to_r.as_mut().copy_from_slice(from_r.as_ref());
        });

        // send RLWE'(-s(X^k)) polynomials to evaluation domain
        let ntt_op = N::new(&value.modulus, ring_size);
        data.iter_rows_mut()
            .for_each(|r| ntt_op.forward(r.as_mut()));

        AutoKeyEvaluationDomain {
            data,
            _phantom: PhantomData,
            modulus: value.modulus.clone(),
        }
    }
}

pub struct ShoupAutoKeyEvaluationDomain<M> {
    data: M,
}

impl<M: Matrix + ToShoup<Modulus = M::MatElement>, Mod: Modulus<Element = M::MatElement>, R, N>
    From<&AutoKeyEvaluationDomain<M, Mod, R, N>> for ShoupAutoKeyEvaluationDomain<M>
{
    fn from(value: &AutoKeyEvaluationDomain<M, Mod, R, N>) -> Self {
        Self {
            data: M::to_shoup(&value.data, value.modulus.q().unwrap()),
        }
    }
}

pub struct RgswCiphertext<M: Matrix, Mod> {
    /// Rgsw ciphertext polynomials
    pub(crate) data: M,
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
                decomposer.a().decomposition_count() * 2 + decomposer.b().decomposition_count() * 2,
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
    > From<&SeededRgswCiphertext<M, R::Seed, Mod>> for RgswCiphertextEvaluationDomain<M, Mod, R, N>
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

impl<M: Matrix, Mod, R, N> Matrix for RgswCiphertextEvaluationDomain<M, Mod, R, N> {
    type MatElement = M::MatElement;
    type R = M::R;

    fn dimension(&self) -> (usize, usize) {
        self.data.dimension()
    }

    fn fits(&self, row: usize, col: usize) -> bool {
        self.data.fits(row, col)
    }
}

impl<M: Matrix, Mod, R, N> AsRef<[M::R]> for RgswCiphertextEvaluationDomain<M, Mod, R, N> {
    fn as_ref(&self) -> &[M::R] {
        self.data.as_ref()
    }
}

pub struct ShoupRgswCiphertextEvaluationDomain<M> {
    pub(crate) data: M,
}

impl<M: Matrix + ToShoup<Modulus = M::MatElement>, Mod: Modulus<Element = M::MatElement>, R, N>
    From<&RgswCiphertextEvaluationDomain<M, Mod, R, N>> for ShoupRgswCiphertextEvaluationDomain<M>
{
    fn from(value: &RgswCiphertextEvaluationDomain<M, Mod, R, N>) -> Self {
        Self {
            data: M::to_shoup(&value.data, value.modulus.q().unwrap()),
        }
    }
}

pub struct SeededRlweCiphertext<R, S, Mod> {
    pub(crate) data: R,
    pub(crate) seed: S,
    pub(crate) modulus: Mod,
}

impl<R: RowEntity, S, Mod> SeededRlweCiphertext<R, S, Mod> {
    pub(crate) fn empty(ring_size: usize, seed: S, modulus: Mod) -> Self {
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
        RandomFillUniformInModulus::random_fill(&mut p_rng, &value.modulus, data.get_row_mut(0));

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
        RandomFillUniformInModulus::random_fill(&mut p_rng, &value.modulus, data.get_row_mut(0));

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

#[cfg(test)]
pub(crate) mod tests {
    use std::{clone, marker::PhantomData, ops::Mul, vec};

    use itertools::{izip, Itertools};
    use rand::{thread_rng, Rng};

    use crate::{
        backend::{GetModulus, ModInit, ModularOpsU64, Modulus, VectorOps},
        decomposer::{Decomposer, DefaultDecomposer, RlweDecomposer},
        ntt::{Ntt, NttBackendU64, NttInit},
        random::{DefaultSecureRng, RandomFillGaussianInModulus, RandomFillUniformInModulus},
        rgsw::{
            galois_auto_shoup, rlwe_by_rgsw_shoup, ShoupAutoKeyEvaluationDomain,
            ShoupRgswCiphertextEvaluationDomain,
        },
        utils::{generate_prime, negacyclic_mul, tests::Stats, TryConvertFrom1},
        Matrix, MatrixMut, Secret,
    };

    use super::{
        keygen::{
            decrypt_rlwe, generate_auto_map, measure_noise, public_key_encrypt_rgsw,
            rlwe_public_key, secret_key_encrypt_rgsw, seeded_auto_key_gen,
            seeded_secret_key_encrypt_rlwe,
        },
        runtime::{rgsw_by_rgsw_inplace, rlwe_auto, rlwe_by_rgsw},
        AutoKeyEvaluationDomain, RgswCiphertext, RgswCiphertextEvaluationDomain, RlweCiphertext,
        RlwePublicKey, RlweSecret, SeededAutoKey, SeededRgswCiphertext, SeededRlweCiphertext,
        SeededRlwePublicKey,
    };

    pub(crate) fn _sk_encrypt_rlwe<T: Modulus<Element = u64> + Clone>(
        m: &[u64],
        s: &[i32],
        ntt_op: &NttBackendU64,
        mod_op: &ModularOpsU64<T>,
    ) -> RlweCiphertext<Vec<Vec<u64>>, DefaultSecureRng> {
        let ring_size = m.len();
        let q = mod_op.modulus();
        assert!(s.len() == ring_size);

        let mut rng = DefaultSecureRng::new();
        let mut rlwe_seed = [0u8; 32];
        rng.fill_bytes(&mut rlwe_seed);
        let mut seeded_rlwe_ct =
            SeededRlweCiphertext::<_, [u8; 32], _>::empty(ring_size as usize, rlwe_seed, q.clone());
        let mut p_rng = DefaultSecureRng::new_seeded(rlwe_seed);
        seeded_secret_key_encrypt_rlwe(
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
    pub(crate) fn _pk_encrypt_rgsw<T: Modulus<Element = u64> + Clone>(
        m: &[u64],
        public_key: &RlwePublicKey<Vec<Vec<u64>>, DefaultSecureRng>,
        decomposer: &(DefaultDecomposer<u64>, DefaultDecomposer<u64>),
        mod_op: &ModularOpsU64<T>,
        ntt_op: &NttBackendU64,
    ) -> RgswCiphertext<Vec<Vec<u64>>, T> {
        let (_, ring_size) = Matrix::dimension(&public_key.data);
        let gadget_vector_a = decomposer.a().gadget_vector();
        let gadget_vector_b = decomposer.b().gadget_vector();

        let mut rng = DefaultSecureRng::new();

        assert!(m.len() == ring_size);

        // public key encrypt RGSW(m1)
        let mut rgsw_ct = RgswCiphertext::empty(ring_size, decomposer, mod_op.modulus().clone());
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
    pub(crate) fn _sk_encrypt_rgsw<T: Modulus<Element = u64> + Clone>(
        m: &[u64],
        s: &[i32],
        decomposer: &(DefaultDecomposer<u64>, DefaultDecomposer<u64>),
        mod_op: &ModularOpsU64<T>,
        ntt_op: &NttBackendU64,
    ) -> SeededRgswCiphertext<Vec<Vec<u64>>, [u8; 32], T> {
        let ring_size = s.len();
        assert!(m.len() == s.len());

        let q = mod_op.modulus();

        let gadget_vector_a = decomposer.a().gadget_vector();
        let gadget_vector_b = decomposer.b().gadget_vector();

        let mut rng = DefaultSecureRng::new();
        let mut rgsw_seed = [0u8; 32];
        rng.fill_bytes(&mut rgsw_seed);
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
        let mut pk_seed = [0u8; 32];
        rng.fill_bytes(&mut pk_seed);
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
        let pk = RlwePublicKey::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_pk);

        // Encrypt m1 as RGSW(m1)
        let rgsw_ct = {
            //TODO(Jay): Figure out better way to test secret key and public key variant of
            // RGSW ciphertext encryption within the same test

            if true {
                // Encryption m1 as RGSW(m1) using secret key
                let seeded_rgsw_ct =
                    _sk_encrypt_rgsw(&m1, s.values(), &decomposer, &mod_op, &ntt_op);
                RgswCiphertextEvaluationDomain::<Vec<Vec<u64>>, _,DefaultSecureRng, NttBackendU64>::from(&seeded_rgsw_ct)
            } else {
                // Encrypt m1 as RGSW(m1) using public key
                let rgsw_ct = _pk_encrypt_rgsw(&m1, &pk, &decomposer, &mod_op, &ntt_op);
                RgswCiphertextEvaluationDomain::<_, _, DefaultSecureRng, NttBackendU64>::from(
                    &rgsw_ct,
                )
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

        // rlwe x rgsw with additional RGSW ciphertexts in shoup repr
        let rlwe_in_ct_shoup = {
            let mut rlwe_in_ct_shoup = RlweCiphertext::<_, DefaultSecureRng> {
                data: rlwe_in_ct.data.clone(),
                is_trivial: rlwe_in_ct.is_trivial,
                _phatom: PhantomData::default(),
            };

            let rgsw_ct_shoup = ShoupRgswCiphertextEvaluationDomain::from(&rgsw_ct);

            rlwe_by_rgsw_shoup(
                &mut rlwe_in_ct_shoup,
                &rgsw_ct.data,
                &rgsw_ct_shoup.data,
                &mut scratch_space,
                &decomposer,
                &ntt_op,
                &mod_op,
            );

            rlwe_in_ct_shoup
        };

        // rlwe x rgsw normal
        {
            rlwe_by_rgsw(
                &mut rlwe_in_ct,
                &rgsw_ct.data,
                &mut scratch_space,
                &decomposer,
                &ntt_op,
                &mod_op,
            );
        }

        // output from both functions must be equal
        {
            assert_eq!(rlwe_in_ct.data, rlwe_in_ct_shoup.data);
        }

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
        let mut seed_rlwe = [0u8; 32];
        rng.fill_bytes(&mut seed_rlwe);
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

        // Generate galois key to key switch from s^k to s
        let decomposer = DefaultDecomposer::new(q, logb, d_rgsw);
        let mut seed_auto = [0u8; 32];
        rng.fill_bytes(&mut seed_auto);
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
            AutoKeyEvaluationDomain::<Vec<Vec<u64>>, _, DefaultSecureRng, NttBackendU64>::from(
                &seeded_auto_key,
            );

        // Send RLWE_{s}(m) -> RLWE_{s}(m^k)
        let mut scratch_space = vec![vec![0u64; ring_size as usize]; d_rgsw + 2];
        let (auto_map_index, auto_map_sign) = generate_auto_map(ring_size as usize, auto_k);

        // galois auto with additional auto key in shoup repr
        let rlwe_m_shoup = {
            let auto_key_shoup = ShoupAutoKeyEvaluationDomain::from(&auto_key);
            let mut rlwe_m_shoup = RlweCiphertext::<_, DefaultSecureRng> {
                data: rlwe_m.data.clone(),
                is_trivial: rlwe_m.is_trivial,
                _phatom: PhantomData::default(),
            };
            galois_auto_shoup(
                &mut rlwe_m_shoup,
                &auto_key.data,
                &auto_key_shoup.data,
                &mut scratch_space,
                &auto_map_index,
                &auto_map_sign,
                &mod_op,
                &ntt_op,
                &decomposer,
            );
            rlwe_m_shoup
        };

        // normal galois auto
        {
            rlwe_auto(
                &mut rlwe_m,
                &auto_key.data,
                &mut scratch_space,
                &auto_map_index,
                &auto_map_sign,
                &mod_op,
                &ntt_op,
                &decomposer,
            );
        }

        // rlwe out from both functions must be same
        assert_eq!(rlwe_m.data, rlwe_m_shoup.data);

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
        let d_rgsw = 12;
        let logb = 5;

        let s = RlweSecret::random((ring_size >> 1) as usize, ring_size as usize);

        let mut rng = DefaultSecureRng::new();
        let ntt_op = NttBackendU64::new(&q, ring_size as usize);
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
            decomposer.a().decomposition_count() * 2
                + decomposer.b().decomposition_count() * 2
                + std::cmp::max(
                    decomposer.a().decomposition_count(),
                    decomposer.b().decomposition_count()
                )
        ];

        // _measure_noise_rgsw(&rgsw_carrym, &carry_m, s.values(), &decomposer, &q);

        for i in 0..2 {
            let mut m = vec![0u64; ring_size as usize];
            m[thread_rng().gen_range(0..ring_size) as usize] = if (i & 1) == 1 { q - 1 } else { 1 };
            let rgsw_m =
                RgswCiphertextEvaluationDomain::<_, _, DefaultSecureRng, NttBackendU64>::from(
                    &_sk_encrypt_rgsw(&m, s.values(), &decomposer, &mod_op, &ntt_op),
                );
            rgsw_by_rgsw_inplace(
                &mut rgsw_carrym,
                decomposer.a().decomposition_count(),
                decomposer.b().decomposition_count(),
                &rgsw_m.data,
                &decomposer,
                &mut scratch_matrix,
                &ntt_op,
                &mod_op,
            );

            // measure noise
            carry_m = negacyclic_mul(&carry_m, &m, mul_mod, q);
            println!("########### Noise RGSW(carrym) in {i}^th loop ###########");
            // _measure_noise_rgsw(&rgsw_carrym, &carry_m, s.values(),
            // &decomposer, &q);
        }
        {
            // RLWE(m) x RGSW(carry_m)
            let mut m = vec![0u64; ring_size as usize];
            RandomFillUniformInModulus::random_fill(&mut rng, &q, m.as_mut_slice());
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
        let logq = 55;
        let ring_size = 1 << 11;
        let q = generate_prime(logq, ring_size as u64, 1u64 << logq).unwrap();
        let d = 2;
        let logb = 12;
        let decomposer = DefaultDecomposer::new(q, logb, d);

        let ntt_op = NttBackendU64::new(&q, ring_size as usize);
        let mod_op = ModularOpsU64::new(q);
        let mut rng = DefaultSecureRng::new();

        let mut stats = Stats::new();

        for _ in 0..10 {
            let mut a = vec![0u64; ring_size];
            RandomFillUniformInModulus::random_fill(&mut rng, &q, a.as_mut());
            let mut m = vec![0u64; ring_size];
            RandomFillGaussianInModulus::random_fill(&mut rng, &q, m.as_mut());

            let mut sk = vec![0u64; ring_size];
            RandomFillGaussianInModulus::random_fill(&mut rng, &q, sk.as_mut());
            let mut sk_eval = sk.clone();
            ntt_op.forward(sk_eval.as_mut_slice());

            let gadget_vector = decomposer.gadget_vector();

            // ksk (beta e)
            let mut ksk_part_b = vec![vec![0u64; ring_size]; decomposer.decomposition_count()];
            let mut ksk_part_a = vec![vec![0u64; ring_size]; decomposer.decomposition_count()];
            izip!(
                ksk_part_b.iter_rows_mut(),
                ksk_part_a.iter_rows_mut(),
                gadget_vector.iter()
            )
            .for_each(|(part_b, part_a, beta)| {
                RandomFillUniformInModulus::random_fill(&mut rng, &q, part_a.as_mut());

                // a * s
                let mut tmp = part_a.to_vec();
                ntt_op.forward(tmp.as_mut());
                mod_op.elwise_mul_mut(tmp.as_mut(), sk_eval.as_ref());
                ntt_op.backward(tmp.as_mut());

                // a*s + e + beta m
                RandomFillGaussianInModulus::random_fill(&mut rng, &q, part_b.as_mut());
                // println!("E: {:?}", &part_b);
                // a*s + e
                mod_op.elwise_add_mut(part_b.as_mut_slice(), tmp.as_ref());
                // a*s + e + beta m
                let mut tmp = m.to_vec();
                mod_op.elwise_scalar_mul_mut(tmp.as_mut_slice(), beta);
                mod_op.elwise_add_mut(part_b.as_mut_slice(), tmp.as_ref());
            });

            // decompose a
            let mut decomposed_a = vec![vec![0u64; ring_size]; decomposer.decomposition_count()];
            a.iter().enumerate().for_each(|(ri, el)| {
                decomposer
                    .decompose_iter(el)
                    .into_iter()
                    .enumerate()
                    .for_each(|(j, d_el)| {
                        decomposed_a[j][ri] = d_el;
                    });
            });

            // println!("Last limb");

            // decomp_a * ksk(beta m)
            ksk_part_b
                .iter_mut()
                .for_each(|r| ntt_op.forward(r.as_mut_slice()));
            ksk_part_a
                .iter_mut()
                .for_each(|r| ntt_op.forward(r.as_mut_slice()));
            decomposed_a
                .iter_mut()
                .for_each(|r| ntt_op.forward(r.as_mut_slice()));
            let mut out = vec![vec![0u64; ring_size]; 2];
            izip!(decomposed_a.iter(), ksk_part_b.iter(), ksk_part_a.iter()).for_each(
                |(d_a, part_b, part_a)| {
                    // out_a += d_a * part_a
                    let mut d_a_clone = d_a.clone();
                    mod_op.elwise_mul_mut(d_a_clone.as_mut_slice(), part_a.as_ref());
                    mod_op.elwise_add_mut(out[0].as_mut_slice(), d_a_clone.as_ref());

                    // out_b += d_a * part_b
                    let mut d_a_clone = d_a.clone();
                    mod_op.elwise_mul_mut(d_a_clone.as_mut_slice(), part_b.as_ref());
                    mod_op.elwise_add_mut(out[1].as_mut_slice(), d_a_clone.as_ref());
                },
            );
            out.iter_mut()
                .for_each(|r| ntt_op.backward(r.as_mut_slice()));

            let out_back = {
                // decrypt
                // a*s
                ntt_op.forward(out[0].as_mut());
                mod_op.elwise_mul_mut(out[0].as_mut(), sk_eval.as_ref());
                ntt_op.backward(out[0].as_mut());

                // b - a*s
                let tmp = (out[0]).clone();
                mod_op.elwise_sub_mut(out[1].as_mut(), tmp.as_ref());
                out.remove(1)
            };

            let out_expected = {
                let mut a_clone = a.clone();
                let mut m_clone = m.clone();

                ntt_op.forward(a_clone.as_mut_slice());
                ntt_op.forward(m_clone.as_mut_slice());

                mod_op.elwise_mul_mut(a_clone.as_mut_slice(), m_clone.as_mut_slice());
                ntt_op.backward(a_clone.as_mut_slice());
                a_clone
            };

            let mut diff = out_expected;
            mod_op.elwise_sub_mut(diff.as_mut_slice(), out_back.as_ref());
            stats.add_more(&Vec::<i64>::try_convert_from(diff.as_ref(), &q));
        }

        println!("Std: {}", stats.std_dev().abs().log2());
    }
}
