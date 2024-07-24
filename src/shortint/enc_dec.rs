use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    bool::BoolEvaluator,
    random::{DefaultSecureRng, RandomFillUniformInModulus},
    utils::WithLocal,
    Decryptor, Encryptor, KeySwitchWithId, Matrix, MatrixEntity, MatrixMut, MultiPartyDecryptor,
    RowMut, SampleExtractor,
};

/// Fhe UInt8
///
/// Note that `Self.data` stores encryptions of bits in little endian (i.e least
/// signficant bit stored at 0th index and most signficant bit stores at 7th
/// index)
#[derive(Clone, Serialize, Deserialize)]
pub struct FheUint8<C> {
    pub(super) data: Vec<C>,
}

impl<C> FheUint8<C> {
    pub(super) fn data(&self) -> &[C] {
        &self.data
    }

    pub(super) fn data_mut(&mut self) -> &mut [C] {
        &mut self.data
    }
}

/// Stores a batch of Fhe Uint8 ciphertext as collection of unseeded RLWE
/// ciphertexts always encrypted under the ideal RLWE secret `s` of the MPC
/// protocol
///
/// To extract Fhe Uint8 ciphertext at `index` call `self.extract(index)`
#[derive(Clone, Serialize, Deserialize)]
pub struct BatchedFheUint8<C> {
    /// Vector of RLWE ciphertexts `C`
    data: Vec<C>,
    /// Count of FheUint8s packed in vector of RLWE ciphertexts
    count: usize,
}

impl<K, C> Encryptor<[u8], BatchedFheUint8<C>> for K
where
    K: Encryptor<[bool], Vec<C>>,
{
    /// Encrypt a batch of uint8s packed in vector of RLWE ciphertexts
    ///
    /// Uint8s can be extracted from `BatchedFheUint8` with `SampleExtractor`
    fn encrypt(&self, m: &[u8]) -> BatchedFheUint8<C> {
        let bool_m = m
            .iter()
            .flat_map(|v| {
                (0..8)
                    .into_iter()
                    .map(|i| ((*v >> i) & 1) == 1)
                    .collect_vec()
            })
            .collect_vec();
        let cts = K::encrypt(&self, &bool_m);
        BatchedFheUint8 {
            data: cts,
            count: m.len(),
        }
    }
}

impl<M: MatrixEntity + MatrixMut<MatElement = u64>> From<&SeededBatchedFheUint8<M::R, [u8; 32]>>
    for BatchedFheUint8<M>
where
    <M as Matrix>::R: RowMut,
{
    /// Unseeds collection of seeded RLWE ciphertext in SeededBatchedFheUint8
    /// and returns as `Self`
    fn from(value: &SeededBatchedFheUint8<M::R, [u8; 32]>) -> Self {
        BoolEvaluator::with_local(|e| {
            let parameters = e.parameters();
            let ring_size = parameters.rlwe_n().0;
            let rlwe_q = parameters.rlwe_q();

            let mut prng = DefaultSecureRng::new_seeded(value.seed);
            let rlwes = value
                .data
                .iter()
                .map(|partb| {
                    let mut rlwe = M::zeros(2, ring_size);

                    // sample A
                    RandomFillUniformInModulus::random_fill(&mut prng, rlwe_q, rlwe.get_row_mut(0));

                    // Copy over B
                    rlwe.get_row_mut(1).copy_from_slice(partb.as_ref());

                    rlwe
                })
                .collect_vec();
            Self {
                data: rlwes,
                count: value.count,
            }
        })
    }
}

impl<C, R> SampleExtractor<FheUint8<R>> for BatchedFheUint8<C>
where
    C: SampleExtractor<R>,
{
    /// Extract Fhe Uint8 ciphertext at `index`
    ///
    /// `Self` stores batch of Fhe uint8 ciphertext as vector of RLWE
    /// ciphertexts. Since Fhe uint8 ciphertext is collection of 8 bool
    /// ciphertexts, Fhe uint8 ciphertext at index `i` is stored in coefficients
    /// `i*8...(i+1)*8`. To extract Fhe uint8 at index `i`, sample extract bool
    /// ciphertext at indices `[i*8, ..., (i+1)*8)`
    fn extract_at(&self, index: usize) -> FheUint8<R> {
        assert!(index < self.count);
        BoolEvaluator::with_local(|e| {
            let ring_size = e.parameters().rlwe_n().0;

            let start_index = index * 8;
            let end_index = (index + 1) * 8;
            let data = (start_index..end_index)
                .map(|i| {
                    let rlwe_index = i / ring_size;
                    let coeff_index = i % ring_size;
                    self.data[rlwe_index].extract_at(coeff_index)
                })
                .collect_vec();
            FheUint8 { data }
        })
    }

    /// Extracts all FheUint8s packed in vector of RLWE ciphertexts of `Self`
    fn extract_all(&self) -> Vec<FheUint8<R>> {
        (0..self.count)
            .map(|index| self.extract_at(index))
            .collect_vec()
    }

    /// Extracts first `how_many` FheUint8s packed in vector of RLWE
    /// ciphertexts of `Self`
    fn extract_many(&self, how_many: usize) -> Vec<FheUint8<R>> {
        (0..how_many)
            .map(|index| self.extract_at(index))
            .collect_vec()
    }
}

/// Stores a batch of FheUint8s packed in a collection unseeded RLWE ciphertexts
///
/// `Self` stores unseeded RLWE ciphertexts encrypted under user's RLWE secret
/// `u_j` and is different from `BatchFheUint8` which stores collection of RLWE
/// ciphertexts under ideal RLWE secret `s` of the (non-interactive/interactive)
/// MPC protocol.
///
/// To extract FheUint8s from `Self`'s collection of RLWE ciphertexts, first
/// switch `Self` to `BatchFheUint8` with `key_switch(user_id)` where `user_id`
/// is user's id. This key switches collection of RLWE ciphertexts from
/// user's RLWE secret `u_j` to ideal RLWE secret `s` of the MPC protocol. Then
/// proceed to use `SampleExtract` on `BatchFheUint8` (for ex, call
/// `extract_at(0)` to extract FheUint8 stored at index 0)
pub struct NonInteractiveBatchedFheUint8<C> {
    /// Vector of RLWE ciphertexts `C`
    data: Vec<C>,
    /// Count of FheUint8s packed in vector of RLWE ciphertexts
    count: usize,
}

impl<M: MatrixEntity + MatrixMut<MatElement = u64>> From<&SeededBatchedFheUint8<M::R, [u8; 32]>>
    for NonInteractiveBatchedFheUint8<M>
where
    <M as Matrix>::R: RowMut,
{
    /// Unseeds collection of seeded RLWE ciphertext in SeededBatchedFheUint8
    /// and returns as `Self`
    fn from(value: &SeededBatchedFheUint8<M::R, [u8; 32]>) -> Self {
        BoolEvaluator::with_local(|e| {
            let parameters = e.parameters();
            let ring_size = parameters.rlwe_n().0;
            let rlwe_q = parameters.rlwe_q();

            let mut prng = DefaultSecureRng::new_seeded(value.seed);
            let rlwes = value
                .data
                .iter()
                .map(|partb| {
                    let mut rlwe = M::zeros(2, ring_size);

                    // sample A
                    RandomFillUniformInModulus::random_fill(&mut prng, rlwe_q, rlwe.get_row_mut(0));

                    // Copy over B
                    rlwe.get_row_mut(1).copy_from_slice(partb.as_ref());

                    rlwe
                })
                .collect_vec();
            Self {
                data: rlwes,
                count: value.count,
            }
        })
    }
}

impl<C> KeySwitchWithId<BatchedFheUint8<C>> for NonInteractiveBatchedFheUint8<C>
where
    C: KeySwitchWithId<C>,
{
    /// Key switch `Self`'s collection of RLWE cihertexts encrypted under user's
    /// RLWE secret `u_j` to ideal RLWE secret `s` of the MPC protocol.
    ///
    /// - user_id: user id of user `j`
    fn key_switch(&self, user_id: usize) -> BatchedFheUint8<C> {
        let data = self
            .data
            .iter()
            .map(|c| c.key_switch(user_id))
            .collect_vec();
        BatchedFheUint8 {
            data,
            count: self.count,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct SeededBatchedFheUint8<C, S> {
    /// Vector of Seeded RLWE ciphertexts `C`.
    ///
    /// If RLWE(m) = [a, b] s.t. m + e = b - as, `a` can be seeded and seeded
    /// RLWE ciphertext only contains `b` polynomial
    data: Vec<C>,
    /// Seed for the ciphertexts
    seed: S,
    /// Count of FheUint8s packed in vector of RLWE ciphertexts
    count: usize,
}

impl<K, C, S> Encryptor<[u8], SeededBatchedFheUint8<C, S>> for K
where
    K: Encryptor<[bool], (Vec<C>, S)>,
{
    /// Encrypt a slice of u8s of arbitray length packed into collection of
    /// seeded RLWE ciphertexts and return `SeededBatchedFheUint8`
    fn encrypt(&self, m: &[u8]) -> SeededBatchedFheUint8<C, S> {
        // convert vector of u8s to vector bools
        let bool_m = m
            .iter()
            .flat_map(|v| (0..8).into_iter().map(|i| (((*v) >> i) & 1) == 1))
            .collect_vec();
        let (cts, seed) = K::encrypt(&self, &bool_m);
        SeededBatchedFheUint8 {
            data: cts,
            seed,
            count: m.len(),
        }
    }
}

impl<C, S> SeededBatchedFheUint8<C, S> {
    /// Unseed collection of seeded RLWE ciphertexts of `Self` and returns
    /// `NonInteractiveBatchedFheUint8` with collection of unseeded RLWE
    /// ciphertexts.
    ///
    /// In non-interactive MPC setting, RLWE ciphertexts are encrypted under
    /// user's RLWE secret `u_j`. The RLWE ciphertexts must be key switched to
    /// ideal RLWE secret `s` of the MPC protocol before use.
    ///
    /// Note that we don't provide `unseed` API from `Self` to
    /// `BatchedFheUint8`. This is because:
    ///
    /// - In non-interactive setting (1) client encrypts private inputs using
    ///   their secret `u_j` as `SeededBatchedFheUint8` and sends it to the
    ///   server. (2) Server unseeds `SeededBatchedFheUint8` into
    ///   `NonInteractiveBatchedFheUint8` indicating that private inputs are
    ///   still encrypted under user's RLWE secret `u_j`. (3) Server key
    ///   switches `NonInteractiveBatchedFheUint8` from user's RLWE secret `u_j`
    ///   to ideal RLWE secret `s` and outputs `BatchedFheUint8`. (4)
    ///   `BatchedFheUint8` always stores RLWE secret under ideal RLWE secret of
    ///   the protocol. Hence, it is safe to extract FheUint8s. Server proceeds
    ///   to extract necessary FheUint8s.
    ///
    /// - In interactive setting (1) client always encrypts private inputs using
    ///   public key corresponding to ideal RLWE secret `s` of the protocol and
    ///   produces `BatchedFheUint8`. (2) Given `BatchedFheUint8` stores
    ///   collection of RLWE ciphertext under ideal RLWE secret `s`, server can
    ///   directly extract necessary FheUint8s to use.
    ///
    /// Thus, there's no need to go directly from `Self` to `BatchedFheUint8`.
    pub fn unseed<M>(&self) -> NonInteractiveBatchedFheUint8<M>
    where
        NonInteractiveBatchedFheUint8<M>: for<'a> From<&'a SeededBatchedFheUint8<C, S>>,
        M: Matrix<R = C>,
    {
        NonInteractiveBatchedFheUint8::from(self)
    }
}

impl<C, K> MultiPartyDecryptor<u8, FheUint8<C>> for K
where
    K: MultiPartyDecryptor<bool, C>,
    <Self as MultiPartyDecryptor<bool, C>>::DecryptionShare: Clone,
{
    type DecryptionShare = Vec<<Self as MultiPartyDecryptor<bool, C>>::DecryptionShare>;
    fn gen_decryption_share(&self, c: &FheUint8<C>) -> Self::DecryptionShare {
        assert!(c.data().len() == 8);
        c.data()
            .iter()
            .map(|bit_c| {
                let decryption_share =
                    MultiPartyDecryptor::<bool, C>::gen_decryption_share(self, bit_c);
                decryption_share
            })
            .collect_vec()
    }

    fn aggregate_decryption_shares(&self, c: &FheUint8<C>, shares: &[Self::DecryptionShare]) -> u8 {
        let mut out = 0u8;

        (0..8).into_iter().for_each(|i| {
            // Collect bit i^th decryption share of each party
            let bit_i_decryption_shares = shares.iter().map(|s| s[i].clone()).collect_vec();
            let bit_i = MultiPartyDecryptor::<bool, C>::aggregate_decryption_shares(
                self,
                &c.data()[i],
                &bit_i_decryption_shares,
            );

            if bit_i {
                out += 1 << i;
            }
        });

        out
    }
}

impl<C, K> Encryptor<u8, FheUint8<C>> for K
where
    K: Encryptor<bool, C>,
{
    fn encrypt(&self, m: &u8) -> FheUint8<C> {
        let cts = (0..8)
            .into_iter()
            .map(|i| {
                let bit = ((m >> i) & 1) == 1;
                K::encrypt(self, &bit)
            })
            .collect_vec();
        FheUint8 { data: cts }
    }
}

impl<K, C> Decryptor<u8, FheUint8<C>> for K
where
    K: Decryptor<bool, C>,
{
    fn decrypt(&self, c: &FheUint8<C>) -> u8 {
        assert!(c.data.len() == 8);
        let mut out = 0u8;
        c.data().iter().enumerate().for_each(|(index, bit_c)| {
            let bool = K::decrypt(self, bit_c);
            if bool {
                out += 1 << index;
            }
        });
        out
    }
}
