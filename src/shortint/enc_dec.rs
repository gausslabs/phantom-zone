use itertools::Itertools;

use crate::{
    bool::BoolEvaluator,
    random::{DefaultSecureRng, RandomFillUniformInModulus},
    utils::{TryConvertFrom1, WithLocal},
    Decryptor, Encryptor, KeySwitchWithId, Matrix, MatrixEntity, MatrixMut, MultiPartyDecryptor,
    RowMut,
};

#[derive(Clone)]
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

pub struct BatchedFheUint8<C> {
    data: Vec<C>,
}

impl<M: MatrixEntity + MatrixMut<MatElement = u64>> From<&SeededBatchedFheUint8<M::R, [u8; 32]>>
    for BatchedFheUint8<M>
where
    <M as Matrix>::R: RowMut,
{
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
            Self { data: rlwes }
        })
    }
}

pub struct SeededBatchedFheUint8<C, S> {
    data: Vec<C>,
    seed: S,
}

impl<C, S> SeededBatchedFheUint8<C, S> {
    pub fn unseed<M>(&self) -> BatchedFheUint8<M>
    where
        BatchedFheUint8<M>: for<'a> From<&'a SeededBatchedFheUint8<C, S>>,
        M: Matrix<R = C>,
    {
        BatchedFheUint8::from(self)
    }
}

impl<K, C, S> Encryptor<[u8], SeededBatchedFheUint8<C, S>> for K
where
    K: Encryptor<[bool], (Vec<C>, S)>,
{
    fn encrypt(&self, m: &[u8]) -> SeededBatchedFheUint8<C, S> {
        // convert vector of u8s to vector bools
        let m = m
            .iter()
            .flat_map(|v| (0..8).into_iter().map(|i| (((*v) >> i) & 1) == 1))
            .collect_vec();
        let (cts, seed) = K::encrypt(&self, &m);
        dbg!(cts.len());
        SeededBatchedFheUint8 { data: cts, seed }
    }
}

impl<K, C> Encryptor<[u8], BatchedFheUint8<C>> for K
where
    K: Encryptor<[bool], Vec<C>>,
{
    fn encrypt(&self, m: &[u8]) -> BatchedFheUint8<C> {
        let m = m
            .iter()
            .flat_map(|v| {
                (0..8)
                    .into_iter()
                    .map(|i| ((*v >> i) & 1) == 1)
                    .collect_vec()
            })
            .collect_vec();
        let cts = K::encrypt(&self, &m);
        BatchedFheUint8 { data: cts }
    }
}

impl<C> KeySwitchWithId<BatchedFheUint8<C>> for BatchedFheUint8<C>
where
    C: KeySwitchWithId<C>,
{
    fn key_switch(&self, user_id: usize) -> BatchedFheUint8<C> {
        let data = self
            .data
            .iter()
            .map(|c| c.key_switch(user_id))
            .collect_vec();
        BatchedFheUint8 { data }
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
