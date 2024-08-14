pub mod evaluator;
pub mod keys;
pub mod parameters;

#[cfg(feature = "interactive_mp")]
mod mp_api;
#[cfg(feature = "non_interactive_mp")]
mod ni_mp_api;

#[cfg(feature = "non_interactive_mp")]
pub use ni_mp_api::*;

#[cfg(feature = "interactive_mp")]
pub use mp_api::*;

use crate::RowEntity;

pub type ClientKey = keys::ClientKey<[u8; 32], u64>;
#[cfg(any(feature = "interactive_mp", feature = "non_interactive_mp"))]
pub type FheBool = impl_bool_frontend::FheBool<Vec<u64>>;

#[cfg(feature = "interactive_mp")]
pub type CollectiveKeyShare = keys::CommonReferenceSeededCollectivePublicKeyShare<
    Vec<u64>,
    [u8; 32],
    parameters::BoolParameters<u64>,
>;
#[cfg(feature = "interactive_mp")]
pub type CollectivePublicKey = keys::PublicKey<
    Vec<Vec<u64>>,
    crate::random::DefaultSecureRng,
    crate::ModularOpsU64<parameters::CiphertextModulus<u64>>,
>;
#[cfg(feature = "interactive_mp")]
pub type ServerKeyShare = keys::CommonReferenceSeededInteractiveMultiPartyServerKeyShare<
    Vec<Vec<u64>>,
    parameters::BoolParameters<u64>,
    evaluator::InteractiveMultiPartyCrs<[u8; 32]>,
>;

#[cfg(feature = "non_interactive_mp")]
pub type ServerKeyShare = keys::CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<
    Vec<Vec<u64>>,
    parameters::BoolParameters<u64>,
    evaluator::NonInteractiveMultiPartyCrs<[u8; 32]>,
>;

pub(crate) trait BooleanGates {
    type Ciphertext: RowEntity;
    type Key;

    fn and_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn nand_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn or_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn nor_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn xor_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn xnor_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn not_inplace(&self, c: &mut Self::Ciphertext);

    fn and(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn nand(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn or(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn nor(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn xor(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn xnor(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn not(&self, c: &Self::Ciphertext) -> Self::Ciphertext;
}

#[cfg(any(feature = "interactive_mp", feature = "non_interactive_mp"))]
mod impl_bool_frontend {
    use serde::{Deserialize, Serialize};

    use crate::MultiPartyDecryptor;

    /// Fhe Bool ciphertext
    #[derive(Default, Clone, Serialize, Deserialize)]
    pub struct FheBool<C> {
        /// LWE bool ciphertext
        pub(crate) data: C,
    }

    impl<C> FheBool<C> {
        pub(crate) fn data(&self) -> &C {
            &self.data
        }

        pub(crate) fn data_mut(&mut self) -> &mut C {
            &mut self.data
        }
    }

    impl<C, K> MultiPartyDecryptor<bool, FheBool<C>> for K
    where
        K: MultiPartyDecryptor<bool, C>,
    {
        type DecryptionShare = <K as MultiPartyDecryptor<bool, C>>::DecryptionShare;

        fn aggregate_decryption_shares(
            &self,
            c: &FheBool<C>,
            shares: &[Self::DecryptionShare],
        ) -> bool {
            self.aggregate_decryption_shares(&c.data(), shares)
        }

        fn gen_decryption_share(&self, c: &FheBool<C>) -> Self::DecryptionShare {
            self.gen_decryption_share(&c.data())
        }
    }

    mod ops {
        use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

        use crate::{
            utils::{Global, WithLocal},
            BooleanGates,
        };

        use super::super::{BoolEvaluator, RuntimeServerKey};

        type FheBool = super::super::FheBool;

        impl BitAnd for &FheBool {
            type Output = FheBool;
            fn bitand(self, rhs: Self) -> Self::Output {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    FheBool {
                        data: e.and(self.data(), rhs.data(), key),
                    }
                })
            }
        }

        impl BitAndAssign for FheBool {
            fn bitand_assign(&mut self, rhs: Self) {
                BoolEvaluator::with_local_mut_mut(&mut |e| {
                    let key = RuntimeServerKey::global();
                    e.and_inplace(&mut self.data_mut(), rhs.data(), key);
                });
            }
        }

        impl BitOr for &FheBool {
            type Output = FheBool;
            fn bitor(self, rhs: Self) -> Self::Output {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    FheBool {
                        data: e.or(self.data(), rhs.data(), key),
                    }
                })
            }
        }

        impl BitOrAssign for FheBool {
            fn bitor_assign(&mut self, rhs: Self) {
                BoolEvaluator::with_local_mut_mut(&mut |e| {
                    let key = RuntimeServerKey::global();
                    e.or_inplace(&mut self.data_mut(), rhs.data(), key);
                });
            }
        }

        impl BitXor for &FheBool {
            type Output = FheBool;
            fn bitxor(self, rhs: Self) -> Self::Output {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    FheBool {
                        data: e.xor(self.data(), rhs.data(), key),
                    }
                })
            }
        }

        impl BitXorAssign for FheBool {
            fn bitxor_assign(&mut self, rhs: Self) {
                BoolEvaluator::with_local_mut_mut(&mut |e| {
                    let key = RuntimeServerKey::global();
                    e.xor_inplace(&mut self.data_mut(), rhs.data(), key);
                });
            }
        }

        impl Not for &FheBool {
            type Output = FheBool;
            fn not(self) -> Self::Output {
                BoolEvaluator::with_local(|e| FheBool {
                    data: e.not(self.data()),
                })
            }
        }

        impl FheBool {
            pub fn nand(&self, rhs: &Self) -> Self {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    FheBool {
                        data: e.nand(self.data(), rhs.data(), key),
                    }
                })
            }

            pub fn xnor(&self, rhs: &Self) -> Self {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    FheBool {
                        data: e.xnor(self.data(), rhs.data(), key),
                    }
                })
            }

            pub fn nor(&self, rhs: &Self) -> Self {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    FheBool {
                        data: e.nor(self.data(), rhs.data(), key),
                    }
                })
            }
        }
    }
}

#[cfg(any(feature = "interactive_mp", feature = "non_interactive_mp"))]
mod common_mp_enc_dec {
    use itertools::Itertools;

    use super::{impl_bool_frontend::FheBool, BoolEvaluator};
    use crate::{
        pbs::{sample_extract, PbsInfo},
        utils::WithLocal,
        Matrix, RowEntity, SampleExtractor,
    };

    type Mat = Vec<Vec<u64>>;

    /// `Self` stores batch of boolean ciphertexts as collection of `unseeded RLWE ciphertexts` encrypted under the ideal RLWE secret key of the protocol.
    ///
    /// Bool ciphertext at `index` can be extracted from the coefficient at `index %
    /// N` of `index / N`th RLWE ciphertext.
    pub struct BatchedFheBools<C> {
        data: Vec<C>,
        count: usize,
    }

    impl<C> BatchedFheBools<C> {
        pub(crate) fn new(data: Vec<C>, count: usize) -> Self {
            Self { data, count }
        }

        pub(crate) fn data(&self) -> &[C] {
            &self.data
        }

        pub(crate) fn count(&self) -> usize {
            self.count
        }
    }

    impl<M: Matrix> SampleExtractor<FheBool<M::R>> for BatchedFheBools<M>
    where
        M: SampleExtractor<M::R>,
    {
        fn extract_all(&self) -> Vec<FheBool<M::R>> {
            if self.data.len() > 0 {
                let ring_size = self.data[0].dimension().1;

                let mut index = 0;
                let mut out = Vec::with_capacity(self.count);

                while index < self.count {
                    let row = index % ring_size;
                    let col = index / ring_size;

                    out.push(FheBool {
                        data: SampleExtractor::extract_at(&self.data[col], row),
                    });
                    index += 1;
                }

                out
            } else {
                vec![]
            }
        }
        fn extract_at(&self, index: usize) -> FheBool<M::R> {
            assert!(self.count > index);

            let ring_size = self.data[0].dimension().1;
            let row = index % ring_size;
            let col = index / ring_size;

            FheBool {
                data: SampleExtractor::extract_at(&self.data[col], row),
            }
        }

        fn extract_many(&self, how_many: usize) -> Vec<FheBool<M::R>> {
            assert!(self.count >= how_many);
            let ring_size = self.data[0].dimension().1;

            let mut index = 0;
            let mut out = Vec::with_capacity(self.count);

            while index < how_many {
                let row = index % ring_size;
                let col = index / ring_size;
                out.push(FheBool {
                    data: SampleExtractor::extract_at(&self.data[col], row),
                });
                index += 1;
            }

            out
        }
    }

    impl SampleExtractor<<Mat as Matrix>::R> for Mat {
        /// Sample extract coefficient at `index` as a LWE ciphertext from RLWE
        /// ciphertext `Self`
        fn extract_at(&self, index: usize) -> <Mat as Matrix>::R {
            // input is RLWE ciphertext
            assert!(self.dimension().0 == 2);

            let ring_size = self.dimension().1;
            assert!(index < ring_size);

            BoolEvaluator::with_local(|e| {
                let mut lwe_out = <Mat as Matrix>::R::zeros(ring_size + 1);
                sample_extract(&mut lwe_out, self, e.pbs_info().modop_rlweq(), index);
                lwe_out
            })
        }

        /// Extract first `how_many` coefficients of `Self` as LWE ciphertexts
        fn extract_many(&self, how_many: usize) -> Vec<<Mat as Matrix>::R> {
            assert!(self.dimension().0 == 2);

            let ring_size = self.dimension().1;
            assert!(how_many <= ring_size);

            (0..how_many)
                .map(|index| {
                    BoolEvaluator::with_local(|e| {
                        let mut lwe_out = <Mat as Matrix>::R::zeros(ring_size + 1);
                        sample_extract(&mut lwe_out, self, e.pbs_info().modop_rlweq(), index);
                        lwe_out
                    })
                })
                .collect_vec()
        }

        /// Extracts all coefficients of `Self` as LWE ciphertexts
        fn extract_all(&self) -> Vec<<Mat as Matrix>::R> {
            assert!(self.dimension().0 == 2);

            let ring_size = self.dimension().1;

            (0..ring_size)
                .map(|index| {
                    BoolEvaluator::with_local(|e| {
                        let mut lwe_out = <Mat as Matrix>::R::zeros(ring_size + 1);
                        sample_extract(&mut lwe_out, self, e.pbs_info().modop_rlweq(), index);
                        lwe_out
                    })
                })
                .collect_vec()
        }
    }
}

#[cfg(test)]
mod print_noise;
