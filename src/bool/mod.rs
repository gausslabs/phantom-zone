pub(crate) mod evaluator;
mod keys;
mod noise;
pub(crate) mod parameters;

use keys::tests;
pub(crate) use keys::PublicKey;

#[cfg(feature = "interactive_mp")]
mod mp_api;
#[cfg(feature = "non_interactive_mp")]
mod ni_mp_api;

#[cfg(feature = "non_interactive_mp")]
pub use ni_mp_api::*;

#[cfg(feature = "interactive_mp")]
pub use mp_api::*;

pub type ClientKey = keys::ClientKey<[u8; 32], u64>;

pub enum ParameterSelector {
    HighCommunicationButFast2Party,
    MultiPartyLessThanOrEqualTo16,
    NonInteractiveMultiPartyLessThanOrEqualTo16,
}

mod common_mp_enc_dec {
    use super::BoolEvaluator;
    use crate::{
        pbs::{sample_extract, PbsInfo},
        utils::WithLocal,
        Matrix, MultiPartyDecryptor, RowEntity, SampleExtractor,
    };

    type Mat = Vec<Vec<u64>>;

    impl<E> MultiPartyDecryptor<bool, <Mat as Matrix>::R> for super::keys::ClientKey<[u8; 32], E> {
        type DecryptionShare = <Mat as Matrix>::MatElement;

        fn gen_decryption_share(&self, c: &<Mat as Matrix>::R) -> Self::DecryptionShare {
            BoolEvaluator::with_local(|e| e.multi_party_decryption_share(c, self))
        }

        fn aggregate_decryption_shares(
            &self,
            c: &<Mat as Matrix>::R,
            shares: &[Self::DecryptionShare],
        ) -> bool {
            BoolEvaluator::with_local(|e| e.multi_party_decrypt(shares, c))
        }
    }

    impl SampleExtractor<<Mat as Matrix>::R> for Mat {
        fn extract(&self, index: usize) -> <Mat as Matrix>::R {
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
    }
}

#[cfg(test)]
mod print_noise;
