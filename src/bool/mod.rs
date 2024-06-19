pub(crate) mod evaluator;
mod keys;
mod mp_api;
mod ni_mp_api;
mod noise;
pub(crate) mod parameters;

pub(crate) use keys::PublicKey;

#[cfg(feature = "non_interactive_mp")]
pub use ni_mp_api::*;

#[cfg(feature = "interactive_mp")]
pub use mp_api::*;

pub type ClientKey = keys::ClientKey<[u8; 32], u64>;

pub enum ParameterSelector {
    MultiPartyLessThanOrEqualTo16,
    NonInteractiveMultiPartyLessThanOrEqualTo16,
}

mod common_mp_enc_dec {
    use super::BoolEvaluator;
    use crate::{utils::WithLocal, Matrix, MultiPartyDecryptor};

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
}
