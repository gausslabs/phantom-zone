pub(crate) mod evaluator;
mod keys;
mod mp_api;
mod ni_mp_api;
mod noise;
pub(crate) mod parameters;
mod sp_api;

pub(crate) use keys::PublicKey;

pub use ni_mp_api::*;
pub type ClientKey = keys::ClientKey<[u8; 32], u64>;

pub enum ParameterSelector {
    MultiPartyLessThanOrEqualTo16,
    NonInteractiveMultiPartyLessThanOrEqualTo16,
}
