pub(crate) mod evaluator;
mod keys;
mod mp_api;
mod ni_mp_api;
mod noise;
pub(crate) mod parameters;

pub(crate) use keys::PublicKey;

pub type FheBool = Vec<u64>;

pub use mp_api::*;
