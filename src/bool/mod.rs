pub(crate) mod evaluator;
pub(crate) mod keys;
mod mp_api;
mod ni_mp_api;
mod noise;
pub(crate) mod parameters;

pub use mp_api::*;

pub type FheBool = Vec<u64>;

use std::{cell::RefCell, sync::OnceLock};

use evaluator::*;
use keys::*;
use parameters::*;
