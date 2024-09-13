use phantom_zone_crypto::scheme::blind_rotation::lmkcdey::{self, interactive};

pub use crate::boolean::{
    evaluator::{
        fhew::{FhewBoolCiphertext, FhewBoolEvaluator},
        BoolEvaluator,
    },
    integer::{FheU16, FheU32, FheU64, FheU8},
};
pub use phantom_zone_crypto::scheme::blind_rotation::lmkcdey::{
    interactive::{aggregate_bs_key_shares, aggregate_pk_shares, bs_key_share_gen, pk_share_gen},
    prepare_bs_key,
};
pub use phantom_zone_crypto::{
    core::{
        lwe::{LweDecryptionShare, LweSecretKey, LweSecretKeyOwned},
        rgsw::RgswDecompositionParam,
        rlwe::{RlwePublicKey, RlwePublicKeyOwned, SeededRlwePublicKey, SeededRlwePublicKeyOwned},
    },
    util::{
        distribution::{NoiseDistribution, SecretDistribution},
        rng::{LweRng, StdLweRng},
    },
};
pub use phantom_zone_math::prelude::*;

pub type FhewBoolParam = lmkcdey::LmkcdeyParam;

pub type FhewBoolKey<T1, T2> = lmkcdey::LmkcdeyKeyOwned<T1, T2>;

pub type FhewBoolMpiParam = interactive::LmkcdeyMpiParam;

pub type FhewBoolMpiCrs<S> = interactive::LmkcdeyMpiCrs<S>;

pub type FhewBoolMpiKeyShare<S1, S2> = interactive::LmkcdeyMpiKeyShare<S1, S2>;

pub type FhewBoolMpiKeyShareCompact = interactive::LmkcdeyMpiKeyShareCompact;

pub type FhewBoolMpiKeyShareOwned<T1, T2> = interactive::LmkcdeyMpiKeyShareOwned<T1, T2>;
