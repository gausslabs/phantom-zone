use phantom_zone_crypto::scheme::blind_rotation::lmkcdey::{
    interactive::{
        LmkcdeyMpiCrs, LmkcdeyMpiKeyShare, LmkcdeyMpiKeyShareCompact, LmkcdeyMpiKeyShareOwned,
        LmkcdeyMpiParam,
    },
    LmkcdeyKeyOwned, LmkcdeyParam,
};

pub use crate::boolean::{
    evaluator::{
        fhew::{FhewBoolCiphertext, FhewBoolCiphertextOwned, FhewBoolEvaluator},
        BoolEvaluator,
    },
    integer::{FheU16, FheU32, FheU64, FheU8, FheUint},
    FheBool,
};
pub use phantom_zone_crypto::scheme::blind_rotation::lmkcdey::{
    interactive::{aggregate_bs_key_shares, aggregate_pk_shares, bs_key_share_gen, pk_share_gen},
    prepare_bs_key,
};
pub use phantom_zone_crypto::{
    core::{
        lwe::{LweDecryptionShare, LweSecretKey, LweSecretKeyOwned},
        rgsw::RgswDecompositionParam,
        rlwe::{
            RlwePublicKey, RlwePublicKeyOwned, RlweSecretKey, RlweSecretKeyOwned,
            SeededRlwePublicKey, SeededRlwePublicKeyOwned,
        },
    },
    util::{
        distribution::{NoiseDistribution, SecretDistribution},
        rng::{HierarchicalSeedableRng, LweRng, StdLweRng},
    },
};
pub use phantom_zone_math::prelude::*;

/// FHEW boolean FHE parameter.
pub type FhewBoolParam = LmkcdeyParam;

/// FHEW boolean FHE bootstrapping key.
pub type FhewBoolKey<T1, T2> = LmkcdeyKeyOwned<T1, T2>;

/// FHEW multi-party boolean FHE (interactive keygen version) parameter.
pub type FhewBoolMpiParam = LmkcdeyMpiParam;

/// FHEW multi-party boolean FHE (interactive keygen version) common reference
/// string.
pub type FhewBoolMpiCrs<S> = LmkcdeyMpiCrs<S>;

/// FHEW multi-party boolean FHE (interactive keygen version) bootstrapping key
/// share.
pub type FhewBoolMpiKeyShare<S1, S2> = LmkcdeyMpiKeyShare<S1, S2>;

/// FHEW multi-party boolean FHE (interactive keygen version) bootstrapping key
/// share in compact format.
pub type FhewBoolMpiKeyShareCompact = LmkcdeyMpiKeyShareCompact;

/// FHEW multi-party boolean FHE (interactive keygen version) bootstrapping key
/// share.
pub type FhewBoolMpiKeyShareOwned<T1, T2> = LmkcdeyMpiKeyShareOwned<T1, T2>;
