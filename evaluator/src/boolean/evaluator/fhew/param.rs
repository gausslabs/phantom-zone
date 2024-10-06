use crate::boolean::fhew::prelude::*;

pub const I_4P: FhewBoolMpiParam = FhewBoolMpiParam {
    param: FhewBoolParam {
        message_bits: 2,
        modulus: Modulus::Prime(18014398509404161),
        ring_size: 2048,
        sk_distribution: SecretDistribution::Ternary(Ternary),
        noise_distribution: NoiseDistribution::Gaussian(Gaussian(3.19)),
        u_distribution: SecretDistribution::Ternary(Ternary),
        auto_decomposition_param: DecompositionParam {
            log_base: 24,
            level: 1,
        },
        rlwe_by_rgsw_decomposition_param: RgswDecompositionParam {
            log_base: 17,
            level_a: 1,
            level_b: 1,
        },
        lwe_modulus: Modulus::PowerOfTwo(16),
        lwe_dimension: 620,
        lwe_sk_distribution: SecretDistribution::Ternary(Ternary),
        lwe_noise_distribution: NoiseDistribution::Gaussian(Gaussian(3.19)),
        lwe_ks_decomposition_param: DecompositionParam {
            log_base: 1,
            level: 13,
        },
        q: 2048,
        g: 5,
        w: 10,
    },
    rgsw_by_rgsw_decomposition_param: RgswDecompositionParam {
        log_base: 6,
        level_a: 7,
        level_b: 6,
    },
    total_shares: 4,
};

pub const I_4P_60: FhewBoolMpiParam = FhewBoolMpiParam {
    param: FhewBoolParam {
        message_bits: 2,
        modulus: Modulus::PowerOfTwo(64),
        ring_size: 1024,
        sk_distribution: SecretDistribution::Ternary(Ternary),
        noise_distribution: NoiseDistribution::Gaussian(Gaussian(3.19)),
        u_distribution: SecretDistribution::Ternary(Ternary),
        auto_decomposition_param: DecompositionParam {
            log_base: 17,
            level: 1,
        },
        rlwe_by_rgsw_decomposition_param: RgswDecompositionParam {
            log_base: 17,
            level_a: 1,
            level_b: 1,
        },
        lwe_modulus: Modulus::PowerOfTwo(18),
        lwe_dimension: 300,
        lwe_sk_distribution: SecretDistribution::Ternary(Ternary),
        lwe_noise_distribution: NoiseDistribution::Gaussian(Gaussian(3.19)),
        lwe_ks_decomposition_param: DecompositionParam {
            log_base: 5,
            level: 3,
        },
        q: 1024,
        g: 5,
        w: 10,
    },
    rgsw_by_rgsw_decomposition_param: RgswDecompositionParam {
        log_base: 16,
        level_a: 3,
        level_b: 2,
    },
    total_shares: 4,
};
