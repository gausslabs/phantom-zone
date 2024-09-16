use core::{array::from_fn, num::Wrapping};
use num_traits::NumOps;
use phantom_zone_evaluator::boolean::fhew::prelude::*;
use rand::{RngCore, SeedableRng};

type Evaluator = FhewBoolEvaluator<NoisyNativeRing, NonNativePowerOfTwoRing>;

const PARAM: FhewBoolParam = FhewBoolParam {
    message_bits: 2,
    modulus: Modulus::PowerOfTwo(64),
    ring_size: 2048,
    sk_distribution: SecretDistribution::Gaussian(Gaussian(3.19)),
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
    lwe_sk_distribution: SecretDistribution::Gaussian(Gaussian(3.19)),
    lwe_noise_distribution: NoiseDistribution::Gaussian(Gaussian(3.19)),
    lwe_ks_decomposition_param: DecompositionParam {
        log_base: 1,
        level: 13,
    },
    q: 2048,
    g: 5,
    w: 10,
};

fn encrypt<'a>(
    evaluator: &'a Evaluator,
    sk: &LweSecretKeyOwned<i32>,
    m: u8,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) -> FheU8<'a, Evaluator> {
    let cts = from_fn(|idx| {
        let m = (m >> idx) & 1 == 1;
        FhewBoolCiphertext::sk_encrypt(evaluator.param(), evaluator.ring(), sk, m, rng)
    });
    FheU8::from_cts(evaluator, cts)
}

fn decrypt(evaluator: &Evaluator, sk: &LweSecretKeyOwned<i32>, ct: FheU8<Evaluator>) -> u8 {
    ct.into_cts()
        .into_iter()
        .rev()
        .map(|ct| ct.decrypt(evaluator.ring(), sk))
        .fold(0, |m, b| (m << 1) | b as u8)
}

fn function<T>(a: &T, b: &T, c: &T, d: &T, e: &T) -> T
where
    T: for<'t> NumOps<&'t T, T>,
    for<'t> &'t T: NumOps<&'t T, T>,
{
    (((a + b) - c) * d) % e
}

fn main() {
    let mut rng = StdLweRng::from_entropy();
    let sk = LweSecretKey::sample(PARAM.ring_size, PARAM.sk_distribution, &mut rng);
    let evaluator = Evaluator::sample(PARAM, &sk, &mut rng);

    let m = from_fn(|_| rng.next_u64() as u8);
    let g = {
        let [a, b, c, d, e] = &m.map(Wrapping);
        function(a, b, c, d, e).0
    };
    let ct_g = {
        let [a, b, c, d, e] = &m.map(|m| encrypt(&evaluator, &sk, m, &mut rng));
        function(a, b, c, d, e)
    };

    assert_eq!(g, decrypt(&evaluator, &sk, ct_g));
}
