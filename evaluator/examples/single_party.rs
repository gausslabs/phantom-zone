use core::{
    ops::{BitAnd, BitOr, BitXor},
    {array::from_fn, num::Wrapping},
};
use num_traits::NumOps;
use phantom_zone_evaluator::boolean::{fhew::prelude::*, FheBool};
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

fn encrypt_bool<'a>(
    evaluator: &'a Evaluator,
    sk: &LweSecretKeyOwned<i32>,
    m: bool,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) -> FheBool<'a, Evaluator> {
    let ct = FhewBoolCiphertext::sk_encrypt(evaluator.param(), evaluator.ring(), sk, m, rng);
    FheBool::new(evaluator, ct)
}

fn decrypt_bool(
    evaluator: &Evaluator,
    sk: &LweSecretKeyOwned<i32>,
    ct: FheBool<Evaluator>,
) -> bool {
    ct.ct().decrypt(evaluator.ring(), sk)
}

fn encrypt_u8<'a>(
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

fn decrypt_u8(evaluator: &Evaluator, sk: &LweSecretKeyOwned<i32>, ct: FheU8<Evaluator>) -> u8 {
    ct.into_cts()
        .into_iter()
        .rev()
        .map(|ct| ct.decrypt(evaluator.ring(), sk))
        .fold(0, |m, b| (m << 1) | b as u8)
}

// Ported from https://github.com/ChihChengLiang/haunted/blob/ba42814b0c444dab222fd4aca51e6efe6eb96381/src/phantom.rs#L453.
trait BitOps<Rhs = Self, Output = Self>:
    BitAnd<Rhs, Output = Output> + BitOr<Rhs, Output = Output> + BitXor<Rhs, Output = Output>
{
}

impl<T, Rhs, Output> BitOps<Rhs, Output> for T where
    T: BitAnd<Rhs, Output = Output> + BitOr<Rhs, Output = Output> + BitXor<Rhs, Output = Output>
{
}

fn bool_function<T>(a: &T, b: &T, c: &T, d: &T) -> T
where
    T: for<'t> BitOps<&'t T, T>,
    for<'t> &'t T: BitOps<&'t T, T>,
{
    ((a | b) & c) ^ d
}

fn wrapping_uint_function<T>(a: &T, b: &T, c: &T, d: &T, e: &T) -> T
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

    // Function with bools

    let m = from_fn(|_| rng.next_u64() & 1 == 1);
    let g = {
        let [a, b, c, d] = &m;
        bool_function(a, b, c, d)
    };
    let ct_g = {
        let [a, b, c, d] = &m.map(|m| encrypt_bool(&evaluator, &sk, m, &mut rng));
        bool_function(a, b, c, d)
    };

    assert_eq!(g, decrypt_bool(&evaluator, &sk, ct_g));

    // Function with wrapping u8s

    let m = from_fn(|_| rng.next_u64() as u8);
    let g = {
        let [a, b, c, d, e] = &m.map(Wrapping);
        wrapping_uint_function(a, b, c, d, e).0
    };
    let ct_g = {
        let [a, b, c, d, e] = &m.map(|m| encrypt_u8(&evaluator, &sk, m, &mut rng));
        wrapping_uint_function(a, b, c, d, e)
    };

    assert_eq!(g, decrypt_u8(&evaluator, &sk, ct_g));
}
