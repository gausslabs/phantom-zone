use std::collections::HashMap;

use bristol_circuit::{BristolCircuit, CircuitInfo};
use frontend::LayeredCircuit;
use phantom_zone_evaluator::boolean::fhew::prelude::*;
use rand::{RngCore, SeedableRng};
use serde_json::from_str;

const CIRCUIT_STR: &str = include_str!("./circuits/gt10/circuit.txt");
const CIRCUIT_INFO_STR: &str = include_str!("./circuits/gt10/circuit_info.json");

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
) -> FheBool<&'a Evaluator> {
    let ct = FhewBoolCiphertext::sk_encrypt(evaluator.param(), evaluator.ring(), sk, m, rng);
    FheBool::new(evaluator, ct)
}

fn decrypt_bool(
    evaluator: &Evaluator,
    sk: &LweSecretKeyOwned<i32>,
    ct: FheBool<&Evaluator>,
) -> bool {
    ct.ct().decrypt(evaluator.ring(), sk)
}

fn main() {
    let circuit = BristolCircuit::from_info_and_bristol_string(
        &from_str::<CircuitInfo>(CIRCUIT_INFO_STR).unwrap(),
        CIRCUIT_STR,
    )
    .unwrap();

    let layered_circuit = LayeredCircuit::from_bristol(&circuit);

    let mut rng = StdLweRng::from_entropy();
    let sk = LweSecretKey::sample(PARAM.ring_size, PARAM.sk_distribution, &mut rng);
    let evaluator = Evaluator::sample(PARAM, &sk, &mut rng);

    let x = vec![true, true, false, false];

    let mut plain_inputs = HashMap::new();
    plain_inputs.insert("x".to_string(), x.clone());
    let plain_outputs = layered_circuit.eval(plain_inputs);
    let plain_main = plain_outputs.get("main").unwrap();

    let encrypted_x = x
        .iter()
        .map(|m| encrypt_bool(&evaluator, &sk, *m, &mut rng))
        .collect::<Vec<_>>();

    // this input is written for use with the greaterThan10.ts circuit (--boolify-width 4),
    // here x is 12 (8+4+0+0), which is gt 10, so the output is 10 (8+0+2+0)
    let inputs = [("x".to_string(), encrypted_x)]
        .iter()
        .cloned()
        .collect::<HashMap<_, _>>();

    let outputs = layered_circuit.eval(inputs);

    let encrypted_main = outputs.get("main").unwrap();

    let main = encrypted_main
        .iter()
        .map(|ct| decrypt_bool(&evaluator, &sk, ct.clone()))
        .collect::<Vec<_>>();

    assert_eq!(plain_main, &main);
}
