use bin_rs::*;
use itertools::Itertools;
use rand::{thread_rng, RngCore};

fn plain_circuit(a: u8, b: u8, c: u8) -> u8 {
    (a + b) * c
}

fn fhe_circuit(fhe_a: &FheUint8, fhe_b: &FheUint8, fhe_c: &FheUint8) -> FheUint8 {
    &(fhe_a + fhe_b) * fhe_c
}

fn main() {
    set_parameter_set(ParameterSelector::MultiPartyLessThan16);
    let no_of_parties = 2;
    let client_keys = (0..no_of_parties)
        .into_iter()
        .map(|_| gen_client_key())
        .collect_vec();

    // set Multi-Party seed
    let mut seed = [0u8; 32];
    thread_rng().fill_bytes(&mut seed);
    set_mp_seed(seed);

    // multi-party key gen round 1
    let pk_shares = client_keys
        .iter()
        .map(|k| gen_mp_keys_phase1(k))
        .collect_vec();

    // create public key
    let public_key = aggregate_public_key_shares(&pk_shares);

    // multi-party key gen round 2
    let server_key_shares = client_keys
        .iter()
        .map(|k| gen_mp_keys_phase2(k, &public_key))
        .collect_vec();

    // server aggregates server key shares and sets it
    let server_key = aggregate_server_key_shares(&server_key_shares);
    server_key.set_server_key();

    // private inputs
    let a = 4u8;
    let b = 6u8;
    let c = 128u8;
    let fhe_a = public_key.encrypt(&a);
    let fhe_b = public_key.encrypt(&b);
    let fhe_c = public_key.encrypt(&c);

    // fhe evaluation
    let now = std::time::Instant::now();
    let fhe_out = fhe_circuit(&fhe_a, &fhe_b, &fhe_c);
    println!("Circuit time: {:?}", now.elapsed());

    // plain evaluation
    let out = plain_circuit(a, b, c);

    // generate decryption shares to decrypt ciphertext fhe_out
    let decryption_shares = client_keys
        .iter()
        .map(|k| k.gen_decryption_share(&fhe_out))
        .collect_vec();

    // decrypt fhe_out using decryption shares
    let got_out = client_keys[0].aggregate_decryption_shares(&fhe_out, &decryption_shares);

    assert_eq!(got_out, out);
}
