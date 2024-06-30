use bin_rs::*;
use itertools::Itertools;
use rand::{thread_rng, Rng, RngCore};

fn circuit(a: u8, b: u8, c: u8, d: u8) -> u8 {
    ((a + b) * c) * d
}

fn fhe_circuit(a: &FheUint8, b: &FheUint8, c: &FheUint8, d: &FheUint8) -> FheUint8 {
    &(&(a + b) * c) * d
}

fn main() {
    set_parameter_set(ParameterSelector::NonInteractiveLTE2Party);

    // set CRS
    let mut seed = [0u8; 32];
    thread_rng().fill_bytes(&mut seed);
    set_common_reference_seed(seed);

    let no_of_parties = 2;

    // Generate client keys
    let cks = (0..no_of_parties).map(|_| gen_client_key()).collect_vec();

    // client 0 encrypts private input
    let c0_a = thread_rng().gen::<u8>();
    let c0_b = thread_rng().gen::<u8>();
    let c0_batched_to_send = cks[0].encrypt(vec![c0_a, c0_b].as_slice());

    // client 1 encrypts private input
    let c1_a = thread_rng().gen::<u8>();
    let c1_b = thread_rng().gen::<u8>();
    let c1_batch_to_send = cks[1].encrypt(vec![c1_a, c1_b].as_slice());

    // Both client indenpendently generate their server key shares
    let server_key_shares = cks
        .iter()
        .enumerate()
        .map(|(id, k)| gen_server_key_share(id, no_of_parties, k))
        .collect_vec();

    // Server side

    // aggregates shares and generates server key
    let server_key = aggregate_server_key_shares(&server_key_shares);
    server_key.set_server_key();

    // extract a and b from client0 inputs
    // let now = std::time::Instant::now();
    let (ct_c0_a, ct_c0_b) = {
        let ct = c0_batched_to_send.unseed::<Vec<Vec<u64>>>().key_switch(0);
        (ct.extract(0), ct.extract(1))
    };
    // println!(
    //     "Time to unseed, key switch, and extract 2 ciphertexts: {:?}",
    //     now.elapsed()
    // );

    // extract a and b from client1 inputs
    let (ct_c1_a, ct_c1_b) = {
        let ct = c1_batch_to_send.unseed::<Vec<Vec<u64>>>().key_switch(1);
        (ct.extract(0), ct.extract(1))
    };

    let now = std::time::Instant::now();
    let c_out = fhe_circuit(&ct_c0_a, &ct_c1_a, &ct_c0_b, &ct_c1_b);
    println!("Circuit Time: {:?}", now.elapsed());

    // decrypt c_out
    let decryption_shares = cks
        .iter()
        .map(|k| k.gen_decryption_share(&c_out))
        .collect_vec();
    let m_out = cks[0].aggregate_decryption_shares(&c_out, &decryption_shares);
    let m_expected = circuit(c0_a, c1_a, c0_b, c1_b);
    assert!(m_expected == m_out);
}
