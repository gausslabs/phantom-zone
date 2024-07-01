use bin_rs::*;
use itertools::Itertools;
use rand::{thread_rng, Rng, RngCore};

fn function1(a: u8, b: u8, c: u8, d: u8) -> u8 {
    ((a + b) * c) * d
}

fn function1_fhe(a: &FheUint8, b: &FheUint8, c: &FheUint8, d: &FheUint8) -> FheUint8 {
    &(&(a + b) * c) * d
}

fn function2(a: u8, b: u8, c: u8, d: u8) -> u8 {
    (a * b) + (c * d)
}

fn function2_fhe(a: &FheUint8, b: &FheUint8, c: &FheUint8, d: &FheUint8) -> FheUint8 {
    &(a * b) + &(c * d)
}

fn main() {
    set_parameter_set(ParameterSelector::NonInteractiveLTE4Party);

    // set application's common reference seed
    let mut seed = [0u8; 32];
    thread_rng().fill_bytes(&mut seed);
    set_common_reference_seed(seed);

    let no_of_parties = 4;

    // Clide side //

    // Generate client keys
    let cks = (0..no_of_parties).map(|_| gen_client_key()).collect_vec();

    // client 0 encrypts its private inputs
    let c0_a = thread_rng().gen::<u8>();
    // Clients encrypt their private inputs in a seeded batched ciphertext
    let c0_enc = cks[0].encrypt(vec![c0_a].as_slice());

    // client 1 encrypts its private inputs
    let c1_a = thread_rng().gen::<u8>();
    let c1_enc = cks[1].encrypt(vec![c1_a].as_slice());

    // client 2 encrypts its private inputs
    let c2_a = thread_rng().gen::<u8>();
    let c2_enc = cks[2].encrypt(vec![c2_a].as_slice());

    // client 1 encrypts its private inputs
    let c3_a = thread_rng().gen::<u8>();
    let c3_enc = cks[3].encrypt(vec![c3_a].as_slice());

    // Clients independently generate their server key shares
    //
    // We assign user_id 0 to client 0, user_id 1 to client 1, user_id 2 to client
    // 2, user_id 3 to client 3.
    let server_key_shares = cks
        .iter()
        .enumerate()
        .map(|(id, k)| gen_server_key_share(id, no_of_parties, k))
        .collect_vec();

    // Each client uploads their server key shares and encrypted private inputs to
    // the server in a single shot message.

    // Server side //

    // Server receives server key shares from each client and proceeds to aggregate
    // them to produce server key. After this point, server can use server key share
    // to evaluate any arbitrary function on encrypted private inputs from the fixed
    // set of clients

    // aggregate shares and generates server key
    let server_key = aggregate_server_key_shares(&server_key_shares);
    server_key.set_server_key();

    // Server proceeds to extract private inputs sent by clients
    //
    // To extract client 0's (with user_id=0) private inputs we first key switch
    // client 0's private inputs from thei secret to ideal secret of the mpc
    // protocol. To indicate we're key switching client 0's private input we
    // supply client 0's user_id i.e. we call `key_switch(0)`. Then we extract
    // the first ciphertext by calling `extract_at(0)`.
    //
    // Since client 0 only encrypted 1 input in batched ciphertext calling
    // extract_at(index) for `index` > 0 will panic. If client 0 had more private
    // inputs then we can either extract them all at once by `extract_all` or first
    // `many` of them by `extract_many(many)`
    let ct_c0_a = c0_enc.unseed::<Vec<Vec<u64>>>().key_switch(0).extract_at(0);

    let ct_c1_a = c1_enc.unseed::<Vec<Vec<u64>>>().key_switch(1).extract_at(0);
    let ct_c2_a = c2_enc.unseed::<Vec<Vec<u64>>>().key_switch(2).extract_at(0);
    let ct_c3_a = c3_enc.unseed::<Vec<Vec<u64>>>().key_switch(3).extract_at(0);

    // After extracting each client's private inputs, server proceeds to evaluate
    // the function1
    let now = std::time::Instant::now();
    let ct_out_f1 = function1_fhe(&ct_c0_a, &ct_c1_a, &ct_c2_a, &ct_c3_a);
    println!("Function1 FHE evaluation time: {:?}", now.elapsed());

    // Server has finished running compute. Clients can proceed to decrypt the
    // output ciphertext using multi-party decryption.

    // Client side //

    // In multi-party decryption, each client needs to come online, download output
    // ciphertext from the server, produce decryption share, and send to other
    // parties (either via p2p or via server). After receving decryption shares
    // for output ciphertext from other parties, client can independently decrypt
    // output ciphertext.

    // each client produces decryption share
    let decryption_shares = cks
        .iter()
        .map(|k| k.gen_decryption_share(&ct_out_f1))
        .collect_vec();

    // With all decrytpion shares, client can aggregate the shares and decrypt the
    // ciphertext
    let out_f1 = cks[0].aggregate_decryption_shares(&ct_out_f1, &decryption_shares);

    // we check that output is correct
    let want_out_f1 = function1(c0_a, c1_a, c2_a, c3_a);
    assert_eq!(out_f1, want_out_f1);

    // -----------

    // Server key can be re-used for different function with different private
    // client inputs for same set of clients. Here we run `function2_fhe` for
    // the same set of client but with new inputs. Client only have to upload their
    // private inputs to the server this time.

    // Each client encrypts their private input
    let c0_a = thread_rng().gen::<u8>();
    let c0_enc = cks[0].encrypt(vec![c0_a].as_slice());
    let c1_a = thread_rng().gen::<u8>();
    let c1_enc = cks[1].encrypt(vec![c1_a].as_slice());
    let c2_a = thread_rng().gen::<u8>();
    let c2_enc = cks[2].encrypt(vec![c2_a].as_slice());
    let c3_a = thread_rng().gen::<u8>();
    let c3_enc = cks[3].encrypt(vec![c3_a].as_slice());

    // Client upload their private inputs to the server

    // Server side //

    // Server receives clients private inputs and extracts them
    let ct_c0_a = c0_enc.unseed::<Vec<Vec<u64>>>().key_switch(0).extract_at(0);
    let ct_c1_a = c1_enc.unseed::<Vec<Vec<u64>>>().key_switch(1).extract_at(0);
    let ct_c2_a = c2_enc.unseed::<Vec<Vec<u64>>>().key_switch(2).extract_at(0);
    let ct_c3_a = c3_enc.unseed::<Vec<Vec<u64>>>().key_switch(3).extract_at(0);

    // Server proceeds to evaluate `function2_fhe`
    let now = std::time::Instant::now();
    let ct_out_f2 = function2_fhe(&ct_c0_a, &ct_c1_a, &ct_c2_a, &ct_c3_a);
    println!("Function2 FHE evaluation time: {:?}", now.elapsed());

    // Client side //

    // Each client generates decrytion share for `ct_out_f2`
    let decryption_shares = cks
        .iter()
        .map(|k| k.gen_decryption_share(&ct_out_f2))
        .collect_vec();

    // Client independently aggregate the shares and decrypt
    let out_f2 = cks[0].aggregate_decryption_shares(&ct_out_f2, &decryption_shares);

    // We check correctness of function2
    let want_out_f2 = function2(c0_a, c1_a, c2_a, c3_a);
    assert_eq!(out_f2, want_out_f2);
}
