use itertools::Itertools;
use phantom_zone::*;
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
    // Select parameter set
    set_parameter_set(ParameterSelector::InteractiveLTE4Party);

    // set application's common reference seed
    let mut seed = [0u8; 32];
    thread_rng().fill_bytes(&mut seed);
    set_common_reference_seed(seed);

    let no_of_parties = 4;

    // Client side //

    // Clients generate their private keys
    let cks = (0..no_of_parties)
        .into_iter()
        .map(|_| gen_client_key())
        .collect_vec();

    // -- Round 1 -- //
    // In round 1 each client generates their share for the collective public key.
    // They send public key shares to each other with or out without the server.
    // After receiving others public key shares clients independently aggregate
    // the shares and produce the collective public key `pk`

    let pk_shares = cks.iter().map(|k| collective_pk_share(k)).collect_vec();

    // Clients aggregate public key shares to produce collective public key `pk`
    let pk = aggregate_public_key_shares(&pk_shares);

    // -- Round 2 -- //
    // In round 2 each client generates server key share using the public key `pk`.
    // Clients may also encrypt their private inputs using collective public key
    // `pk`. Each client then uploads their server key share and private input
    // ciphertexts to the server.

    // Clients generate server key shares
    //
    // We assign user_id 0 to client 0, user_id 1 to client 1, user_id 2 to client
    // 2, and user_id 4 to client 4.
    //
    // Note that `user_id`'s must be unique among the clients and must be less than
    // total number of clients.
    let server_key_shares = cks
        .iter()
        .enumerate()
        .map(|(user_id, k)| collective_server_key_share(k, user_id, no_of_parties, &pk))
        .collect_vec();

    // Each client encrypts their private inputs using the collective public key
    // `pk`. Unlike non-inteactive MPC protocol, private inputs are
    // encrypted using collective public key.
    let c0_a = thread_rng().gen::<u8>();
    let c0_enc = pk.encrypt(vec![c0_a].as_slice());
    let c1_a = thread_rng().gen::<u8>();
    let c1_enc = pk.encrypt(vec![c1_a].as_slice());
    let c2_a = thread_rng().gen::<u8>();
    let c2_enc = pk.encrypt(vec![c2_a].as_slice());
    let c3_a = thread_rng().gen::<u8>();
    let c3_enc = pk.encrypt(vec![c3_a].as_slice());

    // Clients upload their server key along with private encrypted inputs to
    // the server

    // Server side //

    // Server receives server key shares from each client and proceeds to
    // aggregate the shares and produce the server key
    let server_key = aggregate_server_key_shares(&server_key_shares);
    server_key.set_server_key();

    // Server proceeds to extract clients private inputs
    //
    // Clients encrypt their FheUint8s inputs packed in a batched ciphertext.
    // The server must extract clients private inputs from the batch ciphertext
    // either (1) using `extract_at(index)` to extract `index`^{th} FheUint8
    // ciphertext (2) or using `extract_all()` to extract all available FheUint8s
    // (3) or using `extract_many(many)` to extract first `many` available FheUint8s
    let c0_a_enc = c0_enc.extract_at(0);
    let c1_a_enc = c1_enc.extract_at(0);
    let c2_a_enc = c2_enc.extract_at(0);
    let c3_a_enc = c3_enc.extract_at(0);

    // Server proceeds to evaluate function1 on clients private inputs
    let ct_out_f1 = function1_fhe(&c0_a_enc, &c1_a_enc, &c2_a_enc, &c3_a_enc);

    // After server has finished evaluating the circuit on client private
    // inputs, clients can proceed to multi-party decryption protocol to
    // decrypt output ciphertext

    // Client Side //

    // In multi-party decryption protocol, client must come online, download the
    // output ciphertext from the server, product "output ciphertext" dependent
    // decryption share, and send it to other parties. After receiving
    // decryption shares of other parties, clients independently aggregate the
    // decrytion shares and decrypt the output ciphertext.

    // Clients generate decryption shares
    let decryption_shares = cks
        .iter()
        .map(|k| k.gen_decryption_share(&ct_out_f1))
        .collect_vec();

    // After receiving decryption shares from other parties, clients aggregate the
    // shares and decrypt output ciphertext
    let out_f1 = cks[0].aggregate_decryption_shares(&ct_out_f1, &decryption_shares);

    // Check correctness of function1 output
    let want_f1 = function1(c0_a, c1_a, c2_a, c3_a);
    assert!(out_f1 == want_f1);

    // --------

    // Once server key is produced it can be re-used across different functions
    // with different private client inputs for the same set of clients.
    //
    // Here we run `function2_fhe` for the same of clients but with different
    // private inputs. Clients do not need to participate in the 2 round
    // protocol again, instead they only upload their new private inputs to the
    // server.

    // Clients encrypt their private inputs
    let c0_a = thread_rng().gen::<u8>();
    let c0_enc = pk.encrypt(vec![c0_a].as_slice());
    let c1_a = thread_rng().gen::<u8>();
    let c1_enc = pk.encrypt(vec![c1_a].as_slice());
    let c2_a = thread_rng().gen::<u8>();
    let c2_enc = pk.encrypt(vec![c2_a].as_slice());
    let c3_a = thread_rng().gen::<u8>();
    let c3_enc = pk.encrypt(vec![c3_a].as_slice());

    // Clients uploads only their new private inputs to the server

    // Server side //

    // Server receives private inputs from the clients, extracts them, and
    // proceeds to evaluate `function2_fhe`
    let c0_a_enc = c0_enc.extract_at(0);
    let c1_a_enc = c1_enc.extract_at(0);
    let c2_a_enc = c2_enc.extract_at(0);
    let c3_a_enc = c3_enc.extract_at(0);

    let ct_out_f2 = function2_fhe(&c0_a_enc, &c1_a_enc, &c2_a_enc, &c3_a_enc);

    // Client side //

    // Clients generate decryption shares for `ct_out_f2`
    let decryption_shares = cks
        .iter()
        .map(|k| k.gen_decryption_share(&ct_out_f2))
        .collect_vec();

    // Clients aggregate decryption shares and decrypt `ct_out_f2`
    let out_f2 = cks[0].aggregate_decryption_shares(&ct_out_f2, &decryption_shares);

    // We check correctness of function2
    let want_f2 = function2(c0_a, c1_a, c2_a, c3_a);
    assert!(want_f2 == out_f2);
}
