use itertools::Itertools;
use phantom_zone::*;
use rand::{thread_rng, Rng, RngCore};

/// Code that runs when conditional branch is `True`
fn circuit_branch_true(a: &FheUint8, b: &FheUint8) -> FheUint8 {
    a + b
}

/// Code that runs when conditional branch is `False`
fn circuit_branch_false(a: &FheUint8, b: &FheUint8) -> FheUint8 {
    a * b
}

// Conditional branching (ie. If and else) are generally expensive in encrypted
// domain. The code must execute all the branches, and, as apparent, the
// runtime cost grows exponentially with no. of conditional branches.
//
// In general we recommend to write branchless code. In case the code cannot be
// modified to be branchless, the code must execute all branches and use a
// muxer to select correct output at the end.
//
// Below we showcase example of a single conditional branch in encrypted domain.
// The code executes both the branches (i.e. program runs both If and Else) and
// selects output of one of the branches with a mux.
fn main() {
    set_parameter_set(ParameterSelector::NonInteractiveLTE2Party);

    // set application's common reference seed
    let mut seed = [0u8; 32];
    thread_rng().fill_bytes(&mut seed);
    set_common_reference_seed(seed);

    let no_of_parties = 2;

    // Generate client keys
    let cks = (0..no_of_parties).map(|_| gen_client_key()).collect_vec();

    // Generate server key shares
    let server_key_shares = cks
        .iter()
        .enumerate()
        .map(|(id, k)| gen_server_key_share(id, no_of_parties, k))
        .collect_vec();

    // Aggregate server key shares and set the server key
    let server_key = aggregate_server_key_shares(&server_key_shares);
    server_key.set_server_key();

    // -------

    // User 0 encrypts their private input `v_a` and User 1 encrypts their
    // private input `v_b`. We want to execute:
    //
    // if v_a < v_b:
    //      return v_a + v_b
    // else:
    //      return v_a * v_b
    //
    // We define two functions
    //      (1) `circuit_branch_true`: which executes v_a + v_b in encrypted domain.
    //      (2) `circuit_branch_false`: which executes v_a * v_b in encrypted
    //                                  domain.
    //
    // The circuit runs both `circuit_branch_true` and `circuit_branch_false` and
    // then selects the output of `circuit_branch_true` if `v_a < v_b == TRUE`
    // otherwise selects the output of `circuit_branch_false` if `v_a < v_b ==
    // FALSE` using mux.

    // Clients private inputs
    let v_a = thread_rng().gen::<u8>();
    let v_b = thread_rng().gen::<u8>();
    let v_a_enc = cks[0]
        .encrypt(vec![v_a].as_slice())
        .unseed::<Vec<Vec<u64>>>()
        .key_switch(0)
        .extract_at(0);
    let v_b_enc = cks[1]
        .encrypt(vec![v_b].as_slice())
        .unseed::<Vec<Vec<u64>>>()
        .key_switch(1)
        .extract_at(0);

    // Run both branches
    let out_true_enc = circuit_branch_true(&v_a_enc, &v_b_enc);
    let out_false_enc = circuit_branch_false(&v_a_enc, &v_b_enc);

    // define condition select v_a < v_b
    let selector_bit = v_a_enc.lt(&v_b_enc);

    // select output of `circuit_branch_true` if selector_bit == TRUE otherwise
    // select output of `circuit_branch_false`
    let out_enc = out_true_enc.mux(&out_false_enc, &selector_bit);

    let out = cks[0].aggregate_decryption_shares(
        &out_enc,
        &cks.iter()
            .map(|k| k.gen_decryption_share(&out_enc))
            .collect_vec(),
    );
    let want_out = if v_a < v_b {
        v_a.wrapping_add(v_b)
    } else {
        v_a.wrapping_mul(v_b)
    };
    assert_eq!(out, want_out);
}
