use bin_rs::*;
use itertools::Itertools;
use rand::{thread_rng, Rng, RngCore};

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

    // --------

    // We attempt to divide by 0 in encrypted domain and then check whether div by 0
    // error flag is set to True.
    let numerator = thread_rng().gen::<u8>();
    let numerator_enc = cks[0]
        .encrypt(vec![numerator].as_slice())
        .unseed::<Vec<Vec<u64>>>()
        .key_switch(0)
        .extract_at(0);
    let zero_enc = cks[1]
        .encrypt(vec![0].as_slice())
        .unseed::<Vec<Vec<u64>>>()
        .key_switch(1)
        .extract_at(0);

    let (quotient_enc, remainder_enc) = numerator_enc.div_rem(&zero_enc);

    // When attempting to divide by zero, for uint8, quotient is always 255 and
    // remainder = numerator
    let quotient = cks[0].aggregate_decryption_shares(
        &quotient_enc,
        &cks.iter()
            .map(|k| k.gen_decryption_share(&quotient_enc))
            .collect_vec(),
    );
    let remainder = cks[0].aggregate_decryption_shares(
        &remainder_enc,
        &cks.iter()
            .map(|k| k.gen_decryption_share(&remainder_enc))
            .collect_vec(),
    );
    assert!(quotient == 255);
    assert!(remainder == numerator);

    // Div by zero error flag must be True
    let div_by_zero_enc = div_zero_error_flag().expect("We performed division. Flag must be set");
    let div_by_zero = cks[0].aggregate_decryption_shares(
        &div_by_zero_enc,
        &cks.iter()
            .map(|k| k.gen_decryption_share(&div_by_zero_enc))
            .collect_vec(),
    );
    assert!(div_by_zero == true);

    // -------

    // div by zero error flag is thread local. If we were to run another circuit
    // without stopping the thread (i.e. within the same program as previous
    // one), we must reset errors flags set by previous circuit with
    // `reset_error_flags()` to prevent error flags of previous circuit affecting
    // the flags of the next circuit.
    reset_error_flags();

    // We divide again but with non-zero denominator this time and check that div
    // by zero flag is set to False
    let numerator = thread_rng().gen::<u8>();
    let denominator = thread_rng().gen::<u8>();
    let numerator_enc = cks[0]
        .encrypt(vec![numerator].as_slice())
        .unseed::<Vec<Vec<u64>>>()
        .key_switch(0)
        .extract_at(0);
    let denominator_enc = cks[1]
        .encrypt(vec![denominator].as_slice())
        .unseed::<Vec<Vec<u64>>>()
        .key_switch(1)
        .extract_at(0);

    let (quotient_enc, remainder_enc) = numerator_enc.div_rem(&denominator_enc);
    let quotient = cks[0].aggregate_decryption_shares(
        &quotient_enc,
        &cks.iter()
            .map(|k| k.gen_decryption_share(&quotient_enc))
            .collect_vec(),
    );
    let remainder = cks[0].aggregate_decryption_shares(
        &remainder_enc,
        &cks.iter()
            .map(|k| k.gen_decryption_share(&remainder_enc))
            .collect_vec(),
    );
    assert!(quotient == numerator.div_euclid(denominator));
    assert!(remainder == numerator.rem_euclid(denominator));

    // Div by zero error flag must be set to False
    let div_by_zero_enc = div_zero_error_flag().expect("We performed division. Flag must be set");
    let div_by_zero = cks[0].aggregate_decryption_shares(
        &div_by_zero_enc,
        &cks.iter()
            .map(|k| k.gen_decryption_share(&div_by_zero_enc))
            .collect_vec(),
    );
    assert!(div_by_zero == false);
}
