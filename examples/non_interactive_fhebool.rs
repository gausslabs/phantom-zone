use itertools::Itertools;
use rand::{thread_rng, RngCore};

use phantom_zone::*;

fn main() {
    set_parameter_set(ParameterSelector::NonInteractiveLTE2Party);

    let mut seed = [0u8; 32];
    thread_rng().fill_bytes(&mut seed);
    set_common_reference_seed(seed);

    let parties = 2;

    let cks = (0..parties).map(|_| gen_client_key()).collect_vec();

    let s_key_shares = cks
        .iter()
        .enumerate()
        .map(|(user_id, k)| gen_server_key_share(user_id, parties, k))
        .collect_vec();

    let server_key = aggregate_server_key_shares(&s_key_shares);
    server_key.set_server_key();

    let m00 = true;
    let m01 = false;
    let m10 = true;
    let m11 = false;

    let c0 = cks[0].encrypt(vec![m00, m01].as_slice());
    let c1 = cks[1].encrypt(vec![m10, m11].as_slice());

    let (enc_m00, enc_m01) = {
        let mut tmp = c0.unseed::<Vec<Vec<u64>>>().key_switch(0).extract_all();
        (tmp.swap_remove(0), tmp.swap_remove(0))
    };

    let (enc_m10, enc_m11) = {
        let mut tmp = c1.unseed::<Vec<Vec<u64>>>().key_switch(1).extract_all();
        (tmp.swap_remove(0), tmp.swap_remove(0))
    };

    let enc_out = &(&enc_m00 & &enc_m10) | &(&enc_m01 ^ &enc_m11);
    let out = (m00 & m10) | (m10 ^ m11);

    let dec_shares = cks
        .iter()
        .map(|k| k.gen_decryption_share(&enc_out))
        .collect_vec();
    let out_back = cks[0].aggregate_decryption_shares(&enc_out, &dec_shares);

    assert!(out == out_back);
}
