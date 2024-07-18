use itertools::Itertools;
use phantom_zone::*;
use rand::{thread_rng, Rng, RngCore};

/**
 * HIRING MP-FHE MATCHING SPEC
 * - match two people in job market (recruiters, job hunters)
 * - match hunter with recruiter
 * - match hunter salary request < recruiter salary provided
 * - hunter fits all requirements of recruiter
 */

const NUM_CRITERIA: usize = 3;

struct JobCriteria {
    in_market: bool,                // 0 = not in market, 1 = in market
    position: bool,                 // 0 = hunter, 1 = recruiter
    salary: u8,                     // x * $10,000
    criteria: [bool; NUM_CRITERIA], // job criteria as boolean
}

struct FheJobCriteria {
    in_market: FheBool,
    position: FheBool,
    salary: FheUint8,
    criteria: [FheBool; NUM_CRITERIA],
}

fn hiring_match(a: JobCriteria, b: JobCriteria) -> bool {
    // both need to be in the market
    let both_in_market = a.in_market & b.in_market;

    // need to match recruiter with hunter
    let compatible_pos = a.position ^ b.position;

    // if a is recruiter, a's salary upper bound should be higher
    // than b's salary lower bound. vice versa if b is recruiter
    let salary_match = (a.salary > b.salary) ^ b.position;

    // if a is recruiter, their criteria is required to be met for a match
    // to be made, vice versa if b is recruiter
    let mut criteria_match = (!a.position | (!a.criteria[0] | b.criteria[0]))
        | (!b.position | (!b.criteria[0] | a.criteria[0]));

    for i in 1..NUM_CRITERIA {
        criteria_match &= (!a.position | (!a.criteria[i] | b.criteria[i]))
            | (!b.position | (!b.criteria[i] | a.criteria[i]));
    }

    both_in_market & compatible_pos & salary_match & criteria_match
}

fn hiring_match_fhe(a: FheJobCriteria, b: FheJobCriteria) -> FheBool {
    let both_in_market: &FheBool = &(&a.in_market & &b.in_market);

    let compatible_pos: &FheBool = &(&a.position ^ &b.position);

    let salary_match: &FheBool = &((&a.salary.gt(&b.salary)) ^ &b.position);

    let mut criteria_match: FheBool = &(&!&a.position | &(&!&a.criteria[0] | &b.criteria[0]))
        | &(&!&b.position | &(&!&b.criteria[0] | &a.criteria[0]));

    for i in 1..NUM_CRITERIA {
        criteria_match &= &(&!&a.position | &(&!&a.criteria[i] | &b.criteria[i]))
            | &(&!&b.position | &(&!&b.criteria[i] | &a.criteria[i]));
    }

    &(&(both_in_market & compatible_pos) & salary_match) & &criteria_match
}

fn main() {
    println!("Noninteractive MP-FHE Setup");
    set_parameter_set(ParameterSelector::NonInteractiveLTE2Party);

    // set application's common reference seed
    let mut seed = [0u8; 32];
    thread_rng().fill_bytes(&mut seed);
    set_common_reference_seed(seed);

    let no_of_parties = 2;

    // Generate client keys
    let mut now = std::time::Instant::now();
    let cks = (0..no_of_parties).map(|_| gen_client_key()).collect_vec();
    println!(
        "(1) Client keys generated, {:?}ms",
        now.elapsed().as_millis()
    );

    // Clients independently generate their server key shares
    //
    // We assign user_id 0 to client 0, user_id 1 to client 1
    // Note that `user_id`s must be unique among the clients and must be less than
    // total number of clients.
    now = std::time::Instant::now();
    let server_key_shares = cks
        .iter()
        .enumerate()
        .map(|(id, k)| gen_server_key_share(id, no_of_parties, k))
        .collect_vec();
    println!(
        "(2) Clients generate server key shares, {:?}ms",
        now.elapsed().as_millis()
    );

    // Server side //

    // Server receives server key shares from each client and proceeds to aggregate
    // them to produce the server key. After this point, server can use the server
    // key to evaluate any arbitrary function on encrypted private inputs from
    // the fixed set of clients

    // aggregate server shares and generate the server key
    now = std::time::Instant::now();
    let server_key = aggregate_server_key_shares(&server_key_shares);
    server_key.set_server_key();
    println!(
        "(3) Server key aggregated, {:?}ms",
        now.elapsed().as_millis()
    );

    println!("\nF1 computation");

    // In practice, each client would upload their server key shares and encrypted
    // private inputs to the server in a single shot message.

    // Clients encrypt their private inputs in a seeded batched ciphertext using
    // their private RLWE secret `u_j`.

    // client 0 encrypts its private inputs
    now = std::time::Instant::now();
    let a_in_market = true; // in market
    let a_position = false; // looking for job
    let a_criteria = [true; NUM_CRITERIA]; // what they match
    let a_salary: u8 = 100; // requesting 1 million
    let a_bool_enc: NonInteractiveBatchedFheBools<_> = cks[0].encrypt(
        [a_in_market, a_position]
            .iter()
            .copied()
            .chain(a_criteria.iter().copied())
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let a_s_enc = cks[0].encrypt(vec![a_salary].as_slice());

    // client 1 encrypts its private inputs
    let b_in_market = true; // in market
    let b_position = true; // hiring
    let b_criteria = [true; NUM_CRITERIA];
    let b_salary: u8 = 150; // can pay up to 1.5 million
    let b_bool_enc: NonInteractiveBatchedFheBools<_> = cks[1].encrypt(
        [b_in_market, b_position]
            .iter()
            .copied()
            .chain(b_criteria.iter().copied())
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let b_s_enc = cks[1].encrypt(vec![b_salary].as_slice());

    println!(
        "(1) Clients encrypt their input with their own key, {:?}ms",
        now.elapsed().as_millis()
    );

    // Server proceeds to extract private inputs sent by clients
    //
    // To extract client 0's (with user_id=0) private inputs we first key switch
    // client 0's private inputs from their secret `u_j` to ideal secret of the mpc
    // protocol. To indicate we're key switching client 0's private input we
    // supply client 0's `user_id` i.e. we call `key_switch(0)`. Then we extract
    // the first ciphertext by calling `extract_at(0)`.
    //
    // Since client 0 only encrypts 1 input in batched ciphertext, calling
    // extract_at(index) for `index` > 0 will panic. If client 0 had more private
    // inputs then we can either extract them all at once with `extract_all` or
    // first `many` of them with `extract_many(many)`

    now = std::time::Instant::now();

    let a_bool_enc_ks = a_bool_enc.key_switch(0);
    let a_m_ct = FheBool {
        data: a_bool_enc_ks.extract(0),
    };
    let a_p_ct = FheBool {
        data: a_bool_enc_ks.extract(1),
    };
    let mut a_criteria_ct: [FheBool; NUM_CRITERIA] = Default::default();
    for i in 0..NUM_CRITERIA {
        a_criteria_ct[i] = FheBool {
            data: a_bool_enc_ks.extract(i + 2),
        };
    }
    let a_s_ct = a_s_enc
        .unseed::<Vec<Vec<u64>>>()
        .key_switch(0)
        .extract_at(0);

    let b_bool_enc_ks = b_bool_enc.key_switch(1);
    let b_m_ct = FheBool {
        data: b_bool_enc_ks.extract(0),
    };
    let b_p_ct = FheBool {
        data: b_bool_enc_ks.extract(1),
    };
    let mut b_criteria_ct: [FheBool; NUM_CRITERIA] = Default::default();
    for i in 0..NUM_CRITERIA {
        b_criteria_ct[i] = FheBool {
            data: b_bool_enc_ks.extract(i + 2),
        };
    }
    let b_s_ct = b_s_enc
        .unseed::<Vec<Vec<u64>>>()
        .key_switch(1)
        .extract_at(0);

    println!(
        "(2) Client inputs extracted after key switch, {:?}ms",
        now.elapsed().as_millis()
    );

    let a_criteria = JobCriteria {
        in_market: a_in_market,
        position: a_position,
        salary: a_salary,
        criteria: a_criteria,
    };
    let b_criteria = JobCriteria {
        in_market: b_in_market,
        position: b_position,
        salary: b_salary,
        criteria: b_criteria,
    };

    let match_res = hiring_match(a_criteria, b_criteria);

    let a_criteria_fhe = FheJobCriteria {
        in_market: a_m_ct,
        position: a_p_ct,
        salary: a_s_ct,
        criteria: a_criteria_ct,
    };
    let b_criteria_fhe = FheJobCriteria {
        in_market: b_m_ct,
        position: b_p_ct,
        salary: b_s_ct,
        criteria: b_criteria_ct,
    };

    // After extracting each client's private inputs, server proceeds to evaluate
    // function1
    now = std::time::Instant::now();
    let match_res_fhe = hiring_match_fhe(a_criteria_fhe, b_criteria_fhe);
    println!("(3) f1 evaluated, {:?}ms", now.elapsed().as_millis());

    // Server has finished running compute. Clients can proceed to decrypt the
    // output ciphertext using multi-party decryption.

    // Client side //

    // In multi-party decryption, each client needs to come online, download output
    // ciphertext from the server, produce "output ciphertext" dependent decryption
    // share, and send it to other parties (either via p2p or via server). After
    // receving decryption shares from other parties, clients can independently
    // decrypt output ciphertext.

    // each client produces decryption share
    now = std::time::Instant::now();
    let decryption_shares = cks
        .iter()
        .map(|k| k.gen_decryption_share(&match_res_fhe))
        .collect_vec();
    println!(
        "(4) Decryption shares generated, {:?}ms",
        now.elapsed().as_millis()
    );

    // With all decryption shares, clients can aggregate the shares and decrypt the
    // ciphertext
    now = std::time::Instant::now();
    let out_f1 = cks[0].aggregate_decryption_shares(&match_res_fhe, &decryption_shares);
    println!(
        "(5) Decryption shares aggregated, data decrypted by client, {:?}ms",
        now.elapsed().as_millis()
    );

    println!("\nResult comparison");
    println!("Plaintext result: {}", match_res);
    println!("FHE result: {}", out_f1);
}
