use itertools::Itertools;
use phantom_zone::*;
use rand::{thread_rng, Rng, RngCore};
use serde::{Deserialize, Serialize};

/**
 * HIRING MP-FHE MATCHING SPEC
 * - match two people in job market (recruiters, job hunters)
 * - match hunter with recruiter
 * - match hunter salary request < recruiter salary provided
 * - hunter fits all requirements of recruiter
 */

const NUM_CRITERIA: usize = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct JobCriteria {
    in_market: bool,                // 0 = not in market, 1 = in market
    position: bool,                 // 0 = hunter, 1 = recruiter
    salary: u8,                     // x * $10,000
    criteria: [bool; NUM_CRITERIA], // job criteria as boolean
}

#[derive(Clone, Serialize, Deserialize)]
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
    let mut a_criteria_match = !a.criteria[0] | b.criteria[0];
    let mut b_criteria_match = !b.criteria[0] | a.criteria[0];

    for i in 1..NUM_CRITERIA {
        a_criteria_match &= !a.criteria[i] | b.criteria[i];
        b_criteria_match &= !b.criteria[i] | a.criteria[i];
    }

    let criteria_match = (!a.position | a_criteria_match) & (!b.position | b_criteria_match);

    both_in_market & compatible_pos & salary_match & criteria_match
}

fn hiring_match_fhe(a: FheJobCriteria, b: FheJobCriteria) -> FheBool {
    let both_in_market: &FheBool = &(&a.in_market & &b.in_market);

    let compatible_pos: &FheBool = &(&a.position ^ &b.position);

    let salary_match: &FheBool = &((&a.salary.gt(&b.salary)) ^ &b.position);

    let mut a_criteria_match = &!&a.criteria[0] | &b.criteria[0];
    let mut b_criteria_match = &!&b.criteria[0] | &a.criteria[0];

    for i in 1..NUM_CRITERIA {
        a_criteria_match &= &!&a.criteria[i] | &b.criteria[i];
        b_criteria_match &= &!&b.criteria[i] | &a.criteria[i];
    }

    let criteria_match =
        &(&!&a.position | &a_criteria_match) & &(&!&b.position | &b_criteria_match);

    &(&(both_in_market & compatible_pos) & salary_match) & &criteria_match
}

/**
 * FHE SETUP CODE
 */

#[derive(Clone, Serialize, Deserialize)]
struct ClientKeys {
    client_key: ClientKey,
    server_key_share: ServerKeyShare,
}

fn client_setup(id: usize, num_parties: usize) -> ClientKeys {
    let client_key = gen_client_key();
    let server_key_share = gen_server_key_share(id, num_parties, &client_key); // Changed `ck` to `client_key`

    ClientKeys {
        client_key,
        server_key_share,
    }
}

fn server_setup(server_key_shares: Vec<ServerKeyShare>) {
    let server_key = aggregate_server_key_shares(&server_key_shares);
    server_key.set_server_key();
}

/**
 * FHE FUNCTION EVAL CODE
 */

#[derive(Serialize, Deserialize)]
struct ClientEncryptedData {
    bool_enc: NonInteractiveSeededFheBools<Vec<u64>, [u8; 32]>,
    salary_enc: EncFheUint8,
}

fn client_encrypt_job_criteria(jc: JobCriteria, ck: ClientKeys) -> ClientEncryptedData {
    let bool_vec = ([jc.in_market, jc.position].iter().copied())
        .chain(jc.criteria.iter().copied())
        .collect::<Vec<_>>();

    let bool_enc = ck.client_key.encrypt(bool_vec.as_slice());
    let salary_enc = ck.client_key.encrypt(vec![jc.salary].as_slice());

    ClientEncryptedData {
        bool_enc,
        salary_enc,
    }
}

fn server_extract_job_criteria(id: usize, data: ClientEncryptedData) -> FheJobCriteria {
    let tmp = data
        .bool_enc
        .unseed::<Vec<Vec<u64>>>()
        .key_switch(id)
        .extract_all();
    let (in_market, position) = { (tmp[0].clone(), tmp[1].clone()) };

    let mut criteria: [FheBool; NUM_CRITERIA] = Default::default();
    for i in 0..NUM_CRITERIA {
        criteria[i] = tmp[i + 2].clone();
    }

    let salary = data
        .salary_enc
        .unseed::<Vec<Vec<u64>>>()
        .key_switch(id)
        .extract_at(0);

    FheJobCriteria {
        in_market,
        position,
        salary,
        criteria,
    }
}

/**
 * FHE DECRYPTION CODE
 */

fn client_generate_share(ck: ClientKeys, result: FheBool) -> u64 {
    ck.client_key.gen_decryption_share(&result)
}

fn client_full_decrypt(ck: ClientKeys, result: FheBool, shares: [u64; 2]) -> bool {
    ck.client_key.aggregate_decryption_shares(&result, &shares)
}

fn main() {
    set_parameter_set(ParameterSelector::NonInteractiveLTE2Party60Bit);

    /*
     * Phase 1: KEY SETUP
     */
    println!("Noninteractive MP-FHE Key Setup");

    // set application's common reference seed
    let mut seed = [0u8; 32];
    thread_rng().fill_bytes(&mut seed);
    set_common_reference_seed(seed);

    // Client setup
    let mut now = std::time::Instant::now();
    let ck_0 = client_setup(0, 2);
    let ck_1 = client_setup(1, 2);
    println!(
        "(1) Client keys + server shares generated, {:?}ms",
        now.elapsed().as_millis()
    );

    // Server setup
    now = std::time::Instant::now();
    server_setup(vec![
        ck_0.clone().server_key_share,
        ck_1.clone().server_key_share,
    ]);
    println!(
        "(2) Server key aggregated, {:?}ms",
        now.elapsed().as_millis()
    );

    /*
     * Phase 2: FUNCTION COMPUTATION
     */

    println!("\nFunction computation");

    // Client encryption
    now = std::time::Instant::now();

    let jc_0 = JobCriteria {
        in_market: true,
        position: false, // looking for job
        salary: 100,     // asking for at least 1mil
        criteria: [true, true, true],
    };
    let jc_1 = JobCriteria {
        in_market: true,
        position: true, // recruiter
        salary: 150,    // can pay up to 1.5mil
        criteria: [true, false, true],
    };
    let data_0 = client_encrypt_job_criteria(jc_0.clone(), ck_0.clone());
    let data_1 = client_encrypt_job_criteria(jc_1.clone(), ck_1.clone());
    println!(
        "(1) Clients encrypt their input with their own key, {:?}ms",
        now.elapsed().as_millis()
    );

    // Server extracting data from ciphertext
    now = std::time::Instant::now();
    let jc_fhe_0 = server_extract_job_criteria(0, data_0);
    let jc_fhe_1 = server_extract_job_criteria(1, data_1);
    println!(
        "(2) Client inputs extracted after key switch, {:?}ms",
        now.elapsed().as_millis()
    );

    // Server evaluating function
    now = std::time::Instant::now();
    let match_res = hiring_match(jc_0.clone(), jc_1.clone());
    let match_res_fhe = hiring_match_fhe(jc_fhe_0, jc_fhe_1);
    println!("(3) f1 evaluated, {:?}ms", now.elapsed().as_millis());

    // Clients produce decryption share
    now = std::time::Instant::now();
    let decryption_shares = [
        client_generate_share(ck_0.clone(), match_res_fhe.clone()),
        client_generate_share(ck_1, match_res_fhe.clone()),
    ];
    println!(
        "(4) Decryption shares generated, {:?}ms",
        now.elapsed().as_millis()
    );

    // Clients aggregate shares to decrypt
    now = std::time::Instant::now();
    let out_f1 = client_full_decrypt(ck_0, match_res_fhe, decryption_shares);
    println!(
        "(5) Decryption shares aggregated, data decrypted by client, {:?}ms",
        now.elapsed().as_millis()
    );

    println!("\nResult comparison");
    println!("Plaintext result: {}", match_res);
    println!("FHE result: {}", out_f1);
}

#[cfg(test)]
mod tests {
    use super::*;

    // cargo test --release --package phantom-zone --example non_interactive_hiring -- --nocapture
    #[test]
    fn test_hiring_match() {
        set_parameter_set(ParameterSelector::NonInteractiveLTE2Party80Bit);

        println!("Noninteractive MP-FHE Key Setup");

        // set application's common reference seed
        let mut seed = [0u8; 32];
        thread_rng().fill_bytes(&mut seed);
        set_common_reference_seed(seed);

        // Client setup
        let ck_0 = client_setup(0, 2);
        let ck_1 = client_setup(1, 2);

        // Server setup
        server_setup(vec![
            ck_0.clone().server_key_share,
            ck_1.clone().server_key_share,
        ]);

        // try on 25 random cases to ensure same output as plaintext
        for i in 1..25 {
            println!("Running test {}", i);

            // Client encryption
            let jc_0 = JobCriteria {
                in_market: thread_rng().gen_bool(0.95),
                position: thread_rng().gen_bool(0.2),
                salary: thread_rng().gen_range(0..120),
                criteria: [
                    thread_rng().gen_bool(0.8),
                    thread_rng().gen_bool(0.8),
                    thread_rng().gen_bool(0.8),
                ],
            };
            let jc_1 = JobCriteria {
                in_market: thread_rng().gen_bool(0.95),
                position: thread_rng().gen_bool(0.8),
                salary: thread_rng().gen_range(80..200),
                criteria: [
                    thread_rng().gen_bool(0.8),
                    thread_rng().gen_bool(0.2),
                    thread_rng().gen_bool(0.8),
                ],
            };
            let data_0 = client_encrypt_job_criteria(jc_0.clone(), ck_0.clone());
            let data_1 = client_encrypt_job_criteria(jc_1.clone(), ck_1.clone());

            // Server extracting data from ciphertext
            let jc_fhe_0 = server_extract_job_criteria(0, data_0);
            let jc_fhe_1 = server_extract_job_criteria(1, data_1);

            // Server evaluating function
            let match_res = hiring_match(jc_0.clone(), jc_1.clone());
            let match_res_fhe = hiring_match_fhe(jc_fhe_0, jc_fhe_1);

            // Clients produce decryption share
            let decryption_shares = [
                client_generate_share(ck_0.clone(), match_res_fhe.clone()),
                client_generate_share(ck_1.clone(), match_res_fhe.clone()),
            ];

            // Clients aggregate shares to decrypt
            let out_f1 = client_full_decrypt(ck_0.clone(), match_res_fhe, decryption_shares);

            assert_eq!(match_res, out_f1);
            println!("{} {:#?} {:#?}", match_res, jc_0, jc_1);
        }
    }
}
