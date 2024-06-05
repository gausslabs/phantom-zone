use std::cell::RefCell;

mod test {
    use itertools::{izip, Itertools};

    use crate::{
        backend::{ArithmeticOps, ModularOpsU64, Modulus},
        bool::{
            set_parameter_set, BoolEncoding, BoolEvaluator, BooleanGates, CiphertextModulus,
            ClientKey, PublicKey, ServerKeyEvaluationDomain, MP_BOOL_PARAMS, SMALL_MP_BOOL_PARAMS,
        },
        lwe::{decrypt_lwe, LweSecret},
        ntt::NttBackendU64,
        pbs::PbsInfo,
        random::DefaultSecureRng,
        rgsw::RlweSecret,
        utils::Stats,
        Secret,
    };

    #[test]
    fn mp_noise() {
        // set_parameter_set(&SMALL_MP_BOOL_PARAMS);
        let mut evaluator = BoolEvaluator::<
            Vec<Vec<u64>>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
            ModularOpsU64<CiphertextModulus<u64>>,
        >::new(SMALL_MP_BOOL_PARAMS);

        let parties = 2;

        let mut rng = DefaultSecureRng::new();
        let mut pk_cr_seed = [0u8; 32];
        let mut bk_cr_seed = [0u8; 32];
        rng.fill_bytes(&mut pk_cr_seed);
        rng.fill_bytes(&mut bk_cr_seed);

        let cks = (0..parties)
            .into_iter()
            .map(|_| evaluator.client_key())
            .collect_vec();

        // construct ideal rlwe sk for meauring noise
        let ideal_client_key = {
            let mut ideal_rlwe_sk = vec![0i32; evaluator.parameters().rlwe_n().0];
            cks.iter().for_each(|k| {
                izip!(ideal_rlwe_sk.iter_mut(), k.sk_rlwe().values()).for_each(|(ideal_i, s_i)| {
                    *ideal_i = *ideal_i + s_i;
                });
            });
            let mut ideal_lwe_sk = vec![0i32; evaluator.parameters().lwe_n().0];
            cks.iter().for_each(|k| {
                izip!(ideal_lwe_sk.iter_mut(), k.sk_lwe().values()).for_each(|(ideal_i, s_i)| {
                    *ideal_i = *ideal_i + s_i;
                });
            });

            ClientKey::new(
                RlweSecret {
                    values: ideal_rlwe_sk,
                },
                LweSecret {
                    values: ideal_lwe_sk,
                },
            )
        };

        // round 1
        let pk_shares = cks
            .iter()
            .map(|c| evaluator.multi_party_public_key_share(pk_cr_seed, c))
            .collect_vec();

        // public key
        let pk = PublicKey::<Vec<Vec<u64>>, DefaultSecureRng, ModularOpsU64<CiphertextModulus<u64>>>::from(
            pk_shares.as_slice(),
        );

        // round 2
        let server_key_shares = cks
            .iter()
            .map(|c| evaluator.multi_party_server_key_share(bk_cr_seed, &pk.key(), c))
            .collect_vec();

        let server_key = evaluator.aggregate_multi_party_server_key_shares(&server_key_shares);
        let server_key_eval_domain = ServerKeyEvaluationDomain::from(&server_key);

        let mut m0 = false;
        let mut m1 = true;

        let mut c_m0 = evaluator.pk_encrypt(pk.key(), m0);
        let mut c_m1 = evaluator.pk_encrypt(pk.key(), m1);

        let true_el_encoded = evaluator.parameters().rlwe_q().true_el();
        let false_el_encoded = evaluator.parameters().rlwe_q().false_el();

        // let mut stats = Stats::new();

        for _ in 0..1000 {
            let now = std::time::Instant::now();
            let c_out = evaluator.xor(&c_m0, &c_m1, &server_key_eval_domain);
            println!("Gate time: {:?}", now.elapsed());

            // mp decrypt
            // let decryption_shares = cks
            //     .iter()
            //     .map(|c| evaluator.multi_party_decryption_share(&c_out, c))
            //     .collect_vec();
            // let m_out = evaluator.multi_party_decrypt(&decryption_shares, &c_out);
            let m_expected = (m0 ^ m1);
            // assert_eq!(m_expected, m_out, "Expected {m_expected} but got {m_out}");

            // // find noise update
            // {
            //     let out = decrypt_lwe(
            //         &c_out,
            //         ideal_client_key.sk_rlwe().values(),
            //         evaluator.pbs_info().modop_rlweq(),
            //     );

            //     let out_want = {
            //         if m_expected == true {
            //             true_el_encoded
            //         } else {
            //             false_el_encoded
            //         }
            //     };
            //     let diff = evaluator.pbs_info().modop_rlweq().sub(&out, &out_want);

            //     stats.add_more(&vec![evaluator
            //         .pbs_info()
            //         .rlwe_q()
            //         .map_element_to_i64(&diff)]);
            // }

            m1 = m0;
            m0 = m_expected;

            c_m1 = c_m0;
            c_m0 = c_out;
        }

        // println!("log2 std dev {}", stats.std_dev().abs().log2());
    }
}
