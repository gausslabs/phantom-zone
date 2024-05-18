#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::{
        backend::{ArithmeticOps, ModInit, ModularOpsU64},
        decomposer::{Decomposer, DefaultDecomposer},
        ntt::{Ntt, NttBackendU64, NttInit},
        random::{DefaultSecureRng, RandomGaussianDist, RandomUniformDist},
        rgsw::{
            less1_rlwe_by_rgsw, measure_noise, rgsw_by_rgsw_inplace, rlwe_by_rgsw,
            secret_key_encrypt_rgsw, secret_key_encrypt_rlwe, RgswCiphertext,
            RgswCiphertextEvaluationDomain, RlweCiphertext, RlweSecret, SeededRgswCiphertext,
            SeededRlweCiphertext,
        },
        utils::{generate_prime, negacyclic_mul},
        Matrix, Row, Secret,
    };

    // // Test B part with limbd -1 when variance of m is 1
    // #[test]
    // fn trial() {
    //     let logq = 28;
    //     let ring_size = 1 << 10;
    //     let q = generate_prime(logq, (ring_size as u64) << 1, 1 <<
    // logq).unwrap();     let logb = 7;
    //     let d0 = 3;
    //     let d1 = d0 - 1;

    //     let sk = RlweSecret::random((ring_size >> 1) as usize, ring_size as
    // usize);

    //     let mut rng = DefaultSecureRng::new();
    //     let decomposer = DefaultDecomposer::new(q, logb, d0);
    //     let gadget_vector = decomposer.gadget_vector();

    //     for i in 0..100 {
    //         // m should have norm 1
    //         let mut m0 = vec![0u64; ring_size as usize];
    //         m0[thread_rng().gen_range(0..ring_size)] = 1;

    //         let modq_op = ModularOpsU64::new(q);
    //         let nttq_op = NttBackendU64::new(q, ring_size);

    //         // Encrypt RGSW(m0)
    //         let mut rgsw_seed = [0u8; 32];
    //         rng.fill_bytes(&mut rgsw_seed);
    //         let mut seeded_rgsw =
    //             SeededRgswCiphertext::<Vec<Vec<u64>>, _>::empty(ring_size,
    // d0, rgsw_seed, q);         let mut p_rng =
    // DefaultSecureRng::new_seeded(rgsw_seed);
    //         secret_key_encrypt_rgsw(
    //             &mut seeded_rgsw.data,
    //             &m0,
    //             &gadget_vector,
    //             &gadget_vector,
    //             sk.values(),
    //             &modq_op,
    //             &nttq_op,
    //             &mut p_rng,
    //             &mut rng,
    //         );

    //         // Encrypt RLWE(m1)
    //         let mut m1 = vec![0u64; ring_size];
    //         RandomUniformDist::random_fill(&mut rng, &q, m1.as_mut_slice());
    //         let mut rlwe_seed = [0u8; 32];
    //         rng.fill_bytes(&mut rlwe_seed);
    //         let mut seeded_rlwe: SeededRlweCiphertext<Vec<u64>, [u8; 32]> =
    //             SeededRlweCiphertext::<Vec<u64>, _>::empty(ring_size,
    // rlwe_seed, q);         let mut p_rng =
    // DefaultSecureRng::new_seeded(rlwe_seed);
    //         secret_key_encrypt_rlwe(
    //             &m1,
    //             &mut seeded_rlwe.data,
    //             sk.values(),
    //             &modq_op,
    //             &nttq_op,
    //             &mut p_rng,
    //             &mut rng,
    //         );

    //         let mut rlwe = RlweCiphertext::<Vec<Vec<u64>>,
    // DefaultSecureRng>::from(&seeded_rlwe);         let rgsw =
    // RgswCiphertextEvaluationDomain::<_, DefaultSecureRng,
    // NttBackendU64>::from(             &seeded_rgsw,
    //         );

    //         // RLWE(m0m1) = RLWE(m1) x RGSW(m0)
    //         let mut scratch = vec![vec![0u64; ring_size]; d0 + 2];
    //         less1_rlwe_by_rgsw(
    //             &mut rlwe,
    //             &rgsw.data,
    //             &mut scratch,
    //             &decomposer,
    //             &nttq_op,
    //             &modq_op,
    //             0,
    //             1,
    //         );
    //         // rlwe_by_rgsw(
    //         //     &mut rlwe,
    //         //     &rgsw.data,
    //         //     &mut scratch,
    //         //     &decomposer,
    //         //     &nttq_op,
    //         //     &modq_op,
    //         // );

    //         // measure noise
    //         let mul_mod = |v0: &u64, v1: &u64| ((*v0 as u128 * *v1 as u128) %
    // q as u128) as u64;         let m0m1 = negacyclic_mul(&m0, &m1,
    // mul_mod, q);         let noise = measure_noise(&rlwe, &m0m1,
    // &nttq_op, &modq_op, sk.values());         println!("Noise: {noise}");
    //     }
    // }

    // // Test B part with limbd -1 when variance of m is 1
    // #[test]
    // fn rgsw_saver() {
    //     let logq = 60;
    //     let ring_size = 1 << 11;
    //     let q = generate_prime(logq, (ring_size as u64) << 1, 1 <<
    // logq).unwrap();     let logb = 12;
    //     let d0 = 4;

    //     let sk = RlweSecret::random((ring_size >> 1) as usize, ring_size as
    // usize);

    //     let mut rng = DefaultSecureRng::new();

    //     let decomposer = DefaultDecomposer::new(q, logb, d0);
    //     let gadget_vector = decomposer.gadget_vector();

    //     for i in 0..100 {
    //         let modq_op = ModularOpsU64::new(q);
    //         let nttq_op = NttBackendU64::new(q, ring_size);

    //         // Encrypt RGSW(m0)
    //         let mut m0 = vec![0u64; ring_size as usize];
    //         m0[thread_rng().gen_range(0..ring_size)] = 1;
    //         let mut rgsw_seed = [0u8; 32];
    //         rng.fill_bytes(&mut rgsw_seed);
    //         let mut seeded_rgsw0 =
    //             SeededRgswCiphertext::<Vec<Vec<u64>>, _>::empty(ring_size,
    // d0, rgsw_seed, q);         let mut p_rng =
    // DefaultSecureRng::new_seeded(rgsw_seed);
    //         secret_key_encrypt_rgsw(
    //             &mut seeded_rgsw0.data,
    //             &m0,
    //             &gadget_vector,
    //             &gadget_vector,
    //             sk.values(),
    //             &modq_op,
    //             &nttq_op,
    //             &mut p_rng,
    //             &mut rng,
    //         );

    //         // Encrypt RGSW(m1)
    //         let mut m1 = vec![0u64; ring_size as usize];
    //         m1[thread_rng().gen_range(0..ring_size)] = 1;
    //         let mut rgsw_seed = [0u8; 32];
    //         rng.fill_bytes(&mut rgsw_seed);
    //         let mut seeded_rgsw1 =
    //             SeededRgswCiphertext::<Vec<Vec<u64>>, _>::empty(ring_size,
    // d0, rgsw_seed, q);         let mut p_rng =
    // DefaultSecureRng::new_seeded(rgsw_seed);
    //         secret_key_encrypt_rgsw(
    //             &mut seeded_rgsw1.data,
    //             &m1,
    //             &gadget_vector,
    //             &gadget_vector,
    //             sk.values(),
    //             &modq_op,
    //             &nttq_op,
    //             &mut p_rng,
    //             &mut rng,
    //         );

    //         // TODO(Jay): Why cant you create RgswCIphertext from
    // SeededRgswCiphertext?         let mut rgsw0 = {
    //             let mut evl_tmp =
    //                 RgswCiphertextEvaluationDomain::<_, DefaultSecureRng,
    // NttBackendU64>::from(                     &seeded_rgsw0,
    //                 );
    //             evl_tmp
    //                 .data
    //                 .iter_mut()
    //                 .for_each(|ri| nttq_op.backward(ri.as_mut()));
    //             evl_tmp.data
    //         };
    //         let rgsw1 = RgswCiphertextEvaluationDomain::<_, DefaultSecureRng,
    // NttBackendU64>::from(             &seeded_rgsw1,
    //         );
    //         let mut scratch_matrix_d_plus_rgsw_by_ring = vec![vec![0u64;
    // ring_size]; d0 + (d0 * 4)];

    //         // RGSW(m0m1) = RGSW(m0)xRGSW(m1)
    //         rgsw_by_rgsw_inplace(
    //             &mut rgsw0,
    //             &rgsw1.data,
    //             &decomposer,
    //             &decomposer,
    //             &mut scratch_matrix_d_plus_rgsw_by_ring,
    //             &nttq_op,
    //             &modq_op,
    //         );

    //         // send RGSW(m0m1) to Evaluation domain
    //         let mut rgsw01 = rgsw0;
    //         rgsw01
    //             .iter_mut()
    //             .for_each(|v| nttq_op.forward(v.as_mut_slice()));

    //         // RLWE(m2)
    //         let mut m2 = vec![0u64; ring_size as usize];
    //         RandomUniformDist::random_fill(&mut rng, &q, m2.as_mut_slice());
    //         let mut rlwe_seed = [0u8; 32];
    //         rng.fill_bytes(&mut rlwe_seed);
    //         let mut seeded_rlwe =
    //             SeededRlweCiphertext::<Vec<u64>, _>::empty(ring_size,
    // rlwe_seed, q);         let mut p_rng =
    // DefaultSecureRng::new_seeded(rlwe_seed);
    //         secret_key_encrypt_rlwe(
    //             &m2,
    //             &mut seeded_rlwe.data,
    //             sk.values(),
    //             &modq_op,
    //             &nttq_op,
    //             &mut p_rng,
    //             &mut rng,
    //         );

    //         let mut rlwe = RlweCiphertext::<Vec<Vec<u64>>,
    // DefaultSecureRng>::from(&seeded_rlwe);

    //         // RLWE(m0m1m2) = RLWE(m2) x RGSW(m0m1)
    //         let mut scratch_matrix_dplus2_ring = vec![vec![0u64; ring_size];
    // d0 + 2];         less1_rlwe_by_rgsw(
    //             &mut rlwe,
    //             &rgsw01,
    //             &mut scratch_matrix_dplus2_ring,
    //             &decomposer,
    //             &nttq_op,
    //             &modq_op,
    //             1,
    //             2,
    //         );

    //         let mul_mod = |v0: &u64, v1: &u64| ((*v0 as u128 * *v1 as u128) %
    // q as u128) as u64;         let m0m1 = negacyclic_mul(&m0, &m1,
    // mul_mod, q);         let m0m1m2 = negacyclic_mul(&m2, &m0m1, mul_mod,
    // q);         let noise = measure_noise(&rlwe.data, &m0m1m2, &nttq_op,
    // &modq_op, sk.values());

    //         println!("Noise: {noise}");
    //     }
    // }
}
