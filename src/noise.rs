#[cfg(test)]
mod tests {
    use itertools::{izip, Itertools};
    use num_traits::zero;
    use rand::{thread_rng, Rng};

    use crate::{
        bool::keys::ClientKey,
        ntt,
        random::{
            DefaultSecureRng, RandomFill, RandomFillGaussianInModulus, RandomFillUniformInModulus,
        },
        utils::{
            fill_random_ternary_secret_with_hamming_weight, generate_prime, Stats, TryConvertFrom1,
        },
        ArithmeticOps, Decomposer, DefaultDecomposer, ModInit, ModularOpsU64, Ntt, NttBackendU64,
        NttInit, VectorOps,
    };

    #[test]
    fn non_interactive_multi_party() {
        let logq = 56;
        let ring_size = 1usize << 11;
        let q = generate_prime(logq, 2 * ring_size as u64, 1 << logq).unwrap();
        let logb = 1;
        let d = 56;
        let decomposer = DefaultDecomposer::new(q, logb, d);
        let gadget_vec = decomposer.gadget_vector();
        let mut rng = DefaultSecureRng::new();

        let modop = ModularOpsU64::new(q);
        let nttop = NttBackendU64::new(&q, ring_size);

        let no_of_parties = 16;
        let client_secrets = (0..no_of_parties)
            .into_iter()
            .map(|_| {
                let mut sk = vec![0i64; ring_size];
                fill_random_ternary_secret_with_hamming_weight(&mut sk, ring_size >> 1, &mut rng);
                sk
            })
            .collect_vec();

        let mut s_ideal = vec![0i64; ring_size];
        client_secrets.iter().for_each(|s| {
            izip!(s_ideal.iter_mut(), s.iter()).for_each(|(add_to, v)| {
                *add_to = *add_to + *v;
            });
        });

        let sk_poly_ideal = Vec::<u64>::try_convert_from(s_ideal.as_slice(), &q);
        let mut sk_poly_ideal_eval = sk_poly_ideal.clone();
        nttop.forward(&mut sk_poly_ideal_eval);

        let mut ksk_seed = [0u8; 32];
        rng.fill_bytes(&mut ksk_seed);

        // zero encryptions for each party for ksk(u)
        let client_zero_encs = {
            client_secrets
                .iter()
                .map(|sk| {
                    let sk_poly = Vec::<u64>::try_convert_from(sk.as_slice(), &q);
                    let mut sk_poly_eval = sk_poly.clone();
                    nttop.forward(sk_poly_eval.as_mut_slice());

                    let mut zero_encs =
                        vec![vec![0u64; ring_size]; decomposer.decomposition_count()];
                    let mut ksk_prng = DefaultSecureRng::new_seeded(ksk_seed);
                    zero_encs.iter_mut().for_each(|out| {
                        RandomFillUniformInModulus::random_fill(
                            &mut ksk_prng,
                            &q,
                            out.as_mut_slice(),
                        );
                        nttop.forward(out.as_mut_slice());
                        modop.elwise_mul_mut(out.as_mut_slice(), &sk_poly_eval);
                        nttop.backward(out.as_mut_slice());

                        let mut error = vec![0u64; ring_size];
                        RandomFillGaussianInModulus::random_fill(&mut rng, &q, &mut error);

                        modop.elwise_add_mut(out.as_mut_slice(), &error);
                    });
                    zero_encs
                })
                .collect_vec()
        };

        // main values
        let main_a = {
            let mut a = vec![0u64; ring_size];
            RandomFillUniformInModulus::random_fill(&mut rng, &q, &mut a);
            a
        };
        let main_m = {
            let mut main_m = vec![0u64; ring_size];
            RandomFillUniformInModulus::random_fill(&mut rng, &q, &mut main_m);
            main_m
        };

        let mut main_u = vec![0i64; ring_size];
        fill_random_ternary_secret_with_hamming_weight(&mut main_u, ring_size >> 1, &mut rng);
        let u_main_poly = Vec::<u64>::try_convert_from(main_u.as_slice(), &q);
        let mut u_main_poly_eval = u_main_poly.clone();
        nttop.forward(u_main_poly_eval.as_mut_slice());

        // party 0
        let (mut party0_ksk_u, mut rlwe_main_m_parta) = {
            // party 0's secret
            let sk = client_secrets[0].clone();
            let sk_poly = Vec::<u64>::try_convert_from(sk.as_slice(), &q);
            let mut sk_poly_eval = sk_poly.clone();
            nttop.forward(sk_poly_eval.as_mut_slice());

            // `main_a*u + main_m` with ephemeral key u
            let mut rlwe_main_m = main_a.clone();
            nttop.forward(&mut rlwe_main_m);
            modop.elwise_mul_mut(&mut rlwe_main_m, &u_main_poly_eval);
            nttop.backward(&mut rlwe_main_m);
            let mut error = vec![0u64; ring_size];
            RandomFillGaussianInModulus::random_fill(&mut rng, &q, &mut error);
            modop.elwise_add_mut(&mut rlwe_main_m, &error);
            modop.elwise_add_mut(&mut rlwe_main_m, &main_m);

            // Generate KSK(u)
            let mut ksk_prng = DefaultSecureRng::new_seeded(ksk_seed);
            let mut ksk_u = vec![vec![0u64; ring_size]; 2 * decomposer.decomposition_count()];
            let (ksk_u_a, ksk_u_b) = ksk_u.split_at_mut(decomposer.decomposition_count());
            izip!(ksk_u_b.iter_mut(), ksk_u_a.iter_mut(), gadget_vec.iter()).for_each(
                |(row_b, row_a, beta_i)| {
                    // sample a
                    RandomFillUniformInModulus::random_fill(&mut ksk_prng, &q, row_a.as_mut());

                    // s_i * a
                    let mut s_i_a = row_a.clone();
                    nttop.forward(&mut s_i_a);
                    modop.elwise_mul_mut(&mut s_i_a, &sk_poly_eval);
                    nttop.backward(&mut s_i_a);

                    // \beta * u
                    let mut beta_u = u_main_poly.clone();
                    modop.elwise_scalar_mul_mut(beta_u.as_mut_slice(), beta_i);

                    // e
                    RandomFillGaussianInModulus::random_fill(&mut rng, &q, row_b.as_mut_slice());
                    // e + \beta * u
                    modop.elwise_add_mut(row_b.as_mut_slice(), &beta_u);

                    // b = e + \beta * u + a * s_i
                    modop.elwise_add_mut(row_b.as_mut_slice(), &s_i_a);
                },
            );

            // send ksk u from s_0 to s_{ideal}
            ksk_u_b.iter_mut().enumerate().for_each(|(index, out_b)| {
                // note: skip zero encryption of party 0
                client_zero_encs.iter().skip(1).for_each(|encs| {
                    modop.elwise_add_mut(out_b, &encs[index]);
                });
            });

            // // put ksk in fourier domain
            // ksk_u
            //     .iter_mut()
            //     .for_each(|r| nttop.forward(r.as_mut_slice()));
            (ksk_u, rlwe_main_m)
        };

        // Check ksk_u is correct
        // {
        //     let (ksk_a, ksk_b) =
        // party0_ksk_u.split_at_mut(decomposer.decomposition_count());
        //     izip!(
        //         ksk_a.iter(),
        //         ksk_b.iter(),
        //         decomposer.gadget_vector().iter()
        //     )
        //     .for_each(|(row_a, row_b, beta_i)| {
        //         // a * s
        //         let mut sa = row_a.clone();
        //         nttop.forward(&mut sa);
        //         modop.elwise_mul_mut(&mut sa, &sk_poly_ideal_eval);
        //         nttop.backward(&mut sa);

        //         // b - a*s
        //         let mut out = sa;
        //         modop.elwise_neg_mut(&mut out);
        //         modop.elwise_add_mut(&mut out, row_b);

        //         // beta * u
        //         let mut expected = u_main_poly.clone();
        //         modop.elwise_scalar_mul_mut(&mut expected, beta_i);
        //         assert_eq!(expected, out);
        //     });
        // }

        // RLWE(0) = main_a * s + e = \sum main_a*s_i + e_i
        let rlwe_to_switch = {
            let mut sum = vec![0u64; ring_size];
            client_secrets.iter().for_each(|sk| {
                let sk_poly = Vec::<u64>::try_convert_from(sk.as_slice(), &q);
                let mut sk_poly_eval = sk_poly.clone();
                nttop.forward(sk_poly_eval.as_mut_slice());

                // a * s
                let mut rlwe = main_a.clone();
                nttop.forward(&mut rlwe);
                modop.elwise_mul_mut(rlwe.as_mut_slice(), &sk_poly_eval);
                nttop.backward(&mut rlwe);
                // a * s + e
                let mut error = vec![0u64; ring_size];
                RandomFillGaussianInModulus::random_fill(&mut rng, &q, &mut error);
                modop.elwise_add_mut(&mut rlwe, &error);

                modop.elwise_add_mut(&mut sum, &rlwe);
            });
            sum
        };
        // {
        //     let mut tmp = main_a.clone();
        //     nttop.forward(&mut tmp);
        //     modop.elwise_mul_mut(&mut tmp, &sk_poly_ideal_eval);
        //     nttop.backward(&mut tmp);
        //     assert_eq!(&rlwe_to_switch, &tmp);
        // }

        // Key switch \sum decomp<RLWE(0)> * KSK(i)
        let mut decomp_rlwe = vec![vec![0u64; ring_size]; decomposer.decomposition_count()];
        rlwe_to_switch.iter().enumerate().for_each(|(ri, el)| {
            decomposer
                .decompose_iter(el)
                .enumerate()
                .for_each(|(j, d_el)| {
                    decomp_rlwe[j][ri] = d_el;
                });
        });

        // put ksk_u and decomp<RLWE(main_a*s_ideal + e)> in fourier domain
        decomp_rlwe
            .iter_mut()
            .for_each(|r| nttop.forward(r.as_mut_slice()));
        party0_ksk_u
            .iter_mut()
            .for_each(|r| nttop.forward(r.as_mut_slice()));

        let (ksk_u_a, ksk_u_b) = party0_ksk_u.split_at(decomposer.decomposition_count());
        let mut rlwe_main_m_partb_eval = vec![vec![0u64; ring_size]; 2];
        izip!(decomp_rlwe.iter(), ksk_u_a.iter(), ksk_u_b.iter()).for_each(|(o, a, b)| {
            // A part
            // rlwe[0] += o*a
            izip!(rlwe_main_m_partb_eval[0].iter_mut(), o.iter(), a.iter()).for_each(
                |(r, o, a)| {
                    *r = modop.add(r, &modop.mul(o, a));
                },
            );

            // B part
            // rlwe[1] += o*b
            izip!(rlwe_main_m_partb_eval[1].iter_mut(), o.iter(), b.iter()).for_each(
                |(r, o, b)| {
                    *r = modop.add(r, &modop.mul(o, b));
                },
            );
        });

        // construct RLWE_{s_{ideal}}(-sm)
        nttop.forward(rlwe_main_m_parta.as_mut_slice());
        modop.elwise_add_mut(&mut rlwe_main_m_partb_eval[0], &rlwe_main_m_parta);
        let rlwe_main_m_eval = rlwe_main_m_partb_eval;

        // decrypt RLWE_{s_{ideal}}(m) and check
        let mut neg_s_m_main_out = rlwe_main_m_eval[0].clone();
        modop.elwise_mul_mut(&mut neg_s_m_main_out, &sk_poly_ideal_eval);
        modop.elwise_neg_mut(&mut neg_s_m_main_out);
        modop.elwise_add_mut(&mut neg_s_m_main_out, &rlwe_main_m_eval[1]);
        nttop.backward(&mut neg_s_m_main_out);

        let mut neg_s_main_m = main_m.clone();
        nttop.forward(&mut neg_s_main_m);
        modop.elwise_mul_mut(&mut neg_s_main_m, &sk_poly_ideal_eval);
        modop.elwise_neg_mut(&mut neg_s_main_m);
        nttop.backward(&mut neg_s_main_m);

        let mut diff = neg_s_m_main_out.clone();
        modop.elwise_sub_mut(&mut diff, &neg_s_main_m);

        let mut stat = Stats::new();
        stat.add_more(&Vec::<i64>::try_convert_from(&diff, &q));
        println!("Log2 Std: {}", stat.std_dev().abs().log2());
    }
}
