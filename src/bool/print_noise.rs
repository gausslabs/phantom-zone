use std::{fmt::Debug, iter::Sum, thread::panicking};

use itertools::izip;
use num_traits::{FromPrimitive, Pow, PrimInt, Zero};
use rand_distr::uniform::SampleUniform;

use crate::{
    backend::{GetModulus, Modulus},
    decomposer::RlweDecomposer,
    lwe::{decrypt_lwe, lwe_key_switch},
    parameters::{self, BoolParameters, CiphertextModulus},
    pbs::PbsKey,
    random::{DefaultSecureRng, RandomFillUniformInModulus},
    rgsw::{decrypt_rlwe, galois_auto, IsTrivial, RlweCiphertext},
    utils::{encode_x_pow_si_with_emebedding_factor, tests::Stats, TryConvertFrom1, WithLocal},
    ArithmeticOps, ClientKey, Decomposer, DefaultDecomposer, MatrixEntity, MatrixMut, ModInit, Ntt,
    NttInit, RowEntity, RowMut, VectorOps,
};

use super::{
    keys::{
        tests::{ideal_sk_lwe, ideal_sk_rlwe},
        SeededMultiPartyServerKey, ServerKeyEvaluationDomain,
    },
    BoolEvaluator,
};

struct ServerKeyStats<T> {
    brk_rgsw_cts: (Stats<T>, Stats<T>),
    post_1_auto: Stats<T>,
    post_lwe_key_switch: Stats<T>,
}

impl<T: PrimInt + FromPrimitive + Debug + Sum> ServerKeyStats<T>
where
    T: for<'a> Sum<&'a T>,
{
    fn new() -> Self {
        ServerKeyStats {
            brk_rgsw_cts: (Stats::default(), Stats::default()),
            post_1_auto: Stats::default(),
            post_lwe_key_switch: Stats::default(),
        }
    }

    fn add_noise_brk_rgsw_cts_nsm(&mut self, noise: &[T]) {
        self.brk_rgsw_cts.0.add_more(noise);
    }

    fn add_noise_brk_rgsw_cts_m(&mut self, noise: &[T]) {
        self.brk_rgsw_cts.1.add_more(noise);
    }

    fn add_noise_post_1_auto(&mut self, noise: &[T]) {
        self.post_1_auto.add_more(&noise);
    }

    fn add_noise_post_kwe_key_switch(&mut self, noise: &[T]) {
        self.post_lwe_key_switch.add_more(&noise);
    }
}

fn collect_server_key_stats<
    M: MatrixEntity + MatrixMut,
    D: Decomposer<Element = M::MatElement>,
    NttOp: NttInit<CiphertextModulus<M::MatElement>> + Ntt<Element = M::MatElement>,
    ModOp: VectorOps<Element = M::MatElement>
        + ArithmeticOps<Element = M::MatElement>
        + ModInit<M = CiphertextModulus<M::MatElement>>
        + GetModulus<M = CiphertextModulus<M::MatElement>, Element = M::MatElement>,
>(
    parameters: BoolParameters<M::MatElement>,
    client_keys: &[ClientKey],
    server_key: &ServerKeyEvaluationDomain<M, BoolParameters<u64>, DefaultSecureRng, NttOp>,
) -> ServerKeyStats<i64>
where
    M::R: RowMut + RowEntity + TryConvertFrom1<[i32], CiphertextModulus<M::MatElement>> + Clone,
    M::MatElement: Copy + PrimInt + FromPrimitive + SampleUniform + Zero + Debug,
{
    let ideal_sk_rlwe = ideal_sk_rlwe(client_keys);
    let ideal_sk_lwe = ideal_sk_lwe(client_keys);

    let embedding_factor = (2 * parameters.rlwe_n().0) / parameters.br_q();
    let rlwe_n = parameters.rlwe_n().0;
    let rlwe_q = parameters.rlwe_q();
    let lwe_q = parameters.lwe_q();
    let rlwe_modop = ModOp::new(rlwe_q.clone());
    let rlwe_nttop = NttOp::new(rlwe_q, rlwe_n);
    let lwe_modop = ModOp::new(*parameters.lwe_q());

    let rlwe_x_rgsw_decomposer = parameters.rlwe_rgsw_decomposer::<D>();
    let (rlwe_x_rgsw_gadget_a, rlwe_x_rgsw_gadget_b) = (
        rlwe_x_rgsw_decomposer.a().gadget_vector(),
        rlwe_x_rgsw_decomposer.b().gadget_vector(),
    );

    let lwe_ks_decomposer = parameters.lwe_decomposer::<D>();

    let mut server_key_stats = ServerKeyStats::new();

    let mut rng = DefaultSecureRng::new();

    // RGSW ciphertext noise
    // Check noise in RGSW ciphertexts of ideal LWE secret elements
    {
        izip!(ideal_sk_lwe.iter(), server_key.rgsw_cts().iter()).for_each(|(s_i, rgsw_ct_i)| {
            // X^{s[i]}
            let m_si = encode_x_pow_si_with_emebedding_factor::<M::R, _>(
                *s_i,
                embedding_factor,
                rlwe_n,
                rlwe_q,
            );

            // RLWE'(-sm)
            let mut neg_s_eval = M::R::try_convert_from(ideal_sk_rlwe.as_slice(), rlwe_q);
            rlwe_modop.elwise_neg_mut(neg_s_eval.as_mut());
            rlwe_nttop.forward(neg_s_eval.as_mut());

            for j in 0..rlwe_x_rgsw_decomposer.a().decomposition_count() {
                // RLWE(B^{j} * -s[X]*X^{s_lwe[i]})

                // -s[X]*X^{s_lwe[i]}*B_j
                let mut m_ideal = m_si.clone();
                rlwe_nttop.forward(m_ideal.as_mut());
                rlwe_modop.elwise_mul_mut(m_ideal.as_mut(), neg_s_eval.as_ref());
                rlwe_nttop.backward(m_ideal.as_mut());
                rlwe_modop.elwise_scalar_mul_mut(m_ideal.as_mut(), &rlwe_x_rgsw_gadget_a[j]);

                // RLWE(-s*X^{s_lwe[i]}*B_j)
                let mut rlwe_ct = M::zeros(2, rlwe_n);
                rlwe_ct
                    .get_row_mut(0)
                    .copy_from_slice(rgsw_ct_i.get_row_slice(j));
                rlwe_ct.get_row_mut(1).copy_from_slice(
                    rgsw_ct_i.get_row_slice(j + rlwe_x_rgsw_decomposer.a().decomposition_count()),
                );
                // RGSW ciphertexts are in eval domain. We put RLWE ciphertexts back in
                // coefficient domain
                rlwe_ct
                    .iter_rows_mut()
                    .for_each(|r| rlwe_nttop.backward(r.as_mut()));

                let mut m_back = M::R::zeros(rlwe_n);
                decrypt_rlwe(
                    &rlwe_ct,
                    &ideal_sk_rlwe,
                    &mut m_back,
                    &rlwe_nttop,
                    &rlwe_modop,
                );

                // diff
                rlwe_modop.elwise_sub_mut(m_back.as_mut(), m_ideal.as_ref());
                server_key_stats.add_noise_brk_rgsw_cts_nsm(&Vec::<i64>::try_convert_from(
                    m_back.as_ref(),
                    rlwe_q,
                ));
            }

            // RLWE'(m)
            for j in 0..rlwe_x_rgsw_decomposer.b().decomposition_count() {
                // RLWE(B^{j} * X^{s_lwe[i]})

                // X^{s_lwe[i]}*B_j
                let mut m_ideal = m_si.clone();
                rlwe_modop.elwise_scalar_mul_mut(m_ideal.as_mut(), &rlwe_x_rgsw_gadget_b[j]);

                // RLWE(X^{s_lwe[i]}*B_j)
                let mut rlwe_ct = M::zeros(2, rlwe_n);
                rlwe_ct.get_row_mut(0).copy_from_slice(
                    rgsw_ct_i
                        .get_row_slice(j + (2 * rlwe_x_rgsw_decomposer.a().decomposition_count())),
                );
                rlwe_ct
                    .get_row_mut(1)
                    .copy_from_slice(rgsw_ct_i.get_row_slice(
                        j + (2 * rlwe_x_rgsw_decomposer.a().decomposition_count())
                            + rlwe_x_rgsw_decomposer.b().decomposition_count(),
                    ));
                rlwe_ct
                    .iter_rows_mut()
                    .for_each(|r| rlwe_nttop.backward(r.as_mut()));

                let mut m_back = M::R::zeros(rlwe_n);
                decrypt_rlwe(
                    &rlwe_ct,
                    &ideal_sk_rlwe,
                    &mut m_back,
                    &rlwe_nttop,
                    &rlwe_modop,
                );

                // diff
                rlwe_modop.elwise_sub_mut(m_back.as_mut(), m_ideal.as_ref());
                server_key_stats.add_noise_brk_rgsw_cts_m(&Vec::<i64>::try_convert_from(
                    m_back.as_ref(),
                    rlwe_q,
                ));
            }
        });
    }

    // Noise in ciphertext after 1 auto
    // For each auto key g^k. Sample random polynomial m(X) and multiply with
    // -s(X^{g^k}) using key corresponding to auto g^k. Then check the noise in
    // resutling RLWE(m(X) * -s(X^{g^k}))
    {
        let neg_s = {
            let mut s = M::R::try_convert_from(ideal_sk_rlwe.as_slice(), rlwe_q);
            rlwe_modop.elwise_neg_mut(s.as_mut());
            s
        };
        let g = parameters.g();
        let br_q = parameters.br_q();
        let g_dlogs = parameters.auto_element_dlogs();
        let auto_decomposer = parameters.auto_decomposer::<D>();
        let mut scratch_matrix = M::zeros(auto_decomposer.decomposition_count() + 2, rlwe_n);

        g_dlogs.iter().for_each(|k| {
            let g_pow_k = if *k == 0 {
                -(g as isize)
            } else {
                (g.pow(*k as u32) % br_q) as isize
            };

            // Send s(X) -> s(X^{g^k})
            let (auto_index_map, auto_sign_map) = crate::rgsw::generate_auto_map(rlwe_n, g_pow_k);
            let mut neg_s_g_k = M::R::zeros(rlwe_n);
            izip!(
                neg_s.as_ref().iter(),
                auto_index_map.iter(),
                auto_sign_map.iter()
            )
            .for_each(|(el, to_index, to_sign)| {
                if !to_sign {
                    neg_s_g_k.as_mut()[*to_index] = rlwe_modop.neg(el);
                } else {
                    neg_s_g_k.as_mut()[*to_index] = *el;
                }
            });

            let mut m = M::R::zeros(rlwe_n);
            RandomFillUniformInModulus::random_fill(&mut rng, rlwe_q, m.as_mut());

            // We want -m(X^{g^k})s(X^{g^k}) after key switch
            let want_m = {
                let mut m_g_k_eval = M::R::zeros(rlwe_n);
                // send m(X) -> m(X^{g^k})
                izip!(
                    m.as_ref().iter(),
                    auto_index_map.iter(),
                    auto_sign_map.iter()
                )
                .for_each(|(el, to_index, to_sign)| {
                    if !to_sign {
                        m_g_k_eval.as_mut()[*to_index] = rlwe_modop.neg(el);
                    } else {
                        m_g_k_eval.as_mut()[*to_index] = *el;
                    }
                });

                rlwe_nttop.forward(m_g_k_eval.as_mut());
                let mut s_g_k = neg_s_g_k.clone();
                rlwe_nttop.forward(s_g_k.as_mut());
                rlwe_modop.elwise_mul_mut(m_g_k_eval.as_mut(), s_g_k.as_ref());
                rlwe_nttop.backward(m_g_k_eval.as_mut());
                m_g_k_eval
            };

            // RLWE auto sends part A, A(X), of RLWE to A(X^{g^k}) and then multiplies it
            // with -s(X^{g^k}) using auto key. Deliberately set RLWE = (0, m(X))
            // (ie. m in part A) to get back RLWE(-m(X^{g^k})s(X^{g^k}))
            let mut rlwe = RlweCiphertext::<_, DefaultSecureRng>::new_trivial(M::zeros(2, rlwe_n));
            rlwe.data.get_row_mut(0).copy_from_slice(m.as_ref());
            rlwe.set_not_trivial();

            galois_auto(
                &mut rlwe,
                server_key.galois_key_for_auto(*k),
                &mut scratch_matrix,
                &auto_index_map,
                &auto_sign_map,
                &rlwe_modop,
                &rlwe_nttop,
                &auto_decomposer,
            );

            // decrypt RLWE(-m(X)s(X^{g^k]}))
            let mut back_m = M::R::zeros(rlwe_n);
            decrypt_rlwe(&rlwe, &ideal_sk_rlwe, &mut back_m, &rlwe_nttop, &rlwe_modop);

            // check difference
            let mut diff = back_m;
            rlwe_modop.elwise_sub_mut(diff.as_mut(), want_m.as_ref());
            server_key_stats
                .add_noise_post_1_auto(&Vec::<i64>::try_convert_from(diff.as_ref(), rlwe_q));
        });

        // sample random m

        // key switch
    }

    // LWE Key switch
    // LWE key switches LWE_in = LWE_{Q_ks,N, s}(m) = (b, a_0, ... a_N) -> LWE_out =
    // LWE_{Q_{ks}, n, z}(m) = (b', a'_0, ..., a'n)
    // If LWE_in = (0, a = {a_0, ..., a_N}), then LWE_out = LWE(-a \cdot s_{rlwe})
    for _ in 0..1000 {
        let mut lwe_in = M::R::zeros(rlwe_n + 1);
        RandomFillUniformInModulus::random_fill(&mut rng, lwe_q, &mut lwe_in.as_mut()[1..]);

        // Key switch
        let mut lwe_out = M::R::zeros(parameters.lwe_n().0 + 1);
        lwe_key_switch(
            &mut lwe_out,
            &lwe_in,
            server_key.lwe_ksk(),
            &lwe_modop,
            &lwe_ks_decomposer,
        );

        // -a \cdot s
        let mut want_m = M::MatElement::zero();
        izip!(lwe_in.as_ref().iter().skip(1), ideal_sk_rlwe.iter()).for_each(|(a, b)| {
            want_m = lwe_modop.add(
                &want_m,
                &lwe_modop.mul(a, &lwe_q.map_element_from_i64(*b as i64)),
            );
        });
        want_m = lwe_modop.neg(&want_m);

        // decrypt lwe out
        let back_m = decrypt_lwe(&lwe_out, &ideal_sk_lwe, &lwe_modop);

        let noise = lwe_modop.sub(&want_m, &back_m);
        server_key_stats.add_noise_post_kwe_key_switch(&vec![lwe_q.map_element_to_i64(&noise)]);
    }

    server_key_stats
    // Auto keys noise

    // Ksk noise
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::{
        aggregate_public_key_shares, aggregate_server_key_shares,
        bool::keys::ServerKeyEvaluationDomain,
        evaluator::MultiPartyCrs,
        gen_client_key, gen_mp_keys_phase1, gen_mp_keys_phase2,
        parameters::{BoolParameters, CiphertextModulus},
        random::DefaultSecureRng,
        set_mp_seed, set_parameter_set,
        utils::WithLocal,
        BoolEvaluator, DefaultDecomposer, ModularOpsU64, Ntt, NttBackendU64,
    };

    use super::collect_server_key_stats;

    #[test]
    fn qwerty() {
        set_parameter_set(crate::ParameterSelector::HighCommunicationButFast2Party);
        set_mp_seed(MultiPartyCrs::random().seed);
        let parties = 2;
        let cks = (0..parties).map(|_| gen_client_key()).collect_vec();
        let pk_shares = cks.iter().map(|k| gen_mp_keys_phase1(k)).collect_vec();
        let pk = aggregate_public_key_shares(&pk_shares);
        let server_key_shares = cks
            .iter()
            .enumerate()
            .map(|(index, k)| gen_mp_keys_phase2(k, index, parties, &pk))
            .collect_vec();
        let seeded_server_key = aggregate_server_key_shares(&server_key_shares);
        let server_key_eval =
            ServerKeyEvaluationDomain::<_, _, DefaultSecureRng, NttBackendU64>::from(
                &seeded_server_key,
            );

        let parameters = BoolEvaluator::with_local(|e| e.parameters().clone());
        let server_key_stats = collect_server_key_stats::<
            _,
            DefaultDecomposer<u64>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
        >(parameters, &cks, &server_key_eval);

        println!(
            "Rgsw nsm std log2 {}",
            server_key_stats.brk_rgsw_cts.0.std_dev().abs().log2()
        );
        println!(
            "Rgsw m std log2 {}",
            server_key_stats.brk_rgsw_cts.1.std_dev().abs().log2()
        );
        println!(
            "rlwe post 1 auto std log2 {}",
            server_key_stats.post_1_auto.std_dev().abs().log2()
        );
        println!(
            "key switching noise rlwe secret s to lwe secret z std log2 {}",
            server_key_stats.post_lwe_key_switch.std_dev().abs().log2()
        );
    }
}

//     #[test]
//     fn noise_tester() {
//         let bool_evaluator = BoolEvaluator::<
//             Vec<Vec<u64>>,
//             NttBackendU64,
//             ModularOpsU64<CiphertextModulus<u64>>,
//             ModularOpsU64<CiphertextModulus<u64>>,
//             ShoupServerKeyEvaluationDomain<Vec<Vec<u64>>>,
//         >::new(OPTIMISED_SMALL_MP_BOOL_PARAMS);

//         // let (_, collective_pk, _, _, server_key_eval, ideal_client_key) =
//         //     _multi_party_all_keygen(&bool_evaluator, 20);
//         let no_of_parties = 2;
//         let lwe_q = bool_evaluator.pbs_info.parameters.lwe_q();
//         let rlwe_q = bool_evaluator.pbs_info.parameters.rlwe_q();
//         let lwe_n = bool_evaluator.pbs_info.parameters.lwe_n().0;
//         let rlwe_n = bool_evaluator.pbs_info.parameters.rlwe_n().0;
//         let lwe_modop = &bool_evaluator.pbs_info.lwe_modop;
//         let rlwe_nttop = &bool_evaluator.pbs_info.rlwe_nttop;
//         let rlwe_modop = &bool_evaluator.pbs_info.rlwe_modop;

//         // let rgsw_rgsw_decomposer = &bool_evaluator
//         //     .pbs_info
//         //     .parameters
//         //     .rgsw_rgsw_decomposer::<DefaultDecomposer<u64>>();
//         // let rgsw_rgsw_gadget_a = rgsw_rgsw_decomposer.0.gadget_vector();
//         // let rgsw_rgsw_gadget_b = rgsw_rgsw_decomposer.1.gadget_vector();

//         let rlwe_rgsw_decomposer =
// &bool_evaluator.pbs_info.rlwe_rgsw_decomposer;         let rlwe_rgsw_gadget_a
// = rlwe_rgsw_decomposer.0.gadget_vector();         let rlwe_rgsw_gadget_b =
// rlwe_rgsw_decomposer.1.gadget_vector();

//         let auto_decomposer = &bool_evaluator.pbs_info.auto_decomposer;
//         let auto_gadget = auto_decomposer.gadget_vector();

//         let parties = (0..no_of_parties)
//             .map(|_| bool_evaluator.client_key())
//             .collect_vec();

//         let int_mp_seed = MultiPartyCrs::random();

//         let mut ideal_rlwe_sk = vec![0i32; bool_evaluator.pbs_info.rlwe_n()];
//         parties.iter().for_each(|k| {
//             izip!(
//                 ideal_rlwe_sk.iter_mut(),
//                 InteractiveMultiPartyClientKey::sk_rlwe(k).iter()
//             )
//             .for_each(|(ideal_i, s_i)| {
//                 *ideal_i = *ideal_i + s_i;
//             });
//         });
//         let mut ideal_lwe_sk = vec![0i32; bool_evaluator.pbs_info.lwe_n()];
//         parties.iter().for_each(|k| {
//             izip!(
//                 ideal_lwe_sk.iter_mut(),
//                 InteractiveMultiPartyClientKey::sk_lwe(k).iter()
//             )
//             .for_each(|(ideal_i, s_i)| {
//                 *ideal_i = *ideal_i + s_i;
//             });
//         });

//         let mut rng = DefaultSecureRng::new();

//         // check noise in freshly encrypted RLWE ciphertext (ie var_fresh)
//         if false {
//             let mut rng = DefaultSecureRng::new();
//             let mut check = Stats { samples: vec![] };
//             for _ in 0..10 {
//                 // generate a new collective public key
//                 let mut pk_cr_seed = [0u8; 32];
//                 rng.fill_bytes(&mut pk_cr_seed);
//                 let public_key_share = parties
//                     .iter()
//                     .map(|k|
// bool_evaluator.multi_party_public_key_share(&int_mp_seed, k))
// .collect_vec();                 let collective_pk = PublicKey::<
//                     Vec<Vec<u64>>,
//                     DefaultSecureRng,
//                     ModularOpsU64<CiphertextModulus<u64>>,
//                 >::from(public_key_share.as_slice());

//                 let mut m = vec![0u64; rlwe_n];
//                 RandomFillUniformInModulus::random_fill(&mut rng, rlwe_q,
// m.as_mut_slice());                 let mut rlwe_ct = vec![vec![0u64; rlwe_n];
// 2];                 public_key_encrypt_rlwe::<_, _, _, _, i32, _>(
//                     &mut rlwe_ct,
//                     collective_pk.key(),
//                     &m,
//                     rlwe_modop,
//                     rlwe_nttop,
//                     &mut rng,
//                 );

//                 let mut m_back = vec![0u64; rlwe_n];
//                 decrypt_rlwe(
//                     &rlwe_ct,
//                     &ideal_rlwe_sk,
//                     &mut m_back,
//                     rlwe_nttop,
//                     rlwe_modop,
//                 );

//                 rlwe_modop.elwise_sub_mut(m_back.as_mut_slice(),
// m.as_slice());

//                 check.add_more(Vec::<i64>::try_convert_from(&m_back,
// rlwe_q).as_slice());             }

//             println!("Public key Std: {}", check.std_dev().abs().log2());
//         }

//         if true {
//             // Generate server key shares
//             let public_key_share = parties
//                 .iter()
//                 .map(|k|
// bool_evaluator.multi_party_public_key_share(&int_mp_seed, k))
// .collect_vec();             let collective_pk = PublicKey::<
//                 Vec<Vec<u64>>,
//                 DefaultSecureRng,
//                 ModularOpsU64<CiphertextModulus<u64>>,
//             >::from(public_key_share.as_slice());

//             let server_key_shares = parties
//                 .iter()
//                 .enumerate()
//                 .map(|(user_id, k)| {
//                     bool_evaluator.multi_party_server_key_share(
//                         user_id,
//                         no_of_parties,
//                         &int_mp_seed,
//                         collective_pk.key(),
//                         k,
//                     )
//                 })
//                 .collect_vec();

//             let seeded_server_key =
//
// bool_evaluator.aggregate_multi_party_server_key_shares(&server_key_shares);
//         }

//         // server key in Evaluation domain
//         let runtime_server_key =
//             ShoupServerKeyEvaluationDomain::from(ServerKeyEvaluationDomain::<
//                 _,
//                 _,
//                 DefaultSecureRng,
//                 NttBackendU64,
//             >::from(&seeded_server_key));

//         // check noise in RLWE x RGSW(X^{s_i}) where RGSW is accunulated RGSW
// ciphertext         if false {
//             let mut check = Stats { samples: vec![] };

//             ideal_lwe_sk.iter().enumerate().for_each(|(index, s_i)| {
//                 let rgsw_ct_i =
// runtime_server_key.rgsw_ct_lwe_si(index).as_ref();

//                 let mut m = vec![0u64; rlwe_n];
//                 RandomFillUniformInModulus::random_fill(&mut rng, rlwe_q,
// m.as_mut_slice());                 let mut rlwe_ct = vec![vec![0u64; rlwe_n];
// 2];                 public_key_encrypt_rlwe::<_, _, _, _, i32, _>(
//                     &mut rlwe_ct,
//                     collective_pk.key(),
//                     &m,
//                     rlwe_modop,
//                     rlwe_nttop,
//                     &mut rng,
//                 );

//                 let mut rlwe_after = RlweCiphertext::<_, DefaultSecureRng> {
//                     data: rlwe_ct.clone(),
//                     is_trivial: false,
//                     _phatom: PhantomData,
//                 };
//                 let mut scratch = vec![
//                     vec![0u64; rlwe_n];
//                     std::cmp::max(
//                         rlwe_rgsw_decomposer.0.decomposition_count(),
//                         rlwe_rgsw_decomposer.1.decomposition_count()
//                     ) + 2
//                 ];
//                 rlwe_by_rgsw(
//                     &mut rlwe_after,
//                     rgsw_ct_i,
//                     &mut scratch,
//                     rlwe_rgsw_decomposer,
//                     rlwe_nttop,
//                     rlwe_modop,
//                 );

//                 // m1 = X^{s[i]}
//                 let mut m1 = vec![0u64; rlwe_n];
//                 let s_i = *s_i * (bool_evaluator.pbs_info.embedding_factor as
// i32);                 if s_i < 0 {
//                     m1[rlwe_n - (s_i.abs() as usize)] = rlwe_q.neg_one()
//                 } else {
//                     m1[s_i as usize] = 1;
//                 }

//                 // (m+e) * m1
//                 let mut m_plus_e_times_m1 = vec![0u64; rlwe_n];
//                 decrypt_rlwe(
//                     &rlwe_ct,
//                     &ideal_rlwe_sk,
//                     &mut m_plus_e_times_m1,
//                     rlwe_nttop,
//                     rlwe_modop,
//                 );
//                 rlwe_nttop.forward(m_plus_e_times_m1.as_mut_slice());
//                 rlwe_nttop.forward(m1.as_mut_slice());

//                 rlwe_modop.elwise_mul_mut(m_plus_e_times_m1.as_mut_slice(),
// m1.as_slice());
// rlwe_nttop.backward(m_plus_e_times_m1.as_mut_slice());

//                 // Resulting RLWE ciphertext will equal: (m0m1 + em1) +
// e_{rlsw x rgsw}.                 // Hence, resulting rlwe ciphertext will
// have error em1 + e_{rlwe x rgsw}.                 // Here we're only
// concerned with e_{rlwe x rgsw}, that is noise added by                 //
// RLWExRGSW. Also note in practice m1 is a monomial, for ex, X^{s_{i}}, for
//                 // some i and var(em1) = var(e).
//                 let mut m_plus_e_times_m1_more_e = vec![0u64; rlwe_n];
//                 decrypt_rlwe(
//                     &rlwe_after,
//                     &ideal_rlwe_sk,
//                     &mut m_plus_e_times_m1_more_e,
//                     rlwe_nttop,
//                     rlwe_modop,
//                 );

//                 // diff
//                 rlwe_modop.elwise_sub_mut(
//                     m_plus_e_times_m1_more_e.as_mut_slice(),
//                     m_plus_e_times_m1.as_slice(),
//                 );

//                 // let noise = measure_noise(
//                 //     &rlwe_after,
//                 //     &m_plus_e_times_m1,
//                 //     rlwe_nttop,
//                 //     rlwe_modop,
//                 //     ideal_client_key.sk_rlwe.values(),
//                 // );
//                 // print!("NOISE: {}", noise);

//                 check.add_more(&Vec::<i64>::try_convert_from(
//                     &m_plus_e_times_m1_more_e,
//                     rlwe_q,
//                 ));
//             });
//             println!(
//                 "RLWE x RGSW, where RGSW has noise var_brk, std: {} {}",
//                 check.std_dev(),
//                 check.std_dev().abs().log2()
//             )
//         }

//         // check noise in Auto key
//         if false {
//             let mut check = Stats { samples: vec![] };

//             let mut neg_s_poly =
// Vec::<u64>::try_convert_from(ideal_rlwe_sk.as_slice(), rlwe_q);
// rlwe_modop.elwise_neg_mut(neg_s_poly.as_mut_slice());

//             let g = bool_evaluator.pbs_info.g();
//             let br_q = bool_evaluator.pbs_info.br_q();
//             let auto_element_dlogs =
// bool_evaluator.pbs_info.parameters.auto_element_dlogs();             for i in
// auto_element_dlogs.into_iter() {                 let g_pow = if i == 0 {
//                     -g
//                 } else {
//                     (((g as usize).pow(i as u32)) % br_q) as isize
//                 };

//                 // -s[X^k]
//                 let (auto_indices, auto_sign) = generate_auto_map(rlwe_n,
// g_pow);                 let mut neg_s_poly_auto_i = vec![0u64; rlwe_n];
//                 izip!(neg_s_poly.iter(), auto_indices.iter(),
// auto_sign.iter()).for_each(                     |(v, to_i, to_sign)| {
//                         if !to_sign {
//                             neg_s_poly_auto_i[*to_i] = rlwe_modop.neg(v);
//                         } else {
//                             neg_s_poly_auto_i[*to_i] = *v;
//                         }
//                     },
//                 );

//                 let mut auto_key_i =
// runtime_server_key.galois_key_for_auto(i).as_ref().clone(); //send i^th auto
// key to coefficient domain                 auto_key_i
//                     .iter_mut()
//                     .for_each(|r| rlwe_nttop.backward(r.as_mut_slice()));
//                 auto_gadget.iter().enumerate().for_each(|(i, b_i)| {
//                     // B^i * -s[X^k]
//                     let mut m_ideal = neg_s_poly_auto_i.clone();

//                     rlwe_modop.elwise_scalar_mul_mut(m_ideal.as_mut_slice(),
// b_i);

//                     let mut m_out = vec![0u64; rlwe_n];
//                     let mut rlwe_ct = vec![vec![0u64; rlwe_n]; 2];
//                     rlwe_ct[0].copy_from_slice(&auto_key_i[i]);
//                     rlwe_ct[1]
//
// .copy_from_slice(&auto_key_i[auto_decomposer.decomposition_count() + i]);
//                     decrypt_rlwe(&rlwe_ct, &ideal_rlwe_sk, &mut m_out,
// rlwe_nttop, rlwe_modop);

//                     // diff
//                     rlwe_modop.elwise_sub_mut(m_out.as_mut_slice(),
// m_ideal.as_slice());

//                     check.add_more(&Vec::<i64>::try_convert_from(&m_out,
// rlwe_q));                 });
//             }

//             println!("Auto key noise std dev: {}",
// check.std_dev().abs().log2());         }

//         // check noise in RLWE(X^k) after sending RLWE(X) -> RLWE(X^k)using
// collective         // auto key
//         if false {
//             let mut check = Stats { samples: vec![] };
//             let br_q = bool_evaluator.pbs_info.br_q();
//             let g = bool_evaluator.pbs_info.g();
//             let auto_element_dlogs =
// bool_evaluator.pbs_info.parameters.auto_element_dlogs();             for i in
// auto_element_dlogs.into_iter() {                 for _ in 0..10 {
//                     let mut m = vec![0u64; rlwe_n];
//                     RandomFillUniformInModulus::random_fill(&mut rng, rlwe_q,
// m.as_mut_slice());                     let mut rlwe_ct = RlweCiphertext::<_,
// DefaultSecureRng> {                         data: vec![vec![0u64; rlwe_n];
// 2],                         is_trivial: false,
//                         _phatom: PhantomData,
//                     };
//                     public_key_encrypt_rlwe::<_, _, _, _, i32, _>(
//                         &mut rlwe_ct,
//                         collective_pk.key(),
//                         &m,
//                         rlwe_modop,
//                         rlwe_nttop,
//                         &mut rng,
//                     );

//                     // We're only interested in noise increased as a result
// of automorphism.                     // Hence, we take m+e as the bench.
//                     let mut m_plus_e = vec![0u64; rlwe_n];
//                     decrypt_rlwe(
//                         &rlwe_ct,
//                         &ideal_rlwe_sk,
//                         &mut m_plus_e,
//                         rlwe_nttop,
//                         rlwe_modop,
//                     );

//                     let auto_key =
// runtime_server_key.galois_key_for_auto(i).as_ref();                     let
// (auto_map_index, auto_map_sign) = bool_evaluator.pbs_info.rlwe_auto_map(i);
//                     let mut scratch =
//                         vec![vec![0u64; rlwe_n];
// auto_decomposer.decomposition_count() + 2];                     galois_auto(
//                         &mut rlwe_ct,
//                         auto_key,
//                         &mut scratch,
//                         &auto_map_index,
//                         &auto_map_sign,
//                         rlwe_modop,
//                         rlwe_nttop,
//                         auto_decomposer,
//                     );

//                     // send m+e from X to X^k
//                     let mut m_plus_e_auto = vec![0u64; rlwe_n];
//                     izip!(m_plus_e.iter(), auto_map_index.iter(),
// auto_map_sign.iter()).for_each(                         |(v, to_index,
// to_sign)| {                             if !to_sign {
//                                 m_plus_e_auto[*to_index] = rlwe_modop.neg(v);
//                             } else {
//                                 m_plus_e_auto[*to_index] = *v
//                             }
//                         },
//                     );

//                     let mut m_out = vec![0u64; rlwe_n];
//                     decrypt_rlwe(&rlwe_ct, &ideal_rlwe_sk, &mut m_out,
// rlwe_nttop, rlwe_modop);

//                     // diff
//                     rlwe_modop.elwise_sub_mut(m_out.as_mut_slice(),
// m_plus_e_auto.as_slice());

//
// check.add_more(&Vec::<i64>::try_convert_from(m_out.as_slice(), rlwe_q));
//                 }
//             }

//             println!("Rlwe Auto Noise Std: {}",
// check.std_dev().abs().log2());         }

//         // Check noise growth in ksk
//         // TODO check in LWE key switching keys
//         if false {
//             // 1. encrypt LWE ciphertext
//             // 2. Key switching
//             // 3.
//             let mut check = Stats { samples: vec![] };

//             for _ in 0..1024 {
//                 // Encrypt m \in Q_{ks} using RLWE sk
//                 let mut lwe_in_ct = vec![0u64; rlwe_n + 1];
//                 let m = RandomElementInModulus::random(&mut rng,
// &lwe_q.q().unwrap());                 encrypt_lwe(&mut lwe_in_ct, &m,
// &ideal_rlwe_sk, lwe_modop, &mut rng);

//                 // Key switch
//                 let mut lwe_out = vec![0u64; lwe_n + 1];
//                 lwe_key_switch(
//                     &mut lwe_out,
//                     &lwe_in_ct,
//                     runtime_server_key.lwe_ksk(),
//                     lwe_modop,
//                     bool_evaluator.pbs_info.lwe_decomposer(),
//                 );

//                 // We only care about noise added by LWE key switch
//                 // m+e
//                 let m_plus_e = decrypt_lwe(&lwe_in_ct, &ideal_rlwe_sk,
// lwe_modop);

//                 let m_plus_e_plus_lwe_ksk_noise = decrypt_lwe(&lwe_out,
// &ideal_lwe_sk, lwe_modop);

//                 let diff = lwe_modop.sub(&m_plus_e_plus_lwe_ksk_noise,
// &m_plus_e);

//                 check.add_more(&vec![lwe_q.map_element_to_i64(&diff)]);
//             }

//             println!("Lwe ksk std dev: {}", check.std_dev().abs().log2());
//         }
//     }

//     // Check noise in fresh RGSW ciphertexts, ie X^{s_j[i]}, must equalnoise
// in     // // fresh RLWE ciphertext
//     if true {}
//     // test LWE ksk from RLWE -> LWE
//     // if false {
//     //     let logp = 2;
//     //     let mut rng = DefaultSecureRng::new();

//     //     let m = 1;
//     //     let encoded_m = m << (lwe_logq - logp);

//     //     // Encrypt
//     //     let mut lwe_ct = vec![0u64; rlwe_n + 1];
//     //     encrypt_lwe(
//     //         &mut lwe_ct,
//     //         &encoded_m,
//     //         ideal_client_key.sk_rlwe.values(),
//     //         lwe_modop,
//     //         &mut rng,
//     //     );

//     //     // key switch
//     //     let lwe_decomposer = &bool_evaluator.decomposer_lwe;
//     //     let mut lwe_out = vec![0u64; lwe_n + 1];
//     //     lwe_key_switch(
//     //         &mut lwe_out,
//     //         &lwe_ct,
//     //         &server_key_eval.lwe_ksk,
//     //         lwe_modop,
//     //         lwe_decomposer,
//     //     );

//     //     let encoded_m_back = decrypt_lwe(&lwe_out,
//     // ideal_client_key.sk_lwe.values(), lwe_modop);     let m_back
//     // =         ((encoded_m_back as f64 * (1 << logp) as f64) /
//     // (lwe_q as f64)).round() as u64;     dbg!(m_back, m);

//     //     let noise = measure_noise_lwe(
//     //         &lwe_out,
//     //         ideal_client_key.sk_lwe.values(),
//     //         lwe_modop,
//     //         &encoded_m,
//     //     );

//     //     println!("Noise: {noise}");
//     // }

//     // Measure noise in RGSW ciphertexts of ideal LWE secrets
//     // if true {
//     //     let gadget_vec = gadget_vector(
//     //         bool_evaluator.parameters.rlwe_logq,
//     //         bool_evaluator.parameters.logb_rgsw,
//     //         bool_evaluator.parameters.d_rgsw,
//     //     );

//     //     for i in 0..20 {
//     //         // measure noise in RGSW(s[i])
//     //         let si =
//     //             ideal_client_key.sk_lwe.values[i] *
//     // (bool_evaluator.embedding_factor as i32);         let mut
//     // si_poly = vec![0u64; rlwe_n];         if si < 0 {
//     //             si_poly[rlwe_n - (si.abs() as usize)] = rlwe_q - 1;
//     //         } else {
//     //             si_poly[(si.abs() as usize)] = 1;
//     //         }

//     //         let mut rgsw_si = server_key_eval.rgsw_cts[i].clone();
//     //         rgsw_si
//     //             .iter_mut()
//     //             .for_each(|ri| rlwe_nttop.backward(ri.as_mut()));

//     //         println!("####### Noise in RGSW(X^s_{i}) #######");
//     //         _measure_noise_rgsw(
//     //             &rgsw_si,
//     //             &si_poly,
//     //             ideal_client_key.sk_rlwe.values(),
//     //             &gadget_vec,
//     //             rlwe_q,
//     //         );
//     //         println!("####### ##################### #######");
//     //     }
//     // }

//     // // measure noise grwoth in RLWExRGSW
//     // if true {
//     //     let mut rng = DefaultSecureRng::new();
//     //     let mut carry_m = vec![0u64; rlwe_n];
//     //     RandomUniformDist1::random_fill(&mut rng, &rlwe_q,
//     // carry_m.as_mut_slice());

//     //     // RGSW(carrym)
//     //     let trivial_rlwect = vec![vec![0u64; rlwe_n],carry_m.clone()];
//     //     let mut rlwe_ct = RlweCiphertext::<_,
//     // DefaultSecureRng>::from_raw(trivial_rlwect, true);

//     //     let mut scratch_matrix_dplus2_ring = vec![vec![0u64; rlwe_n];
//     // d_rgsw + 2];     let mul_mod =
//     //         |v0: &u64, v1: &u64| (((*v0 as u128 * *v1 as u128) % (rlwe_q
// as u128)) as u64);

//     //     for i in 0..bool_evaluator.parameters.lwe_n {
//     //         rlwe_by_rgsw(
//     //             &mut rlwe_ct,
//     //             server_key_eval.rgsw_ct_lwe_si(i),
//     //             &mut scratch_matrix_dplus2_ring,
//     //             rlwe_decomposer,
//     //             rlwe_nttop,
//     //             rlwe_modop,
//     //         );

//     //         // carry_m[X] * s_i[X]
//     //         let si =
//     //             ideal_client_key.sk_lwe.values[i] *
//     // (bool_evaluator.embedding_factor as i32);         let mut
//     // si_poly = vec![0u64; rlwe_n];         if si < 0 {
//     //             si_poly[rlwe_n - (si.abs() as usize)] = rlwe_q - 1;
//     //         } else {
//     //             si_poly[(si.abs() as usize)] = 1;
//     //         }
//     //         carry_m = negacyclic_mul(&carry_m, &si_poly, mul_mod,
//     // rlwe_q);

//     //         let noise = measure_noise(
//     //             &rlwe_ct,
//     //             &carry_m,
//     //             rlwe_nttop,
//     //             rlwe_modop,
//     //             ideal_client_key.sk_rlwe.values(),
//     //         );
//     //         println!("Noise RLWE(carry_m) accumulating {i}^th secret
//     // monomial: {noise}");     }
//     // }

//     // // Check galois keys
//     // if false {
//     //     let g = bool_evaluator.g() as isize;
//     //     let mut rng = DefaultSecureRng::new();
//     //     let mut scratch_matrix_dplus2_ring = vec![vec![0u64; rlwe_n];
//     // d_rgsw + 2];     for i in [g, -g] {
//     //         let mut m = vec![0u64; rlwe_n];
//     //         RandomUniformDist1::random_fill(&mut rng, &rlwe_q,
//     // m.as_mut_slice());         let mut rlwe_ct = {
//     //             let mut data = vec![vec![0u64; rlwe_n]; 2];
//     //             public_key_encrypt_rlwe(
//     //                 &mut data,
//     //                 &collective_pk.key,
//     //                 &m,
//     //                 rlwe_modop,
//     //                 rlwe_nttop,
//     //                 &mut rng,
//     //             );
//     //             RlweCiphertext::<_, DefaultSecureRng>::from_raw(data,
//     // false)         };

//     //         let auto_key = server_key_eval.galois_key_for_auto(i);
//     //         let (auto_map_index, auto_map_sign) =
//     // generate_auto_map(rlwe_n, i);         galois_auto(
//     //             &mut rlwe_ct,
//     //             auto_key,
//     //             &mut scratch_matrix_dplus2_ring,
//     //             &auto_map_index,
//     //             &auto_map_sign,
//     //             rlwe_modop,
//     //             rlwe_nttop,
//     //             rlwe_decomposer,
//     //         );

//     //         // send m(X) -> m(X^i)
//     //         let mut m_k = vec![0u64; rlwe_n];
//     //         izip!(m.iter(), auto_map_index.iter(),
//     // auto_map_sign.iter()).for_each(             |(mi, to_index,to_sign)|
//     // // {                 if !to_sign {
//     // m_k[*to_index] = rlwe_q - *mi;                 } else {
//     //                     m_k[*to_index] = *mi;
//     //                 }
//     //             },
//     //         );

//     //         // measure noise
//     //         let noise = measure_noise(
//     //             &rlwe_ct,
//     //             &m_k,
//     //             rlwe_nttop,
//     //             rlwe_modop,
//     //             ideal_client_key.sk_rlwe.values(),
//     //         );

//     //         println!("Noise after auto k={i}: {noise}");
//     //     }
//     // }
// }
