mod impl_enc_dec {
    use crate::{
        bool::{
            evaluator::{BoolEncoding, BoolEvaluator},
            keys::NonInteractiveMultiPartyClientKey,
            parameters::CiphertextModulus,
        },
        pbs::PbsInfo,
        random::{DefaultSecureRng, NewWithSeed},
        rgsw::secret_key_encrypt_rlwe,
        utils::{TryConvertFrom1, WithLocal},
        Encryptor, Matrix, RowEntity,
    };
    use num_traits::Zero;

    trait SeededCiphertext<M, S> {
        fn new_with_seed(data: M, seed: S) -> Self;
    }

    type Mat = Vec<Vec<u64>>;

    impl<K, C> Encryptor<[bool], C> for K
    where
        K: NonInteractiveMultiPartyClientKey,
        C: SeededCiphertext<<Mat as Matrix>::R, <DefaultSecureRng as NewWithSeed>::Seed>,
        <Mat as Matrix>::R:
            TryConvertFrom1<[K::Element], CiphertextModulus<<Mat as Matrix>::MatElement>>,
    {
        fn encrypt(&self, m: &[bool]) -> C {
            BoolEvaluator::with_local(|e| {
                let parameters = e.parameters();
                assert!(m.len() <= parameters.rlwe_n().0);

                let mut message = vec![<Mat as Matrix>::MatElement::zero(); parameters.rlwe_n().0];
                m.iter().enumerate().for_each(|(i, v)| {
                    if *v {
                        message[i] = parameters.rlwe_q().true_el()
                    } else {
                        message[i] = parameters.rlwe_q().false_el()
                    }
                });

                DefaultSecureRng::with_local_mut(|rng| {
                    let mut seed = <DefaultSecureRng as NewWithSeed>::Seed::default();
                    rng.fill_bytes(&mut seed);
                    let mut prng = DefaultSecureRng::new_seeded(seed);

                    let mut rlwe_out =
                        <<Mat as Matrix>::R as RowEntity>::zeros(parameters.rlwe_n().0);

                    secret_key_encrypt_rlwe(
                        &message,
                        &mut rlwe_out,
                        &self.sk_u_rlwe(),
                        e.pbs_info().modop_rlweq(),
                        e.pbs_info().nttop_rlweq(),
                        &mut prng,
                        rng,
                    );

                    C::new_with_seed(rlwe_out, seed)
                })
            })
        }
    }
}
