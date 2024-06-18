use std::{cell::RefCell, sync::OnceLock};

use crate::{
    backend::ModulusPowerOf2,
    bool::parameters::ParameterVariant,
    random::DefaultSecureRng,
    utils::{Global, WithLocal},
    ModularOpsU64, NttBackendU64,
};

use super::{
    evaluator::NonInteractiveMultiPartyCrs,
    keys::{
        CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare,
        NonInteractiveServerKeyEvaluationDomain, SeededNonInteractiveMultiPartyServerKey,
        ShoupNonInteractiveServerKeyEvaluationDomain,
    },
    parameters::{BoolParameters, CiphertextModulus, NON_INTERACTIVE_SMALL_MP_BOOL_PARAMS},
    ClientKey, ParameterSelector,
};

pub type BoolEvaluator = super::evaluator::BoolEvaluator<
    Vec<Vec<u64>>,
    NttBackendU64,
    ModularOpsU64<CiphertextModulus<u64>>,
    ModulusPowerOf2<CiphertextModulus<u64>>,
    ShoupNonInteractiveServerKeyEvaluationDomain<Vec<Vec<u64>>>,
>;

thread_local! {
    static BOOL_EVALUATOR: RefCell<Option<BoolEvaluator>> = RefCell::new(None);

}
static BOOL_SERVER_KEY: OnceLock<ShoupNonInteractiveServerKeyEvaluationDomain<Vec<Vec<u64>>>> =
    OnceLock::new();

static MULTI_PARTY_CRS: OnceLock<NonInteractiveMultiPartyCrs<[u8; 32]>> = OnceLock::new();

pub fn set_parameter_set(select: ParameterSelector) {
    match select {
        ParameterSelector::NonInteractiveMultiPartyLessThanOrEqualTo16 => {
            BOOL_EVALUATOR.with_borrow_mut(|v| {
                *v = Some(BoolEvaluator::new(NON_INTERACTIVE_SMALL_MP_BOOL_PARAMS))
            });
        }
        _ => {
            panic!("Paramerters not supported")
        }
    }
}

pub fn set_common_reference_seed(seed: [u8; 32]) {
    BoolEvaluator::with_local(|e| {
        assert_eq!(
            e.parameters().variant(),
            &ParameterVariant::NonInteractiveMultiParty,
            "Set parameters do not support Non interactive multi-party"
        );
    });

    assert!(
        MULTI_PARTY_CRS
            .set(NonInteractiveMultiPartyCrs { seed: seed })
            .is_ok(),
        "Attempted to set MP SEED twice."
    )
}

pub fn gen_client_key() -> ClientKey {
    BoolEvaluator::with_local(|e| e.client_key())
}

pub fn gen_server_key_share(
    user_id: usize,
    total_users: usize,
    client_key: &ClientKey,
) -> CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<
    Vec<Vec<u64>>,
    NonInteractiveMultiPartyCrs<[u8; 32]>,
> {
    BoolEvaluator::with_local(|e| {
        let cr_seed = NonInteractiveMultiPartyCrs::global();
        e.non_interactive_multi_party_key_share(cr_seed, user_id, total_users, client_key)
    })
}

pub fn aggregate_server_key_shares(
    shares: &[CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<
        Vec<Vec<u64>>,
        NonInteractiveMultiPartyCrs<[u8; 32]>,
    >],
) -> SeededNonInteractiveMultiPartyServerKey<
    Vec<Vec<u64>>,
    NonInteractiveMultiPartyCrs<[u8; 32]>,
    BoolParameters<u64>,
> {
    BoolEvaluator::with_local(|e| {
        let cr_seed = NonInteractiveMultiPartyCrs::global();
        e.aggregate_non_interactive_multi_party_key_share(cr_seed, shares.len(), shares)
    })
}

impl
    SeededNonInteractiveMultiPartyServerKey<
        Vec<Vec<u64>>,
        NonInteractiveMultiPartyCrs<[u8; 32]>,
        BoolParameters<u64>,
    >
{
    pub fn set_server_key(&self) {
        let eval_key = NonInteractiveServerKeyEvaluationDomain::<
            _,
            BoolParameters<u64>,
            DefaultSecureRng,
            NttBackendU64,
        >::from(self);
        assert!(
            BOOL_SERVER_KEY
                .set(ShoupNonInteractiveServerKeyEvaluationDomain::from(eval_key))
                .is_ok(),
            "Attempted to set server key twice!"
        );
    }
}

impl Global for NonInteractiveMultiPartyCrs<[u8; 32]> {
    fn global() -> &'static Self {
        MULTI_PARTY_CRS
            .get()
            .expect("Non-interactive multi-party common reference string not set")
    }
}

// BOOL EVALUATOR //
impl WithLocal for BoolEvaluator {
    fn with_local<F, R>(func: F) -> R
    where
        F: Fn(&Self) -> R,
    {
        BOOL_EVALUATOR.with_borrow(|s| func(s.as_ref().expect("Parameters not set")))
    }

    fn with_local_mut<F, R>(func: F) -> R
    where
        F: Fn(&mut Self) -> R,
    {
        BOOL_EVALUATOR.with_borrow_mut(|s| func(s.as_mut().expect("Parameters not set")))
    }

    fn with_local_mut_mut<F, R>(func: &mut F) -> R
    where
        F: FnMut(&mut Self) -> R,
    {
        BOOL_EVALUATOR.with_borrow_mut(|s| func(s.as_mut().expect("Parameters not set")))
    }
}

pub(crate) type RuntimeServerKey = ShoupNonInteractiveServerKeyEvaluationDomain<Vec<Vec<u64>>>;
impl Global for RuntimeServerKey {
    fn global() -> &'static Self {
        BOOL_SERVER_KEY.get().expect("Server key not set!")
    }
}

/// Non interactive multi-party specfic encryptor decryptor routines
mod impl_enc_dec {
    use crate::{
        bool::{evaluator::BoolEncoding, keys::NonInteractiveMultiPartyClientKey},
        pbs::{sample_extract, PbsInfo, WithShoupRepr},
        random::{DefaultSecureRng, NewWithSeed, RandomFillUniformInModulus},
        rgsw::{key_switch, secret_key_encrypt_rlwe},
        utils::{TryConvertFrom1, WithLocal},
        Encryptor, KeySwitchWithId, Matrix, MatrixEntity, MatrixMut, MultiPartyDecryptor,
        RowEntity, RowMut,
    };
    use itertools::Itertools;
    use num_traits::{ToPrimitive, Zero};

    use super::*;

    type Mat = Vec<Vec<u64>>;

    pub(super) struct BatchedFheBools<C> {
        pub(super) data: Vec<C>,
    }

    impl<C: MatrixMut<MatElement = u64>> BatchedFheBools<C>
    where
        C::R: RowEntity + RowMut,
    {
        pub(super) fn extract(&self, index: usize) -> C::R {
            BoolEvaluator::with_local(|e| {
                let ring_size = e.parameters().rlwe_n().0;
                let ct_index = index / ring_size;
                let coeff_index = index % ring_size;
                let mut lwe_out = C::R::zeros(e.parameters().rlwe_n().0 + 1);
                sample_extract(
                    &mut lwe_out,
                    &self.data[ct_index],
                    e.pbs_info().modop_rlweq(),
                    coeff_index,
                );
                lwe_out
            })
        }
    }

    pub(super) struct NonInteractiveBatchedFheBools<C> {
        data: Vec<C>,
    }

    impl<M: MatrixEntity + MatrixMut<MatElement = u64>> From<&(Vec<M::R>, [u8; 32])>
        for NonInteractiveBatchedFheBools<M>
    where
        <M as Matrix>::R: RowMut,
    {
        fn from(value: &(Vec<M::R>, [u8; 32])) -> Self {
            BoolEvaluator::with_local(|e| {
                let parameters = e.parameters();
                let ring_size = parameters.rlwe_n().0;
                let rlwe_q = parameters.rlwe_q();

                let mut prng = DefaultSecureRng::new_seeded(value.1);
                let rlwes = value
                    .0
                    .iter()
                    .map(|partb| {
                        let mut rlwe = M::zeros(2, ring_size);

                        // sample A
                        RandomFillUniformInModulus::random_fill(
                            &mut prng,
                            rlwe_q,
                            rlwe.get_row_mut(0),
                        );

                        // Copy over B
                        rlwe.get_row_mut(1).copy_from_slice(partb.as_ref());

                        rlwe
                    })
                    .collect_vec();
                Self { data: rlwes }
            })
        }
    }

    impl<K> Encryptor<[bool], NonInteractiveBatchedFheBools<Mat>> for K
    where
        K: Encryptor<[bool], (Mat, [u8; 32])>,
    {
        fn encrypt(&self, m: &[bool]) -> NonInteractiveBatchedFheBools<Mat> {
            NonInteractiveBatchedFheBools::from(&K::encrypt(&self, m))
        }
    }

    impl<K> Encryptor<[bool], (Mat, [u8; 32])> for K
    where
        K: NonInteractiveMultiPartyClientKey,
        <Mat as Matrix>::R:
            TryConvertFrom1<[K::Element], CiphertextModulus<<Mat as Matrix>::MatElement>>,
    {
        fn encrypt(&self, m: &[bool]) -> (Mat, [u8; 32]) {
            BoolEvaluator::with_local(|e| {
                DefaultSecureRng::with_local_mut(|rng| {
                    let parameters = e.parameters();
                    let ring_size = parameters.rlwe_n().0;

                    let rlwe_count = ((m.len() as f64 / ring_size as f64).ceil())
                        .to_usize()
                        .unwrap();

                    let mut seed = <DefaultSecureRng as NewWithSeed>::Seed::default();
                    rng.fill_bytes(&mut seed);
                    let mut prng = DefaultSecureRng::new_seeded(seed);

                    let sk_u = self.sk_u_rlwe();

                    // encrypt `m` into ceil(len(m)/N) RLWE ciphertexts
                    let rlwes = (0..rlwe_count)
                        .map(|index| {
                            let mut message = vec![<Mat as Matrix>::MatElement::zero(); ring_size];
                            m[(index * ring_size)..std::cmp::min(m.len(), (index + 1) * ring_size)]
                                .iter()
                                .enumerate()
                                .for_each(|(i, v)| {
                                    if *v {
                                        message[i] = parameters.rlwe_q().true_el()
                                    } else {
                                        message[i] = parameters.rlwe_q().false_el()
                                    }
                                });

                            // encrypt message
                            let mut rlwe_out =
                                <<Mat as Matrix>::R as RowEntity>::zeros(parameters.rlwe_n().0);

                            secret_key_encrypt_rlwe(
                                &message,
                                &mut rlwe_out,
                                &sk_u,
                                e.pbs_info().modop_rlweq(),
                                e.pbs_info().nttop_rlweq(),
                                &mut prng,
                                rng,
                            );

                            rlwe_out
                        })
                        .collect_vec();

                    (rlwes, seed)
                })
            })
        }
    }

    impl KeySwitchWithId<Mat> for Mat {
        fn key_switch(&self, user_id: usize) -> Mat {
            BoolEvaluator::with_local(|e| {
                let server_key = BOOL_SERVER_KEY.get().unwrap();
                let ksk = server_key.ui_to_s_ksk(user_id);
                let decomposer = e.ni_ui_to_s_ks_decomposer().as_ref().unwrap();

                // perform key switch
                key_switch(
                    self,
                    ksk.as_ref(),
                    ksk.shoup_repr(),
                    decomposer,
                    e.pbs_info().nttop_rlweq(),
                    e.pbs_info().modop_rlweq(),
                )
            })
        }
    }

    impl<C> KeySwitchWithId<BatchedFheBools<C>> for NonInteractiveBatchedFheBools<C>
    where
        C: KeySwitchWithId<C>,
    {
        fn key_switch(&self, user_id: usize) -> BatchedFheBools<C> {
            let data = self
                .data
                .iter()
                .map(|c| c.key_switch(user_id))
                .collect_vec();
            BatchedFheBools { data }
        }
    }

    impl<E> MultiPartyDecryptor<bool, <Mat as Matrix>::R>
        for super::super::keys::ClientKey<[u8; 32], E>
    {
        type DecryptionShare = <Mat as Matrix>::MatElement;

        fn gen_decryption_share(&self, c: &<Mat as Matrix>::R) -> Self::DecryptionShare {
            BoolEvaluator::with_local(|e| e.multi_party_decryption_share(c, self))
        }

        fn aggregate_decryption_shares(
            &self,
            c: &<Mat as Matrix>::R,
            shares: &[Self::DecryptionShare],
        ) -> bool {
            BoolEvaluator::with_local(|e| e.multi_party_decrypt(shares, c))
        }
    }
}

#[cfg(test)]
mod tests {
    use impl_enc_dec::NonInteractiveBatchedFheBools;
    use itertools::{izip, Itertools};
    use num_traits::ToPrimitive;
    use rand::{thread_rng, RngCore};

    use crate::{
        backend::Modulus,
        bool::{
            evaluator::{BoolEncoding, BooleanGates},
            keys::SinglePartyClientKey,
        },
        lwe::decrypt_lwe,
        rgsw::decrypt_rlwe,
        utils::{Stats, TryConvertFrom1},
        ArithmeticOps, Encryptor, KeySwitchWithId, ModInit, MultiPartyDecryptor, NttInit,
        VectorOps,
    };

    use super::*;

    #[test]
    fn non_interactive_mp_bool_nand() {
        set_parameter_set(ParameterSelector::NonInteractiveMultiPartyLessThanOrEqualTo16);
        let mut seed = [0u8; 32];
        thread_rng().fill_bytes(&mut seed);
        set_common_reference_seed(seed);

        let parties = 2;

        let cks = (0..parties).map(|_| gen_client_key()).collect_vec();

        let key_shares = cks
            .iter()
            .enumerate()
            .map(|(user_index, ck)| gen_server_key_share(user_index, parties, ck))
            .collect_vec();

        let seeded_server_key = aggregate_server_key_shares(&key_shares);
        seeded_server_key.set_server_key();

        let parameters = BoolEvaluator::with_local(|e| e.parameters().clone());
        let nttop = NttBackendU64::new(parameters.rlwe_q(), parameters.rlwe_n().0);
        let rlwe_q_modop = ModularOpsU64::new(*parameters.rlwe_q());

        let mut ideal_rlwe_sk = vec![0i32; parameters.rlwe_n().0];
        cks.iter().for_each(|k| {
            let sk_rlwe = k.sk_rlwe();
            izip!(ideal_rlwe_sk.iter_mut(), sk_rlwe.iter()).for_each(|(a, b)| {
                *a = *a + b;
            });
        });

        let mut m0 = false;
        let mut m1 = true;

        let mut ct0 = {
            let ct: NonInteractiveBatchedFheBools<_> = cks[0].encrypt(vec![m0].as_slice());
            let ct = ct.key_switch(0);
            ct.extract(0)
        };
        let mut ct1 = {
            let ct: NonInteractiveBatchedFheBools<_> = cks[1].encrypt(vec![m1].as_slice());
            let ct = ct.key_switch(1);
            ct.extract(0)
        };

        for _ in 0..100 {
            let ct_out =
                BoolEvaluator::with_local_mut(|e| e.xor(&ct0, &ct1, RuntimeServerKey::global()));

            let decryption_shares = cks
                .iter()
                .map(|k| k.gen_decryption_share(&ct_out))
                .collect_vec();
            let m_out = cks[0].aggregate_decryption_shares(&ct_out, &decryption_shares);

            let m_expected = (m0 ^ m1);

            {
                let noisy_m = decrypt_lwe(&ct_out, &ideal_rlwe_sk, &rlwe_q_modop);
                let noise = if m_expected {
                    rlwe_q_modop.sub(&parameters.rlwe_q().true_el(), &noisy_m)
                } else {
                    rlwe_q_modop.sub(&parameters.rlwe_q().false_el(), &noisy_m)
                };
                println!(
                    "Noise: {}",
                    parameters
                        .rlwe_q()
                        .map_element_to_i64(&noise)
                        .abs()
                        .to_f64()
                        .unwrap()
                        .log2()
                )
            }

            assert!(m_out == m_expected, "Expected {m_expected} but got {m_out}");

            m1 = m0;
            m0 = m_out;

            ct1 = ct0;
            ct0 = ct_out;
        }
    }

    #[test]
    fn trialtest() {
        set_parameter_set(ParameterSelector::NonInteractiveMultiPartyLessThanOrEqualTo16);
        set_common_reference_seed([2; 32]);

        let parties = 2;

        let cks = (0..parties).map(|_| gen_client_key()).collect_vec();

        let key_shares = cks
            .iter()
            .enumerate()
            .map(|(user_index, ck)| gen_server_key_share(user_index, parties, ck))
            .collect_vec();

        let seeded_server_key = aggregate_server_key_shares(&key_shares);
        seeded_server_key.set_server_key();

        let m = vec![false, true];
        let ct: NonInteractiveBatchedFheBools<_> = cks[0].encrypt(m.as_slice());
        let ct = ct.key_switch(0);

        let parameters = BoolEvaluator::with_local(|e| e.parameters().clone());
        let nttop = NttBackendU64::new(parameters.rlwe_q(), parameters.rlwe_n().0);
        let rlwe_q_modop = ModularOpsU64::new(*parameters.rlwe_q());

        let mut ideal_rlwe_sk = vec![0i32; parameters.rlwe_n().0];
        cks.iter().for_each(|k| {
            let sk_rlwe = k.sk_rlwe();
            izip!(ideal_rlwe_sk.iter_mut(), sk_rlwe.iter()).for_each(|(a, b)| {
                *a = *a + b;
            });
        });

        let message = m
            .iter()
            .map(|b| {
                if *b {
                    parameters.rlwe_q().true_el()
                } else {
                    parameters.rlwe_q().false_el()
                }
            })
            .collect_vec();

        let mut m_out = vec![0u64; parameters.rlwe_n().0];
        decrypt_rlwe(
            &ct.data[0],
            &ideal_rlwe_sk,
            &mut m_out,
            &nttop,
            &rlwe_q_modop,
        );

        let mut diff = m_out;
        rlwe_q_modop.elwise_sub_mut(diff.as_mut_slice(), message.as_ref());

        let mut stats = Stats::new();
        stats.add_more(&Vec::<i64>::try_convert_from(
            diff.as_slice(),
            parameters.rlwe_q(),
        ));
        println!("Noise: {}", stats.std_dev().abs().log2());
    }
}
