use std::{cell::RefCell, sync::OnceLock};

use serde::{Deserialize, Serialize};

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
    parameters::{BoolParameters, CiphertextModulus, NI_2P, NI_4P_HB_FR, NI_8P},
    ClientKey,
};

pub(crate) type BoolEvaluator = super::evaluator::BoolEvaluator<
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

pub enum ParameterSelector {
    NonInteractiveLTE2Party,
    NonInteractiveLTE4Party,
    NonInteractiveLTE8Party,
}

pub fn set_parameter_set(select: ParameterSelector) {
    match select {
        ParameterSelector::NonInteractiveLTE2Party => {
            BOOL_EVALUATOR.with_borrow_mut(|v| *v = Some(BoolEvaluator::new(NI_2P)));
        }
        ParameterSelector::NonInteractiveLTE4Party => {
            BOOL_EVALUATOR.with_borrow_mut(|v| *v = Some(BoolEvaluator::new(NI_4P_HB_FR)));
        }
        ParameterSelector::NonInteractiveLTE8Party => {
            BOOL_EVALUATOR.with_borrow_mut(|v| *v = Some(BoolEvaluator::new(NI_8P)));
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
    BoolParameters<u64>,
    NonInteractiveMultiPartyCrs<[u8; 32]>,
> {
    BoolEvaluator::with_local(|e| {
        let cr_seed = NonInteractiveMultiPartyCrs::global();
        e.gen_non_interactive_multi_party_key_share(cr_seed, user_id, total_users, client_key)
    })
}

pub fn aggregate_server_key_shares(
    shares: &[CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<
        Vec<Vec<u64>>,
        BoolParameters<u64>,
        NonInteractiveMultiPartyCrs<[u8; 32]>,
    >],
) -> SeededNonInteractiveMultiPartyServerKey<
    Vec<Vec<u64>>,
    NonInteractiveMultiPartyCrs<[u8; 32]>,
    BoolParameters<u64>,
> {
    BoolEvaluator::with_local(|e| {
        let cr_seed = NonInteractiveMultiPartyCrs::global();
        e.aggregate_non_interactive_multi_party_server_key_shares(cr_seed, shares)
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

/// Batch of bool ciphertexts stored as vector of RLWE ciphertext under user j's
/// RLWE secret `u_j`
///
/// To use the bool ciphertexts in multi-party protocol first key switch the
/// ciphertexts from u_j to ideal RLWE secret `s` with
/// `self.key_switch(user_id)` where `user_id` is user j's id. Key switch
/// returns `BatchedFheBools` that stored key vector of key switched RLWE
/// ciphertext.
#[derive(Serialize, Deserialize)]
pub struct NonInteractiveBatchedFheBools<C> {
    data: Vec<C>,
}

/// Batch of Bool cipphertexts stored as vector of RLWE ciphertexts under the
/// ideal RLWE secret key `s` of the protocol
///
/// Bool ciphertext at `index` can be extracted from the coefficient at `index %
/// N` of `index / N`th RLWE ciphertext.
///
/// To extract bool ciphertext at `index` as LWE ciphertext use
/// `self.extract(index)`
#[derive(Serialize, Deserialize)]
pub struct BatchedFheBools<C> {
    pub data: Vec<C>,
}

/// Non interactive multi-party specfic encryptor decryptor routines
mod impl_enc_dec {
    use crate::{
        bool::{evaluator::BoolEncoding, keys::NonInteractiveMultiPartyClientKey},
        multi_party::{
            multi_party_aggregate_decryption_shares_and_decrypt, multi_party_decryption_share,
        },
        pbs::{sample_extract, PbsInfo, WithShoupRepr},
        random::{NewWithSeed, RandomFillUniformInModulus},
        rgsw::{rlwe_key_switch, seeded_secret_key_encrypt_rlwe},
        utils::TryConvertFrom1,
        Encryptor, KeySwitchWithId, Matrix, MatrixEntity, MatrixMut, MultiPartyDecryptor,
        RowEntity, RowMut,
    };
    use itertools::Itertools;
    use num_traits::{ToPrimitive, Zero};

    use super::*;

    type Mat = Vec<Vec<u64>>;

    // Implement `extract` to extract Bool LWE ciphertext at `index` from
    // `BatchedFheBools`
    impl<C: MatrixMut<MatElement = u64>> BatchedFheBools<C>
    where
        C::R: RowEntity + RowMut,
    {
        pub fn extract(&self, index: usize) -> C::R {
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

    impl<M: MatrixEntity + MatrixMut<MatElement = u64>> From<&(Vec<M::R>, [u8; 32])>
        for NonInteractiveBatchedFheBools<M>
    where
        <M as Matrix>::R: RowMut,
    {
        /// Derive `NonInteractiveBatchedFheBools` from a vector seeded RLWE
        /// ciphertexts (Vec<RLWE>, Seed)
        ///
        /// Unseed the RLWE ciphertexts and store them as vector RLWE
        /// ciphertexts in `NonInteractiveBatchedFheBools`
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
        /// Encrypt a vector bool of arbitrary length as vector of unseeded RLWE
        /// ciphertexts in `NonInteractiveBatchedFheBools`
        fn encrypt(&self, m: &[bool]) -> NonInteractiveBatchedFheBools<Mat> {
            NonInteractiveBatchedFheBools::from(&K::encrypt(&self, m))
        }
    }

    impl<K> Encryptor<[bool], (Vec<<Mat as Matrix>::R>, [u8; 32])> for K
    where
        K: NonInteractiveMultiPartyClientKey,
        <Mat as Matrix>::R:
            TryConvertFrom1<[K::Element], CiphertextModulus<<Mat as Matrix>::MatElement>>,
    {
        /// Encrypt a vector of bool of arbitrary length as vector of seeded
        /// RLWE ciphertexts and returns (Vec<RLWE>, Seed)
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

                            seeded_secret_key_encrypt_rlwe(
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

    impl<K> MultiPartyDecryptor<bool, <Mat as Matrix>::R> for K
    where
        K: NonInteractiveMultiPartyClientKey,
        <Mat as Matrix>::R:
            TryConvertFrom1<[K::Element], CiphertextModulus<<Mat as Matrix>::MatElement>>,
    {
        type DecryptionShare = <Mat as Matrix>::MatElement;

        fn gen_decryption_share(&self, c: &<Mat as Matrix>::R) -> Self::DecryptionShare {
            BoolEvaluator::with_local(|e| {
                DefaultSecureRng::with_local_mut(|rng| {
                    multi_party_decryption_share(
                        c,
                        self.sk_rlwe().as_slice(),
                        e.pbs_info().modop_rlweq(),
                        rng,
                    )
                })
            })
        }

        fn aggregate_decryption_shares(
            &self,
            c: &<Mat as Matrix>::R,
            shares: &[Self::DecryptionShare],
        ) -> bool {
            BoolEvaluator::with_local(|e| {
                let noisy_m = multi_party_aggregate_decryption_shares_and_decrypt(
                    c,
                    shares,
                    e.pbs_info().modop_rlweq(),
                );

                e.pbs_info().rlwe_q().decode(noisy_m)
            })
        }
    }

    impl KeySwitchWithId<Mat> for Mat {
        /// Key switch RLWE ciphertext `Self` from user j's RLWE secret u_j
        /// to ideal RLWE secret `s` of non-interactive multi-party protocol.
        ///
        /// - user_id: user j's user_id in the protocol
        fn key_switch(&self, user_id: usize) -> Mat {
            BoolEvaluator::with_local(|e| {
                assert!(self.dimension() == (2, e.parameters().rlwe_n().0));
                let server_key = BOOL_SERVER_KEY.get().unwrap();
                let ksk = server_key.ui_to_s_ksk(user_id);
                let decomposer = e.ni_ui_to_s_ks_decomposer().as_ref().unwrap();

                // perform key switch
                rlwe_key_switch(
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
        /// Key switch `Self`'s vector of RLWE ciphertexts from user j's RLWE
        /// secret u_j to ideal RLWE secret `s` of non-interactive
        /// multi-party protocol.
        ///
        /// Returns vector of key switched RLWE ciphertext as `BatchedFheBools`
        /// which can then be used to extract individual Bool LWE ciphertexts.
        ///
        /// - user_id: user j's user_id in the protocol
        fn key_switch(&self, user_id: usize) -> BatchedFheBools<C> {
            let data = self
                .data
                .iter()
                .map(|c| c.key_switch(user_id))
                .collect_vec();
            BatchedFheBools { data }
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::{thread_rng, RngCore};

    use crate::{
        backend::Modulus,
        bool::{
            keys::tests::{ideal_sk_rlwe, measure_noise_lwe},
            BooleanGates,
        },
        utils::tests::Stats,
        Encoder, Encryptor, KeySwitchWithId, MultiPartyDecryptor,
    };

    use super::*;
}
