use std::{cell::RefCell, ops::Mul, sync::OnceLock};

use crate::{
    backend::{ModularOpsU64, ModulusPowerOf2},
    ntt::NttBackendU64,
    random::{DefaultSecureRng, NewWithSeed},
    utils::{Global, WithLocal},
};

use super::{evaluator::InteractiveMultiPartyCrs, keys::*, parameters::*, ClientKey};

pub(crate) type BoolEvaluator = super::evaluator::BoolEvaluator<
    Vec<Vec<u64>>,
    NttBackendU64,
    ModularOpsU64<CiphertextModulus<u64>>,
    ModulusPowerOf2<CiphertextModulus<u64>>,
    ShoupServerKeyEvaluationDomain<Vec<Vec<u64>>>,
>;

thread_local! {
    static BOOL_EVALUATOR: RefCell<Option<BoolEvaluator>> = RefCell::new(None);

}
static BOOL_SERVER_KEY: OnceLock<ShoupServerKeyEvaluationDomain<Vec<Vec<u64>>>> = OnceLock::new();

static MULTI_PARTY_CRS: OnceLock<InteractiveMultiPartyCrs<[u8; 32]>> = OnceLock::new();

pub enum ParameterSelector {
    InteractiveLTE2Party,
}

/// Select Interactive multi-party parameter variant
pub fn set_parameter_set(select: ParameterSelector) {
    match select {
        ParameterSelector::InteractiveLTE2Party => {
            BOOL_EVALUATOR.with_borrow_mut(|v| *v = Some(BoolEvaluator::new(I_2P)));
        }

        _ => {
            panic!("Paramerter not supported")
        }
    }
}

/// Set application specific interactive multi-party common reference string
pub fn set_mp_seed(seed: [u8; 32]) {
    assert!(
        MULTI_PARTY_CRS
            .set(InteractiveMultiPartyCrs { seed: seed })
            .is_ok(),
        "Attempted to set MP SEED twice."
    )
}

/// Generate client key for interactive multi-party protocol
pub fn gen_client_key() -> ClientKey {
    BoolEvaluator::with_local(|e| e.client_key())
}

/// Generate client's share for collective public key, i.e round 1, of the
/// protocol
pub fn gen_mp_keys_phase1(
    ck: &ClientKey,
) -> CommonReferenceSeededCollectivePublicKeyShare<Vec<u64>, [u8; 32], BoolParameters<u64>> {
    BoolEvaluator::with_local(|e| {
        let pk_share = e.multi_party_public_key_share(InteractiveMultiPartyCrs::global(), ck);
        pk_share
    })
}

/// Generate clients share for collective server key, i.e. round 2, of the
/// protocol
pub fn gen_mp_keys_phase2<R, ModOp>(
    ck: &ClientKey,
    user_id: usize,
    total_users: usize,
    pk: &PublicKey<Vec<Vec<u64>>, R, ModOp>,
) -> CommonReferenceSeededInteractiveMultiPartyServerKeyShare<
    Vec<Vec<u64>>,
    BoolParameters<u64>,
    InteractiveMultiPartyCrs<[u8; 32]>,
> {
    BoolEvaluator::with_local_mut(|e| {
        let server_key_share = e.gen_interactive_multi_party_server_key_share(
            user_id,
            total_users,
            InteractiveMultiPartyCrs::global(),
            pk.key(),
            ck,
        );
        server_key_share
    })
}

/// Aggregate public key shares from all parties.
///
/// Public key shares are generated per client in round 1. Aggregation of public
/// key shares marks the end of round 1.
pub fn aggregate_public_key_shares(
    shares: &[CommonReferenceSeededCollectivePublicKeyShare<
        Vec<u64>,
        [u8; 32],
        BoolParameters<u64>,
    >],
) -> PublicKey<Vec<Vec<u64>>, DefaultSecureRng, ModularOpsU64<CiphertextModulus<u64>>> {
    PublicKey::from(shares)
}

/// Aggregate server key shares
pub fn aggregate_server_key_shares(
    shares: &[CommonReferenceSeededInteractiveMultiPartyServerKeyShare<
        Vec<Vec<u64>>,
        BoolParameters<u64>,
        InteractiveMultiPartyCrs<[u8; 32]>,
    >],
) -> SeededInteractiveMultiPartyServerKey<
    Vec<Vec<u64>>,
    InteractiveMultiPartyCrs<[u8; 32]>,
    BoolParameters<u64>,
> {
    BoolEvaluator::with_local(|e| e.aggregate_interactive_multi_party_server_key_shares(shares))
}

impl
    SeededInteractiveMultiPartyServerKey<
        Vec<Vec<u64>>,
        InteractiveMultiPartyCrs<<DefaultSecureRng as NewWithSeed>::Seed>,
        BoolParameters<u64>,
    >
{
    /// Sets the server key as a global reference for circuit evaluation
    pub fn set_server_key(&self) {
        assert!(
            BOOL_SERVER_KEY
                .set(ShoupServerKeyEvaluationDomain::from(
                    ServerKeyEvaluationDomain::<_, _, DefaultSecureRng, NttBackendU64>::from(self),
                ))
                .is_ok(),
            "Attempted to set server key twice."
        );
    }
}

//  MULTIPARTY CRS  //
impl Global for InteractiveMultiPartyCrs<[u8; 32]> {
    fn global() -> &'static Self {
        MULTI_PARTY_CRS
            .get()
            .expect("Multi Party Common Reference String not set")
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

pub(crate) type RuntimeServerKey = ShoupServerKeyEvaluationDomain<Vec<Vec<u64>>>;
impl Global for RuntimeServerKey {
    fn global() -> &'static Self {
        BOOL_SERVER_KEY.get().expect("Server key not set!")
    }
}

mod impl_enc_dec {
    use crate::{
        bool::evaluator::BoolEncoding,
        pbs::{sample_extract, PbsInfo},
        rgsw::public_key_encrypt_rlwe,
        Encryptor, Matrix, MatrixEntity, RowEntity,
    };
    use itertools::Itertools;
    use num_traits::{ToPrimitive, Zero};

    use super::*;

    type Mat = Vec<Vec<u64>>;

    impl<Rng, ModOp> Encryptor<[bool], Vec<Mat>> for PublicKey<Mat, Rng, ModOp> {
        fn encrypt(&self, m: &[bool]) -> Vec<Mat> {
            BoolEvaluator::with_local(|e| {
                DefaultSecureRng::with_local_mut(|rng| {
                    let parameters = e.parameters();
                    let ring_size = parameters.rlwe_n().0;

                    let rlwe_count = ((m.len() as f64 / ring_size as f64).ceil())
                        .to_usize()
                        .unwrap();

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
                                <Mat as MatrixEntity>::zeros(2, parameters.rlwe_n().0);

                            public_key_encrypt_rlwe::<_, _, _, _, i32, _>(
                                &mut rlwe_out,
                                self.key(),
                                &message,
                                e.pbs_info().modop_rlweq(),
                                e.pbs_info().nttop_rlweq(),
                                rng,
                            );

                            rlwe_out
                        })
                        .collect_vec();
                    rlwes
                })
            })
        }
    }

    impl<Rng, ModOp> Encryptor<bool, <Mat as Matrix>::R> for PublicKey<Mat, Rng, ModOp> {
        fn encrypt(&self, m: &bool) -> <Mat as Matrix>::R {
            let m = vec![*m];
            let rlwe = &self.encrypt(m.as_slice())[0];
            BoolEvaluator::with_local(|e| {
                let mut lwe = <Mat as Matrix>::R::zeros(e.parameters().rlwe_n().0 + 1);
                sample_extract(&mut lwe, rlwe, e.pbs_info().modop_rlweq(), 0);
                lwe
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread::panicking;

    use itertools::Itertools;
    use rand::{thread_rng, RngCore};

    use crate::{
        bool::{
            evaluator::BooleanGates,
            keys::tests::{ideal_sk_rlwe, measure_noise_lwe},
        },
        evaluator::{BoolEncoding, BoolEvaluator},
        Encryptor, MultiPartyDecryptor,
    };

    use super::*;

    #[test]
    fn multi_party_bool_gates() {
        set_parameter_set(ParameterSelector::InteractiveLTE2Party);
        let mut seed = [0u8; 32];
        thread_rng().fill_bytes(&mut seed);
        set_mp_seed(seed);

        let parties = 2;
        let cks = (0..parties).map(|_| gen_client_key()).collect_vec();

        // round 1
        let pk_shares = cks.iter().map(|k| gen_mp_keys_phase1(k)).collect_vec();

        // collective pk
        let pk = aggregate_public_key_shares(&pk_shares);

        // round 2
        let server_key_shares = cks
            .iter()
            .enumerate()
            .map(|(user_id, k)| gen_mp_keys_phase2(k, user_id, parties, &pk))
            .collect_vec();

        // server key
        let server_key = aggregate_server_key_shares(&server_key_shares);
        server_key.set_server_key();

        let mut m0 = false;
        let mut m1 = true;

        let mut ct0 = pk.encrypt(&m0);
        let mut ct1 = pk.encrypt(&m1);

        let ideal_sk_rlwe = ideal_sk_rlwe(&cks);
        let parameters = BoolEvaluator::with_local(|e| e.parameters().clone());
        let rlwe_modop = parameters.default_rlwe_modop();

        for _ in 0..500 {
            let ct_out =
                BoolEvaluator::with_local_mut(|e| e.nand(&ct0, &ct1, RuntimeServerKey::global()));

            let m_expected = !(m0 && m1);

            let decryption_shares = cks
                .iter()
                .map(|k| k.gen_decryption_share(&ct_out))
                .collect_vec();
            let m_out = cks[0].aggregate_decryption_shares(&ct_out, &decryption_shares);

            assert!(m_out == m_expected, "Expected {m_expected}, got {m_out}");
            {
                let m_expected_el = if m_expected == true {
                    parameters.rlwe_q().true_el()
                } else {
                    parameters.rlwe_q().false_el()
                };
                let noise = measure_noise_lwe(&ct_out, m_expected_el, &ideal_sk_rlwe, &rlwe_modop);
                println!("NAND Noise: {noise}");
            }

            m1 = m0;
            m0 = m_expected;

            ct1 = ct0;
            ct0 = ct_out;
        }

        for _ in 0..500 {
            let ct_out =
                BoolEvaluator::with_local_mut(|e| e.xnor(&ct0, &ct1, RuntimeServerKey::global()));

            let m_expected = !(m0 ^ m1);

            let decryption_shares = cks
                .iter()
                .map(|k| k.gen_decryption_share(&ct_out))
                .collect_vec();
            let m_out = cks[0].aggregate_decryption_shares(&ct_out, &decryption_shares);

            assert!(m_out == m_expected, "Expected {m_expected}, got {m_out}");

            m1 = m0;
            m0 = m_expected;

            ct1 = ct0;
            ct0 = ct_out;
        }
    }

    mod sp_api {
        use num_traits::ToPrimitive;
        use rand::Rng;

        use crate::{
            backend::ModulusPowerOf2, evaluator::BoolEncoding, pbs::PbsInfo,
            rgsw::seeded_secret_key_encrypt_rlwe, utils::WithLocal, Decryptor, ModularOpsU64,
            NttBackendU64, SampleExtractor,
        };

        use super::*;

        pub(crate) fn set_single_party_parameter_sets(parameter: BoolParameters<u64>) {
            BOOL_EVALUATOR.with_borrow_mut(|e| *e = Some(BoolEvaluator::new(parameter)));
        }

        // SERVER KEY EVAL (/SHOUP) DOMAIN //
        impl SeededSinglePartyServerKey<Vec<Vec<u64>>, BoolParameters<u64>, [u8; 32]> {
            pub fn set_server_key(&self) {
                assert!(
                    BOOL_SERVER_KEY
                        .set(
                            ShoupServerKeyEvaluationDomain::from(ServerKeyEvaluationDomain::<
                                _,
                                _,
                                DefaultSecureRng,
                                NttBackendU64,
                            >::from(
                                self
                            ),)
                        )
                        .is_ok(),
                    "Attempted to set server key twice."
                );
            }
        }

        pub(crate) fn gen_keys() -> (
            ClientKey,
            SeededSinglePartyServerKey<Vec<Vec<u64>>, BoolParameters<u64>, [u8; 32]>,
        ) {
            super::BoolEvaluator::with_local_mut(|e| {
                let ck = e.client_key();
                let sk = e.single_party_server_key(&ck);

                (ck, sk)
            })
        }

        impl<K: SinglePartyClientKey<Element = i32>> Encryptor<bool, Vec<u64>> for K {
            fn encrypt(&self, m: &bool) -> Vec<u64> {
                BoolEvaluator::with_local(|e| e.sk_encrypt(*m, self))
            }
        }

        impl<K: SinglePartyClientKey<Element = i32>> Decryptor<bool, Vec<u64>> for K {
            fn decrypt(&self, c: &Vec<u64>) -> bool {
                BoolEvaluator::with_local(|e| e.sk_decrypt(c, self))
            }
        }
        impl<K> Encryptor<[bool], (Vec<Vec<u64>>, [u8; 32])> for K
        where
            K: SinglePartyClientKey<Element = i32>,
        {
            fn encrypt(&self, m: &[bool]) -> (Vec<Vec<u64>>, [u8; 32]) {
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

                        let sk_u = self.sk_rlwe();

                        // encrypt `m` into ceil(len(m)/N) RLWE ciphertexts
                        let rlwes = (0..rlwe_count)
                            .map(|index| {
                                let mut message = vec![0; ring_size];
                                m[(index * ring_size)
                                    ..std::cmp::min(m.len(), (index + 1) * ring_size)]
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
                                let mut rlwe_out = vec![0u64; parameters.rlwe_n().0];
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

        #[test]
        fn batch_extract_works() {
            set_single_party_parameter_sets(SP_TEST_BOOL_PARAMS);

            let (ck, sk) = gen_keys();
            sk.set_server_key();

            let batch_size = (SP_TEST_BOOL_PARAMS.rlwe_n().0 * 3 + 123);
            let m = (0..batch_size)
                .map(|_| thread_rng().gen::<u8>())
                .collect_vec();

            let seeded_ct = ck.encrypt(m.as_slice());
            let ct = seeded_ct.unseed::<Vec<Vec<u64>>>();

            let m_back = (0..batch_size)
                .map(|i| ck.decrypt(&ct.extract(i)))
                .collect_vec();

            assert_eq!(m, m_back);
        }

        #[test]
        #[cfg(feature = "interactive_mp")]
        fn all_uint8_apis() {
            use num_traits::Euclid;

            set_single_party_parameter_sets(SP_TEST_BOOL_PARAMS);

            let (ck, sk) = gen_keys();
            sk.set_server_key();

            for i in 144..=255 {
                for j in 100..=255 {
                    let m0 = i;
                    let m1 = j;
                    let c0 = ck.encrypt(&m0);
                    let c1 = ck.encrypt(&m1);

                    assert!(ck.decrypt(&c0) == m0);
                    assert!(ck.decrypt(&c1) == m1);

                    // Arithmetic
                    {
                        {
                            // Add
                            let c_add = &c0 + &c1;
                            let m0_plus_m1 = ck.decrypt(&c_add);
                            assert_eq!(
                                m0_plus_m1,
                                m0.wrapping_add(m1),
                                "Expected {} but got {m0_plus_m1} for {i}+{j}",
                                m0.wrapping_add(m1)
                            );
                        }
                        {
                            // Sub
                            let c_sub = &c0 - &c1;
                            let m0_sub_m1 = ck.decrypt(&c_sub);
                            assert_eq!(
                                m0_sub_m1,
                                m0.wrapping_sub(m1),
                                "Expected {} but got {m0_sub_m1} for {i}-{j}",
                                m0.wrapping_sub(m1)
                            );
                        }

                        {
                            // Mul
                            let c_m0m1 = &c0 * &c1;
                            let m0m1 = ck.decrypt(&c_m0m1);
                            assert_eq!(
                                m0m1,
                                m0.wrapping_mul(m1),
                                "Expected {} but got {m0m1} for {i}x{j}",
                                m0.wrapping_mul(m1)
                            );
                        }

                        // Div & Rem
                        {
                            let (c_quotient, c_rem) = c0.div_rem(&c1);
                            let m_quotient = ck.decrypt(&c_quotient);
                            let m_remainder = ck.decrypt(&c_rem);
                            if j != 0 {
                                let (q, r) = i.div_rem_euclid(&j);
                                assert_eq!(
                                    m_quotient, q,
                                    "Expected {} but got {m_quotient} for {i}/{j}",
                                    q
                                );
                                assert_eq!(
                                    m_remainder, r,
                                    "Expected {} but got {m_quotient} for {i}%{j}",
                                    r
                                );
                            } else {
                                assert_eq!(
                                    m_quotient, 255,
                                    "Expected 255 but got {m_quotient}. Case div by zero"
                                );
                                assert_eq!(
                                    m_remainder, i,
                                    "Expected {i} but got {m_quotient}. Case div by zero"
                                )
                            }
                        }
                    }

                    // Comparisons
                    {
                        {
                            let c_eq = c0.eq(&c1);
                            let is_eq = ck.decrypt(&c_eq);
                            assert_eq!(
                                is_eq,
                                i == j,
                                "Expected {} but got {is_eq} for {i}=={j}",
                                i == j
                            );
                        }

                        {
                            let c_gt = c0.gt(&c1);
                            let is_gt = ck.decrypt(&c_gt);
                            assert_eq!(
                                is_gt,
                                i > j,
                                "Expected {} but got {is_gt} for {i}>{j}",
                                i > j
                            );
                        }

                        {
                            let c_lt = c0.lt(&c1);
                            let is_lt = ck.decrypt(&c_lt);
                            assert_eq!(
                                is_lt,
                                i < j,
                                "Expected {} but got {is_lt} for {i}<{j}",
                                i < j
                            );
                        }

                        {
                            let c_ge = c0.ge(&c1);
                            let is_ge = ck.decrypt(&c_ge);
                            assert_eq!(
                                is_ge,
                                i >= j,
                                "Expected {} but got {is_ge} for {i}>={j}",
                                i >= j
                            );
                        }

                        {
                            let c_le = c0.le(&c1);
                            let is_le = ck.decrypt(&c_le);
                            assert_eq!(
                                is_le,
                                i <= j,
                                "Expected {} but got {is_le} for {i}<={j}",
                                i <= j
                            );
                        }
                    }
                }
            }
        }
    }
}
