use std::{cell::RefCell, sync::OnceLock};

use crate::{
    backend::{ModularOpsU64, ModulusPowerOf2},
    ntt::NttBackendU64,
    random::{DefaultSecureRng, NewWithSeed},
    utils::{Global, WithLocal},
};

use super::{evaluator::MultiPartyCrs, keys::*, parameters::*, ClientKey, ParameterSelector};

pub type BoolEvaluator = super::evaluator::BoolEvaluator<
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

static MULTI_PARTY_CRS: OnceLock<MultiPartyCrs<[u8; 32]>> = OnceLock::new();

pub fn set_parameter_set(select: ParameterSelector) {
    match select {
        ParameterSelector::MultiPartyLessThanOrEqualTo16 => {
            BOOL_EVALUATOR.with_borrow_mut(|v| *v = Some(BoolEvaluator::new(SMALL_MP_BOOL_PARAMS)));
        }
        _ => {
            panic!("Paramerters not supported")
        }
    }
}

pub fn set_mp_seed(seed: [u8; 32]) {
    assert!(
        MULTI_PARTY_CRS.set(MultiPartyCrs { seed: seed }).is_ok(),
        "Attempted to set MP SEED twice."
    )
}

fn set_server_key(key: ShoupServerKeyEvaluationDomain<Vec<Vec<u64>>>) {
    assert!(
        BOOL_SERVER_KEY.set(key).is_ok(),
        "Attempted to set server key twice."
    );
}

pub(crate) fn gen_keys() -> (
    ClientKey,
    SeededSinglePartyServerKey<Vec<Vec<u64>>, BoolParameters<u64>, [u8; 32]>,
) {
    BoolEvaluator::with_local_mut(|e| {
        let ck = e.client_key();
        let sk = e.single_party_server_key(&ck);

        (ck, sk)
    })
}

pub fn gen_client_key() -> ClientKey {
    BoolEvaluator::with_local(|e| e.client_key())
}

pub fn gen_mp_keys_phase1(
    ck: &ClientKey,
) -> CommonReferenceSeededCollectivePublicKeyShare<Vec<u64>, [u8; 32], BoolParameters<u64>> {
    let seed = MultiPartyCrs::global().public_key_share_seed::<DefaultSecureRng>();
    BoolEvaluator::with_local(|e| {
        let pk_share = e.multi_party_public_key_share(seed, ck);
        pk_share
    })
}

pub fn gen_mp_keys_phase2<R, ModOp>(
    ck: &ClientKey,
    pk: &PublicKey<Vec<Vec<u64>>, R, ModOp>,
) -> CommonReferenceSeededMultiPartyServerKeyShare<Vec<Vec<u64>>, BoolParameters<u64>, [u8; 32]> {
    let seed = MultiPartyCrs::global().server_key_share_seed::<DefaultSecureRng>();
    BoolEvaluator::with_local_mut(|e| {
        let server_key_share = e.multi_party_server_key_share(seed, pk.key(), ck);
        server_key_share
    })
}

pub fn aggregate_public_key_shares(
    shares: &[CommonReferenceSeededCollectivePublicKeyShare<
        Vec<u64>,
        [u8; 32],
        BoolParameters<u64>,
    >],
) -> PublicKey<Vec<Vec<u64>>, DefaultSecureRng, ModularOpsU64<CiphertextModulus<u64>>> {
    PublicKey::from(shares)
}

pub fn aggregate_server_key_shares(
    shares: &[CommonReferenceSeededMultiPartyServerKeyShare<
        Vec<Vec<u64>>,
        BoolParameters<u64>,
        [u8; 32],
    >],
) -> SeededMultiPartyServerKey<Vec<Vec<u64>>, [u8; 32], BoolParameters<u64>> {
    BoolEvaluator::with_local(|e| e.aggregate_multi_party_server_key_shares(shares))
}

// SERVER KEY EVAL (/SHOUP) DOMAIN //
impl SeededSinglePartyServerKey<Vec<Vec<u64>>, BoolParameters<u64>, [u8; 32]> {
    pub fn set_server_key(&self) {
        let eval = ServerKeyEvaluationDomain::<_, _, DefaultSecureRng, NttBackendU64>::from(self);
        set_server_key(ShoupServerKeyEvaluationDomain::from(eval));
    }
}

impl
    SeededMultiPartyServerKey<
        Vec<Vec<u64>>,
        <DefaultSecureRng as NewWithSeed>::Seed,
        BoolParameters<u64>,
    >
{
    pub fn set_server_key(&self) {
        set_server_key(ShoupServerKeyEvaluationDomain::from(
            ServerKeyEvaluationDomain::<_, _, DefaultSecureRng, NttBackendU64>::from(self),
        ))
    }
}

//  MULTIPARTY CRS  //
impl Global for MultiPartyCrs<[u8; 32]> {
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
        Encryptor, Matrix, MatrixEntity, MultiPartyDecryptor, RowEntity,
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
