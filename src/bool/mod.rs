pub(crate) mod evaluator;
pub(crate) mod keys;
pub mod noise;
pub(crate) mod parameters;

pub type FheBool = Vec<u64>;

use std::{cell::RefCell, sync::OnceLock};

use evaluator::*;
use keys::*;
use parameters::*;

use crate::{
    backend::ModularOpsU64,
    ntt::NttBackendU64,
    random::{DefaultSecureRng, NewWithSeed},
    utils::{Global, WithLocal},
};

thread_local! {
    static BOOL_EVALUATOR: RefCell<Option<BoolEvaluator<Vec<Vec<u64>>, NttBackendU64, ModularOpsU64<CiphertextModulus<u64>>,  ModularOpsU64<CiphertextModulus<u64>>, ShoupServerKeyEvaluationDomain<Vec<Vec<u64>>>>>> = RefCell::new(None);

}
static BOOL_SERVER_KEY: OnceLock<ShoupServerKeyEvaluationDomain<Vec<Vec<u64>>>> = OnceLock::new();

static MULTI_PARTY_CRS: OnceLock<MultiPartyCrs<[u8; 32]>> = OnceLock::new();

pub fn set_parameter_set(parameter: &BoolParameters<u64>) {
    BOOL_EVALUATOR.with_borrow_mut(|v| *v = Some(BoolEvaluator::new(parameter.clone())));
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
    SeededServerKey<Vec<Vec<u64>>, BoolParameters<u64>, [u8; 32]>,
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
        let pk_share = e.multi_party_public_key_share(seed, &ck);
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
impl SeededServerKey<Vec<Vec<u64>>, BoolParameters<u64>, [u8; 32]> {
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
impl WithLocal
    for BoolEvaluator<
        Vec<Vec<u64>>,
        NttBackendU64,
        ModularOpsU64<CiphertextModulus<u64>>,
        ModularOpsU64<CiphertextModulus<u64>>,
        ShoupServerKeyEvaluationDomain<Vec<Vec<u64>>>,
    >
{
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
