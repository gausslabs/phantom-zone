use std::{cell::RefCell, sync::OnceLock};

use crate::{
    backend::{ModularOpsU64, ModulusPowerOf2},
    ntt::NttBackendU64,
    random::{DefaultSecureRng, NewWithSeed},
    utils::{Global, WithLocal},
};

use super::{evaluator::*, keys::*, parameters::*};

thread_local! {
    static BOOL_EVALUATOR: RefCell<Option<BoolEvaluator<Vec<Vec<u64>>, NttBackendU64, ModularOpsU64<CiphertextModulus<u64>>,  ModulusPowerOf2<CiphertextModulus<u64>>, ShoupServerKeyEvaluationDomain<Vec<Vec<u64>>>>>> = RefCell::new(None);

}
static BOOL_SERVER_KEY: OnceLock<ShoupServerKeyEvaluationDomain<Vec<Vec<u64>>>> = OnceLock::new();

static MULTI_PARTY_CRS: OnceLock<MultiPartyCrs<[u8; 32]>> = OnceLock::new();

pub type ClientKey = super::keys::ClientKey<[u8; 32], u64>;

pub enum ParameterSelector {
    MultiPartyLessThanOrEqualTo16,
}

pub fn set_parameter_set(select: ParameterSelector) {
    match select {
        ParameterSelector::MultiPartyLessThanOrEqualTo16 => {
            BOOL_EVALUATOR.with_borrow_mut(|v| *v = Some(BoolEvaluator::new(SMALL_MP_BOOL_PARAMS)));
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
impl WithLocal
    for BoolEvaluator<
        Vec<Vec<u64>>,
        NttBackendU64,
        ModularOpsU64<CiphertextModulus<u64>>,
        ModulusPowerOf2<CiphertextModulus<u64>>,
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

mod impl_enc_dec {
    use crate::{
        pbs::{sample_extract, PbsInfo},
        rgsw::public_key_encrypt_rlwe,
        Decryptor, Encryptor, Matrix, MatrixEntity, MultiPartyDecryptor, RowEntity,
    };
    use num_traits::Zero;

    use super::*;

    type Mat = Vec<Vec<u64>>;

    impl<E> Encryptor<bool, Vec<u64>> for super::super::keys::ClientKey<[u8; 32], E> {
        fn encrypt(&self, m: &bool) -> Vec<u64> {
            BoolEvaluator::with_local(|e| e.sk_encrypt(*m, self))
        }
    }

    impl<E> Decryptor<bool, Vec<u64>> for super::super::keys::ClientKey<[u8; 32], E> {
        fn decrypt(&self, c: &Vec<u64>) -> bool {
            BoolEvaluator::with_local(|e| e.sk_decrypt(c, self))
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

    impl<Rng, ModOp> Encryptor<[bool], Mat> for PublicKey<Mat, Rng, ModOp> {
        fn encrypt(&self, m: &[bool]) -> Mat {
            BoolEvaluator::with_local(|e| {
                DefaultSecureRng::with_local_mut(|rng| {
                    let parameters = e.parameters();
                    let mut rlwe_out = <Mat as MatrixEntity>::zeros(2, parameters.rlwe_n().0);
                    assert!(m.len() <= parameters.rlwe_n().0);

                    let mut message =
                        vec![<Mat as Matrix>::MatElement::zero(); parameters.rlwe_n().0];
                    m.iter().enumerate().for_each(|(i, v)| {
                        if *v {
                            message[i] = parameters.rlwe_q().true_el()
                        } else {
                            message[i] = parameters.rlwe_q().false_el()
                        }
                    });

                    // e.pk_encrypt_batched(self.key(), m)
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
            })
        }
    }

    impl<Rng, ModOp> Encryptor<bool, <Mat as Matrix>::R> for PublicKey<Mat, Rng, ModOp> {
        fn encrypt(&self, m: &bool) -> <Mat as Matrix>::R {
            let m = vec![*m];
            let rlwe = self.encrypt(m.as_slice());
            BoolEvaluator::with_local(|e| {
                let mut lwe = <Mat as Matrix>::R::zeros(e.parameters().rlwe_n().0 + 1);
                sample_extract(&mut lwe, &rlwe, e.pbs_info().modop_rlweq(), 0);
                lwe
            })
        }
    }
}
