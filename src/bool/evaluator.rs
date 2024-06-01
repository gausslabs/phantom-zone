use std::{
    cell::{OnceCell, RefCell},
    clone,
    collections::HashMap,
    fmt::{Debug, Display},
    iter::Once,
    marker::PhantomData,
    ops::Shr,
    sync::OnceLock,
};

use itertools::{izip, partition, Itertools};
use num_traits::{FromPrimitive, Num, One, Pow, PrimInt, ToPrimitive, WrappingSub, Zero};
use rand_distr::uniform::SampleUniform;

use crate::{
    backend::{ArithmeticOps, GetModulus, ModInit, ModularOpsU64, Modulus, VectorOps},
    bool::parameters::{MP_BOOL_PARAMS, SP_BOOL_PARAMS},
    decomposer::{Decomposer, DefaultDecomposer, NumInfo, RlweDecomposer},
    lwe::{decrypt_lwe, encrypt_lwe, lwe_key_switch, lwe_ksk_keygen, measure_noise_lwe, LweSecret},
    multi_party::public_key_share,
    ntt::{self, Ntt, NttBackendU64, NttInit},
    pbs::{pbs, sample_extract, PbsInfo, PbsKey},
    random::{
        DefaultSecureRng, NewWithSeed, RandomFill, RandomFillGaussianInModulus,
        RandomFillUniformInModulus, RandomGaussianElementInModulus,
    },
    rgsw::{
        decrypt_rlwe, galois_auto, galois_key_gen, generate_auto_map, public_key_encrypt_rgsw,
        rgsw_by_rgsw_inplace, rlwe_by_rgsw, secret_key_encrypt_rgsw, IsTrivial, RgswCiphertext,
        RlweCiphertext, RlweSecret,
    },
    utils::{
        fill_random_ternary_secret_with_hamming_weight, generate_prime, mod_exponent, Global,
        TryConvertFrom1, WithLocal,
    },
    Decryptor, Encryptor, Matrix, MatrixEntity, MatrixMut, MultiPartyDecryptor, Row, RowEntity,
    RowMut, Secret,
};

use super::parameters::{self, BoolParameters, CiphertextModulus};

thread_local! {
    pub(crate) static BOOL_EVALUATOR: RefCell<BoolEvaluator<Vec<Vec<u64>>, NttBackendU64, ModularOpsU64<CiphertextModulus<u64>>,  ModularOpsU64<CiphertextModulus<u64>>>> = RefCell::new(BoolEvaluator::new(MP_BOOL_PARAMS));

}
pub(crate) static BOOL_SERVER_KEY: OnceLock<
    ServerKeyEvaluationDomain<Vec<Vec<u64>>, DefaultSecureRng, NttBackendU64>,
> = OnceLock::new();

pub(crate) static MULTI_PARTY_CRS: OnceLock<MultiPartyCrs<[u8; 32]>> = OnceLock::new();

pub fn set_parameter_set(parameter: &BoolParameters<u64>) {
    BoolEvaluator::with_local_mut(|e| *e = BoolEvaluator::new(parameter.clone()))
}

pub fn set_mp_seed(seed: [u8; 32]) {
    assert!(
        MULTI_PARTY_CRS.set(MultiPartyCrs { seed: seed }).is_ok(),
        "Attempted to set MP SEED twice."
    )
}

fn set_server_key(key: ServerKeyEvaluationDomain<Vec<Vec<u64>>, DefaultSecureRng, NttBackendU64>) {
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
        let sk = e.server_key(&ck);

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
        let server_key_share = e.multi_party_server_key_share(seed, &pk.key, ck);
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

// GENERIC BELOW

pub struct MultiPartyCrs<S> {
    seed: S,
}

impl<S: Default + Copy> MultiPartyCrs<S> {
    /// Seed to generate public key share using MultiPartyCrs as the main seed.
    ///
    /// Public key seed equals the 1st seed extracted from PRNG Seeded with
    /// MiltiPartyCrs's seed.
    fn public_key_share_seed<Rng: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut prng = Rng::new_with_seed(self.seed);

        let mut seed = S::default();
        RandomFill::<S>::random_fill(&mut prng, &mut seed);
        seed
    }

    /// Seed to generate server key share using MultiPartyCrs as the main seed.
    ///
    /// Server key seed equals the 2nd seed extracted from PRNG Seeded with
    /// MiltiPartyCrs's seed.
    fn server_key_share_seed<Rng: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut prng = Rng::new_with_seed(self.seed);

        let mut seed = S::default();
        RandomFill::<S>::random_fill(&mut prng, &mut seed);
        RandomFill::<S>::random_fill(&mut prng, &mut seed);
        seed
    }
}

impl Global for MultiPartyCrs<[u8; 32]> {
    fn global() -> &'static Self {
        MULTI_PARTY_CRS
            .get()
            .expect("Multi Party Common Reference String not set")
    }
}

pub(crate) trait BooleanGates {
    type Ciphertext: RowEntity;
    type Key;

    fn and_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn nand_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn or_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn nor_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn xor_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn xnor_inplace(&mut self, c0: &mut Self::Ciphertext, c1: &Self::Ciphertext, key: &Self::Key);
    fn not_inplace(&mut self, c: &mut Self::Ciphertext);

    fn and(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn nand(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn or(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn nor(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn xor(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn xnor(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext;
    fn not(&mut self, c: &Self::Ciphertext) -> Self::Ciphertext;
}

impl WithLocal
    for BoolEvaluator<
        Vec<Vec<u64>>,
        NttBackendU64,
        ModularOpsU64<CiphertextModulus<u64>>,
        ModularOpsU64<CiphertextModulus<u64>>,
    >
{
    fn with_local<F, R>(func: F) -> R
    where
        F: Fn(&Self) -> R,
    {
        BOOL_EVALUATOR.with_borrow(|s| func(s))
    }

    fn with_local_mut<F, R>(func: F) -> R
    where
        F: Fn(&mut Self) -> R,
    {
        BOOL_EVALUATOR.with_borrow_mut(|s| func(s))
    }

    fn with_local_mut_mut<F, R>(func: &mut F) -> R
    where
        F: FnMut(&mut Self) -> R,
    {
        BOOL_EVALUATOR.with_borrow_mut(|s| func(s))
    }
}

impl Global for ServerKeyEvaluationDomain<Vec<Vec<u64>>, DefaultSecureRng, NttBackendU64> {
    fn global() -> &'static Self {
        BOOL_SERVER_KEY.get().unwrap()
    }
}

struct ScratchMemory<M>
where
    M: Matrix,
{
    lwe_vector: M::R,
    decomposition_matrix: M,
}

impl<M: MatrixEntity> ScratchMemory<M>
where
    M::R: RowEntity,
{
    fn new(parameters: &BoolParameters<M::MatElement>) -> Self {
        // Vector to store LWE ciphertext with LWE dimesnion n
        let lwe_vector = M::R::zeros(parameters.lwe_n().0 + 1);

        // Matrix to store decomposed polynomials
        // Max decompistion count + space for temporary RLWE
        let d = std::cmp::max(
            parameters.auto_decomposition_count().0,
            std::cmp::max(
                parameters.rlwe_rgsw_decomposition_count().0 .0,
                parameters.rlwe_rgsw_decomposition_count().1 .0,
            ),
        ) + 2;
        let decomposition_matrix = M::zeros(d, parameters.rlwe_n().0);

        Self {
            lwe_vector,
            decomposition_matrix,
        }
    }
}

// thread_local! {
//     pub(crate) static CLIENT_KEY: RefCell<ClientKey> =
// RefCell::new(ClientKey::random()); }

trait BoolEncoding {
    type Element;
    fn true_el(&self) -> Self::Element;
    fn false_el(&self) -> Self::Element;
    fn qby4(&self) -> Self::Element;
    fn decode(&self, m: Self::Element) -> bool;
}

impl<T> BoolEncoding for CiphertextModulus<T>
where
    CiphertextModulus<T>: Modulus<Element = T>,
    T: PrimInt,
{
    type Element = T;

    fn qby4(&self) -> Self::Element {
        if self.is_native() {
            T::one() << (CiphertextModulus::<T>::_bits() - 2)
        } else {
            self.q().unwrap() >> 2
        }
    }
    /// Q/8
    fn true_el(&self) -> Self::Element {
        if self.is_native() {
            T::one() << (CiphertextModulus::<T>::_bits() - 3)
        } else {
            self.q().unwrap() >> 3
        }
    }
    /// -Q/8
    fn false_el(&self) -> Self::Element {
        self.largest_unsigned_value() - self.true_el() + T::one()
    }
    fn decode(&self, m: Self::Element) -> bool {
        let qby8 = self.true_el();
        let m = (((m + qby8).to_f64().unwrap() * 4.0f64) / self.q_as_f64().unwrap()).round()
            as usize
            % 4usize;

        if m == 0 {
            return false;
        } else if m == 1 {
            return true;
        } else {
            panic!("Incorrect bool decryption. Got m={m} but expected m to be 0 or 1")
        }
    }
}

#[derive(Clone)]
pub struct ClientKey {
    sk_rlwe: RlweSecret,
    sk_lwe: LweSecret,
}

impl ClientKey {
    fn random() -> Self {
        let sk_rlwe = RlweSecret::random(0, 0);
        let sk_lwe = LweSecret::random(0, 0);
        Self { sk_rlwe, sk_lwe }
    }
}

impl Encryptor<bool, Vec<u64>> for ClientKey {
    fn encrypt(&self, m: &bool) -> Vec<u64> {
        BoolEvaluator::with_local(|e| e.sk_encrypt(*m, self))
    }
}

impl Decryptor<bool, Vec<u64>> for ClientKey {
    fn decrypt(&self, c: &Vec<u64>) -> bool {
        BoolEvaluator::with_local(|e| e.sk_decrypt(c, self))
    }
}

impl MultiPartyDecryptor<bool, Vec<u64>> for ClientKey {
    type DecryptionShare = u64;

    fn gen_decryption_share(&self, c: &Vec<u64>) -> Self::DecryptionShare {
        BoolEvaluator::with_local(|e| e.multi_party_decryption_share(c, &self))
    }

    fn aggregate_decryption_shares(&self, c: &Vec<u64>, shares: &[Self::DecryptionShare]) -> bool {
        BoolEvaluator::with_local(|e| e.multi_party_decrypt(shares, c))
    }
}

pub struct CommonReferenceSeededCollectivePublicKeyShare<R, S, P> {
    share: R,
    cr_seed: S,
    parameters: P,
}

struct SeededPublicKey<R, S, P, ModOp> {
    part_b: R,
    seed: S,
    parameters: P,
    _phantom: PhantomData<ModOp>,
}

impl<R, S, ModOp>
    From<&[CommonReferenceSeededCollectivePublicKeyShare<R, S, BoolParameters<R::Element>>]>
    for SeededPublicKey<R, S, BoolParameters<R::Element>, ModOp>
where
    ModOp: VectorOps<Element = R::Element> + ModInit<M = CiphertextModulus<R::Element>>,
    S: PartialEq + Clone,
    R: RowMut + RowEntity + Clone,
    R::Element: Clone + PartialEq,
{
    fn from(
        value: &[CommonReferenceSeededCollectivePublicKeyShare<R, S, BoolParameters<R::Element>>],
    ) -> Self {
        assert!(value.len() > 0);

        let parameters = &value[0].parameters;
        let cr_seed = value[0].cr_seed.clone();

        // Sum all Bs
        let rlweq_modop = ModOp::new(parameters.rlwe_q().clone());
        let mut part_b = value[0].share.clone();
        value.iter().skip(1).for_each(|share_i| {
            assert!(&share_i.cr_seed == &cr_seed);
            assert!(&share_i.parameters == parameters);

            rlweq_modop.elwise_add_mut(part_b.as_mut(), share_i.share.as_ref());
        });

        Self {
            part_b,
            seed: cr_seed,
            parameters: parameters.clone(),
            _phantom: PhantomData,
        }
    }
}

pub struct PublicKey<M, Rng, ModOp> {
    key: M,
    _phantom: PhantomData<(Rng, ModOp)>,
}

impl<Rng, ModOp> Encryptor<bool, Vec<u64>> for PublicKey<Vec<Vec<u64>>, Rng, ModOp> {
    fn encrypt(&self, m: &bool) -> Vec<u64> {
        BoolEvaluator::with_local(|e| e.pk_encrypt(&self.key, *m))
    }
}

impl<Rng, ModOp> Encryptor<[bool], Vec<Vec<u64>>> for PublicKey<Vec<Vec<u64>>, Rng, ModOp> {
    fn encrypt(&self, m: &[bool]) -> Vec<Vec<u64>> {
        BoolEvaluator::with_local(|e| e.pk_encrypt_batched(&self.key, m))
    }
}

impl<
        M: MatrixMut + MatrixEntity,
        Rng: NewWithSeed + RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>>,
        ModOp,
    > From<SeededPublicKey<M::R, Rng::Seed, BoolParameters<M::MatElement>, ModOp>>
    for PublicKey<M, Rng, ModOp>
where
    <M as Matrix>::R: RowMut,
    M::MatElement: Copy,
{
    fn from(value: SeededPublicKey<M::R, Rng::Seed, BoolParameters<M::MatElement>, ModOp>) -> Self {
        let mut prng = Rng::new_with_seed(value.seed);

        let mut key = M::zeros(2, value.parameters.rlwe_n().0);
        // sample A
        RandomFillUniformInModulus::random_fill(
            &mut prng,
            value.parameters.rlwe_q(),
            key.get_row_mut(0),
        );
        // Copy over B
        key.get_row_mut(1).copy_from_slice(value.part_b.as_ref());

        PublicKey {
            key,
            _phantom: PhantomData,
        }
    }
}

impl<
        M: MatrixMut + MatrixEntity,
        Rng: NewWithSeed + RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>>,
        ModOp: VectorOps<Element = M::MatElement> + ModInit<M = CiphertextModulus<M::MatElement>>,
    >
    From<
        &[CommonReferenceSeededCollectivePublicKeyShare<
            M::R,
            Rng::Seed,
            BoolParameters<M::MatElement>,
        >],
    > for PublicKey<M, Rng, ModOp>
where
    <M as Matrix>::R: RowMut,
    Rng::Seed: Copy + PartialEq,
    M::MatElement: PartialEq + Copy,
{
    fn from(
        value: &[CommonReferenceSeededCollectivePublicKeyShare<
            M::R,
            Rng::Seed,
            BoolParameters<M::MatElement>,
        >],
    ) -> Self {
        assert!(value.len() > 0);

        let parameters = &value[0].parameters;
        let mut key = M::zeros(2, parameters.rlwe_n().0);

        // sample A
        let seed = value[0].cr_seed;
        let mut main_rng = Rng::new_with_seed(seed);
        RandomFillUniformInModulus::random_fill(
            &mut main_rng,
            parameters.rlwe_q(),
            key.get_row_mut(0),
        );

        // Sum all Bs
        let rlweq_modop = ModOp::new(parameters.rlwe_q().clone());
        value.iter().for_each(|share_i| {
            assert!(share_i.cr_seed == seed);
            assert!(&share_i.parameters == parameters);

            rlweq_modop.elwise_add_mut(key.get_row_mut(1), share_i.share.as_ref());
        });

        PublicKey {
            key,
            _phantom: PhantomData,
        }
    }
}

pub struct CommonReferenceSeededMultiPartyServerKeyShare<M: Matrix, P, S> {
    rgsw_cts: Vec<M>,
    /// Auto keys. Key corresponding to g^{k} is at index `k`. Key corresponding
    /// to -g is at 0
    auto_keys: HashMap<usize, M>,
    lwe_ksk: M::R,
    /// Common reference seed
    cr_seed: S,
    parameters: P,
}
pub struct SeededMultiPartyServerKey<M: Matrix, S, P> {
    rgsw_cts: Vec<M>,
    /// Auto keys. Key corresponding to g^{k} is at index `k`. Key corresponding
    /// to -g is at 0
    auto_keys: HashMap<usize, M>,
    lwe_ksk: M::R,
    cr_seed: S,
    parameters: P,
}

impl
    SeededMultiPartyServerKey<
        Vec<Vec<u64>>,
        <DefaultSecureRng as NewWithSeed>::Seed,
        BoolParameters<u64>,
    >
{
    pub fn set_server_key(&self) {
        set_server_key(ServerKeyEvaluationDomain::<
            Vec<Vec<u64>>,
            DefaultSecureRng,
            NttBackendU64,
        >::from(self))
    }
}

/// Seeded single party server key
pub struct SeededServerKey<M: Matrix, P, S> {
    /// Rgsw cts of LWE secret elements
    pub(crate) rgsw_cts: Vec<M>,
    /// Auto keys. Key corresponding to g^{k} is at index `k`. Key corresponding
    /// to -g is at 0
    pub(crate) auto_keys: HashMap<usize, M>,
    /// LWE ksk to key switching LWE ciphertext from RLWE secret to LWE secret
    pub(crate) lwe_ksk: M::R,
    /// Parameters
    pub(crate) parameters: P,
    /// Main seed
    pub(crate) seed: S,
}

impl<M: Matrix, S> SeededServerKey<M, BoolParameters<M::MatElement>, S> {
    pub(crate) fn from_raw(
        auto_keys: HashMap<usize, M>,
        rgsw_cts: Vec<M>,
        lwe_ksk: M::R,
        parameters: BoolParameters<M::MatElement>,
        seed: S,
    ) -> Self {
        // sanity checks
        auto_keys.iter().for_each(|v| {
            assert!(
                v.1.dimension()
                    == (
                        parameters.auto_decomposition_count().0,
                        parameters.rlwe_n().0
                    )
            )
        });

        let (part_a_d, part_b_d) = parameters.rlwe_rgsw_decomposition_count();
        rgsw_cts.iter().for_each(|v| {
            assert!(v.dimension() == (part_a_d.0 * 2 + part_b_d.0, parameters.rlwe_n().0))
        });
        assert!(
            lwe_ksk.as_ref().len()
                == (parameters.lwe_decomposition_count().0 * parameters.rlwe_n().0)
        );

        SeededServerKey {
            rgsw_cts,
            auto_keys,
            lwe_ksk,
            parameters,
            seed,
        }
    }
}

impl SeededServerKey<Vec<Vec<u64>>, BoolParameters<u64>, [u8; 32]> {
    pub fn set_server_key(&self) {
        set_server_key(ServerKeyEvaluationDomain::<
            _,
            DefaultSecureRng,
            NttBackendU64,
        >::from(self));
    }
}

/// Server key in evaluation domain
pub(crate) struct ServerKeyEvaluationDomain<M, R, N> {
    /// Rgsw cts of LWE secret elements
    rgsw_cts: Vec<M>,
    /// Auto keys. Key corresponding to g^{k} is at index `k`. Key corresponding
    /// to -g is at 0
    galois_keys: HashMap<usize, M>,
    /// LWE ksk to key switching LWE ciphertext from RLWE secret to LWE secret
    lwe_ksk: M,
    _phanton: PhantomData<(R, N)>,
}

impl<
        M: MatrixMut + MatrixEntity,
        R: RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>> + NewWithSeed,
        N: NttInit<CiphertextModulus<M::MatElement>> + Ntt<Element = M::MatElement>,
    > From<&SeededServerKey<M, BoolParameters<M::MatElement>, R::Seed>>
    for ServerKeyEvaluationDomain<M, R, N>
where
    <M as Matrix>::R: RowMut,
    M::MatElement: Copy,
    R::Seed: Clone,
{
    fn from(value: &SeededServerKey<M, BoolParameters<M::MatElement>, R::Seed>) -> Self {
        let mut main_prng = R::new_with_seed(value.seed.clone());
        let parameters = &value.parameters;
        let g = parameters.g() as isize;
        let ring_size = value.parameters.rlwe_n().0;
        let lwe_n = value.parameters.lwe_n().0;
        let rlwe_q = value.parameters.rlwe_q();
        let lwq_q = value.parameters.lwe_q();

        let nttop = N::new(rlwe_q, ring_size);

        // galois keys
        let mut auto_keys = HashMap::new();
        let auto_decomp_count = parameters.auto_decomposition_count().0;
        let auto_element_dlogs = parameters.auto_element_dlogs();
        for i in auto_element_dlogs.into_iter() {
            let seeded_auto_key = value.auto_keys.get(&i).unwrap();
            assert!(seeded_auto_key.dimension() == (auto_decomp_count, ring_size));

            let mut data = M::zeros(auto_decomp_count * 2, ring_size);

            // sample RLWE'_A(-s(X^k))
            data.iter_rows_mut().take(auto_decomp_count).for_each(|ri| {
                RandomFillUniformInModulus::random_fill(&mut main_prng, &rlwe_q, ri.as_mut())
            });

            // copy over RLWE'B_(-s(X^k))
            izip!(
                data.iter_rows_mut().skip(auto_decomp_count),
                seeded_auto_key.iter_rows()
            )
            .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

            // Send to Evaluation domain
            data.iter_rows_mut()
                .for_each(|ri| nttop.forward(ri.as_mut()));

            auto_keys.insert(i, data);
        }

        // RGSW ciphertexts
        let (rlrg_a_decomp, rlrg_b_decomp) = parameters.rlwe_rgsw_decomposition_count();
        let rgsw_cts = value
            .rgsw_cts
            .iter()
            .map(|seeded_rgsw_si| {
                assert!(
                    seeded_rgsw_si.dimension()
                        == (rlrg_a_decomp.0 * 2 + rlrg_b_decomp.0, ring_size)
                );

                let mut data = M::zeros(rlrg_a_decomp.0 * 2 + rlrg_b_decomp.0 * 2, ring_size);

                // copy over RLWE'(-sm)
                izip!(
                    data.iter_rows_mut().take(rlrg_a_decomp.0 * 2),
                    seeded_rgsw_si.iter_rows().take(rlrg_a_decomp.0 * 2)
                )
                .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

                // sample RLWE'_A(m)
                data.iter_rows_mut()
                    .skip(rlrg_a_decomp.0 * 2)
                    .take(rlrg_b_decomp.0)
                    .for_each(|ri| {
                        RandomFillUniformInModulus::random_fill(
                            &mut main_prng,
                            &rlwe_q,
                            ri.as_mut(),
                        )
                    });

                // copy over RLWE'_B(m)
                izip!(
                    data.iter_rows_mut()
                        .skip(rlrg_a_decomp.0 * 2 + rlrg_b_decomp.0),
                    seeded_rgsw_si.iter_rows().skip(rlrg_a_decomp.0 * 2)
                )
                .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

                // send polynomials to evaluation domain
                data.iter_rows_mut()
                    .for_each(|ri| nttop.forward(ri.as_mut()));

                data
            })
            .collect_vec();

        // LWE ksk
        let lwe_ksk = {
            let d = parameters.lwe_decomposition_count().0;
            assert!(value.lwe_ksk.as_ref().len() == d * ring_size);

            let mut data = M::zeros(d * ring_size, lwe_n + 1);
            izip!(data.iter_rows_mut(), value.lwe_ksk.as_ref().iter()).for_each(|(lwe_i, bi)| {
                RandomFillUniformInModulus::random_fill(
                    &mut main_prng,
                    &lwq_q,
                    &mut lwe_i.as_mut()[1..],
                );
                lwe_i.as_mut()[0] = *bi;
            });

            data
        };

        ServerKeyEvaluationDomain {
            rgsw_cts,
            galois_keys: auto_keys,
            lwe_ksk,
            _phanton: PhantomData,
        }
    }
}

impl<
        M: MatrixMut + MatrixEntity,
        Rng: NewWithSeed,
        N: NttInit<CiphertextModulus<M::MatElement>> + Ntt<Element = M::MatElement>,
    > From<&SeededMultiPartyServerKey<M, Rng::Seed, BoolParameters<M::MatElement>>>
    for ServerKeyEvaluationDomain<M, Rng, N>
where
    <M as Matrix>::R: RowMut,
    Rng::Seed: Copy,
    Rng: RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>>,
    M::MatElement: Copy,
{
    fn from(
        value: &SeededMultiPartyServerKey<M, Rng::Seed, BoolParameters<M::MatElement>>,
    ) -> Self {
        let g = value.parameters.g() as isize;
        let rlwe_n = value.parameters.rlwe_n().0;
        let lwe_n = value.parameters.lwe_n().0;
        let rlwe_q = value.parameters.rlwe_q();
        let lwe_q = value.parameters.lwe_q();

        let mut main_prng = Rng::new_with_seed(value.cr_seed);

        let rlwe_nttop = N::new(rlwe_q, rlwe_n);

        // auto keys
        let mut auto_keys = HashMap::new();
        let auto_d_count = value.parameters.auto_decomposition_count().0;
        let auto_element_dlogs = value.parameters.auto_element_dlogs();
        for i in auto_element_dlogs.into_iter() {
            let mut key = M::zeros(auto_d_count * 2, rlwe_n);

            // sample a
            key.iter_rows_mut().take(auto_d_count).for_each(|ri| {
                RandomFillUniformInModulus::random_fill(&mut main_prng, &rlwe_q, ri.as_mut())
            });

            let key_part_b = value.auto_keys.get(&i).unwrap();
            assert!(key_part_b.dimension() == (auto_d_count, rlwe_n));
            izip!(
                key.iter_rows_mut().skip(auto_d_count),
                key_part_b.iter_rows()
            )
            .for_each(|(to_ri, from_ri)| {
                to_ri.as_mut().copy_from_slice(from_ri.as_ref());
            });

            // send to evaluation domain
            key.iter_rows_mut()
                .for_each(|ri| rlwe_nttop.forward(ri.as_mut()));

            auto_keys.insert(i, key);
        }

        // rgsw cts
        let (rlrg_d_a, rlrg_d_b) = value.parameters.rlwe_rgsw_decomposition_count();
        let rgsw_ct_out = rlrg_d_a.0 * 2 + rlrg_d_b.0 * 2;
        let rgsw_cts = value
            .rgsw_cts
            .iter()
            .map(|ct_i_in| {
                assert!(ct_i_in.dimension() == (rgsw_ct_out, rlwe_n));
                let mut eval_ct_i_out = M::zeros(rgsw_ct_out, rlwe_n);

                izip!(eval_ct_i_out.iter_rows_mut(), ct_i_in.iter_rows()).for_each(
                    |(to_ri, from_ri)| {
                        to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                        rlwe_nttop.forward(to_ri.as_mut());
                    },
                );

                eval_ct_i_out
            })
            .collect_vec();

        // lwe ksk
        let d_lwe = value.parameters.lwe_decomposition_count().0;
        let mut lwe_ksk = M::zeros(rlwe_n * d_lwe, lwe_n + 1);
        izip!(lwe_ksk.iter_rows_mut(), value.lwe_ksk.as_ref().iter()).for_each(|(lwe_i, bi)| {
            RandomFillUniformInModulus::random_fill(
                &mut main_prng,
                &lwe_q,
                &mut lwe_i.as_mut()[1..],
            );
            lwe_i.as_mut()[0] = *bi;
        });

        ServerKeyEvaluationDomain {
            rgsw_cts,
            galois_keys: auto_keys,
            lwe_ksk,
            _phanton: PhantomData,
        }
    }
}

impl<M: Matrix, R, N> PbsKey for ServerKeyEvaluationDomain<M, R, N> {
    type M = M;
    fn galois_key_for_auto(&self, k: usize) -> &Self::M {
        self.galois_keys.get(&k).unwrap()
    }
    fn rgsw_ct_lwe_si(&self, si: usize) -> &Self::M {
        &self.rgsw_cts[si]
    }

    fn lwe_ksk(&self) -> &Self::M {
        &self.lwe_ksk
    }
}

struct BoolPbsInfo<M: Matrix, Ntt, RlweModOp, LweModOp> {
    auto_decomposer: DefaultDecomposer<M::MatElement>,
    rlwe_rgsw_decomposer: (
        DefaultDecomposer<M::MatElement>,
        DefaultDecomposer<M::MatElement>,
    ),
    lwe_decomposer: DefaultDecomposer<M::MatElement>,
    g_k_dlog_map: Vec<usize>,
    rlwe_nttop: Ntt,
    rlwe_modop: RlweModOp,
    lwe_modop: LweModOp,
    embedding_factor: usize,
    rlwe_qby4: M::MatElement,
    rlwe_auto_maps: Vec<(Vec<usize>, Vec<bool>)>,
    parameters: BoolParameters<M::MatElement>,
}

impl<M: Matrix, NttOp, RlweModOp, LweModOp> PbsInfo for BoolPbsInfo<M, NttOp, RlweModOp, LweModOp>
where
    M::MatElement: PrimInt + WrappingSub + NumInfo + FromPrimitive,
    RlweModOp: ArithmeticOps<Element = M::MatElement> + VectorOps<Element = M::MatElement>,
    LweModOp: ArithmeticOps<Element = M::MatElement> + VectorOps<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement>,
{
    type Modulus = CiphertextModulus<M::MatElement>;
    type Element = M::MatElement;
    type D = DefaultDecomposer<M::MatElement>;
    type RlweModOp = RlweModOp;
    type LweModOp = LweModOp;
    type NttOp = NttOp;
    fn rlwe_auto_map(&self, k: usize) -> &(Vec<usize>, Vec<bool>) {
        &self.rlwe_auto_maps[k]
    }
    fn br_q(&self) -> usize {
        *self.parameters.br_q()
    }
    fn lwe_decomposer(&self) -> &Self::D {
        &self.lwe_decomposer
    }
    fn rlwe_rgsw_decomposer(&self) -> &(Self::D, Self::D) {
        &self.rlwe_rgsw_decomposer
    }
    fn auto_decomposer(&self) -> &Self::D {
        &self.auto_decomposer
    }
    fn embedding_factor(&self) -> usize {
        self.embedding_factor
    }
    fn g(&self) -> isize {
        self.parameters.g() as isize
    }
    fn w(&self) -> usize {
        self.parameters.w()
    }
    fn g_k_dlog_map(&self) -> &[usize] {
        &self.g_k_dlog_map
    }
    fn lwe_n(&self) -> usize {
        self.parameters.lwe_n().0
    }
    fn lwe_q(&self) -> &Self::Modulus {
        self.parameters.lwe_q()
    }
    fn rlwe_n(&self) -> usize {
        self.parameters.rlwe_n().0
    }
    fn rlwe_q(&self) -> &Self::Modulus {
        self.parameters.rlwe_q()
    }
    fn modop_lweq(&self) -> &Self::LweModOp {
        &self.lwe_modop
    }
    fn modop_rlweq(&self) -> &Self::RlweModOp {
        &self.rlwe_modop
    }
    fn nttop_rlweq(&self) -> &Self::NttOp {
        &self.rlwe_nttop
    }
}

pub(crate) struct BoolEvaluator<M, Ntt, RlweModOp, LweModOp>
where
    M: Matrix,
{
    pbs_info: BoolPbsInfo<M, Ntt, RlweModOp, LweModOp>,
    scratch_memory: ScratchMemory<M>,
    nand_test_vec: M::R,
    and_test_vec: M::R,
    or_test_vec: M::R,
    nor_test_vec: M::R,
    xor_test_vec: M::R,
    xnor_test_vec: M::R,
    _phantom: PhantomData<M>,
}

impl<M: Matrix, NttOp, RlweModOp, LweModOp> BoolEvaluator<M, NttOp, RlweModOp, LweModOp> {}

impl<M: Matrix, NttOp, RlweModOp, LweModOp> BoolEvaluator<M, NttOp, RlweModOp, LweModOp>
where
    M: MatrixEntity + MatrixMut,
    M::MatElement:
        PrimInt + Debug + Display + NumInfo + FromPrimitive + WrappingSub + SampleUniform,
    NttOp: Ntt<Element = M::MatElement>,
    RlweModOp: ArithmeticOps<Element = M::MatElement>
        + VectorOps<Element = M::MatElement>
        + GetModulus<Element = M::MatElement, M = CiphertextModulus<M::MatElement>>,
    LweModOp: ArithmeticOps<Element = M::MatElement>
        + VectorOps<Element = M::MatElement>
        + GetModulus<Element = M::MatElement, M = CiphertextModulus<M::MatElement>>,
    M::R: TryConvertFrom1<[i32], CiphertextModulus<M::MatElement>> + RowEntity + Debug,
    <M as Matrix>::R: RowMut,
{
    fn new(parameters: BoolParameters<M::MatElement>) -> Self
    where
        RlweModOp: ModInit<M = CiphertextModulus<M::MatElement>>,
        LweModOp: ModInit<M = CiphertextModulus<M::MatElement>>,
        NttOp: NttInit<CiphertextModulus<M::MatElement>>,
    {
        //TODO(Jay): Run sanity checks for modulus values in parameters

        // generates dlog map s.t. (+/-)g^{k} % q = a, for all a \in Z*_{q} and k \in
        // [0, q/4). We store the dlog `k` at index `a`. This makes it easier to
        // simply look up `k` at runtime as vec[a]. If a = g^{k} then dlog is
        // stored as k. If a = -g^{k} then dlog is stored as k = q/4. This is done to
        // differentiate sign.
        let g = parameters.g();
        let q = *parameters.br_q();
        let mut g_k_dlog_map = vec![0usize; q];
        for i in 0..q / 4 {
            let v = mod_exponent(g as u64, i as u64, q as u64) as usize;
            // g^i
            g_k_dlog_map[v] = i;
            // -(g^i)
            g_k_dlog_map[q - v] = i + (q / 4);
        }

        let embedding_factor = (2 * parameters.rlwe_n().0) / q;

        let rlwe_nttop = NttOp::new(parameters.rlwe_q(), parameters.rlwe_n().0);
        let rlwe_modop = RlweModOp::new(*parameters.rlwe_q());
        let lwe_modop = LweModOp::new(*parameters.lwe_q());

        let q = *parameters.br_q();
        let qby2 = q >> 1;
        let qby8 = q >> 3;
        // Q/8 (Q: rlwe_q)
        let true_m_el = parameters.rlwe_q().true_el();
        // -Q/8
        let false_m_el = parameters.rlwe_q().false_el();
        let (auto_map_index, auto_map_sign) = generate_auto_map(qby2, -(g as isize));

        let init_test_vec = |partition_el: usize,
                             before_partition_el: M::MatElement,
                             after_partition_el: M::MatElement| {
            let mut test_vec = M::R::zeros(qby2);
            for i in 0..qby2 {
                if i < partition_el {
                    test_vec.as_mut()[i] = before_partition_el;
                } else {
                    test_vec.as_mut()[i] = after_partition_el;
                }
            }

            // v(X) -> v(X^{-g})
            let mut test_vec_autog = M::R::zeros(qby2);
            izip!(
                test_vec.as_ref().iter(),
                auto_map_index.iter(),
                auto_map_sign.iter()
            )
            .for_each(|(v, to_index, to_sign)| {
                if !to_sign {
                    // negate
                    test_vec_autog.as_mut()[*to_index] = rlwe_modop.neg(v);
                } else {
                    test_vec_autog.as_mut()[*to_index] = *v;
                }
            });

            return test_vec_autog;
        };

        let nand_test_vec = init_test_vec(3 * qby8, true_m_el, false_m_el);
        let and_test_vec = init_test_vec(3 * qby8, false_m_el, true_m_el);
        let or_test_vec = init_test_vec(qby8, false_m_el, true_m_el);
        let nor_test_vec = init_test_vec(qby8, true_m_el, false_m_el);
        let xor_test_vec = init_test_vec(qby8, false_m_el, true_m_el);
        let xnor_test_vec = init_test_vec(qby8, true_m_el, false_m_el);

        // auto map indices and sign
        // Auto maps are stored as [-g, g^{1}, g^{2}, ..., g^{w}]
        let mut rlwe_auto_maps = vec![];
        let ring_size = parameters.rlwe_n().0;
        let g = parameters.g();
        let br_q = parameters.br_q();
        let auto_element_dlogs = parameters.auto_element_dlogs();
        assert!(auto_element_dlogs[0] == 0);
        for i in auto_element_dlogs.into_iter() {
            let el = if i == 0 {
                -(g as isize)
            } else {
                (g.pow(i as u32) % br_q) as isize
            };
            rlwe_auto_maps.push(generate_auto_map(ring_size, el))
        }

        let rlwe_qby4 = parameters.rlwe_q().qby4();

        let scratch_memory = ScratchMemory::new(&parameters);

        let pbs_info = BoolPbsInfo {
            auto_decomposer: parameters.auto_decomposer(),
            lwe_decomposer: parameters.lwe_decomposer(),
            rlwe_rgsw_decomposer: parameters.rlwe_rgsw_decomposer(),
            g_k_dlog_map,
            embedding_factor,
            lwe_modop,
            rlwe_modop,
            rlwe_nttop,
            rlwe_qby4,
            rlwe_auto_maps,
            parameters: parameters,
        };

        BoolEvaluator {
            pbs_info,
            scratch_memory,
            nand_test_vec,
            and_test_vec,
            or_test_vec,
            nor_test_vec,
            xnor_test_vec,
            xor_test_vec,
            _phantom: PhantomData,
        }
    }

    fn client_key(&self) -> ClientKey {
        let sk_lwe = LweSecret::random(
            self.pbs_info.parameters.lwe_n().0 >> 1,
            self.pbs_info.parameters.lwe_n().0,
        );
        let sk_rlwe = RlweSecret::random(
            self.pbs_info.parameters.rlwe_n().0 >> 1,
            self.pbs_info.parameters.rlwe_n().0,
        );
        ClientKey { sk_rlwe, sk_lwe }
    }

    fn server_key(
        &self,
        client_key: &ClientKey,
    ) -> SeededServerKey<M, BoolParameters<M::MatElement>, [u8; 32]> {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut main_seed = [0u8; 32];
            rng.fill_bytes(&mut main_seed);

            let mut main_prng = DefaultSecureRng::new_seeded(main_seed);

            let rlwe_n = self.pbs_info.parameters.rlwe_n().0;
            let sk_rlwe = &client_key.sk_rlwe;
            let sk_lwe = &client_key.sk_lwe;

            // generate auto keys
            let mut auto_keys = HashMap::new();
            let auto_gadget = self.pbs_info.auto_decomposer.gadget_vector();
            let g = self.pbs_info.parameters.g();
            let br_q = self.pbs_info.parameters.br_q();
            let auto_els = self.pbs_info.parameters.auto_element_dlogs();
            for i in auto_els.into_iter() {
                let g_pow = if i == 0 {
                    -(g as isize)
                } else {
                    (g.pow(i as u32) % br_q) as isize
                };
                let mut gk = M::zeros(self.pbs_info.auto_decomposer.decomposition_count(), rlwe_n);
                galois_key_gen(
                    &mut gk,
                    sk_rlwe.values(),
                    g_pow,
                    &auto_gadget,
                    &self.pbs_info.rlwe_modop,
                    &self.pbs_info.rlwe_nttop,
                    &mut main_prng,
                    rng,
                );
                auto_keys.insert(i, gk);
            }

            // generate rgsw ciphertexts RGSW(si) where si is i^th LWE secret element
            let ring_size = self.pbs_info.parameters.rlwe_n().0;
            let rlwe_q = self.pbs_info.parameters.rlwe_q();
            let (rlrg_d_a, rlrg_d_b) = (
                self.pbs_info.rlwe_rgsw_decomposer.0.decomposition_count(),
                self.pbs_info.rlwe_rgsw_decomposer.1.decomposition_count(),
            );
            let rlrg_gadget_a = self.pbs_info.rlwe_rgsw_decomposer.0.gadget_vector();
            let rlrg_gadget_b = self.pbs_info.rlwe_rgsw_decomposer.1.gadget_vector();
            let rgsw_cts = sk_lwe
                .values()
                .iter()
                .map(|si| {
                    // X^{si}; assume |emebedding_factor * si| < N
                    let mut m = M::R::zeros(ring_size);
                    let si = (self.pbs_info.embedding_factor as i32) * si;
                    // dbg!(si);
                    if si < 0 {
                        // X^{-i} = X^{2N - i} = -X^{N-i}
                        m.as_mut()[ring_size - (si.abs() as usize)] = rlwe_q.neg_one();
                    } else {
                        // X^{i}
                        m.as_mut()[si.abs() as usize] = M::MatElement::one();
                    }

                    let mut rgsw_si = M::zeros(rlrg_d_a * 2 + rlrg_d_b, ring_size);
                    secret_key_encrypt_rgsw(
                        &mut rgsw_si,
                        m.as_ref(),
                        &rlrg_gadget_a,
                        &rlrg_gadget_b,
                        sk_rlwe.values(),
                        &self.pbs_info.rlwe_modop,
                        &self.pbs_info.rlwe_nttop,
                        &mut main_prng,
                        rng,
                    );

                    rgsw_si
                })
                .collect_vec();

            // LWE KSK from RLWE secret s -> LWE secret z
            let d_lwe_gadget = self.pbs_info.lwe_decomposer.gadget_vector();
            let mut lwe_ksk =
                M::R::zeros(self.pbs_info.lwe_decomposer.decomposition_count() * ring_size);
            lwe_ksk_keygen(
                &sk_rlwe.values(),
                &sk_lwe.values(),
                &mut lwe_ksk,
                &d_lwe_gadget,
                &self.pbs_info.lwe_modop,
                &mut main_prng,
                rng,
            );

            SeededServerKey::from_raw(
                auto_keys,
                rgsw_cts,
                lwe_ksk,
                self.pbs_info.parameters.clone(),
                main_seed,
            )
        })
    }

    fn multi_party_server_key_share(
        &self,
        cr_seed: [u8; 32],
        collective_pk: &M,
        client_key: &ClientKey,
    ) -> CommonReferenceSeededMultiPartyServerKeyShare<M, BoolParameters<M::MatElement>, [u8; 32]>
    {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut main_prng = DefaultSecureRng::new_seeded(cr_seed);

            let sk_rlwe = &client_key.sk_rlwe;
            let sk_lwe = &client_key.sk_lwe;

            let g = self.pbs_info.parameters.g();
            let ring_size = self.pbs_info.parameters.rlwe_n().0;
            let rlwe_q = self.pbs_info.parameters.rlwe_q();
            let lwe_q = self.pbs_info.parameters.lwe_q();

            let rlweq_modop = &self.pbs_info.rlwe_modop;
            let rlweq_nttop = &self.pbs_info.rlwe_nttop;

            // sanity check
            assert!(sk_rlwe.values().len() == ring_size);
            assert!(sk_lwe.values().len() == self.pbs_info.parameters.lwe_n().0);

            // auto keys
            let mut auto_keys = HashMap::new();
            let auto_gadget = self.pbs_info.auto_decomposer.gadget_vector();
            let auto_element_dlogs = self.pbs_info.parameters.auto_element_dlogs();
            let br_q = self.pbs_info.parameters.br_q();
            for i in auto_element_dlogs.into_iter() {
                let g_pow = if i == 0 {
                    -(g as isize)
                } else {
                    (g.pow(i as u32) % br_q) as isize
                };

                let mut ksk_out = M::zeros(
                    self.pbs_info.auto_decomposer.decomposition_count(),
                    ring_size,
                );
                galois_key_gen(
                    &mut ksk_out,
                    sk_rlwe.values(),
                    g_pow,
                    &auto_gadget,
                    rlweq_modop,
                    rlweq_nttop,
                    &mut main_prng,
                    rng,
                );
                auto_keys.insert(i, ksk_out);
            }

            // rgsw ciphertexts of lwe secret elements
            let rgsw_rgsw_decomposer = self
                .pbs_info
                .parameters
                .rgsw_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();
            let (rgrg_d_a, rgrg_d_b) = (
                rgsw_rgsw_decomposer.0.decomposition_count(),
                rgsw_rgsw_decomposer.1.decomposition_count(),
            );
            let (rgrg_gadget_a, rgrg_gadget_b) = (
                rgsw_rgsw_decomposer.0.gadget_vector(),
                rgsw_rgsw_decomposer.1.gadget_vector(),
            );
            let rgsw_cts = sk_lwe
                .values()
                .iter()
                .map(|si| {
                    let mut m = M::R::zeros(ring_size);
                    //TODO(Jay): It will be nice to have a function that returns polynomial
                    // (monomial infact!) corresponding to secret element embedded in ring X^{2N+1}.
                    // Save lots of mistakes where one forgest to emebed si in bigger ring.
                    let si = *si * (self.pbs_info.embedding_factor as i32);
                    if si < 0 {
                        // X^{-si} = X^{2N-si} = -X^{N-si}, assuming abs(si) < N
                        // (which it is given si is secret element)
                        m.as_mut()[ring_size - (si.abs() as usize)] = rlwe_q.neg_one();
                    } else {
                        m.as_mut()[si as usize] = M::MatElement::one();
                    }

                    // public key RGSW encryption has no part that can be seeded, unlike secret key
                    // RGSW encryption where RLWE'_A(m) is seeded
                    let mut out_rgsw = M::zeros(rgrg_d_a * 2 + rgrg_d_b * 2, ring_size);
                    public_key_encrypt_rgsw(
                        &mut out_rgsw,
                        &m.as_ref(),
                        collective_pk,
                        &rgrg_gadget_a,
                        &rgrg_gadget_b,
                        rlweq_modop,
                        rlweq_nttop,
                        rng,
                    );

                    out_rgsw
                })
                .collect_vec();

            // LWE ksk
            let mut lwe_ksk =
                M::R::zeros(self.pbs_info.lwe_decomposer.decomposition_count() * ring_size);
            let lwe_modop = &self.pbs_info.lwe_modop;
            let d_lwe_gadget_vec = self.pbs_info.lwe_decomposer.gadget_vector();
            lwe_ksk_keygen(
                sk_rlwe.values(),
                sk_lwe.values(),
                &mut lwe_ksk,
                &d_lwe_gadget_vec,
                lwe_modop,
                &mut main_prng,
                rng,
            );

            CommonReferenceSeededMultiPartyServerKeyShare {
                auto_keys,
                rgsw_cts,
                lwe_ksk,
                cr_seed,
                parameters: self.pbs_info.parameters.clone(),
            }
        })
    }

    fn multi_party_public_key_share(
        &self,
        cr_seed: [u8; 32],
        client_key: &ClientKey,
    ) -> CommonReferenceSeededCollectivePublicKeyShare<
        <M as Matrix>::R,
        [u8; 32],
        BoolParameters<<M as Matrix>::MatElement>,
    > {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut share_out = M::R::zeros(self.pbs_info.parameters.rlwe_n().0);
            let modop = &self.pbs_info.rlwe_modop;
            let nttop = &self.pbs_info.rlwe_nttop;
            let mut main_prng = DefaultSecureRng::new_seeded(cr_seed);
            public_key_share(
                &mut share_out,
                client_key.sk_rlwe.values(),
                modop,
                nttop,
                &mut main_prng,
                rng,
            );

            CommonReferenceSeededCollectivePublicKeyShare {
                share: share_out,
                cr_seed: cr_seed,
                parameters: self.pbs_info.parameters.clone(),
            }
        })
    }

    fn multi_party_decryption_share(
        &self,
        lwe_ct: &M::R,
        client_key: &ClientKey,
    ) -> <M as Matrix>::MatElement {
        assert!(lwe_ct.as_ref().len() == self.pbs_info.parameters.rlwe_n().0 + 1);
        let modop = &self.pbs_info.rlwe_modop;
        let mut neg_s = M::R::try_convert_from(
            client_key.sk_rlwe.values(),
            &self.pbs_info.parameters.rlwe_q(),
        );
        modop.elwise_neg_mut(neg_s.as_mut());

        let mut neg_sa = M::MatElement::zero();
        izip!(lwe_ct.as_ref().iter().skip(1), neg_s.as_ref().iter()).for_each(|(ai, nsi)| {
            neg_sa = modop.add(&neg_sa, &modop.mul(ai, nsi));
        });

        let e = DefaultSecureRng::with_local_mut(|rng| {
            let mut e =
                RandomGaussianElementInModulus::random(rng, self.pbs_info.parameters.rlwe_q());
            e
        });
        let share = modop.add(&neg_sa, &e);

        share
    }

    pub(crate) fn multi_party_decrypt(&self, shares: &[M::MatElement], lwe_ct: &M::R) -> bool {
        let modop = &self.pbs_info.rlwe_modop;
        let mut sum_a = M::MatElement::zero();
        shares
            .iter()
            .for_each(|share_i| sum_a = modop.add(&sum_a, &share_i));

        let encoded_m = modop.add(&lwe_ct.as_ref()[0], &sum_a);
        self.pbs_info.parameters.rlwe_q().decode(encoded_m)
    }

    pub(crate) fn pk_encrypt(&self, pk: &M, m: bool) -> M::R {
        self.pk_encrypt_batched(pk, &vec![m]).remove(0)
    }

    /// Encrypts a batch booleans as multiple LWE ciphertexts.
    ///
    /// For public key encryption we first encrypt `m` as a RLWE ciphertext and
    /// then sample extract LWE samples at required indices.
    ///
    /// - TODO(Jay:) Communication can be improved by not sample exctracting and
    ///   instead just truncate degree 0 values (part Bs)
    pub(crate) fn pk_encrypt_batched(&self, pk: &M, m: &[bool]) -> Vec<M::R> {
        DefaultSecureRng::with_local_mut(|rng| {
            let ring_size = self.pbs_info.parameters.rlwe_n().0;
            assert!(
                m.len() <= ring_size,
                "Cannot batch encrypt > ring_size{ring_size} elements at once"
            );

            let modop = &self.pbs_info.rlwe_modop;
            let nttop = &self.pbs_info.rlwe_nttop;

            // RLWE(0)
            // sample ephemeral key u
            let mut u = vec![0i32; ring_size];
            fill_random_ternary_secret_with_hamming_weight(u.as_mut(), ring_size >> 1, rng);
            let mut u = M::R::try_convert_from(&u, &self.pbs_info.parameters.rlwe_q());
            nttop.forward(u.as_mut());

            let mut ua = M::R::zeros(ring_size);
            ua.as_mut().copy_from_slice(pk.get_row_slice(0));
            let mut ub = M::R::zeros(ring_size);
            ub.as_mut().copy_from_slice(pk.get_row_slice(1));

            // a*u
            nttop.forward(ua.as_mut());
            modop.elwise_mul_mut(ua.as_mut(), u.as_ref());
            nttop.backward(ua.as_mut());

            // b*u
            nttop.forward(ub.as_mut());
            modop.elwise_mul_mut(ub.as_mut(), u.as_ref());
            nttop.backward(ub.as_mut());

            let mut rlwe = M::zeros(2, ring_size);
            // sample error
            rlwe.iter_rows_mut().for_each(|ri| {
                RandomFillGaussianInModulus::<[M::MatElement], CiphertextModulus<M::MatElement>>::random_fill(
                    rng,
                    &self.pbs_info.parameters.rlwe_q(),
                    ri.as_mut(),
                );
            });

            // a*u + e0
            modop.elwise_add_mut(rlwe.get_row_mut(0), ua.as_ref());
            // b*u + e1
            modop.elwise_add_mut(rlwe.get_row_mut(1), ub.as_ref());

            //FIXME(Jay): Figure out a way to get Q/8 form modulus
            let mut m_vec = M::R::zeros(ring_size);
            izip!(m_vec.as_mut().iter_mut(), m.iter()).for_each(|(m_el, m_bool)| {
                if *m_bool {
                    // Q/8
                    *m_el = self.pbs_info.rlwe_q().true_el()
                } else {
                    // -Q/8
                    *m_el = self.pbs_info.rlwe_q().false_el()
                }
            });

            // b*u + e1 + m
            modop.elwise_add_mut(rlwe.get_row_mut(1), m_vec.as_ref());
            // rlwe.set(1, 0, modop.add(rlwe.get(1, 0), &m));

            // sample extract index required indices
            let samples = m.len();
            (0..samples)
                .into_iter()
                .map(|i| {
                    let mut lwe_out = M::R::zeros(ring_size + 1);
                    sample_extract(&mut lwe_out, &rlwe, modop, i);
                    lwe_out
                })
                .collect_vec()
        })
    }

    /// TODO(Jay): Fetch client key from thread local
    pub fn sk_encrypt(&self, m: bool, client_key: &ClientKey) -> M::R {
        //FIXME(Jay): Figure out a way to get Q/8 form modulus
        let m = if m {
            // Q/8
            self.pbs_info.rlwe_q().true_el()
        } else {
            // -Q/8
            self.pbs_info.rlwe_q().false_el()
        };

        DefaultSecureRng::with_local_mut(|rng| {
            let mut lwe_out = M::R::zeros(self.pbs_info.parameters.rlwe_n().0 + 1);
            encrypt_lwe(
                &mut lwe_out,
                &m,
                client_key.sk_rlwe.values(),
                &self.pbs_info.rlwe_modop,
                rng,
            );
            lwe_out
        })
    }

    pub fn sk_decrypt(&self, lwe_ct: &M::R, client_key: &ClientKey) -> bool {
        let m = decrypt_lwe(
            lwe_ct,
            client_key.sk_rlwe.values(),
            &self.pbs_info.rlwe_modop,
        );
        self.pbs_info.rlwe_q().decode(m)
    }

    fn aggregate_multi_party_server_key_shares<S>(
        &self,
        shares: &[CommonReferenceSeededMultiPartyServerKeyShare<
            M,
            BoolParameters<M::MatElement>,
            S,
        >],
    ) -> SeededMultiPartyServerKey<M, S, BoolParameters<M::MatElement>>
    where
        S: PartialEq + Clone,
        M: Clone,
    {
        assert!(shares.len() > 0);
        let parameters = shares[0].parameters.clone();
        let cr_seed = &shares[0].cr_seed;

        let rlwe_n = parameters.rlwe_n().0;
        let g = parameters.g() as isize;
        let rlwe_q = parameters.rlwe_q();
        let lwe_q = parameters.lwe_q();

        // sanity checks
        shares.iter().skip(1).for_each(|s| {
            assert!(s.parameters == parameters);
            assert!(&s.cr_seed == cr_seed);
        });

        let rlweq_modop = &self.pbs_info.rlwe_modop;
        let rlweq_nttop = &self.pbs_info.rlwe_nttop;

        // auto keys
        let mut auto_keys = HashMap::new();
        let auto_elements_dlog = parameters.auto_element_dlogs();
        for i in auto_elements_dlog.into_iter() {
            let mut key = M::zeros(parameters.auto_decomposition_count().0, rlwe_n);

            shares.iter().for_each(|s| {
                let auto_key_share_i = s.auto_keys.get(&i).expect("Auto key {i} missing");
                assert!(
                    auto_key_share_i.dimension()
                        == (parameters.auto_decomposition_count().0, rlwe_n)
                );
                izip!(key.iter_rows_mut(), auto_key_share_i.iter_rows()).for_each(
                    |(partb_out, partb_share)| {
                        rlweq_modop.elwise_add_mut(partb_out.as_mut(), partb_share.as_ref());
                    },
                );
            });

            auto_keys.insert(i, key);
        }

        // rgsw ciphertext (most expensive part!)
        let lwe_n = parameters.lwe_n().0;
        let rgsw_by_rgsw_decomposer =
            parameters.rgsw_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();
        let mut scratch_matrix = M::zeros(
            std::cmp::max(
                rgsw_by_rgsw_decomposer.a().decomposition_count(),
                rgsw_by_rgsw_decomposer.b().decomposition_count(),
            ) + (rgsw_by_rgsw_decomposer.a().decomposition_count() * 2
                + rgsw_by_rgsw_decomposer.b().decomposition_count() * 2),
            rlwe_n,
        );

        let mut tmp_rgsw =
            RgswCiphertext::<M, _>::empty(rlwe_n, &rgsw_by_rgsw_decomposer, rlwe_q.clone()).data;
        let rgsw_cts = (0..lwe_n).into_iter().map(|index| {
            // copy over rgsw ciphertext for index^th secret element from first share and
            // treat it as accumulating rgsw ciphertext
            let mut rgsw_i = shares[0].rgsw_cts[index].clone();

            shares.iter().skip(1).for_each(|si| {
                // copy over si's RGSW[index] ciphertext and send to evaluation domain
                izip!(tmp_rgsw.iter_rows_mut(), si.rgsw_cts[index].iter_rows()).for_each(
                    |(to_ri, from_ri)| {
                        to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                        rlweq_nttop.forward(to_ri.as_mut())
                    },
                );

                rgsw_by_rgsw_inplace(
                    &mut rgsw_i,
                    &tmp_rgsw,
                    &rgsw_by_rgsw_decomposer,
                    &mut scratch_matrix,
                    rlweq_nttop,
                    rlweq_modop,
                );
            });

            rgsw_i
        });
        // d_a and d_b may differ for RGSWxRGSW multiplication and RLWExRGSW
        // multiplication. After this point RGSW ciphertexts will only be used for
        // RLWExRGSW multiplication (in blind rotation). Thus we drop any additional
        // RLWE ciphertexts in RGSW ciphertexts after RGSw x RGSW multiplication
        let (rgswrgsw_d_a, rgswrgsw_d_b) = self.pbs_info.parameters.rgsw_rgsw_decomposition_count();
        let (rlrg_d_a, rlrg_d_b) = self.pbs_info.parameters.rlwe_rgsw_decomposition_count();
        let rgsw_ct_rows_in = rgswrgsw_d_a.0 * 2 + rgswrgsw_d_b.0 * 2;
        let rgsw_ct_rows_out = rlrg_d_a.0 * 2 + rlrg_d_b.0 * 2;
        assert!(rgswrgsw_d_a.0 >= rlrg_d_a.0, "RGSWxRGSW part A decomposition count {} must be >= RLWExRGSW part A decomposition count {}", rgswrgsw_d_a.0 , rlrg_d_a.0);
        assert!(rgswrgsw_d_b.0 >= rlrg_d_b.0, "RGSWxRGSW part B decomposition count {} must be >= RLWExRGSW part B decomposition count {}", rgswrgsw_d_b.0 , rlrg_d_b.0);
        let rgsw_cts = rgsw_cts
            .map(|ct_i_in| {
                assert!(ct_i_in.dimension() == (rgsw_ct_rows_in, rlwe_n));
                let mut reduced_ct_i_out = M::zeros(rgsw_ct_rows_out, rlwe_n);

                // RLWE'(-sm) part A
                izip!(
                    reduced_ct_i_out.iter_rows_mut().take(rlrg_d_a.0),
                    ct_i_in
                        .iter_rows()
                        .skip(rgswrgsw_d_a.0 - rlrg_d_a.0)
                        .take(rlrg_d_a.0)
                )
                .for_each(|(to_ri, from_ri)| {
                    to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                });

                // RLWE'(-sm) part B
                izip!(
                    reduced_ct_i_out
                        .iter_rows_mut()
                        .skip(rlrg_d_a.0)
                        .take(rlrg_d_a.0),
                    ct_i_in
                        .iter_rows()
                        .skip(rgswrgsw_d_a.0 + (rgswrgsw_d_a.0 - rlrg_d_a.0))
                        .take(rlrg_d_a.0)
                )
                .for_each(|(to_ri, from_ri)| {
                    to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                });

                // RLWE'(m) Part A
                izip!(
                    reduced_ct_i_out
                        .iter_rows_mut()
                        .skip(rlrg_d_a.0 * 2)
                        .take(rlrg_d_b.0),
                    ct_i_in
                        .iter_rows()
                        .skip(rgswrgsw_d_a.0 * 2 + (rgswrgsw_d_b.0 - rlrg_d_b.0))
                        .take(rlrg_d_b.0)
                )
                .for_each(|(to_ri, from_ri)| {
                    to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                });

                // RLWE'(m) Part B
                izip!(
                    reduced_ct_i_out
                        .iter_rows_mut()
                        .skip(rlrg_d_a.0 * 2 + rlrg_d_b.0)
                        .take(rlrg_d_b.0),
                    ct_i_in
                        .iter_rows()
                        .skip(rgswrgsw_d_a.0 * 2 + rgswrgsw_d_b.0 + (rgswrgsw_d_b.0 - rlrg_d_b.0))
                        .take(rlrg_d_b.0)
                )
                .for_each(|(to_ri, from_ri)| {
                    to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                });

                reduced_ct_i_out
            })
            .collect_vec();

        // LWE ksks
        let mut lwe_ksk = M::R::zeros(rlwe_n * parameters.lwe_decomposition_count().0);
        let lweq_modop = &self.pbs_info.lwe_modop;
        shares.iter().for_each(|si| {
            assert!(si.lwe_ksk.as_ref().len() == rlwe_n * parameters.lwe_decomposition_count().0);
            lweq_modop.elwise_add_mut(lwe_ksk.as_mut(), si.lwe_ksk.as_ref())
        });

        SeededMultiPartyServerKey {
            rgsw_cts,
            auto_keys,
            lwe_ksk,
            cr_seed: cr_seed.clone(),
            parameters: parameters,
        }
    }
}

impl<M, NttOp, RlweModOp, LweModOp> BoolEvaluator<M, NttOp, RlweModOp, LweModOp>
where
    M: MatrixMut + MatrixEntity,
    M::R: RowMut + RowEntity,
    M::MatElement: PrimInt + FromPrimitive + One + Copy + Zero + Display + WrappingSub + NumInfo,
    RlweModOp: VectorOps<Element = M::MatElement>
        + ArithmeticOps<Element = M::MatElement>
        + GetModulus<Element = M::MatElement, M = CiphertextModulus<M::MatElement>>,
    LweModOp: VectorOps<Element = M::MatElement>
        + ArithmeticOps<Element = M::MatElement>
        + GetModulus<Element = M::MatElement, M = CiphertextModulus<M::MatElement>>,
    NttOp: Ntt<Element = M::MatElement>,
{
    /// Returns c0 + c1 + Q/4
    fn _add_and_shift_lwe_cts(&self, c0: &mut M::R, c1: &M::R) {
        let modop = &self.pbs_info.rlwe_modop;
        modop.elwise_add_mut(c0.as_mut(), c1.as_ref());
        // +Q/4
        c0.as_mut()[0] = modop.add(&c0.as_ref()[0], &self.pbs_info.rlwe_qby4);
    }

    /// Returns 2(c0 - c1) + Q/4
    fn _subtract_double_lwe_cts(&self, c0: &mut M::R, c1: &M::R) {
        let modop = &self.pbs_info.rlwe_modop;
        // c0 - c1
        modop.elwise_sub_mut(c0.as_mut(), c1.as_ref());

        // double
        c0.as_mut().iter_mut().for_each(|v| *v = modop.add(v, v));
    }
}

impl<M, NttOp, RlweModOp, LweModOp> BooleanGates for BoolEvaluator<M, NttOp, RlweModOp, LweModOp>
where
    M: MatrixMut + MatrixEntity,
    M::R: RowMut + RowEntity + Clone,
    M::MatElement: PrimInt + FromPrimitive + One + Copy + Zero + Display + WrappingSub + NumInfo,
    RlweModOp: VectorOps<Element = M::MatElement>
        + ArithmeticOps<Element = M::MatElement>
        + GetModulus<Element = M::MatElement, M = CiphertextModulus<M::MatElement>>,
    LweModOp: VectorOps<Element = M::MatElement>
        + ArithmeticOps<Element = M::MatElement>
        + GetModulus<Element = M::MatElement, M = CiphertextModulus<M::MatElement>>,
    NttOp: Ntt<Element = M::MatElement>,
{
    type Ciphertext = M::R;
    type Key = ServerKeyEvaluationDomain<M, DefaultSecureRng, NttOp>;

    fn nand_inplace(
        &mut self,
        c0: &mut M::R,
        c1: &M::R,
        server_key: &ServerKeyEvaluationDomain<M, DefaultSecureRng, NttOp>,
    ) {
        self._add_and_shift_lwe_cts(c0, c1);

        // PBS
        pbs(
            &self.pbs_info,
            &self.nand_test_vec,
            c0,
            server_key,
            &mut self.scratch_memory.lwe_vector,
            &mut self.scratch_memory.decomposition_matrix,
        );
    }

    fn and_inplace(
        &mut self,
        c0: &mut M::R,
        c1: &M::R,
        server_key: &ServerKeyEvaluationDomain<M, DefaultSecureRng, NttOp>,
    ) {
        self._add_and_shift_lwe_cts(c0, c1);

        // PBS
        pbs(
            &self.pbs_info,
            &self.and_test_vec,
            c0,
            server_key,
            &mut self.scratch_memory.lwe_vector,
            &mut self.scratch_memory.decomposition_matrix,
        );
    }

    fn or_inplace(
        &mut self,
        c0: &mut M::R,
        c1: &M::R,
        server_key: &ServerKeyEvaluationDomain<M, DefaultSecureRng, NttOp>,
    ) {
        self._add_and_shift_lwe_cts(c0, c1);

        // PBS
        pbs(
            &self.pbs_info,
            &self.or_test_vec,
            c0,
            server_key,
            &mut self.scratch_memory.lwe_vector,
            &mut self.scratch_memory.decomposition_matrix,
        );
    }

    fn nor_inplace(
        &mut self,
        c0: &mut M::R,
        c1: &M::R,
        server_key: &ServerKeyEvaluationDomain<M, DefaultSecureRng, NttOp>,
    ) {
        self._add_and_shift_lwe_cts(c0, c1);

        // PBS
        pbs(
            &self.pbs_info,
            &self.nor_test_vec,
            c0,
            server_key,
            &mut self.scratch_memory.lwe_vector,
            &mut self.scratch_memory.decomposition_matrix,
        )
    }

    fn xor_inplace(
        &mut self,
        c0: &mut M::R,
        c1: &M::R,
        server_key: &ServerKeyEvaluationDomain<M, DefaultSecureRng, NttOp>,
    ) {
        self._subtract_double_lwe_cts(c0, c1);

        // PBS
        pbs(
            &self.pbs_info,
            &self.xor_test_vec,
            c0,
            server_key,
            &mut self.scratch_memory.lwe_vector,
            &mut self.scratch_memory.decomposition_matrix,
        );
    }

    fn xnor_inplace(
        &mut self,
        c0: &mut M::R,
        c1: &M::R,
        server_key: &ServerKeyEvaluationDomain<M, DefaultSecureRng, NttOp>,
    ) {
        self._subtract_double_lwe_cts(c0, c1);

        // PBS
        pbs(
            &self.pbs_info,
            &self.xnor_test_vec,
            c0,
            server_key,
            &mut self.scratch_memory.lwe_vector,
            &mut self.scratch_memory.decomposition_matrix,
        );
    }

    fn not_inplace(&mut self, c0: &mut M::R) {
        let modop = &self.pbs_info.rlwe_modop;
        c0.as_mut().iter_mut().for_each(|v| *v = modop.neg(v));
    }

    fn and(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext {
        let mut out = c0.clone();
        self.and_inplace(&mut out, c1, key);
        out
    }

    fn nand(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext {
        let mut out = c0.clone();
        self.nand_inplace(&mut out, c1, key);
        out
    }

    fn or(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext {
        let mut out = c0.clone();
        self.or_inplace(&mut out, c1, key);
        out
    }

    fn nor(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext {
        let mut out = c0.clone();
        self.nor_inplace(&mut out, c1, key);
        out
    }

    fn xnor(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext {
        let mut out = c0.clone();
        self.xnor_inplace(&mut out, c1, key);
        out
    }

    fn xor(
        &mut self,
        c0: &Self::Ciphertext,
        c1: &Self::Ciphertext,
        key: &Self::Key,
    ) -> Self::Ciphertext {
        let mut out = c0.clone();
        self.xor_inplace(&mut out, c1, key);
        out
    }

    fn not(&mut self, c: &Self::Ciphertext) -> Self::Ciphertext {
        let mut out = c.clone();
        self.not_inplace(&mut out);
        out
    }
}

thread_local! {
    static PBS_TRACER: RefCell<PBSTracer<Vec<Vec<u64>>>> =
RefCell::new(PBSTracer::default()); }

#[derive(Default)]
struct PBSTracer<M>
where
    M: Matrix + Default,
{
    pub(crate) ct_rlwe_q_mod: M::R,
    pub(crate) ct_lwe_q_mod: M::R,
    pub(crate) ct_lwe_q_mod_after_ksk: M::R,
    pub(crate) ct_br_q_mod: Vec<u64>,
}

impl PBSTracer<Vec<Vec<u64>>> {
    fn trace(&self, parameters: &BoolParameters<u64>, sk_lwe: &[i32], sk_rlwe: &[i32]) {
        assert!(parameters.rlwe_n().0 == sk_rlwe.len());
        assert!(parameters.lwe_n().0 == sk_lwe.len());

        let modop_rlweq = ModularOpsU64::new(*parameters.rlwe_q());
        // noise after mod down Q -> Q_ks
        let m_back0 = decrypt_lwe(&self.ct_rlwe_q_mod, sk_rlwe, &modop_rlweq);

        let modop_lweq = ModularOpsU64::<CiphertextModulus<u64>>::new(*parameters.lwe_q());
        // noise after mod down Q -> Q_ks
        let m_back1 = decrypt_lwe(&self.ct_lwe_q_mod, sk_rlwe, &modop_lweq);
        // noise after key switch from RLWE -> LWE
        let m_back2 = decrypt_lwe(&self.ct_lwe_q_mod_after_ksk, sk_lwe, &modop_lweq);

        // noise after mod down odd from Q_ks -> q
        let modop_br_q = ModularOpsU64::<u64>::new(*parameters.br_q() as u64);
        let m_back3 = decrypt_lwe(&self.ct_br_q_mod, sk_lwe, &modop_br_q);

        println!(
            "
            M initial mod Q: {m_back0},
            M after mod down Q -> Q_ks: {m_back1},
            M after key switch from RLWE -> LWE: {m_back2},
            M after mod dwon Q_ks -> q: {m_back3}
        "
        );
    }
}

impl WithLocal for PBSTracer<Vec<Vec<u64>>> {
    fn with_local<F, R>(func: F) -> R
    where
        F: Fn(&Self) -> R,
    {
        PBS_TRACER.with_borrow(|t| func(t))
    }

    fn with_local_mut<F, R>(func: F) -> R
    where
        F: Fn(&mut Self) -> R,
    {
        PBS_TRACER.with_borrow_mut(|t| func(t))
    }

    fn with_local_mut_mut<F, R>(func: &mut F) -> R
    where
        F: FnMut(&mut Self) -> R,
    {
        PBS_TRACER.with_borrow_mut(|t| func(t))
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use rand_distr::Uniform;

    use crate::{
        backend::{GetModulus, ModInit, ModularOpsU64, WordSizeModulus},
        bool,
        ntt::NttBackendU64,
        random::{RandomElementInModulus, DEFAULT_RNG},
        rgsw::{
            self, measure_noise, public_key_encrypt_rlwe, secret_key_encrypt_rlwe,
            tests::{_measure_noise_rgsw, _sk_encrypt_rlwe},
            RgswCiphertext, RgswCiphertextEvaluationDomain, SeededRgswCiphertext,
            SeededRlweCiphertext,
        },
        utils::{negacyclic_mul, Stats},
    };

    use super::*;

    #[test]
    fn tri() {
        let bool_evaluator = BoolEvaluator::<
            Vec<Vec<u64>>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
            ModularOpsU64<CiphertextModulus<u64>>,
        >::new(SP_BOOL_PARAMS);
        let mut v = bool_evaluator.pbs_info.g_k_dlog_map.clone();
        // v.sort();
        println!("{:?}", v);
        let client_key = bool_evaluator.client_key();
        let server_key = bool_evaluator.server_key(&client_key);
        let server_key_eval_domain =
            ServerKeyEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(&server_key);

        let ring_size = bool_evaluator.pbs_info.parameters.rlwe_n().0;
        let rlwe_q = bool_evaluator.pbs_info.rlwe_q().q().unwrap();
        let mut rng = DefaultSecureRng::new();

        let mut m = vec![0u64; ring_size as usize];
        RandomFillUniformInModulus::random_fill(&mut rng, &rlwe_q, m.as_mut_slice());

        let ntt_op = &bool_evaluator.pbs_info.rlwe_nttop;
        let mod_op = &bool_evaluator.pbs_info.rlwe_modop;

        // RLWE_{s}(m)
        let mut seed_rlwe = [0u8; 32];
        rng.fill_bytes(&mut seed_rlwe);
        let mut seeded_rlwe_m = SeededRlweCiphertext::empty(ring_size as usize, seed_rlwe, rlwe_q);
        let mut p_rng = DefaultSecureRng::new_seeded(seed_rlwe);
        secret_key_encrypt_rlwe(
            &m,
            &mut seeded_rlwe_m.data,
            client_key.sk_rlwe.values(),
            mod_op,
            ntt_op,
            &mut p_rng,
            &mut rng,
        );
        let mut rlwe_m = RlweCiphertext::<Vec<Vec<u64>>, DefaultSecureRng>::from(&seeded_rlwe_m);

        let k = 1;
        let auto_k = (5usize).pow(k as u32);
        // let auto_k = -5;

        let decomposer = bool_evaluator.pbs_info.auto_decomposer();

        // Send RLWE_{s}(m) -> RLWE_{s}(m^k)
        let mut scratch_space =
            vec![vec![0u64; ring_size as usize]; decomposer.decomposition_count() + 2];
        let (auto_map_index, auto_map_sign) = bool_evaluator.pbs_info.rlwe_auto_map(k);
        galois_auto(
            &mut rlwe_m,
            server_key_eval_domain.galois_key_for_auto(k),
            &mut scratch_space,
            &auto_map_index,
            &auto_map_sign,
            mod_op,
            ntt_op,
            decomposer,
        );

        let rlwe_m_k = rlwe_m;

        // Decrypt RLWE_{s}(m^k) and check
        let mut encoded_m_k_back = vec![0u64; ring_size as usize];
        decrypt_rlwe(
            &rlwe_m_k,
            client_key.sk_rlwe.values(),
            &mut encoded_m_k_back,
            ntt_op,
            mod_op,
        );

        {
            let mut m_k = vec![0u64; ring_size];
            let (auto_map_index, auto_map_sign) = generate_auto_map(ring_size, auto_k as isize);
            izip!(m.iter(), auto_map_index.iter(), auto_map_sign.iter()).for_each(
                |(v, to_index, sign)| {
                    if !*sign {
                        m_k[*to_index] = (rlwe_q - *v);
                    } else {
                        m_k[*to_index] = *v;
                    }
                },
            );
            let noise = measure_noise(&rlwe_m_k, &m_k, ntt_op, mod_op, client_key.sk_rlwe.values());
            println!("Ksk noise: {noise}");
        }
    }

    #[test]
    fn bool_encrypt_decrypt_works() {
        let bool_evaluator = BoolEvaluator::<
            Vec<Vec<u64>>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
            ModularOpsU64<CiphertextModulus<u64>>,
        >::new(SP_BOOL_PARAMS);
        let client_key = bool_evaluator.client_key();

        let mut m = true;
        for _ in 0..1000 {
            let lwe_ct = bool_evaluator.sk_encrypt(m, &client_key);
            let m_back = bool_evaluator.sk_decrypt(&lwe_ct, &client_key);
            assert_eq!(m, m_back);
            m = !m;
        }
    }

    #[test]
    fn bool_nand() {
        DefaultSecureRng::with_local_mut(|r| {
            let rng = DefaultSecureRng::new_seeded([19u8; 32]);
            *r = rng;
        });

        // let mog = WordSizeModulus::<CiphertextModulus<u64>>::new(12u64);

        let mut bool_evaluator = BoolEvaluator::<
            Vec<Vec<u64>>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
            ModularOpsU64<CiphertextModulus<u64>>,
        >::new(SP_BOOL_PARAMS);

        // println!("{:?}", bool_evaluator.nand_test_vec);
        let client_key = bool_evaluator.client_key();
        let seeded_server_key = bool_evaluator.server_key(&client_key);
        let server_key_eval_domain =
            ServerKeyEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(
                &seeded_server_key,
            );

        let mut m0 = false;
        let mut m1 = true;

        let mut ct0 = bool_evaluator.sk_encrypt(m0, &client_key);
        let mut ct1 = bool_evaluator.sk_encrypt(m1, &client_key);

        for _ in 0..500 {
            let ct_back = bool_evaluator.nand(&ct0, &ct1, &server_key_eval_domain);

            let m_out = !(m0 && m1);

            // Trace and measure PBS noise
            {
                let noise0 = {
                    let ideal = if m0 {
                        bool_evaluator.pbs_info.parameters.rlwe_q().true_el()
                    } else {
                        bool_evaluator.pbs_info.parameters.rlwe_q().false_el()
                    };
                    let n = measure_noise_lwe(
                        &ct0,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                        &ideal,
                    );
                    let v = decrypt_lwe(
                        &ct0,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                    );
                    (n, v)
                };
                let noise1 = {
                    let ideal = if m1 {
                        bool_evaluator.pbs_info.parameters.rlwe_q().true_el()
                    } else {
                        bool_evaluator.pbs_info.parameters.rlwe_q().false_el()
                    };
                    let n = measure_noise_lwe(
                        &ct1,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                        &ideal,
                    );
                    let v = decrypt_lwe(
                        &ct1,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                    );
                    (n, v)
                };

                // // // Trace PBS
                // PBSTracer::with_local(|t| {
                //     t.trace(
                //         &SP_BOOL_PARAMS,
                //         &client_key.sk_lwe.values(),
                //         client_key.sk_rlwe.values(),
                //     )
                // });

                // Calculate noise in ciphertext post PBS
                let noise_out = {
                    let ideal = if m_out {
                        bool_evaluator.pbs_info.parameters.rlwe_q().true_el()
                    } else {
                        bool_evaluator.pbs_info.parameters.rlwe_q().false_el()
                    };
                    let n = measure_noise_lwe(
                        &ct_back,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                        &ideal,
                    );
                    let v = decrypt_lwe(
                        &ct_back,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                    );
                    (n, v)
                };
                dbg!(m0, m1, m_out);
                println!(
                    "ct0 (noise, message): {:?}  \n ct1 (noise, message): {:?} \n PBS (noise, message): {:?}", noise0, noise1, noise_out
                );
            }
            let m_back = bool_evaluator.sk_decrypt(&ct_back, &client_key);
            assert!(m_out == m_back, "Expected {m_out}, got {m_back}");
            println!("----------");

            m1 = m0;
            m0 = m_out;

            ct1 = ct0;
            ct0 = ct_back;
        }
    }

    #[test]
    fn bool_xor() {
        let mut bool_evaluator = BoolEvaluator::<
            Vec<Vec<u64>>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
            ModularOpsU64<CiphertextModulus<u64>>,
        >::new(SP_BOOL_PARAMS);

        // println!("{:?}", bool_evaluator.nand_test_vec);
        let client_key = bool_evaluator.client_key();
        let seeded_server_key = bool_evaluator.server_key(&client_key);
        let server_key_eval_domain =
            ServerKeyEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(
                &seeded_server_key,
            );

        let mut m0 = false;
        let mut m1 = true;

        let mut ct0 = bool_evaluator.sk_encrypt(m0, &client_key);
        let mut ct1 = bool_evaluator.sk_encrypt(m1, &client_key);

        for _ in 0..1000 {
            let ct_back = bool_evaluator.xor(&ct0, &ct1, &server_key_eval_domain);
            let m_out = (m0 ^ m1);

            let m_back = bool_evaluator.sk_decrypt(&ct_back, &client_key);
            assert!(m_out == m_back, "Expected {m_out}, got {m_back}");

            m1 = m0;
            m0 = m_out;

            ct1 = ct0;
            ct0 = ct_back;
        }
    }

    #[test]
    fn multi_party_encryption_decryption() {
        let bool_evaluator = BoolEvaluator::<
            Vec<Vec<u64>>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
            ModularOpsU64<CiphertextModulus<u64>>,
        >::new(MP_BOOL_PARAMS);

        let no_of_parties = 500;
        let parties = (0..no_of_parties)
            .map(|_| bool_evaluator.client_key())
            .collect_vec();

        let mut ideal_rlwe_sk = vec![0i32; bool_evaluator.pbs_info.rlwe_n()];
        parties.iter().for_each(|k| {
            izip!(ideal_rlwe_sk.iter_mut(), k.sk_rlwe.values()).for_each(|(ideal_i, s_i)| {
                *ideal_i = *ideal_i + s_i;
            });
        });

        println!("{:?}", &ideal_rlwe_sk);

        let mut m = true;
        for i in 0..100 {
            let pk_cr_seed = [0u8; 32];

            let public_key_share = parties
                .iter()
                .map(|k| bool_evaluator.multi_party_public_key_share(pk_cr_seed, k))
                .collect_vec();

            let collective_pk = PublicKey::<
                Vec<Vec<u64>>,
                DefaultSecureRng,
                ModularOpsU64<CiphertextModulus<u64>>,
            >::from(public_key_share.as_slice());
            let lwe_ct = bool_evaluator.pk_encrypt(&collective_pk.key, m);

            let decryption_shares = parties
                .iter()
                .map(|k| bool_evaluator.multi_party_decryption_share(&lwe_ct, k))
                .collect_vec();

            let m_back = bool_evaluator.multi_party_decrypt(&decryption_shares, &lwe_ct);

            {
                let ideal_m = if m {
                    bool_evaluator.pbs_info.parameters.rlwe_q().true_el()
                } else {
                    bool_evaluator.pbs_info.parameters.rlwe_q().false_el()
                };
                let noise = measure_noise_lwe(
                    &lwe_ct,
                    &ideal_rlwe_sk,
                    &bool_evaluator.pbs_info.rlwe_modop,
                    &ideal_m,
                );
                println!("Noise: {noise}");
            }

            assert_eq!(m_back, m);
            m = !m;
        }
    }

    fn _collecitve_public_key_gen(rlwe_q: u64, parties_rlwe_sk: &[RlweSecret]) -> Vec<Vec<u64>> {
        let ring_size = parties_rlwe_sk[0].values.len();
        assert!(ring_size.is_power_of_two());
        let mut rng = DefaultSecureRng::new();
        let nttop = NttBackendU64::new(&rlwe_q, ring_size);
        let modop = ModularOpsU64::new(rlwe_q);

        // Generate Pk shares
        let pk_seed = [0u8; 32];
        let pk_shares = parties_rlwe_sk.iter().map(|sk| {
            let mut p_rng = DefaultSecureRng::new_seeded(pk_seed);
            let mut share_out = vec![0u64; ring_size];
            public_key_share(
                &mut share_out,
                sk.values(),
                &modop,
                &nttop,
                &mut p_rng,
                &mut rng,
            );
            share_out
        });

        let mut pk_part_b = vec![0u64; ring_size];
        pk_shares.for_each(|share| modop.elwise_add_mut(&mut pk_part_b, &share));
        let mut pk_part_a = vec![0u64; ring_size];
        let mut p_rng = DefaultSecureRng::new_seeded(pk_seed);
        RandomFillUniformInModulus::random_fill(&mut p_rng, &rlwe_q, pk_part_a.as_mut_slice());

        vec![pk_part_a, pk_part_b]
    }

    fn _multi_party_all_keygen(
        bool_evaluator: &BoolEvaluator<
            Vec<Vec<u64>>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
            ModularOpsU64<CiphertextModulus<u64>>,
        >,
        no_of_parties: usize,
    ) -> (
        Vec<ClientKey>,
        PublicKey<Vec<Vec<u64>>, DefaultSecureRng, ModularOpsU64<CiphertextModulus<u64>>>,
        Vec<
            CommonReferenceSeededMultiPartyServerKeyShare<
                Vec<Vec<u64>>,
                BoolParameters<u64>,
                [u8; 32],
            >,
        >,
        SeededMultiPartyServerKey<Vec<Vec<u64>>, [u8; 32], BoolParameters<u64>>,
        ServerKeyEvaluationDomain<Vec<Vec<u64>>, DefaultSecureRng, NttBackendU64>,
        ClientKey,
    ) {
        let parties = (0..no_of_parties)
            .map(|_| bool_evaluator.client_key())
            .collect_vec();

        let mut rng = DefaultSecureRng::new();

        // Collective public key
        let mut pk_cr_seed = [0u8; 32];
        rng.fill_bytes(&mut pk_cr_seed);
        let public_key_share = parties
            .iter()
            .map(|k| bool_evaluator.multi_party_public_key_share(pk_cr_seed, k))
            .collect_vec();
        let collective_pk =
            PublicKey::<Vec<Vec<u64>>, DefaultSecureRng, _>::from(public_key_share.as_slice());

        // Server key
        let mut pbs_cr_seed = [0u8; 32];
        rng.fill_bytes(&mut pbs_cr_seed);
        let server_key_shares = parties
            .iter()
            .map(|k| {
                bool_evaluator.multi_party_server_key_share(pbs_cr_seed, &collective_pk.key, k)
            })
            .collect_vec();
        let seeded_server_key =
            bool_evaluator.aggregate_multi_party_server_key_shares(&server_key_shares);
        let server_key_eval = ServerKeyEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(
            &seeded_server_key,
        );

        // construct ideal rlwe sk for meauring noise
        let ideal_client_key = {
            let mut ideal_rlwe_sk = vec![0i32; bool_evaluator.pbs_info.rlwe_n()];
            parties.iter().for_each(|k| {
                izip!(ideal_rlwe_sk.iter_mut(), k.sk_rlwe.values()).for_each(|(ideal_i, s_i)| {
                    *ideal_i = *ideal_i + s_i;
                });
            });
            let mut ideal_lwe_sk = vec![0i32; bool_evaluator.pbs_info.lwe_n()];
            parties.iter().for_each(|k| {
                izip!(ideal_lwe_sk.iter_mut(), k.sk_lwe.values()).for_each(|(ideal_i, s_i)| {
                    *ideal_i = *ideal_i + s_i;
                });
            });

            ClientKey {
                sk_lwe: LweSecret {
                    values: ideal_lwe_sk,
                },
                sk_rlwe: RlweSecret {
                    values: ideal_rlwe_sk,
                },
            }
        };

        (
            parties,
            collective_pk,
            server_key_shares,
            seeded_server_key,
            server_key_eval,
            ideal_client_key,
        )
    }

    #[test]
    fn multi_party_nand() {
        let mut bool_evaluator = BoolEvaluator::<
            Vec<Vec<u64>>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
            ModularOpsU64<CiphertextModulus<u64>>,
        >::new(MP_BOOL_PARAMS);

        let (parties, collective_pk, _, _, server_key_eval, ideal_client_key) =
            _multi_party_all_keygen(&bool_evaluator, 64);

        let mut m0 = true;
        let mut m1 = false;

        let mut lwe0 = bool_evaluator.pk_encrypt(&collective_pk.key, m0);
        let mut lwe1 = bool_evaluator.pk_encrypt(&collective_pk.key, m1);

        for _ in 0..2000 {
            let lwe_out = bool_evaluator.nand(&lwe0, &lwe1, &server_key_eval);

            let m_expected = !(m0 & m1);

            // measure noise
            {
                let noise0 = {
                    let ideal = if m0 {
                        bool_evaluator.pbs_info.rlwe_q().true_el()
                    } else {
                        bool_evaluator.pbs_info.rlwe_q().false_el()
                    };
                    let n = measure_noise_lwe(
                        &lwe0,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                        &ideal,
                    );
                    let v = decrypt_lwe(
                        &lwe0,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                    );
                    (n, v)
                };
                let noise1 = {
                    let ideal = if m1 {
                        bool_evaluator.pbs_info.rlwe_q().true_el()
                    } else {
                        bool_evaluator.pbs_info.rlwe_q().false_el()
                    };
                    let n = measure_noise_lwe(
                        &lwe1,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                        &ideal,
                    );
                    let v = decrypt_lwe(
                        &lwe1,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                    );
                    (n, v)
                };

                // // Trace PBS
                // PBSTracer::with_local(|t| {
                //     t.trace(
                //         &MP_BOOL_PARAMS,
                //         &ideal_client_key.sk_lwe.values(),
                //         &ideal_client_key.sk_rlwe.values(),
                //     )
                // });

                let noise_out = {
                    let ideal_m = if m_expected {
                        bool_evaluator.pbs_info.rlwe_q().true_el()
                    } else {
                        bool_evaluator.pbs_info.rlwe_q().false_el()
                    };
                    let n = measure_noise_lwe(
                        &lwe_out,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                        &ideal_m,
                    );
                    let v = decrypt_lwe(
                        &lwe_out,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.pbs_info.rlwe_modop,
                    );
                    (n, v)
                };
                dbg!(m0, m1, m_expected);
                println!(
                    "ct0 (noise, message): {:?}  \n ct1 (noise, message): {:?} \n PBS (noise, message): {:?}", noise0, noise1, noise_out
                );
            }

            // multi-party decrypt
            let decryption_shares = parties
                .iter()
                .map(|k| bool_evaluator.multi_party_decryption_share(&lwe_out, k))
                .collect_vec();
            let m_back = bool_evaluator.multi_party_decrypt(&decryption_shares, &lwe_out);

            // let m_back = bool_evaluator.sk_decrypt(&lwe_out, &ideal_client_key);

            assert!(m_expected == m_back, "Expected {m_expected}, got {m_back}");
            m1 = m0;
            m0 = m_expected;

            lwe1 = lwe0;
            lwe0 = lwe_out;
        }
    }

    #[test]
    fn noise_tester() {
        let bool_evaluator = BoolEvaluator::<
            Vec<Vec<u64>>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
            ModularOpsU64<CiphertextModulus<u64>>,
        >::new(MP_BOOL_PARAMS);

        // let (_, collective_pk, _, _, server_key_eval, ideal_client_key) =
        //     _multi_party_all_keygen(&bool_evaluator, 20);
        let no_of_parties = 32;
        let lwe_q = bool_evaluator.pbs_info.parameters.lwe_q();
        let rlwe_q = bool_evaluator.pbs_info.parameters.rlwe_q();
        let lwe_n = bool_evaluator.pbs_info.parameters.lwe_n().0;
        let rlwe_n = bool_evaluator.pbs_info.parameters.rlwe_n().0;
        let lwe_modop = &bool_evaluator.pbs_info.lwe_modop;
        let rlwe_nttop = &bool_evaluator.pbs_info.rlwe_nttop;
        let rlwe_modop = &bool_evaluator.pbs_info.rlwe_modop;

        // let rgsw_rgsw_decomposer = &bool_evaluator
        //     .pbs_info
        //     .parameters
        //     .rgsw_rgsw_decomposer::<DefaultDecomposer<u64>>();
        // let rgsw_rgsw_gadget_a = rgsw_rgsw_decomposer.0.gadget_vector();
        // let rgsw_rgsw_gadget_b = rgsw_rgsw_decomposer.1.gadget_vector();

        let rlwe_rgsw_decomposer = &bool_evaluator.pbs_info.rlwe_rgsw_decomposer;
        let rlwe_rgsw_gadget_a = rlwe_rgsw_decomposer.0.gadget_vector();
        let rlwe_rgsw_gadget_b = rlwe_rgsw_decomposer.1.gadget_vector();

        let auto_decomposer = &bool_evaluator.pbs_info.auto_decomposer;
        let auto_gadget = auto_decomposer.gadget_vector();

        let parties = (0..no_of_parties)
            .map(|_| bool_evaluator.client_key())
            .collect_vec();

        let ideal_client_key = {
            let mut ideal_rlwe_sk = vec![0i32; bool_evaluator.pbs_info.rlwe_n()];
            parties.iter().for_each(|k| {
                izip!(ideal_rlwe_sk.iter_mut(), k.sk_rlwe.values()).for_each(|(ideal_i, s_i)| {
                    *ideal_i = *ideal_i + s_i;
                });
            });
            let mut ideal_lwe_sk = vec![0i32; bool_evaluator.pbs_info.lwe_n()];
            parties.iter().for_each(|k| {
                izip!(ideal_lwe_sk.iter_mut(), k.sk_lwe.values()).for_each(|(ideal_i, s_i)| {
                    *ideal_i = *ideal_i + s_i;
                });
            });

            ClientKey {
                sk_lwe: LweSecret {
                    values: ideal_lwe_sk,
                },
                sk_rlwe: RlweSecret {
                    values: ideal_rlwe_sk,
                },
            }
        };

        // check noise in freshly encrypted RLWE ciphertext (ie var_fresh)
        if true {
            let mut rng = DefaultSecureRng::new();
            let mut check = Stats { samples: vec![] };
            for _ in 0..10 {
                // generate a new collective public key
                let mut pk_cr_seed = [0u8; 32];
                rng.fill_bytes(&mut pk_cr_seed);
                let public_key_share = parties
                    .iter()
                    .map(|k| bool_evaluator.multi_party_public_key_share(pk_cr_seed, k))
                    .collect_vec();
                let collective_pk = PublicKey::<
                    Vec<Vec<u64>>,
                    DefaultSecureRng,
                    ModularOpsU64<CiphertextModulus<u64>>,
                >::from(public_key_share.as_slice());

                let mut m = vec![0u64; rlwe_n];
                RandomFillUniformInModulus::random_fill(&mut rng, rlwe_q, m.as_mut_slice());
                let mut rlwe_ct = vec![vec![0u64; rlwe_n]; 2];
                public_key_encrypt_rlwe::<_, _, _, _, i32, _>(
                    &mut rlwe_ct,
                    &collective_pk.key,
                    &m,
                    rlwe_modop,
                    rlwe_nttop,
                    &mut rng,
                );

                let mut m_back = vec![0u64; rlwe_n];
                decrypt_rlwe(
                    &rlwe_ct,
                    ideal_client_key.sk_rlwe.values(),
                    &mut m_back,
                    rlwe_nttop,
                    rlwe_modop,
                );

                rlwe_modop.elwise_sub_mut(m_back.as_mut_slice(), m.as_slice());

                check.add_more(Vec::<i64>::try_convert_from(&m_back, rlwe_q).as_slice());
            }

            println!("Public key Std: {}", check.std_dev().abs().log2());
        }

        if true {
            // Generate server key shares
            let mut rng = DefaultSecureRng::new();
            let mut pk_cr_seed = [0u8; 32];
            rng.fill_bytes(&mut pk_cr_seed);
            let public_key_share = parties
                .iter()
                .map(|k| bool_evaluator.multi_party_public_key_share(pk_cr_seed, k))
                .collect_vec();
            let collective_pk = PublicKey::<
                Vec<Vec<u64>>,
                DefaultSecureRng,
                ModularOpsU64<CiphertextModulus<u64>>,
            >::from(public_key_share.as_slice());

            let pbs_cr_seed = [0u8; 32];
            rng.fill_bytes(&mut pk_cr_seed);
            let server_key_shares = parties
                .iter()
                .map(|k| {
                    bool_evaluator.multi_party_server_key_share(pbs_cr_seed, &collective_pk.key, k)
                })
                .collect_vec();

            let seeded_server_key =
                bool_evaluator.aggregate_multi_party_server_key_shares(&server_key_shares);

            // Check noise in RGSW ciphertexts of ideal LWE secret elements
            if true {
                let mut check = Stats { samples: vec![] };
                izip!(
                    ideal_client_key.sk_lwe.values.iter(),
                    seeded_server_key.rgsw_cts.iter()
                )
                .for_each(|(s_i, rgsw_ct_i)| {
                    // X^{s[i]}
                    let mut m_si = vec![0u64; rlwe_n];
                    let s_i = *s_i * (bool_evaluator.pbs_info.embedding_factor as i32);
                    if s_i < 0 {
                        m_si[rlwe_n - (s_i.abs() as usize)] = rlwe_q.neg_one();
                    } else {
                        m_si[s_i as usize] = 1;
                    }

                    // RLWE'(-sm)
                    let mut neg_s_eval =
                        Vec::<u64>::try_convert_from(ideal_client_key.sk_rlwe.values(), rlwe_q);
                    rlwe_modop.elwise_neg_mut(&mut neg_s_eval);
                    rlwe_nttop.forward(&mut neg_s_eval);
                    for j in 0..rlwe_rgsw_decomposer.a().decomposition_count() {
                        // RLWE(B^{j} * -s[X]*X^{s_lwe[i]})

                        // -s[X]*X^{s_lwe[i]}*B_j
                        let mut m_ideal = m_si.clone();
                        rlwe_nttop.forward(m_ideal.as_mut_slice());
                        rlwe_modop.elwise_mul_mut(m_ideal.as_mut_slice(), neg_s_eval.as_slice());
                        rlwe_nttop.backward(m_ideal.as_mut_slice());
                        rlwe_modop
                            .elwise_scalar_mul_mut(m_ideal.as_mut_slice(), &rlwe_rgsw_gadget_a[j]);

                        // RLWE(-s*X^{s_lwe[i]}*B_j)
                        let mut rlwe_ct = vec![vec![0u64; rlwe_n]; 2];
                        rlwe_ct[0].copy_from_slice(&rgsw_ct_i[j]);
                        rlwe_ct[1].copy_from_slice(
                            &rgsw_ct_i[j + rlwe_rgsw_decomposer.a().decomposition_count()],
                        );

                        let mut m_back = vec![0u64; rlwe_n];
                        decrypt_rlwe(
                            &rlwe_ct,
                            ideal_client_key.sk_rlwe.values(),
                            &mut m_back,
                            rlwe_nttop,
                            rlwe_modop,
                        );

                        // diff
                        rlwe_modop.elwise_sub_mut(&mut m_back, &m_ideal);
                        check.add_more(&Vec::<i64>::try_convert_from(&m_back, rlwe_q));
                    }

                    // RLWE'(m)
                    for j in 0..rlwe_rgsw_decomposer.b().decomposition_count() {
                        // RLWE(B^{j} * X^{s_lwe[i]})

                        // X^{s_lwe[i]}*B_j
                        let mut m_ideal = m_si.clone();
                        rlwe_modop
                            .elwise_scalar_mul_mut(m_ideal.as_mut_slice(), &rlwe_rgsw_gadget_b[j]);

                        // RLWE(X^{s_lwe[i]}*B_j)
                        let mut rlwe_ct = vec![vec![0u64; rlwe_n]; 2];
                        rlwe_ct[0].copy_from_slice(
                            &rgsw_ct_i[j + (2 * rlwe_rgsw_decomposer.a().decomposition_count())],
                        );
                        rlwe_ct[1].copy_from_slice(
                            &rgsw_ct_i[j
                                + (2 * rlwe_rgsw_decomposer.a().decomposition_count()
                                    + rlwe_rgsw_decomposer.b().decomposition_count())],
                        );

                        let mut m_back = vec![0u64; rlwe_n];
                        decrypt_rlwe(
                            &rlwe_ct,
                            ideal_client_key.sk_rlwe.values(),
                            &mut m_back,
                            rlwe_nttop,
                            rlwe_modop,
                        );

                        // diff
                        rlwe_modop.elwise_sub_mut(&mut m_back, &m_ideal);
                        check.add_more(&Vec::<i64>::try_convert_from(&m_back, rlwe_q));
                    }
                });
                println!(
                    "RGSW Std: {} {} ;; max={}",
                    check.mean(),
                    check.std_dev().abs().log2(),
                    check.samples.iter().max().unwrap()
                );
            }

            // server key in Evaluation domain
            let server_key_eval_domain =
                ServerKeyEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(
                    &seeded_server_key,
                );

            // check noise in RLWE x RGSW(X^{s_i}) where RGSW is accunulated RGSW ciphertext
            if true {
                let mut check = Stats { samples: vec![] };

                izip!(
                    ideal_client_key.sk_lwe.values(),
                    server_key_eval_domain.rgsw_cts.iter()
                )
                .for_each(|(s_i, rgsw_ct_i)| {
                    let mut m = vec![0u64; rlwe_n];
                    RandomFillUniformInModulus::random_fill(&mut rng, rlwe_q, m.as_mut_slice());
                    let mut rlwe_ct = vec![vec![0u64; rlwe_n]; 2];
                    public_key_encrypt_rlwe::<_, _, _, _, i32, _>(
                        &mut rlwe_ct,
                        &collective_pk.key,
                        &m,
                        rlwe_modop,
                        rlwe_nttop,
                        &mut rng,
                    );

                    // RLWE(m*X^{s[i]}) = RLWE(m) x RGSW(X^{s[i]})
                    // let mut rlwe_after = RlweCiphertext::<_, DefaultSecureRng>::new_trivial(vec![
                    //     vec![0u64; rlwe_n],
                    //     m.clone(),
                    // ]);
                    let mut rlwe_after = RlweCiphertext::<_, DefaultSecureRng> {
                        data: rlwe_ct.clone(),
                        is_trivial: false,
                        _phatom: PhantomData,
                    };
                    let mut scratch = vec![
                        vec![0u64; rlwe_n];
                        std::cmp::max(
                            rlwe_rgsw_decomposer.0.decomposition_count(),
                            rlwe_rgsw_decomposer.1.decomposition_count()
                        ) + 2
                    ];
                    rlwe_by_rgsw(
                        &mut rlwe_after,
                        rgsw_ct_i,
                        &mut scratch,
                        rlwe_rgsw_decomposer,
                        rlwe_nttop,
                        rlwe_modop,
                    );

                    // m1 = X^{s[i]}
                    let mut m1 = vec![0u64; rlwe_n];
                    let s_i = *s_i * (bool_evaluator.pbs_info.embedding_factor as i32);
                    if s_i < 0 {
                        m1[rlwe_n - (s_i.abs() as usize)] = rlwe_q.neg_one()
                    } else {
                        m1[s_i as usize] = 1;
                    }

                    // (m+e) * m1
                    let mut m_plus_e_times_m1 = vec![0u64; rlwe_n];
                    decrypt_rlwe(
                        &rlwe_ct,
                        ideal_client_key.sk_rlwe.values(),
                        &mut m_plus_e_times_m1,
                        rlwe_nttop,
                        rlwe_modop,
                    );
                    rlwe_nttop.forward(m_plus_e_times_m1.as_mut_slice());
                    rlwe_nttop.forward(m1.as_mut_slice());
                    rlwe_modop.elwise_mul_mut(m_plus_e_times_m1.as_mut_slice(), m1.as_slice());
                    rlwe_nttop.backward(m_plus_e_times_m1.as_mut_slice());

                    // Resulting RLWE ciphertext will equal: (m0m1 + em1) + e_{rlsw x rgsw}.
                    // Hence, resulting rlwe ciphertext will have error em1 + e_{rlwe x rgsw}.
                    // Here we're only concerned with e_{rlwe x rgsw}, that is noise added by
                    // RLWExRGSW. Also note in practice m1 is a monomial, for ex, X^{s_{i}}, for
                    // some i and var(em1) = var(e).
                    let mut m_plus_e_times_m1_more_e = vec![0u64; rlwe_n];
                    decrypt_rlwe(
                        &rlwe_after,
                        ideal_client_key.sk_rlwe.values(),
                        &mut m_plus_e_times_m1_more_e,
                        rlwe_nttop,
                        rlwe_modop,
                    );

                    // diff
                    rlwe_modop.elwise_sub_mut(
                        m_plus_e_times_m1_more_e.as_mut_slice(),
                        m_plus_e_times_m1.as_slice(),
                    );

                    // let noise = measure_noise(
                    //     &rlwe_after,
                    //     &m_plus_e_times_m1,
                    //     rlwe_nttop,
                    //     rlwe_modop,
                    //     ideal_client_key.sk_rlwe.values(),
                    // );
                    // print!("NOISE: {}", noise);

                    check.add_more(&Vec::<i64>::try_convert_from(
                        &m_plus_e_times_m1_more_e,
                        rlwe_q,
                    ));
                });
                println!(
                    "RLWE x RGSW, where RGSW has noise var_brk, std: {} {}",
                    check.std_dev(),
                    check.std_dev().abs().log2()
                )
            }

            // check noise in Auto key
            if true {
                let mut check = Stats { samples: vec![] };

                let mut neg_s_poly =
                    Vec::<u64>::try_convert_from(ideal_client_key.sk_rlwe.values(), rlwe_q);
                rlwe_modop.elwise_neg_mut(neg_s_poly.as_mut_slice());

                let g = bool_evaluator.pbs_info.g();
                let br_q = bool_evaluator.pbs_info.br_q();
                let auto_element_dlogs = bool_evaluator.pbs_info.parameters.auto_element_dlogs();
                for i in auto_element_dlogs.into_iter() {
                    let g_pow = if i == 0 {
                        -g
                    } else {
                        (((g as usize).pow(i as u32)) % br_q) as isize
                    };

                    // -s[X^k]
                    let (auto_indices, auto_sign) = generate_auto_map(rlwe_n, g_pow);
                    let mut neg_s_poly_auto_i = vec![0u64; rlwe_n];
                    izip!(neg_s_poly.iter(), auto_indices.iter(), auto_sign.iter()).for_each(
                        |(v, to_i, to_sign)| {
                            if !to_sign {
                                neg_s_poly_auto_i[*to_i] = rlwe_modop.neg(v);
                            } else {
                                neg_s_poly_auto_i[*to_i] = *v;
                            }
                        },
                    );

                    let mut auto_key_i = server_key_eval_domain.galois_key_for_auto(i).clone();
                    // send i^th auto key to coefficient domain
                    auto_key_i
                        .iter_mut()
                        .for_each(|r| rlwe_nttop.backward(r.as_mut_slice()));
                    auto_gadget.iter().enumerate().for_each(|(i, b_i)| {
                        // B^i * -s[X^k]
                        let mut m_ideal = neg_s_poly_auto_i.clone();
                        rlwe_modop.elwise_scalar_mul_mut(m_ideal.as_mut_slice(), b_i);

                        let mut m_out = vec![0u64; rlwe_n];
                        let mut rlwe_ct = vec![vec![0u64; rlwe_n]; 2];
                        rlwe_ct[0].copy_from_slice(&auto_key_i[i]);
                        rlwe_ct[1].copy_from_slice(
                            &auto_key_i[auto_decomposer.decomposition_count() + i],
                        );
                        decrypt_rlwe(
                            &rlwe_ct,
                            ideal_client_key.sk_rlwe.values(),
                            &mut m_out,
                            rlwe_nttop,
                            rlwe_modop,
                        );

                        // diff
                        rlwe_modop.elwise_sub_mut(m_out.as_mut_slice(), m_ideal.as_slice());

                        check.add_more(&Vec::<i64>::try_convert_from(&m_out, rlwe_q));
                    });
                }

                println!("Auto key noise std dev: {}", check.std_dev().abs().log2());
            }

            // check noise in RLWE(X^k) after sending RLWE(X) -> RLWE(X^k) using collective
            // auto key
            if true {
                let mut check = Stats { samples: vec![] };

                let g = bool_evaluator.pbs_info.g();
                let auto_element_dlogs = bool_evaluator.pbs_info.parameters.auto_element_dlogs();
                for i in auto_element_dlogs.into_iter() {
                    for _ in 0..10 {
                        let mut m = vec![0u64; rlwe_n];
                        RandomFillUniformInModulus::random_fill(&mut rng, rlwe_q, m.as_mut_slice());
                        let mut rlwe_ct = RlweCiphertext::<_, DefaultSecureRng> {
                            data: vec![vec![0u64; rlwe_n]; 2],
                            is_trivial: false,
                            _phatom: PhantomData,
                        };
                        public_key_encrypt_rlwe::<_, _, _, _, i32, _>(
                            &mut rlwe_ct,
                            &collective_pk.key,
                            &m,
                            rlwe_modop,
                            rlwe_nttop,
                            &mut rng,
                        );

                        // We're only interested in noise increased as a result of automorphism.
                        // Hence, we take m+e as the bench.
                        let mut m_plus_e = vec![0u64; rlwe_n];
                        decrypt_rlwe(
                            &rlwe_ct,
                            ideal_client_key.sk_rlwe.values(),
                            &mut m_plus_e,
                            rlwe_nttop,
                            rlwe_modop,
                        );

                        let auto_key = server_key_eval_domain.galois_key_for_auto(i);
                        let g_pow = if i == 0 { -g } else { g.pow(i as u32) };
                        let (auto_map_index, auto_map_sign) = generate_auto_map(rlwe_n, g_pow);
                        let mut scratch =
                            vec![vec![0u64; rlwe_n]; auto_decomposer.decomposition_count() + 2];
                        galois_auto(
                            &mut rlwe_ct,
                            auto_key,
                            &mut scratch,
                            &auto_map_index,
                            &auto_map_sign,
                            rlwe_modop,
                            rlwe_nttop,
                            auto_decomposer,
                        );

                        // send m+e from X to X^k
                        let mut m_plus_e_auto = vec![0u64; rlwe_n];
                        izip!(m_plus_e.iter(), auto_map_index.iter(), auto_map_sign.iter())
                            .for_each(|(v, to_index, to_sign)| {
                                if !to_sign {
                                    m_plus_e_auto[*to_index] = rlwe_modop.neg(v);
                                } else {
                                    m_plus_e_auto[*to_index] = *v
                                }
                            });

                        let mut m_out = vec![0u64; rlwe_n];
                        decrypt_rlwe(
                            &rlwe_ct,
                            ideal_client_key.sk_rlwe.values(),
                            &mut m_out,
                            rlwe_nttop,
                            rlwe_modop,
                        );

                        // diff
                        rlwe_modop.elwise_sub_mut(m_out.as_mut_slice(), m_plus_e_auto.as_slice());

                        check.add_more(&Vec::<i64>::try_convert_from(m_out.as_slice(), rlwe_q));
                    }
                }

                println!("Rlwe Auto Noise Std: {}", check.std_dev().abs().log2());
            }

            // Check noise growth in ksk
            // TODO check in LWE key switching keys
            if true {
                // 1. encrypt LWE ciphertext
                // 2. Key switching
                // 3.
                let mut check = Stats { samples: vec![] };

                for _ in 0..1024 {
                    // Encrypt m \in Q_{ks} using RLWE sk
                    let mut lwe_in_ct = vec![0u64; rlwe_n + 1];
                    let m = RandomElementInModulus::random(&mut rng, &lwe_q.q().unwrap());
                    encrypt_lwe(
                        &mut lwe_in_ct,
                        &m,
                        ideal_client_key.sk_rlwe.values(),
                        lwe_modop,
                        &mut rng,
                    );

                    // Key switch
                    let mut lwe_out = vec![0u64; lwe_n + 1];
                    lwe_key_switch(
                        &mut lwe_out,
                        &lwe_in_ct,
                        server_key_eval_domain.lwe_ksk(),
                        lwe_modop,
                        bool_evaluator.pbs_info.lwe_decomposer(),
                    );

                    // We only care about noise added by LWE key switch
                    // m+e
                    let m_plus_e =
                        decrypt_lwe(&lwe_in_ct, ideal_client_key.sk_rlwe.values(), lwe_modop);

                    let m_plus_e_plus_lwe_ksk_noise =
                        decrypt_lwe(&lwe_out, ideal_client_key.sk_lwe.values(), lwe_modop);

                    let diff = lwe_modop.sub(&m_plus_e_plus_lwe_ksk_noise, &m_plus_e);

                    check.add_more(&vec![lwe_q.map_element_to_i64(&diff)]);
                }

                println!("Lwe ksk std dev: {}", check.std_dev().abs().log2());
            }
        }

        // Check noise in fresh RGSW ciphertexts, ie X^{s_j[i]}, must equal noise in
        // fresh RLWE ciphertext
        if true {}
        // test LWE ksk from RLWE -> LWE
        // if false {
        //     let logp = 2;
        //     let mut rng = DefaultSecureRng::new();

        //     let m = 1;
        //     let encoded_m = m << (lwe_logq - logp);

        //     // Encrypt
        //     let mut lwe_ct = vec![0u64; rlwe_n + 1];
        //     encrypt_lwe(
        //         &mut lwe_ct,
        //         &encoded_m,
        //         ideal_client_key.sk_rlwe.values(),
        //         lwe_modop,
        //         &mut rng,
        //     );

        //     // key switch
        //     let lwe_decomposer = &bool_evaluator.decomposer_lwe;
        //     let mut lwe_out = vec![0u64; lwe_n + 1];
        //     lwe_key_switch(
        //         &mut lwe_out,
        //         &lwe_ct,
        //         &server_key_eval.lwe_ksk,
        //         lwe_modop,
        //         lwe_decomposer,
        //     );

        //     let encoded_m_back = decrypt_lwe(&lwe_out,
        // ideal_client_key.sk_lwe.values(), lwe_modop);     let m_back
        // =         ((encoded_m_back as f64 * (1 << logp) as f64) /
        // (lwe_q as f64)).round() as u64;     dbg!(m_back, m);

        //     let noise = measure_noise_lwe(
        //         &lwe_out,
        //         ideal_client_key.sk_lwe.values(),
        //         lwe_modop,
        //         &encoded_m,
        //     );

        //     println!("Noise: {noise}");
        // }

        // Measure noise in RGSW ciphertexts of ideal LWE secrets
        // if true {
        //     let gadget_vec = gadget_vector(
        //         bool_evaluator.parameters.rlwe_logq,
        //         bool_evaluator.parameters.logb_rgsw,
        //         bool_evaluator.parameters.d_rgsw,
        //     );

        //     for i in 0..20 {
        //         // measure noise in RGSW(s[i])
        //         let si =
        //             ideal_client_key.sk_lwe.values[i] *
        // (bool_evaluator.embedding_factor as i32);         let mut
        // si_poly = vec![0u64; rlwe_n];         if si < 0 {
        //             si_poly[rlwe_n - (si.abs() as usize)] = rlwe_q - 1;
        //         } else {
        //             si_poly[(si.abs() as usize)] = 1;
        //         }

        //         let mut rgsw_si = server_key_eval.rgsw_cts[i].clone();
        //         rgsw_si
        //             .iter_mut()
        //             .for_each(|ri| rlwe_nttop.backward(ri.as_mut()));

        //         println!("####### Noise in RGSW(X^s_{i}) #######");
        //         _measure_noise_rgsw(
        //             &rgsw_si,
        //             &si_poly,
        //             ideal_client_key.sk_rlwe.values(),
        //             &gadget_vec,
        //             rlwe_q,
        //         );
        //         println!("####### ##################### #######");
        //     }
        // }

        // // measure noise grwoth in RLWExRGSW
        // if true {
        //     let mut rng = DefaultSecureRng::new();
        //     let mut carry_m = vec![0u64; rlwe_n];
        //     RandomUniformDist1::random_fill(&mut rng, &rlwe_q,
        // carry_m.as_mut_slice());

        //     // RGSW(carrym)
        //     let trivial_rlwect = vec![vec![0u64; rlwe_n], carry_m.clone()];
        //     let mut rlwe_ct = RlweCiphertext::<_,
        // DefaultSecureRng>::from_raw(trivial_rlwect, true);

        //     let mut scratch_matrix_dplus2_ring = vec![vec![0u64; rlwe_n];
        // d_rgsw + 2];     let mul_mod =
        //         |v0: &u64, v1: &u64| (((*v0 as u128 * *v1 as u128) % (rlwe_q as u128)) as u64);

        //     for i in 0..bool_evaluator.parameters.lwe_n {
        //         rlwe_by_rgsw(
        //             &mut rlwe_ct,
        //             server_key_eval.rgsw_ct_lwe_si(i),
        //             &mut scratch_matrix_dplus2_ring,
        //             rlwe_decomposer,
        //             rlwe_nttop,
        //             rlwe_modop,
        //         );

        //         // carry_m[X] * s_i[X]
        //         let si =
        //             ideal_client_key.sk_lwe.values[i] *
        // (bool_evaluator.embedding_factor as i32);         let mut
        // si_poly = vec![0u64; rlwe_n];         if si < 0 {
        //             si_poly[rlwe_n - (si.abs() as usize)] = rlwe_q - 1;
        //         } else {
        //             si_poly[(si.abs() as usize)] = 1;
        //         }
        //         carry_m = negacyclic_mul(&carry_m, &si_poly, mul_mod,
        // rlwe_q);

        //         let noise = measure_noise(
        //             &rlwe_ct,
        //             &carry_m,
        //             rlwe_nttop,
        //             rlwe_modop,
        //             ideal_client_key.sk_rlwe.values(),
        //         );
        //         println!("Noise RLWE(carry_m) accumulating {i}^th secret
        // monomial: {noise}");     }
        // }

        // // Check galois keys
        // if false {
        //     let g = bool_evaluator.g() as isize;
        //     let mut rng = DefaultSecureRng::new();
        //     let mut scratch_matrix_dplus2_ring = vec![vec![0u64; rlwe_n];
        // d_rgsw + 2];     for i in [g, -g] {
        //         let mut m = vec![0u64; rlwe_n];
        //         RandomUniformDist1::random_fill(&mut rng, &rlwe_q,
        // m.as_mut_slice());         let mut rlwe_ct = {
        //             let mut data = vec![vec![0u64; rlwe_n]; 2];
        //             public_key_encrypt_rlwe(
        //                 &mut data,
        //                 &collective_pk.key,
        //                 &m,
        //                 rlwe_modop,
        //                 rlwe_nttop,
        //                 &mut rng,
        //             );
        //             RlweCiphertext::<_, DefaultSecureRng>::from_raw(data,
        // false)         };

        //         let auto_key = server_key_eval.galois_key_for_auto(i);
        //         let (auto_map_index, auto_map_sign) =
        // generate_auto_map(rlwe_n, i);         galois_auto(
        //             &mut rlwe_ct,
        //             auto_key,
        //             &mut scratch_matrix_dplus2_ring,
        //             &auto_map_index,
        //             &auto_map_sign,
        //             rlwe_modop,
        //             rlwe_nttop,
        //             rlwe_decomposer,
        //         );

        //         // send m(X) -> m(X^i)
        //         let mut m_k = vec![0u64; rlwe_n];
        //         izip!(m.iter(), auto_map_index.iter(),
        // auto_map_sign.iter()).for_each(             |(mi, to_index, to_sign)|
        // {                 if !to_sign {
        //                     m_k[*to_index] = rlwe_q - *mi;
        //                 } else {
        //                     m_k[*to_index] = *mi;
        //                 }
        //             },
        //         );

        //         // measure noise
        //         let noise = measure_noise(
        //             &rlwe_ct,
        //             &m_k,
        //             rlwe_nttop,
        //             rlwe_modop,
        //             ideal_client_key.sk_rlwe.values(),
        //         );

        //         println!("Noise after auto k={i}: {noise}");
        //     }
        // }
    }
}
