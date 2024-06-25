use std::{
    borrow::BorrowMut,
    cell::{OnceCell, RefCell},
    clone,
    collections::HashMap,
    fmt::{Debug, Display},
    iter::Once,
    marker::PhantomData,
    ops::Shr,
    sync::OnceLock,
    usize,
};

use itertools::{izip, partition, Itertools};
use num_traits::{
    FromPrimitive, Num, One, Pow, PrimInt, ToPrimitive, WrappingAdd, WrappingSub, Zero,
};
use rand::Rng;
use rand_distr::uniform::SampleUniform;

use crate::{
    backend::{
        ArithmeticOps, GetModulus, ModInit, ModularOpsU64, Modulus, ShoupMatrixFMA, VectorOps,
    },
    bool::parameters::ParameterVariant,
    decomposer::{Decomposer, DefaultDecomposer, NumInfo, RlweDecomposer},
    lwe::{decrypt_lwe, encrypt_lwe, lwe_key_switch, lwe_ksk_keygen, measure_noise_lwe, LweSecret},
    multi_party::{
        non_interactive_ksk_gen, non_interactive_ksk_zero_encryptions_for_other_party_i,
        non_interactive_rgsw_ct, public_key_share,
    },
    ntt::{self, Ntt, NttBackendU64, NttInit},
    pbs::{pbs, sample_extract, PbsInfo, PbsKey, WithShoupRepr},
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
        encode_x_pow_si_with_emebedding_factor, fill_random_ternary_secret_with_hamming_weight,
        generate_prime, mod_exponent, puncture_p_rng, Global, TryConvertFrom1, WithLocal,
    },
    Decryptor, Encoder, Encryptor, Matrix, MatrixEntity, MatrixMut, MultiPartyDecryptor, Row,
    RowEntity, RowMut, Secret,
};

use super::{
    keys::{
        ClientKey, CommonReferenceSeededCollectivePublicKeyShare,
        CommonReferenceSeededMultiPartyServerKeyShare,
        CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare,
        InteractiveMultiPartyClientKey, NonInteractiveMultiPartyClientKey,
        SeededMultiPartyServerKey, SeededNonInteractiveMultiPartyServerKey,
        SeededSinglePartyServerKey, ServerKeyEvaluationDomain, ShoupServerKeyEvaluationDomain,
        SinglePartyClientKey,
    },
    parameters::{
        BoolParameters, CiphertextModulus, DecompositionCount, DecompostionLogBase,
        DoubleDecomposerParams,
    },
};

/// Common reference seed used for Interactive multi-party,
///
/// Seeds for public key shares and differents parts of server key shares are
/// derived from common reference seed with different puncture rountines.
///
/// ## Punctures
///
///     Initial Seed:
///         Puncture 1 -> Public key share seed
///         Puncture 2 -> Main server key share seed
///             Puncture 1 -> RGSW cuphertexts seed
///             Puncture 2 -> Auto keys cipertexts seed
///             Puncture 3 -> LWE ksk seed
#[derive(Clone, PartialEq)]
pub struct MultiPartyCrs<S> {
    pub(super) seed: S,
}

impl MultiPartyCrs<[u8; 32]> {
    pub(super) fn random() -> Self {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);
            Self { seed }
        })
    }
}
impl<S: Default + Copy> MultiPartyCrs<S> {
    /// Seed to generate public key share
    fn public_key_share_seed<Rng: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut prng = Rng::new_with_seed(self.seed);
        puncture_p_rng(&mut prng, 1)
    }

    /// Main server key share seed
    fn key_seed<Rng: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut prng = Rng::new_with_seed(self.seed);
        puncture_p_rng(&mut prng, 2)
    }

    pub(super) fn rgsw_cts_seed<Rng: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut key_prng = Rng::new_with_seed(self.key_seed::<Rng>());
        puncture_p_rng(&mut key_prng, 1)
    }

    pub(super) fn auto_keys_cts_seed<Rng: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut key_prng = Rng::new_with_seed(self.key_seed::<Rng>());
        puncture_p_rng(&mut key_prng, 2)
    }

    pub(super) fn lwe_ksk_cts_seed_seed<Rng: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut key_prng = Rng::new_with_seed(self.key_seed::<Rng>());
        puncture_p_rng(&mut key_prng, 3)
    }
}

/// Common reference seed used for non-interactive multi-party.
///
/// Initial Seed
///     Puncture 1 -> Key Seed
///         Puncture 1 -> Rgsw ciphertext seed
///         Puncture 2 -> auto keys seed
///         Puncture 3 -> Lwe key switching key seed
///     Puncture 2 -> user specific seed for u_j to s ksk
///         Punture j+1 -> user j's seed    
#[derive(Clone)]
pub struct NonInteractiveMultiPartyCrs<S> {
    pub(super) seed: S,
}

impl NonInteractiveMultiPartyCrs<[u8; 32]> {
    pub(super) fn random() -> Self {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);
            Self { seed }
        })
    }
}

impl<S: Default + Copy> NonInteractiveMultiPartyCrs<S> {
    fn key_seed<R: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut p_rng = R::new_with_seed(self.seed);
        puncture_p_rng(&mut p_rng, 1)
    }

    pub(crate) fn rgsw_cts_seed<R: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut p_rng = R::new_with_seed(self.key_seed::<R>());
        puncture_p_rng(&mut p_rng, 1)
    }

    pub(crate) fn auto_keys_cts_seed<R: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut p_rng = R::new_with_seed(self.key_seed::<R>());
        puncture_p_rng(&mut p_rng, 2)
    }

    pub(crate) fn lwe_ksk_cts_seed<R: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut p_rng = R::new_with_seed(self.key_seed::<R>());
        puncture_p_rng(&mut p_rng, 3)
    }

    fn ui_to_s_ks_seed<R: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut p_rng = R::new_with_seed(self.seed);
        puncture_p_rng(&mut p_rng, 2)
    }

    pub(crate) fn ui_to_s_ks_seed_for_user_i<R: NewWithSeed<Seed = S> + RandomFill<S>>(
        &self,
        user_i: usize,
    ) -> S {
        let ks_seed = self.ui_to_s_ks_seed::<R>();
        let mut p_rng = R::new_with_seed(ks_seed);

        puncture_p_rng(&mut p_rng, user_i + 1)
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

pub(super) trait BoolEncoding {
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

impl<B> Encoder<bool, B::Element> for B
where
    B: BoolEncoding,
{
    fn encode(&self, v: bool) -> B::Element {
        if v {
            self.true_el()
        } else {
            self.false_el()
        }
    }
}

pub(super) struct BoolPbsInfo<M: Matrix, Ntt, RlweModOp, LweModOp> {
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
    M::MatElement: PrimInt
        + WrappingSub
        + NumInfo
        + FromPrimitive
        + From<bool>
        + Display
        + WrappingAdd
        + Debug,
    RlweModOp: ArithmeticOps<Element = M::MatElement> + ShoupMatrixFMA<M::R>,
    LweModOp: ArithmeticOps<Element = M::MatElement> + VectorOps<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement>,
{
    type M = M;
    type Modulus = CiphertextModulus<M::MatElement>;
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

pub(crate) struct BoolEvaluator<M, Ntt, RlweModOp, LweModOp, SKey>
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
    /// Non-interactive u_i -> s key switch decomposer
    ni_ui_to_s_ks_decomposer: Option<DefaultDecomposer<M::MatElement>>,
    _phantom: PhantomData<SKey>,
}

impl<M: Matrix, NttOp, RlweModOp, LweModOp, Skey>
    BoolEvaluator<M, NttOp, RlweModOp, LweModOp, Skey>
{
    pub(crate) fn parameters(&self) -> &BoolParameters<M::MatElement> {
        &self.pbs_info.parameters
    }

    pub(super) fn pbs_info(&self) -> &BoolPbsInfo<M, NttOp, RlweModOp, LweModOp> {
        &self.pbs_info
    }

    pub(super) fn ni_ui_to_s_ks_decomposer(&self) -> &Option<DefaultDecomposer<M::MatElement>> {
        &self.ni_ui_to_s_ks_decomposer
    }
}

fn trim_rgsw_ct_matrix_from_rgrg_to_rlrg<
    M: MatrixMut + MatrixEntity,
    D: DoubleDecomposerParams<Count = DecompositionCount>,
>(
    rgsw_ct_in: M,
    rgrg_params: D,
    rlrg_params: D,
) -> M
where
    M::R: RowMut,
    M::MatElement: Copy,
{
    let (rgswrgsw_d_a, rgswrgsw_d_b) = (
        rgrg_params.decomposition_count_a(),
        rgrg_params.decomposition_count_b(),
    );
    let (rlrg_d_a, rlrg_d_b) = (
        rlrg_params.decomposition_count_a(),
        rlrg_params.decomposition_count_b(),
    );
    let rgsw_ct_rows_in = rgswrgsw_d_a.0 * 2 + rgswrgsw_d_b.0 * 2;
    let rgsw_ct_rows_out = rlrg_d_a.0 * 2 + rlrg_d_b.0 * 2;
    assert!(rgsw_ct_in.dimension().0 == rgsw_ct_rows_in);
    assert!(rgswrgsw_d_a.0 >= rlrg_d_a.0, "RGSWxRGSW part A decomposition count {} must be >= RLWExRGSW part A decomposition count {}", rgswrgsw_d_a.0 , rlrg_d_a.0);
    assert!(rgswrgsw_d_b.0 >= rlrg_d_b.0, "RGSWxRGSW part B decomposition count {} must be >= RLWExRGSW part B decomposition count {}", rgswrgsw_d_b.0 , rlrg_d_b.0);

    let mut reduced_ct_i_out = M::zeros(rgsw_ct_rows_out, rgsw_ct_in.dimension().1);

    // RLWE'(-sm) part A
    izip!(
        reduced_ct_i_out.iter_rows_mut().take(rlrg_d_a.0),
        rgsw_ct_in
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
        rgsw_ct_in
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
        rgsw_ct_in
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
        rgsw_ct_in
            .iter_rows()
            .skip(rgswrgsw_d_a.0 * 2 + rgswrgsw_d_b.0 + (rgswrgsw_d_b.0 - rlrg_d_b.0))
            .take(rlrg_d_b.0)
    )
    .for_each(|(to_ri, from_ri)| {
        to_ri.as_mut().copy_from_slice(from_ri.as_ref());
    });

    reduced_ct_i_out
}

/// Assigns user with user_id segement of LWE secret indices for which they
/// generate RGSW(X^{s[i]}) as the leader (i.e. for RLWExRGSW). If returned
/// tuple is (start, end), user's segment is [start, end)
pub(super) fn interactive_mult_party_user_id_lwe_segment(
    user_id: usize,
    total_users: usize,
    lwe_n: usize,
) -> (usize, usize) {
    let per_user = (lwe_n as f64 / total_users as f64)
        .ceil()
        .to_usize()
        .unwrap();
    (
        per_user * user_id,
        std::cmp::min(per_user * (user_id + 1), lwe_n),
    )
}

impl<M: Matrix, NttOp, RlweModOp, LweModOp, SKey> BoolEvaluator<M, NttOp, RlweModOp, LweModOp, SKey>
where
    M: MatrixEntity + MatrixMut,
    M::MatElement: PrimInt
        + Debug
        + Display
        + NumInfo
        + FromPrimitive
        + WrappingSub
        + WrappingAdd
        + SampleUniform
        + From<bool>,
    NttOp: Ntt<Element = M::MatElement>,
    RlweModOp: ArithmeticOps<Element = M::MatElement>
        + VectorOps<Element = M::MatElement>
        + GetModulus<Element = M::MatElement, M = CiphertextModulus<M::MatElement>>
        + ShoupMatrixFMA<M::R>,
    LweModOp: ArithmeticOps<Element = M::MatElement>
        + VectorOps<Element = M::MatElement>
        + GetModulus<Element = M::MatElement, M = CiphertextModulus<M::MatElement>>,
    M::R: TryConvertFrom1<[i32], CiphertextModulus<M::MatElement>> + RowEntity + Debug,
    <M as Matrix>::R: RowMut,
{
    pub(super) fn new(parameters: BoolParameters<M::MatElement>) -> Self
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

        let ni_ui_to_s_ks_decomposer = if parameters.variant()
            == &ParameterVariant::NonInteractiveMultiParty
        {
            Some(parameters
                .non_interactive_ui_to_s_key_switch_decomposer::<DefaultDecomposer<M::MatElement>>())
        } else {
            None
        };

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
            ni_ui_to_s_ks_decomposer,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn client_key(
        &self,
    ) -> ClientKey<<DefaultSecureRng as NewWithSeed>::Seed, M::MatElement> {
        ClientKey::new(self.parameters().clone())
    }

    pub(super) fn single_party_server_key<K: SinglePartyClientKey<Element = i32>>(
        &self,
        client_key: &K,
    ) -> SeededSinglePartyServerKey<M, BoolParameters<M::MatElement>, [u8; 32]> {
        assert_eq!(self.parameters().variant(), &ParameterVariant::SingleParty);

        DefaultSecureRng::with_local_mut(|rng| {
            let mut main_seed = [0u8; 32];
            rng.fill_bytes(&mut main_seed);

            let mut main_prng = DefaultSecureRng::new_seeded(main_seed);

            let rlwe_n = self.pbs_info.parameters.rlwe_n().0;
            let sk_rlwe = client_key.sk_rlwe();
            let sk_lwe = client_key.sk_lwe();

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
                    &sk_rlwe,
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
                        &sk_rlwe,
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
                &sk_rlwe,
                &sk_lwe,
                &mut lwe_ksk,
                &d_lwe_gadget,
                &self.pbs_info.lwe_modop,
                &mut main_prng,
                rng,
            );

            SeededSinglePartyServerKey::from_raw(
                auto_keys,
                rgsw_cts,
                lwe_ksk,
                self.pbs_info.parameters.clone(),
                main_seed,
            )
        })
    }

    pub(super) fn multi_party_server_key_share<K: InteractiveMultiPartyClientKey<Element = i32>>(
        &self,
        user_id: usize,
        total_users: usize,
        cr_seed: &MultiPartyCrs<[u8; 32]>,
        collective_pk: &M,
        client_key: &K,
    ) -> CommonReferenceSeededMultiPartyServerKeyShare<
        M,
        BoolParameters<M::MatElement>,
        MultiPartyCrs<[u8; 32]>,
    > {
        assert_eq!(self.parameters().variant(), &ParameterVariant::MultiParty);
        assert!(user_id < total_users);

        let sk_rlwe = client_key.sk_rlwe();
        let sk_lwe = client_key.sk_lwe();

        let g = self.pbs_info.parameters.g();
        let ring_size = self.pbs_info.parameters.rlwe_n().0;
        let rlwe_q = self.pbs_info.parameters.rlwe_q();
        let lwe_q = self.pbs_info.parameters.lwe_q();

        let rlweq_modop = &self.pbs_info.rlwe_modop;
        let rlweq_nttop = &self.pbs_info.rlwe_nttop;

        // sanity check
        assert!(sk_rlwe.len() == ring_size);
        assert!(sk_lwe.len() == self.pbs_info.parameters.lwe_n().0);

        // auto keys
        let auto_keys = self._common_rountine_multi_party_auto_keys_share_gen(
            cr_seed.auto_keys_cts_seed::<DefaultSecureRng>(),
            &sk_rlwe,
        );

        // rgsw ciphertexts of lwe secret elements
        let (self_leader_rgsws, not_self_leader_rgsws) = DefaultSecureRng::with_local_mut(|rng| {
            let mut self_leader_rgsw = vec![];
            let mut not_self_leader_rgsws = vec![];

            let (segment_start, segment_end) = interactive_mult_party_user_id_lwe_segment(
                user_id,
                total_users,
                self.pbs_info().lwe_n(),
            );

            // self LWE secret indices
            {
                // LWE secret indices for which user is the leader they need to send RGSW(m) for
                // RLWE x RGSW multiplication
                let rlrg_decomposer = self.pbs_info().rlwe_rgsw_decomposer();
                let (rlrg_d_a, rlrg_d_b) = (
                    rlrg_decomposer.a().decomposition_count(),
                    rlrg_decomposer.b().decomposition_count(),
                );
                let (gadget_a, gadget_b) = (
                    rlrg_decomposer.a().gadget_vector(),
                    rlrg_decomposer.b().gadget_vector(),
                );
                for s_index in segment_start..segment_end {
                    let mut out_rgsw = M::zeros(rlrg_d_a * 2 + rlrg_d_b * 2, ring_size);
                    public_key_encrypt_rgsw(
                        &mut out_rgsw,
                        &encode_x_pow_si_with_emebedding_factor::<
                            M::R,
                            CiphertextModulus<M::MatElement>,
                        >(
                            sk_lwe[s_index],
                            self.pbs_info().embedding_factor(),
                            ring_size,
                            self.pbs_info().rlwe_q(),
                        )
                        .as_ref(),
                        collective_pk,
                        &gadget_a,
                        &gadget_b,
                        rlweq_modop,
                        rlweq_nttop,
                        rng,
                    );
                    self_leader_rgsw.push(out_rgsw);
                }
            }

            // not self LWE secret indices
            {
                // LWE secret indices for which user isn't the leader, they need to send RGSW(m)
                // for RGSW x RGSW multiplcation
                let rgsw_rgsw_decomposer = self
                    .pbs_info
                    .parameters
                    .rgsw_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();
                let (rgrg_d_a, rgrg_d_b) = (
                    rgsw_rgsw_decomposer.a().decomposition_count(),
                    rgsw_rgsw_decomposer.b().decomposition_count(),
                );
                let (rgrg_gadget_a, rgrg_gadget_b) = (
                    rgsw_rgsw_decomposer.a().gadget_vector(),
                    rgsw_rgsw_decomposer.b().gadget_vector(),
                );

                for s_index in (0..segment_start).chain(segment_end..self.parameters().lwe_n().0) {
                    let mut out_rgsw = M::zeros(rgrg_d_a * 2 + rgrg_d_b * 2, ring_size);
                    public_key_encrypt_rgsw(
                        &mut out_rgsw,
                        &encode_x_pow_si_with_emebedding_factor::<
                            M::R,
                            CiphertextModulus<M::MatElement>,
                        >(
                            sk_lwe[s_index],
                            self.pbs_info().embedding_factor(),
                            ring_size,
                            self.pbs_info().rlwe_q(),
                        )
                        .as_ref(),
                        collective_pk,
                        &rgrg_gadget_a,
                        &rgrg_gadget_b,
                        rlweq_modop,
                        rlweq_nttop,
                        rng,
                    );

                    not_self_leader_rgsws.push(out_rgsw);
                }
            }

            (self_leader_rgsw, not_self_leader_rgsws)
        });

        // LWE Ksk
        let lwe_ksk = self._common_rountine_multi_party_lwe_ksk_share_gen(
            cr_seed.lwe_ksk_cts_seed_seed::<DefaultSecureRng>(),
            &sk_rlwe,
            &sk_lwe,
        );

        CommonReferenceSeededMultiPartyServerKeyShare::new(
            self_leader_rgsws,
            not_self_leader_rgsws,
            auto_keys,
            lwe_ksk,
            cr_seed.clone(),
            self.pbs_info.parameters.clone(),
            user_id,
        )
    }

    pub(super) fn aggregate_multi_party_server_key_shares<S>(
        &self,
        shares: &[CommonReferenceSeededMultiPartyServerKeyShare<
            M,
            BoolParameters<M::MatElement>,
            MultiPartyCrs<S>,
        >],
    ) -> SeededMultiPartyServerKey<M, MultiPartyCrs<S>, BoolParameters<M::MatElement>>
    where
        S: PartialEq + Clone,
        M: Clone,
    {
        assert_eq!(self.parameters().variant(), &ParameterVariant::MultiParty);
        assert!(shares.len() > 0);

        let total_users = shares.len();

        let parameters = shares[0].parameters().clone();
        let cr_seed = shares[0].cr_seed();

        let rlwe_n = parameters.rlwe_n().0;
        let g = parameters.g() as isize;
        let rlwe_q = parameters.rlwe_q();
        let lwe_q = parameters.lwe_q();

        // sanity checks
        shares.iter().skip(1).for_each(|s| {
            assert!(s.parameters() == &parameters);
            assert!(s.cr_seed() == cr_seed);
        });

        let rlweq_modop = &self.pbs_info.rlwe_modop;
        let rlweq_nttop = &self.pbs_info.rlwe_nttop;

        // auto keys
        let mut auto_keys = HashMap::new();
        let auto_elements_dlog = parameters.auto_element_dlogs();
        for i in auto_elements_dlog.into_iter() {
            let mut key = M::zeros(parameters.auto_decomposition_count().0, rlwe_n);

            shares.iter().for_each(|s| {
                let auto_key_share_i = s.auto_keys().get(&i).expect("Auto key {i} missing");
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
        let rgsw_cts = {
            let rgsw_by_rgsw_decomposer =
                parameters.rgsw_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();
            let rlwe_x_rgsw_decomposer = self.pbs_info().rlwe_rgsw_decomposer();
            let rgsw_x_rgsw_dimension = (
                rgsw_by_rgsw_decomposer.a().decomposition_count() * 2
                    + rgsw_by_rgsw_decomposer.b().decomposition_count() * 2,
                rlwe_n,
            );
            let rlwe_x_rgsw_dimension = (
                rlwe_x_rgsw_decomposer.a().decomposition_count() * 2
                    + rlwe_x_rgsw_decomposer.b().decomposition_count() * 2,
                rlwe_n,
            );
            let mut rgsw_x_rgsw_scratch_mat = M::zeros(
                std::cmp::max(
                    rgsw_by_rgsw_decomposer.a().decomposition_count(),
                    rgsw_by_rgsw_decomposer.b().decomposition_count(),
                ) + rlwe_x_rgsw_dimension.0,
                rlwe_n,
            );

            let shares_in_correct_order = (0..total_users)
                .map(|i| shares.iter().find(|s| s.user_id() == i).unwrap())
                .collect_vec();

            let lwe_n = self.parameters().lwe_n().0;
            let (users_segments, users_segments_sizes): (Vec<(usize, usize)>, Vec<usize>) = (0
                ..total_users)
                .map(|(user_id)| {
                    let (start_index, end_index) =
                        interactive_mult_party_user_id_lwe_segment(user_id, total_users, lwe_n);
                    ((start_index, end_index), end_index - start_index)
                })
                .unzip();

            let mut rgsw_cts = Vec::with_capacity(lwe_n);
            users_segments
                .iter()
                .enumerate()
                .for_each(|(user_id, user_segment)| {
                    let share = shares_in_correct_order[user_id];
                    for secret_index in user_segment.0..user_segment.1 {
                        let mut rgsw_i =
                            share.self_leader_rgsws()[secret_index - user_segment.0].clone();
                        // assert already exists in RGSW x RGSW rountine
                        assert!(rgsw_i.dimension() == rlwe_x_rgsw_dimension);

                        // multiply leader's RGSW ct at `secret_index` with RGSW cts of other users
                        // for lwe index `secret_index`
                        (0..total_users)
                            .filter(|i| i != &user_id)
                            .for_each(|other_user_id| {
                                let mut offset = 0;
                                if other_user_id < user_id {
                                    offset = users_segments_sizes[other_user_id];
                                }

                                let mut other_rgsw_i = shares_in_correct_order[other_user_id]
                                    .not_self_leader_rgsws()
                                    [secret_index.checked_sub(offset).unwrap()]
                                .clone();
                                // assert already exists in RGSW x RGSW rountine
                                assert!(other_rgsw_i.dimension() == rgsw_x_rgsw_dimension);

                                // send to evaluation domain for RGSwxRGSW mul
                                other_rgsw_i
                                    .iter_rows_mut()
                                    .for_each(|r| rlweq_nttop.forward(r.as_mut()));

                                rgsw_by_rgsw_inplace(
                                    &mut rgsw_i,
                                    rlwe_x_rgsw_decomposer.a().decomposition_count(),
                                    rlwe_x_rgsw_decomposer.b().decomposition_count(),
                                    &other_rgsw_i,
                                    &rgsw_by_rgsw_decomposer,
                                    &mut rgsw_x_rgsw_scratch_mat,
                                    rlweq_nttop,
                                    rlweq_modop,
                                )
                            });

                        rgsw_cts.push(rgsw_i);
                    }
                });

            rgsw_cts
        };

        // LWE ksks
        let mut lwe_ksk = M::R::zeros(rlwe_n * parameters.lwe_decomposition_count().0);
        let lweq_modop = &self.pbs_info.lwe_modop;
        shares.iter().for_each(|si| {
            assert!(si.lwe_ksk().as_ref().len() == rlwe_n * parameters.lwe_decomposition_count().0);
            lweq_modop.elwise_add_mut(lwe_ksk.as_mut(), si.lwe_ksk().as_ref())
        });

        SeededMultiPartyServerKey::new(rgsw_cts, auto_keys, lwe_ksk, cr_seed.clone(), parameters)
    }

    pub(super) fn aggregate_non_interactive_multi_party_key_share(
        &self,
        cr_seed: &NonInteractiveMultiPartyCrs<[u8; 32]>,
        total_users: usize,
        key_shares: &[CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<
            M,
            NonInteractiveMultiPartyCrs<[u8; 32]>,
        >],
    ) -> SeededNonInteractiveMultiPartyServerKey<
        M,
        NonInteractiveMultiPartyCrs<[u8; 32]>,
        BoolParameters<M::MatElement>,
    >
    where
        M: Clone + Debug,
    {
        assert_eq!(
            self.parameters().variant(),
            &ParameterVariant::NonInteractiveMultiParty
        );

        // sanity checks
        let key_order = {
            let existing_key_order = key_shares.iter().map(|s| s.user_index()).collect_vec();

            // record the order s.t. key_order[i] stores the position of i^th
            // users key share in existing order
            let mut key_order = Vec::with_capacity(existing_key_order.len());
            (0..total_users).for_each(|i| {
                // find i
                let index = existing_key_order
                    .iter()
                    .position(|x| x == &i)
                    .expect(&format!("Missing user {i}'s key!"));
                key_order.push(index);
            });

            key_order
        };

        let rlwe_modop = &self.pbs_info().rlwe_modop;
        let nttop = &self.pbs_info().rlwe_nttop;
        let ring_size = self.parameters().rlwe_n().0;
        let rlwe_q = self.parameters().rlwe_q();
        let lwe_modop = self.pbs_info().modop_lweq();

        // genrate key switching key from u_i to s
        let ui_to_s_ksk_decomposition_count = self
            .parameters()
            .non_interactive_ui_to_s_key_switch_decomposition_count();
        let mut ui_to_s_ksks = key_shares
            .iter()
            .map(|share| {
                let mut useri_ui_to_s_ksk = share.ui_to_s_ksk().clone();
                assert!(
                    useri_ui_to_s_ksk.dimension() == (ui_to_s_ksk_decomposition_count.0, ring_size)
                );
                key_shares
                    .iter()
                    .filter(|x| x.user_index() != share.user_index())
                    .for_each(|(other_share)| {
                        let op2 = other_share.ui_to_s_ksk_zero_encs_for_user_i(share.user_index());
                        assert!(op2.dimension() == (ui_to_s_ksk_decomposition_count.0, ring_size));
                        izip!(useri_ui_to_s_ksk.iter_rows_mut(), op2.iter_rows()).for_each(
                            |(add_to, add_from)| {
                                rlwe_modop.elwise_add_mut(add_to.as_mut(), add_from.as_ref())
                            },
                        );
                    });
                useri_ui_to_s_ksk
            })
            .collect_vec();

        let rgsw_cts = {
            let mut rgsw_prng =
                DefaultSecureRng::new_seeded(cr_seed.rgsw_cts_seed::<DefaultSecureRng>());

            let rgsw_by_rgsw_decomposer = self
                .parameters()
                .rgsw_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();

            // Generate RGSW Cts
            let rgsw_cts_all_users_eval = {
                // temporarily put ui_to_s in evaluation domain and sample a_i's for u_i to s
                // ksk for upcomong key switches
                ui_to_s_ksks.iter_mut().for_each(|ksk_i| {
                    ksk_i
                        .iter_rows_mut()
                        .for_each(|r| nttop.forward(r.as_mut()))
                });
                let ui_to_s_ksks_part_a_eval = key_shares
                    .iter()
                    .map(|share| {
                        let mut ksk_prng = DefaultSecureRng::new_seeded(
                            cr_seed
                                .ui_to_s_ks_seed_for_user_i::<DefaultSecureRng>(share.user_index()),
                        );
                        let mut ais = M::zeros(ui_to_s_ksk_decomposition_count.0, ring_size);

                        ais.iter_rows_mut().for_each(|r_ai| {
                            RandomFillUniformInModulus::random_fill(
                                &mut ksk_prng,
                                rlwe_q,
                                r_ai.as_mut(),
                            );

                            nttop.forward(r_ai.as_mut())
                        });
                        ais
                    })
                    .collect_vec();

                let max_rgrg_deocmposer = if rgsw_by_rgsw_decomposer.a().decomposition_count()
                    > rgsw_by_rgsw_decomposer.b().decomposition_count()
                {
                    rgsw_by_rgsw_decomposer.a()
                } else {
                    rgsw_by_rgsw_decomposer.b()
                };

                let ui_to_s_ksk_decomposer = self
                .parameters()
                .non_interactive_ui_to_s_key_switch_decomposer::<DefaultDecomposer<M::MatElement>>(
                );

                // Generate a_i*s + E = \sum_{j \in P} a_i*s_j + e for all rlwes in each
                // non-interactive rgsws. Then decompose and put decompositions in evaluation
                // for u_i -> s key switch.
                let decomp_ni_rgsws_part_1_acc = {
                    let mut tmp_space = M::R::zeros(ring_size);

                    (0..self.parameters().lwe_n().0)
                        .into_iter()
                        .map(|lwe_index| {
                            (0..max_rgrg_deocmposer.decomposition_count())
                                .into_iter()
                                .map(|d_index| {
                                    let mut sum = M::zeros(
                                        ui_to_s_ksk_decomposer.decomposition_count(),
                                        ring_size,
                                    );

                                    // set temp_space to all zeros
                                    tmp_space.as_mut().fill(M::MatElement::zero());

                                    // a_i*s + E
                                    key_shares.iter().for_each(|s| {
                                        rlwe_modop.elwise_add_mut(
                                            tmp_space.as_mut(),
                                            s.ni_rgsw_cts().1[lwe_index].get_row_slice(d_index),
                                        );
                                    });

                                    tmp_space.as_ref().iter().enumerate().for_each(|(ri, el)| {
                                        ui_to_s_ksk_decomposer
                                            .decompose_iter(el)
                                            .enumerate()
                                            .for_each(|(row_j, d_el)| {
                                                (sum.as_mut()[row_j]).as_mut()[ri] = d_el;
                                            });
                                    });

                                    sum.iter_rows_mut().for_each(|r| nttop.forward(r.as_mut()));

                                    sum
                                })
                                .collect_vec()
                        })
                        .collect_vec()
                };

                // Sample a_i's are used to generate non-interactive rgsw cts for all lwe
                // indices. Since a_i are just needed for key switches , decompose them
                // and put them in evaluation domain
                // Decomposition count used for RGSW ct in non-interactive key share gen equals
                // max of A and B decomposition required in RGSWxRGSW. This is because same
                // polynomials are used to generate RLWE'(m) and RLWE'(-sm)
                let decomp_ni_rgsw_neg_ais = {
                    let mut tmp_space = M::R::zeros(ring_size);
                    (0..self.parameters().lwe_n().0)
                        .into_iter()
                        .map(|_| {
                            // FIXME(Jay): well well, ais are only required for key switching to
                            // generate RLWE'(m) and RLWE'(m) itself
                            // requires RLWE(\beta^i m) for i \in RGSWxRGSW
                            // part B decomposition count. However, we still need to puncture prng
                            // for ais corresponding to ignored limbs.
                            // Probably it will be nice idea to
                            // avoid decomposition after punturing for a_i's corresponding to
                            // ignored limbs. Moreover, note that, for
                            // RGSWxRGSW often times decompostion count for part A
                            // > part B. Hence, it's very likely that we are doing
                            // unecesary decompositions for ignored limbs all the time.
                            (0..max_rgrg_deocmposer.decomposition_count())
                                .map(|_| {
                                    RandomFillUniformInModulus::random_fill(
                                        &mut rgsw_prng,
                                        rlwe_q,
                                        tmp_space.as_mut(),
                                    );

                                    // negate
                                    rlwe_modop.elwise_neg_mut(tmp_space.as_mut());

                                    // decomposer a_i for ui -> s key switch
                                    let mut decomp_neg_ai = M::zeros(
                                        ui_to_s_ksk_decomposer.decomposition_count(),
                                        ring_size,
                                    );
                                    tmp_space.as_ref().iter().enumerate().for_each(
                                        |(index, el)| {
                                            ui_to_s_ksk_decomposer
                                                .decompose_iter(el)
                                                .enumerate()
                                                .for_each(|(row_j, d_el)| {
                                                    (decomp_neg_ai.as_mut()[row_j]).as_mut()
                                                        [index] = d_el;
                                                });
                                        },
                                    );

                                    // put in evaluation domain
                                    decomp_neg_ai
                                        .iter_rows_mut()
                                        .for_each(|r| nttop.forward(r.as_mut()));

                                    decomp_neg_ai
                                })
                                .collect_vec()
                        })
                        .collect_vec()
                };

                // genrate RGSW cts
                let rgsw_cts_all_users_eval = izip!(
                    key_shares.iter(),
                    ui_to_s_ksks.iter(),
                    ui_to_s_ksks_part_a_eval.iter()
                )
                .map(
                    |(share, user_uitos_ksk_partb_eval, user_uitos_ksk_parta_eval)| {
                        // RGSW_s(X^{s[i]})
                        let rgsw_cts_user_i_eval = izip!(
                            share.ni_rgsw_cts().0.iter(),
                            decomp_ni_rgsw_neg_ais.iter(),
                            decomp_ni_rgsws_part_1_acc.iter()
                        )
                        .map(
                            |(
                                m_encs_under_ui,
                                decomposed_rgsw_i_neg_ais,
                                decomposed_acc_rgsw_part1,
                            )| {
                                let d_a = rgsw_by_rgsw_decomposer.a().decomposition_count();
                                let d_b = rgsw_by_rgsw_decomposer.b().decomposition_count();
                                let max_d = std::cmp::max(d_a, d_b);

                                assert!(decomposed_rgsw_i_neg_ais.len() == max_d);

                                // To be RGSW(X^{s[i]}) = [RLWE'(-sm), RLWE'(m)]
                                let mut rgsw_ct_eval = M::zeros(d_a * 2 + d_b * 2, ring_size);
                                let (rlwe_dash_nsm, rlwe_dash_m) =
                                    rgsw_ct_eval.split_at_row_mut(d_a * 2);

                                let mut scratch_row = M::R::zeros(ring_size);

                                let mut m_encs_under_ui_eval = m_encs_under_ui.clone();
                                m_encs_under_ui_eval
                                    .iter_rows_mut()
                                    .for_each(|r| nttop.forward(r.as_mut()));

                                // RLWE(-sm)
                                {
                                    // Recall that we have RLWE(a_i * s + e). We key
                                    // switch RLWE(a_i * s + e) using ksk(u_i -> s) to
                                    // get RLWE(a_i*s*u_i + e*u_i).
                                    // Given (u_i * a_i + e + \beta m), we obtain
                                    // RLWE(-sm) = RLWE(a_i*s*u_i + e*u_i) + (0, (u_i *
                                    // a_i + e + \beta m))
                                    //
                                    // Again RLWE'(-sm) only cares for RLWE(\beta -sm)
                                    // with scaling factor \beta corresponding to most
                                    // signficant d_a limbs. Hence, we skip (d_max -
                                    // d_a) least signficant limbs

                                    let (rlwe_dash_neg_sm_part_a, rlwe_dash_neg_sm_part_b) =
                                        rlwe_dash_nsm.split_at_mut(d_a);

                                    izip!(
                                        rlwe_dash_neg_sm_part_a.iter_mut(),
                                        rlwe_dash_neg_sm_part_b.iter_mut(),
                                        decomposed_acc_rgsw_part1.iter().skip(max_d - d_a),
                                        m_encs_under_ui_eval.iter_rows().skip(max_d - d_a)
                                    )
                                    .for_each(
                                        |(rlwe_a, rlwe_b, decomp_ai_s, beta_m_enc_ui)| {
                                            // RLWE_s(a_i * s * u_i + u_i * e) = decomp<a_i * s + e>
                                            // * ksk(u_i
                                            // -> s)
                                            izip!(
                                                decomp_ai_s.iter_rows(),
                                                user_uitos_ksk_partb_eval.iter_rows(),
                                                user_uitos_ksk_parta_eval.iter_rows()
                                            )
                                            .for_each(
                                                |(a0, part_b, part_a)| {
                                                    // rlwe_b += decomp<a_i * s + e>[i] * ksk
                                                    // part_b[i]
                                                    rlwe_modop.elwise_mul(
                                                        scratch_row.as_mut(),
                                                        a0.as_ref(),
                                                        part_b.as_ref(),
                                                    );
                                                    rlwe_modop.elwise_add_mut(
                                                        rlwe_b.as_mut(),
                                                        scratch_row.as_ref(),
                                                    );

                                                    // rlwe_a += decomp<a_i * s + e>[i] * ksk
                                                    // part_a[i]
                                                    rlwe_modop.elwise_mul(
                                                        scratch_row.as_mut(),
                                                        a0.as_ref(),
                                                        part_a.as_ref(),
                                                    );
                                                    rlwe_modop.elwise_add_mut(
                                                        rlwe_a.as_mut(),
                                                        scratch_row.as_ref(),
                                                    );
                                                },
                                            );

                                            // RLWE_s(-sm) = RLWE_s(a_i * s * u_i + u_i
                                            // * e) + (0, a_i * u + e + m)
                                            rlwe_modop.elwise_add_mut(
                                                rlwe_a.as_mut(),
                                                beta_m_enc_ui.as_ref(),
                                            );
                                        },
                                    );
                                }

                                // RLWE(m)
                                {
                                    // Routine:
                                    // Let RLWE(-a_i * u_i) = (decomp<-a_i> \cdot ksk(u_i -> s)),
                                    // then RLWE(m) = RLWE(a'*s + e + m) = (a_i
                                    // * u_i + e + m, 0) + RLWE(-a_i * u_i)
                                    //
                                    // Since RLWE'(m) only cares for RLWE ciphertexts corresponding
                                    // to higher d_b limbs, we skip routine 1 for lower max(d_a,
                                    // d_b) - d_b limbs
                                    let (rlwe_dash_m_part_a, rlwe_dash_m_part_b) =
                                        rlwe_dash_m.split_at_mut(d_b);

                                    izip!(
                                        rlwe_dash_m_part_a.iter_mut(),
                                        rlwe_dash_m_part_b.iter_mut(),
                                        decomposed_rgsw_i_neg_ais.iter().skip(max_d - d_b),
                                        m_encs_under_ui_eval.iter_rows().skip(max_d - d_b)
                                    )
                                    .for_each(
                                        |(rlwe_a, rlwe_b, decomp_neg_ai, beta_m_enc_ui)| {
                                            // RLWE_s(-a_i * ui) = decomp<-a_i> \cdot ksk(ui -> s)
                                            izip!(
                                                decomp_neg_ai.iter_rows(),
                                                user_uitos_ksk_partb_eval.iter_rows(),
                                                user_uitos_ksk_parta_eval.iter_rows()
                                            )
                                            .for_each(
                                                |(a0, part_b, part_a)| {
                                                    // rlwe_b += decomp<a>[i] * ksk part_b[i]
                                                    rlwe_modop.elwise_mul(
                                                        scratch_row.as_mut(),
                                                        a0.as_ref(),
                                                        part_b.as_ref(),
                                                    );
                                                    rlwe_modop.elwise_add_mut(
                                                        rlwe_b.as_mut(),
                                                        scratch_row.as_ref(),
                                                    );

                                                    // rlwe_a += decomp<a>[i] * ksk
                                                    // part_a[i]
                                                    rlwe_modop.elwise_mul(
                                                        scratch_row.as_mut(),
                                                        a0.as_ref(),
                                                        part_a.as_ref(),
                                                    );
                                                    rlwe_modop.elwise_add_mut(
                                                        rlwe_a.as_mut(),
                                                        scratch_row.as_ref(),
                                                    );
                                                },
                                            );

                                            // RLWE_s(m) = (a_i * ui + e +  \beta m, 0)
                                            // +
                                            // RLWE_s(-a_i * ui)
                                            rlwe_modop.elwise_add_mut(
                                                rlwe_b.as_mut(),
                                                beta_m_enc_ui.as_ref(),
                                            );
                                        },
                                    );
                                }

                                rgsw_ct_eval
                            },
                        )
                        .collect_vec();
                        rgsw_cts_user_i_eval
                    },
                )
                .collect_vec();

                // put u_i -> s ksks back in coefficient domain
                ui_to_s_ksks.iter_mut().for_each(|ksk_i| {
                    ksk_i
                        .iter_rows_mut()
                        .for_each(|r| nttop.backward(r.as_mut()))
                });

                rgsw_cts_all_users_eval
            };

            // RGSW x RGSW
            let lwe_n = self.parameters().lwe_n().0;
            let mut scratch_matrix = M::zeros(
                std::cmp::max(
                    rgsw_by_rgsw_decomposer.a().decomposition_count(),
                    rgsw_by_rgsw_decomposer.b().decomposition_count(),
                ) + (rgsw_by_rgsw_decomposer.a().decomposition_count() * 2
                    + rgsw_by_rgsw_decomposer.b().decomposition_count() * 2),
                ring_size,
            );
            let rgsw_cts_untrimmed = (0..lwe_n).map(|s_index| {
                // copy over s_index^th rgsw ct of user 0. Use it to accumulate RGSW products of
                // all RGSW ciphertexts at s_index
                let mut rgsw_i = rgsw_cts_all_users_eval[0][s_index].clone();
                rgsw_i
                    .iter_rows_mut()
                    .for_each(|r| nttop.backward(r.as_mut()));

                rgsw_cts_all_users_eval
                    .iter()
                    .skip(1)
                    .for_each(|user_i_rgsws| {
                        rgsw_by_rgsw_inplace(
                            &mut rgsw_i,
                            rgsw_by_rgsw_decomposer.a().decomposition_count(),
                            rgsw_by_rgsw_decomposer.b().decomposition_count(),
                            &user_i_rgsws[s_index],
                            &rgsw_by_rgsw_decomposer,
                            &mut scratch_matrix,
                            nttop,
                            rlwe_modop,
                        );
                    });

                rgsw_i
            });

            // After this point we don't require RGSW cts for RGSWxRGSW
            // multiplicaiton anymore. So we trim them to suit RLWExRGSW require  for PBS
            let rgsw_cts = rgsw_cts_untrimmed
                .map(|rgsw_ct| {
                    trim_rgsw_ct_matrix_from_rgrg_to_rlrg(
                        rgsw_ct,
                        self.parameters().rgsw_by_rgsw_decomposition_params(),
                        self.parameters().rlwe_by_rgsw_decomposition_params(),
                    )
                })
                .collect_vec();
            rgsw_cts
        };

        // auto keys
        let auto_keys = {
            let mut auto_keys = HashMap::new();
            let auto_elements_dlog = self.parameters().auto_element_dlogs();
            for i in auto_elements_dlog.into_iter() {
                let mut key = M::zeros(self.parameters().auto_decomposition_count().0, ring_size);

                key_shares.iter().for_each(|s| {
                    let auto_key_share_i =
                        s.auto_keys_share().get(&i).expect("Auto key {i} missing");
                    assert!(
                        auto_key_share_i.dimension()
                            == (self.parameters().auto_decomposition_count().0, ring_size)
                    );
                    izip!(key.iter_rows_mut(), auto_key_share_i.iter_rows()).for_each(
                        |(partb_out, partb_share)| {
                            rlwe_modop.elwise_add_mut(partb_out.as_mut(), partb_share.as_ref());
                        },
                    );
                });

                auto_keys.insert(i, key);
            }
            auto_keys
        };

        // LWE ksk
        let lwe_ksk = {
            let mut lwe_ksk =
                M::R::zeros(self.parameters().lwe_decomposition_count().0 * ring_size);
            key_shares.iter().for_each(|s| {
                assert!(
                    s.lwe_ksk_share().as_ref().len()
                        == self.parameters().lwe_decomposition_count().0 * ring_size
                );
                lwe_modop.elwise_add_mut(lwe_ksk.as_mut(), s.lwe_ksk_share().as_ref());
            });
            lwe_ksk
        };

        SeededNonInteractiveMultiPartyServerKey::new(
            ui_to_s_ksks,
            key_order,
            rgsw_cts,
            auto_keys,
            lwe_ksk,
            cr_seed.clone(),
            self.parameters().clone(),
        )
    }

    pub(super) fn non_interactive_multi_party_key_share<
        K: NonInteractiveMultiPartyClientKey<Element = i32>,
    >(
        &self,
        // TODO(Jay): Should get a common reference seed here and derive the rest.
        cr_seed: &NonInteractiveMultiPartyCrs<[u8; 32]>,
        self_index: usize,
        total_users: usize,
        client_key: &K,
    ) -> CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<
        M,
        NonInteractiveMultiPartyCrs<[u8; 32]>,
    > {
        assert_eq!(
            self.parameters().variant(),
            &ParameterVariant::NonInteractiveMultiParty
        );

        // TODO:  check whether parameters support `total_users`
        let nttop = self.pbs_info().nttop_rlweq();
        let rlwe_modop = self.pbs_info().modop_rlweq();
        let ring_size = self.pbs_info().rlwe_n();
        let rlwe_q = self.parameters().rlwe_q();

        let sk_rlwe = client_key.sk_rlwe();
        let sk_u_rlwe = client_key.sk_u_rlwe();
        let sk_lwe = client_key.sk_lwe();

        let (ui_to_s_ksk, zero_encs_for_others) = DefaultSecureRng::with_local_mut(|rng| {
            // ui_to_s_ksk
            let non_interactive_decomposer = self
                .parameters()
                .non_interactive_ui_to_s_key_switch_decomposer::<DefaultDecomposer<M::MatElement>>(
                );
            let non_interactive_gadget_vec = non_interactive_decomposer.gadget_vector();
            let ui_to_s_ksk = {
                let mut p_rng = DefaultSecureRng::new_seeded(
                    cr_seed.ui_to_s_ks_seed_for_user_i::<DefaultSecureRng>(self_index),
                );

                non_interactive_ksk_gen::<M, _, _, _, _, _>(
                    &sk_rlwe,
                    &sk_u_rlwe,
                    &non_interactive_gadget_vec,
                    &mut p_rng,
                    rng,
                    nttop,
                    rlwe_modop,
                )
            };

            // zero encryptions for others uj_to_s ksk
            let all_users_except_self = (0..total_users).filter(|x| *x != self_index);
            let zero_encs_for_others = all_users_except_self
                .map(|other_user_index| {
                    let mut p_rng = DefaultSecureRng::new_seeded(
                        cr_seed.ui_to_s_ks_seed_for_user_i::<DefaultSecureRng>(other_user_index),
                    );
                    let zero_encs =
                        non_interactive_ksk_zero_encryptions_for_other_party_i::<M, _, _, _, _, _>(
                            &sk_rlwe,
                            &non_interactive_gadget_vec,
                            &mut p_rng,
                            rng,
                            nttop,
                            rlwe_modop,
                        );
                    zero_encs
                })
                .collect_vec();

            (ui_to_s_ksk, zero_encs_for_others)
        });

        // Non-interactive RGSW cts = (a_i * u_j + e + \beta X^{s[i]}, a_i * s_j + e')
        let ni_rgsw_cts = DefaultSecureRng::with_local_mut(|rng| {
            let mut rgsw_cts_prng =
                DefaultSecureRng::new_seeded(cr_seed.rgsw_cts_seed::<DefaultSecureRng>());
            // generate non-interactive rgsw cts
            let rgsw_by_rgsw_decomposer = self
                .parameters()
                .rgsw_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();

            let ni_rgrg_gadget_vec = {
                if rgsw_by_rgsw_decomposer.a().decomposition_count()
                    > rgsw_by_rgsw_decomposer.b().decomposition_count()
                {
                    rgsw_by_rgsw_decomposer.a().gadget_vector()
                } else {
                    rgsw_by_rgsw_decomposer.b().gadget_vector()
                }
            };
            let ni_rgsw_cts: (Vec<M>, Vec<M>) = sk_lwe
                .iter()
                .map(|s_i| {
                    // X^{s[i]}
                    let mut m = M::R::zeros(ring_size);
                    let s_i = s_i * (self.pbs_info().embedding_factor() as i32);
                    if s_i < 0 {
                        // X^{-s[i]} -> -X^{N+s[i]}
                        m.as_mut()[ring_size - (s_i.abs() as usize)] = rlwe_q.neg_one();
                    } else {
                        m.as_mut()[s_i as usize] = M::MatElement::one();
                    }

                    non_interactive_rgsw_ct::<M, _, _, _, _, _>(
                        &sk_rlwe,
                        &sk_u_rlwe,
                        m.as_ref(),
                        &ni_rgrg_gadget_vec,
                        &mut rgsw_cts_prng,
                        rng,
                        nttop,
                        rlwe_modop,
                    )
                })
                .unzip();
            ni_rgsw_cts
        });

        // Auto key share
        let auto_keys_share = {
            let auto_seed = cr_seed.auto_keys_cts_seed::<DefaultSecureRng>();
            self._common_rountine_multi_party_auto_keys_share_gen(auto_seed, &sk_rlwe)
        };

        // Lwe Ksk share
        let lwe_ksk_share = {
            let lwe_ksk_seed = cr_seed.lwe_ksk_cts_seed::<DefaultSecureRng>();
            self._common_rountine_multi_party_lwe_ksk_share_gen(lwe_ksk_seed, &sk_rlwe, &sk_lwe)
        };

        CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare::new(
            ni_rgsw_cts,
            ui_to_s_ksk,
            zero_encs_for_others,
            auto_keys_share,
            lwe_ksk_share,
            self_index,
            cr_seed.clone(),
        )
    }

    fn _common_rountine_multi_party_auto_keys_share_gen(
        &self,
        auto_seed: <DefaultSecureRng as NewWithSeed>::Seed,
        sk_rlwe: &[i32],
    ) -> HashMap<usize, M> {
        let g = self.pbs_info.parameters.g();
        let ring_size = self.pbs_info.parameters.rlwe_n().0;
        let br_q = self.pbs_info.parameters.br_q();
        let rlweq_modop = &self.pbs_info.rlwe_modop;
        let rlweq_nttop = &self.pbs_info.rlwe_nttop;

        DefaultSecureRng::with_local_mut(|rng| {
            let mut p_rng = DefaultSecureRng::new_seeded(auto_seed);

            let mut auto_keys = HashMap::new();
            let auto_gadget = self.pbs_info.auto_decomposer.gadget_vector();
            let auto_element_dlogs = self.pbs_info.parameters.auto_element_dlogs();

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
                    sk_rlwe,
                    g_pow,
                    &auto_gadget,
                    rlweq_modop,
                    rlweq_nttop,
                    &mut p_rng,
                    rng,
                );
                auto_keys.insert(i, ksk_out);
            }

            auto_keys
        })
    }

    fn _common_rountine_multi_party_lwe_ksk_share_gen(
        &self,
        lwe_ksk_seed: <DefaultSecureRng as NewWithSeed>::Seed,
        sk_rlwe: &[i32],
        sk_lwe: &[i32],
    ) -> M::R {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut p_rng = DefaultSecureRng::new_seeded(lwe_ksk_seed);
            let mut lwe_ksk = M::R::zeros(
                self.pbs_info.lwe_decomposer.decomposition_count() * self.parameters().rlwe_n().0,
            );
            let lwe_modop = &self.pbs_info.lwe_modop;
            let d_lwe_gadget_vec = self.pbs_info.lwe_decomposer.gadget_vector();
            lwe_ksk_keygen(
                sk_rlwe,
                sk_lwe,
                &mut lwe_ksk,
                &d_lwe_gadget_vec,
                lwe_modop,
                &mut p_rng,
                rng,
            );
            lwe_ksk
        })
    }

    pub(super) fn multi_party_public_key_share<K: InteractiveMultiPartyClientKey<Element = i32>>(
        &self,
        cr_seed: &MultiPartyCrs<[u8; 32]>,
        client_key: &K,
    ) -> CommonReferenceSeededCollectivePublicKeyShare<
        <M as Matrix>::R,
        [u8; 32],
        BoolParameters<<M as Matrix>::MatElement>,
    > {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut share_out = M::R::zeros(self.pbs_info.parameters.rlwe_n().0);
            let modop = &self.pbs_info.rlwe_modop;
            let nttop = &self.pbs_info.rlwe_nttop;
            let pk_seed = cr_seed.public_key_share_seed::<DefaultSecureRng>();
            let mut main_prng = DefaultSecureRng::new_seeded(pk_seed);
            public_key_share(
                &mut share_out,
                &client_key.sk_rlwe(),
                modop,
                nttop,
                &mut main_prng,
                rng,
            );
            CommonReferenceSeededCollectivePublicKeyShare::new(
                share_out,
                pk_seed,
                self.pbs_info.parameters.clone(),
            )
        })
    }

    pub(super) fn multi_party_decryption_share<K: InteractiveMultiPartyClientKey<Element = i32>>(
        &self,
        lwe_ct: &M::R,
        client_key: &K,
    ) -> <M as Matrix>::MatElement {
        assert!(lwe_ct.as_ref().len() == self.pbs_info.parameters.rlwe_n().0 + 1);
        let modop = &self.pbs_info.rlwe_modop;
        let mut neg_s =
            M::R::try_convert_from(&client_key.sk_rlwe(), &self.pbs_info.parameters.rlwe_q());
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
    pub fn sk_encrypt<K: SinglePartyClientKey<Element = i32>>(
        &self,
        m: bool,
        client_key: &K,
    ) -> M::R {
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
                &client_key.sk_rlwe(),
                &self.pbs_info.rlwe_modop,
                rng,
            );
            lwe_out
        })
    }

    pub fn sk_decrypt<K: SinglePartyClientKey<Element = i32>>(
        &self,
        lwe_ct: &M::R,
        client_key: &K,
    ) -> bool {
        let m = decrypt_lwe(lwe_ct, &client_key.sk_rlwe(), &self.pbs_info.rlwe_modop);
        self.pbs_info.rlwe_q().decode(m)
    }
}

impl<M, NttOp, RlweModOp, LweModOp, Skey> BoolEvaluator<M, NttOp, RlweModOp, LweModOp, Skey>
where
    M: MatrixMut + MatrixEntity,
    M::R: RowMut + RowEntity,
    M::MatElement: PrimInt + FromPrimitive + One + Copy + Zero + Display + WrappingSub + NumInfo,
    RlweModOp: VectorOps<Element = M::MatElement> + ArithmeticOps<Element = M::MatElement>,
    LweModOp: VectorOps<Element = M::MatElement> + ArithmeticOps<Element = M::MatElement>,
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

impl<M, NttOp, RlweModOp, LweModOp, Skey> BooleanGates
    for BoolEvaluator<M, NttOp, RlweModOp, LweModOp, Skey>
where
    M: MatrixMut + MatrixEntity,
    M::R: RowMut + RowEntity + Clone,
    M::MatElement: PrimInt
        + FromPrimitive
        + One
        + Copy
        + Zero
        + Display
        + WrappingSub
        + NumInfo
        + From<bool>
        + WrappingAdd
        + Debug,
    RlweModOp: VectorOps<Element = M::MatElement>
        + ArithmeticOps<Element = M::MatElement>
        + ShoupMatrixFMA<M::R>,
    LweModOp: VectorOps<Element = M::MatElement> + ArithmeticOps<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement>,
    Skey: PbsKey<AutoKey = <Skey as PbsKey>::RgswCt, LweKskKey = M>,
    <Skey as PbsKey>::RgswCt: WithShoupRepr<M = M>,
{
    type Ciphertext = M::R;
    type Key = Skey;

    fn nand_inplace(&mut self, c0: &mut M::R, c1: &M::R, server_key: &Self::Key) {
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

    fn and_inplace(&mut self, c0: &mut M::R, c1: &M::R, server_key: &Self::Key) {
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

    fn or_inplace(&mut self, c0: &mut M::R, c1: &M::R, server_key: &Self::Key) {
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

    fn nor_inplace(&mut self, c0: &mut M::R, c1: &M::R, server_key: &Self::Key) {
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

    fn xor_inplace(&mut self, c0: &mut M::R, c1: &M::R, server_key: &Self::Key) {
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

    fn xnor_inplace(&mut self, c0: &mut M::R, c1: &M::R, server_key: &Self::Key) {
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

#[cfg(test)]
mod tests {

    use rand::{thread_rng, Rng};
    use rand_distr::Uniform;

    use crate::{
        backend::ModulusPowerOf2,
        bool::{
            keys::{
                NonInteractiveServerKeyEvaluationDomain, PublicKey,
                ShoupNonInteractiveServerKeyEvaluationDomain,
            },
            parameters::{MP_BOOL_PARAMS, NI_2P, SMALL_MP_BOOL_PARAMS, SP_TEST_BOOL_PARAMS},
        },
        evaluator,
        ntt::NttBackendU64,
        parameters::I_2P,
        random::{RandomElementInModulus, DEFAULT_RNG},
        rgsw::{
            self, measure_noise, public_key_encrypt_rlwe, secret_key_encrypt_rlwe,
            tests::{_measure_noise_rgsw, _sk_encrypt_rlwe},
            RgswCiphertext, RgswCiphertextEvaluationDomain, SeededRgswCiphertext,
            SeededRlweCiphertext,
        },
        utils::{negacyclic_mul, tests::Stats},
    };

    use super::*;

    #[test]
    fn testtest() {
        let evaluator = BoolEvaluator::<
            Vec<Vec<u64>>,
            NttBackendU64,
            ModularOpsU64<CiphertextModulus<u64>>,
            ModulusPowerOf2<CiphertextModulus<u64>>,
            ShoupNonInteractiveServerKeyEvaluationDomain<Vec<Vec<u64>>>,
        >::new(NI_2P);
        let mp_seed = NonInteractiveMultiPartyCrs { seed: [1u8; 32] };

        let ring_size = evaluator.parameters().rlwe_n().0;
        let rlwe_q = evaluator.parameters().rlwe_q();
        let rlwe_modop = evaluator.pbs_info().modop_rlweq();
        let nttop = evaluator.pbs_info().nttop_rlweq();

        let parties = 2;

        let cks = (0..parties).map(|_| evaluator.client_key()).collect_vec();

        let key_shares = (0..parties)
            .map(|i| evaluator.non_interactive_multi_party_key_share(&mp_seed, i, parties, &cks[i]))
            .collect_vec();
        // dbg!(key_shares[1].user_index);

        let seeded_server_key = evaluator.aggregate_non_interactive_multi_party_key_share(
            &mp_seed,
            parties,
            &key_shares,
        );
        let server_key_evaluation_domain =
            NonInteractiveServerKeyEvaluationDomain::<_, _, DefaultSecureRng, NttBackendU64>::from(
                &seeded_server_key,
            );

        let mut ideal_rlwe = vec![0; ring_size];
        cks.iter().for_each(|k| {
            izip!(
                ideal_rlwe.iter_mut(),
                NonInteractiveMultiPartyClientKey::sk_rlwe(k).iter()
            )
            .for_each(|(a, b)| {
                *a = *a + b;
            });
        });

        let mut ideal_lwe = vec![0; evaluator.parameters().lwe_n().0];
        cks.iter().for_each(|k| {
            izip!(
                ideal_lwe.iter_mut(),
                NonInteractiveMultiPartyClientKey::sk_lwe(k).iter()
            )
            .for_each(|(a, b)| {
                *a = *a + b;
            });
        });

        let mut stats = Stats::new();

        {
            let (rlrg_decomp_a, rlrg_decomp_b) = evaluator
                .parameters()
                .rlwe_rgsw_decomposer::<DefaultDecomposer<u64>>();
            let gadget_vec_a = rlrg_decomp_a.gadget_vector();
            let gadget_vec_b = rlrg_decomp_b.gadget_vector();
            let d_a = rlrg_decomp_a.decomposition_count();
            let d_b = rlrg_decomp_b.decomposition_count();

            // s[X]
            let s_poly = Vec::<u64>::try_convert_from(ideal_rlwe.as_slice(), rlwe_q);

            // -s[X]
            let mut neg_s_poly_eval = s_poly.clone();
            rlwe_modop.elwise_neg_mut(&mut neg_s_poly_eval);
            nttop.forward(neg_s_poly_eval.as_mut());

            server_key_evaluation_domain
                .rgsw_cts()
                .iter()
                .enumerate()
                .for_each(|(s_index, ct)| {
                    // X^{lwe_s[i]}
                    let mut m = vec![0u64; ring_size];
                    let s_i = ideal_lwe[s_index] * (evaluator.pbs_info().embedding_factor() as i32);
                    if s_i < 0 {
                        m[ring_size - (s_i.abs() as usize)] = rlwe_q.neg_one();
                    } else {
                        m[(s_i as usize)] = 1;
                    }

                    let mut neg_sm = m.clone();
                    nttop.forward(&mut neg_sm);
                    rlwe_modop.elwise_mul_mut(&mut neg_sm, &neg_s_poly_eval);
                    nttop.backward(&mut neg_sm);

                    // RLWE'(-sm)
                    gadget_vec_a.iter().enumerate().for_each(|(index, beta)| {
                        // RLWE(\beta -sm)

                        // \beta * -sX^[lwe_s[i]]
                        let mut beta_neg_sm = neg_sm.clone();
                        rlwe_modop.elwise_scalar_mul_mut(&mut beta_neg_sm, beta);

                        // extract RLWE(-sm \beta)
                        let mut rlwe = vec![vec![0u64; ring_size]; 2];
                        rlwe[0].copy_from_slice(&ct[index]);
                        rlwe[1].copy_from_slice(&ct[index + d_a]);
                        // send back to coefficient domain
                        rlwe.iter_rows_mut()
                            .for_each(|r| nttop.backward(r.as_mut_slice()));

                        // decrypt
                        let mut m_out = vec![0u64; ring_size];
                        decrypt_rlwe(&rlwe, &ideal_rlwe, &mut m_out, nttop, rlwe_modop); // println!("{:?}", &beta_neg_sm);

                        let mut diff = m_out;
                        rlwe_modop.elwise_sub_mut(&mut diff, &beta_neg_sm);

                        stats.add_more(&Vec::<i64>::try_convert_from(&diff, rlwe_q));
                    });
                });
        }
        println!("Stats: {}", stats.std_dev().abs().log2());
    }
}
