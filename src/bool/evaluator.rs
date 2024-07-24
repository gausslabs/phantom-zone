use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    marker::PhantomData,
    usize,
};

use itertools::{izip, Itertools};
use num_traits::{FromPrimitive, One, PrimInt, ToPrimitive, WrappingAdd, WrappingSub, Zero};
use rand_distr::uniform::SampleUniform;
use serde::{Deserialize, Serialize};

use crate::{
    backend::{ArithmeticOps, GetModulus, ModInit, Modulus, ShoupMatrixFMA, VectorOps},
    bool::parameters::ParameterVariant,
    decomposer::{Decomposer, DefaultDecomposer, NumInfo, RlweDecomposer},
    lwe::{decrypt_lwe, encrypt_lwe, seeded_lwe_ksk_keygen},
    multi_party::{
        non_interactive_ksk_gen, non_interactive_ksk_zero_encryptions_for_other_party_i,
        public_key_share,
    },
    ntt::{Ntt, NttInit},
    pbs::{pbs, PbsInfo, PbsKey, WithShoupRepr},
    random::{
        DefaultSecureRng, NewWithSeed, RandomFill, RandomFillGaussianInModulus,
        RandomFillUniformInModulus,
    },
    rgsw::{
        generate_auto_map, public_key_encrypt_rgsw, rgsw_by_rgsw_inplace, rgsw_x_rgsw_scratch_rows,
        rlwe_auto_scratch_rows, rlwe_x_rgsw_scratch_rows, secret_key_encrypt_rgsw,
        seeded_auto_key_gen, RgswCiphertextMutRef, RgswCiphertextRef, RuntimeScratchMutRef,
    },
    utils::{
        encode_x_pow_si_with_emebedding_factor, mod_exponent, puncture_p_rng, TryConvertFrom1,
        WithLocal,
    },
    BooleanGates, Encoder, Matrix, MatrixEntity, MatrixMut, RowEntity, RowMut,
};

use super::{
    keys::{
        ClientKey, CommonReferenceSeededCollectivePublicKeyShare,
        CommonReferenceSeededInteractiveMultiPartyServerKeyShare,
        CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare,
        InteractiveMultiPartyClientKey, NonInteractiveMultiPartyClientKey,
        SeededInteractiveMultiPartyServerKey, SeededNonInteractiveMultiPartyServerKey,
        SeededSinglePartyServerKey, SinglePartyClientKey,
    },
    parameters::{BoolParameters, CiphertextModulus, DecompositionCount, DoubleDecomposerParams},
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
///             Puncture 1 -> Auto keys cipertexts seed
///             Puncture 2 -> LWE ksk seed
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractiveMultiPartyCrs<S> {
    pub(super) seed: S,
}

impl InteractiveMultiPartyCrs<[u8; 32]> {
    pub(super) fn random() -> Self {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);
            Self { seed }
        })
    }
}

impl<S: Default + Copy> InteractiveMultiPartyCrs<S> {
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

    pub(super) fn auto_keys_cts_seed<Rng: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut key_prng = Rng::new_with_seed(self.key_seed::<Rng>());
        puncture_p_rng(&mut key_prng, 1)
    }

    pub(super) fn lwe_ksk_cts_seed_seed<Rng: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut key_prng = Rng::new_with_seed(self.key_seed::<Rng>());
        puncture_p_rng(&mut key_prng, 2)
    }
}

/// Common reference seed used for non-interactive multi-party.
///
/// Initial Seed
///     Puncture 1 -> Key Seed
///         Puncture 1 -> Rgsw ciphertext seed
///             Puncture l+1 -> Seed for zero encs and non-interactive
///                             multi-party RGSW ciphertexts of
///                             l^th LWE index.
///         Puncture 2 -> auto keys seed
///         Puncture 3 -> Lwe key switching key seed
///     Puncture 2 -> user specific seed for u_j to s ksk
///         Punture j+1 -> user j's seed    
#[derive(Clone, PartialEq, Serialize, Deserialize)]
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

    pub(crate) fn ni_rgsw_cts_main_seed<R: NewWithSeed<Seed = S> + RandomFill<S>>(&self) -> S {
        let mut p_rng = R::new_with_seed(self.key_seed::<R>());
        puncture_p_rng(&mut p_rng, 1)
    }

    pub(crate) fn ni_rgsw_ct_seed_for_index<R: NewWithSeed<Seed = S> + RandomFill<S>>(
        &self,
        lwe_index: usize,
    ) -> S {
        let mut p_rng = R::new_with_seed(self.ni_rgsw_cts_main_seed::<R>());
        puncture_p_rng(&mut p_rng, lwe_index + 1)
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

        // PBS perform two operations at runtime: RLWE x RGW and RLWE auto. Since the
        // operations are performed serially same scratch space can be used for both.
        // Hence we create scratch space that contains maximum amount of rows that
        // suffices for RLWE x RGSW and RLWE auto
        let decomposition_matrix = M::zeros(
            std::cmp::max(
                rlwe_x_rgsw_scratch_rows(parameters.rlwe_by_rgsw_decomposition_params()),
                rlwe_auto_scratch_rows(parameters.auto_decomposition_param()),
            ),
            parameters.rlwe_n().0,
        );

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
    T: PrimInt + NumInfo,
{
    type Element = T;

    fn qby4(&self) -> Self::Element {
        if self.is_native() {
            T::one() << ((T::BITS as usize) - 2)
        } else {
            self.q().unwrap() >> 2
        }
    }
    /// Q/8
    fn true_el(&self) -> Self::Element {
        if self.is_native() {
            T::one() << ((T::BITS as usize) - 3)
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

fn produce_rgsw_ciphertext_from_ni_rgsw<
    M: MatrixMut + MatrixEntity,
    D: RlweDecomposer,
    ModOp: VectorOps<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement>,
>(
    ni_rgsw_ct: &M,
    aggregated_decomposed_ni_rgsw_zero_encs: &[M],
    decomposed_neg_ais: &[M],
    decomposer: &D,
    parameters: &BoolParameters<M::MatElement>,
    uj_to_s_ksk: (&M, &M),
    rlwe_modop: &ModOp,
    nttop: &NttOp,
    out_eval: bool,
) -> M
where
    <M as Matrix>::R: RowMut + Clone,
{
    let max_decomposer =
        if decomposer.a().decomposition_count().0 > decomposer.b().decomposition_count().0 {
            decomposer.a()
        } else {
            decomposer.b()
        };

    assert!(
        ni_rgsw_ct.dimension()
            == (
                max_decomposer.decomposition_count().0,
                parameters.rlwe_n().0
            )
    );
    assert!(
        aggregated_decomposed_ni_rgsw_zero_encs.len() == decomposer.a().decomposition_count().0,
    );
    assert!(decomposed_neg_ais.len() == decomposer.b().decomposition_count().0);

    let mut rgsw_i = M::zeros(
        decomposer.a().decomposition_count().0 * 2 + decomposer.b().decomposition_count().0 * 2,
        parameters.rlwe_n().0,
    );
    let (rlwe_dash_nsm, rlwe_dash_m) =
        rgsw_i.split_at_row_mut(decomposer.a().decomposition_count().0 * 2);

    // RLWE'_{s}(-sm)
    // Key switch `s * a_{i, l} + e` using ksk(u_j -> s) to produce RLWE(s *
    // u_{j=user_id} * a_{i, l}).
    //
    // Then set RLWE_{s}(-s B^i m) = (0, u_{j=user_id} * a_{i, l} + e + B^i m) +
    // RLWE(s * u_{j=user_id} * a_{i, l})
    {
        let (rlwe_dash_nsm_parta, rlwe_dash_nsm_partb) =
            rlwe_dash_nsm.split_at_mut(decomposer.a().decomposition_count().0);
        izip!(
            rlwe_dash_nsm_parta.iter_mut(),
            rlwe_dash_nsm_partb.iter_mut(),
            ni_rgsw_ct.iter_rows().skip(
                max_decomposer.decomposition_count().0 - decomposer.a().decomposition_count().0
            ),
            aggregated_decomposed_ni_rgsw_zero_encs.iter()
        )
        .for_each(|(rlwe_a, rlwe_b, ni_rlwe_ct, decomp_zero_enc)| {
            // KS(s * a_{i, l} + e) = RLWE(s * u_j *
            // a_{i, l}) using user j's Ksk
            izip!(
                decomp_zero_enc.iter_rows(),
                uj_to_s_ksk.0.iter_rows(),
                uj_to_s_ksk.1.iter_rows()
            )
            .for_each(|(c, pb, pa)| {
                rlwe_modop.elwise_fma_mut(rlwe_b.as_mut(), pb.as_ref(), c.as_ref());
                rlwe_modop.elwise_fma_mut(rlwe_a.as_mut(), pa.as_ref(), c.as_ref());
            });

            // RLWE(-s beta^i m) = (0, u_j * a_{j, l} +
            // e + beta^i m) + RLWE(s * u_j * a_{i, l})
            if out_eval {
                let mut ni_rlwe_ct = ni_rlwe_ct.clone();
                nttop.forward(ni_rlwe_ct.as_mut());
                rlwe_modop.elwise_add_mut(rlwe_a.as_mut(), ni_rlwe_ct.as_ref());
            } else {
                nttop.backward(rlwe_a.as_mut());
                nttop.backward(rlwe_b.as_mut());
                rlwe_modop.elwise_add_mut(rlwe_a.as_mut(), ni_rlwe_ct.as_ref());
            }
        });
    }

    // RLWE'_{s}(m)
    {
        let (rlwe_dash_m_parta, rlwe_dash_partb) =
            rlwe_dash_m.split_at_mut(decomposer.b().decomposition_count().0);
        izip!(
            rlwe_dash_m_parta.iter_mut(),
            rlwe_dash_partb.iter_mut(),
            ni_rgsw_ct.iter_rows().skip(
                max_decomposer.decomposition_count().0 - decomposer.b().decomposition_count().0
            ),
            decomposed_neg_ais.iter()
        )
        .for_each(|(rlwe_a, rlwe_b, ni_rlwe_ct, decomp_neg_ai)| {
            // KS(-a_{i, l}) = RLWE(u_i * -a_{i,l}) using user j's Ksk
            izip!(
                decomp_neg_ai.iter_rows(),
                uj_to_s_ksk.0.iter_rows(),
                uj_to_s_ksk.1.iter_rows()
            )
            .for_each(|(c, pb, pa)| {
                rlwe_modop.elwise_fma_mut(rlwe_b.as_mut(), pb.as_ref(), c.as_ref());
                rlwe_modop.elwise_fma_mut(rlwe_a.as_mut(), pa.as_ref(), c.as_ref());
            });

            // RLWE_{s}(beta^i m) = (u_j * a_{i, l} + e + beta^i m, 0) -
            // RLWE(-a_{i, l} u_j)
            if out_eval {
                let mut ni_rlwe_ct = ni_rlwe_ct.clone();
                nttop.forward(ni_rlwe_ct.as_mut());
                rlwe_modop.elwise_add_mut(rlwe_b.as_mut(), ni_rlwe_ct.as_ref());
            } else {
                nttop.backward(rlwe_a.as_mut());
                nttop.backward(rlwe_b.as_mut());
                rlwe_modop.elwise_add_mut(rlwe_b.as_mut(), ni_rlwe_ct.as_ref());
            }
        });
    }

    rgsw_i
}

/// Assigns user with user_id segement of LWE secret indices for which they
/// generate RGSW(X^{s[i]}) as the leader (i.e. for RLWExRGSW). If returned
/// tuple is (start, end), user's segment is [start, end)
pub(super) fn multi_party_user_id_lwe_segment(
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
                let mut gk = M::zeros(
                    self.pbs_info.auto_decomposer.decomposition_count().0,
                    rlwe_n,
                );
                seeded_auto_key_gen(
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
                self.pbs_info.rlwe_rgsw_decomposer.0.decomposition_count().0,
                self.pbs_info.rlwe_rgsw_decomposer.1.decomposition_count().0,
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
            let lwe_ksk = seeded_lwe_ksk_keygen(
                &sk_rlwe,
                &sk_lwe,
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

    pub(super) fn gen_interactive_multi_party_server_key_share<
        K: InteractiveMultiPartyClientKey<Element = i32>,
    >(
        &self,
        user_id: usize,
        total_users: usize,
        cr_seed: &InteractiveMultiPartyCrs<[u8; 32]>,
        collective_pk: &M,
        client_key: &K,
    ) -> CommonReferenceSeededInteractiveMultiPartyServerKeyShare<
        M,
        BoolParameters<M::MatElement>,
        InteractiveMultiPartyCrs<[u8; 32]>,
    > {
        assert_eq!(
            self.parameters().variant(),
            &ParameterVariant::InteractiveMultiParty
        );
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

            let (segment_start, segment_end) =
                multi_party_user_id_lwe_segment(user_id, total_users, self.pbs_info().lwe_n());

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
                    let mut out_rgsw = M::zeros(rlrg_d_a.0 * 2 + rlrg_d_b.0 * 2, ring_size);
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
                    let mut out_rgsw = M::zeros(rgrg_d_a.0 * 2 + rgrg_d_b.0 * 2, ring_size);
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

        CommonReferenceSeededInteractiveMultiPartyServerKeyShare::new(
            self_leader_rgsws,
            not_self_leader_rgsws,
            auto_keys,
            lwe_ksk,
            cr_seed.clone(),
            self.pbs_info.parameters.clone(),
            user_id,
        )
    }

    pub(super) fn aggregate_interactive_multi_party_server_key_shares<S>(
        &self,
        shares: &[CommonReferenceSeededInteractiveMultiPartyServerKeyShare<
            M,
            BoolParameters<M::MatElement>,
            InteractiveMultiPartyCrs<S>,
        >],
    ) -> SeededInteractiveMultiPartyServerKey<
        M,
        InteractiveMultiPartyCrs<S>,
        BoolParameters<M::MatElement>,
    >
    where
        S: PartialEq + Clone,
        M: Clone,
    {
        assert_eq!(
            self.parameters().variant(),
            &ParameterVariant::InteractiveMultiParty
        );
        assert!(shares.len() > 0);

        let total_users = shares.len();

        let parameters = shares[0].parameters().clone();
        let cr_seed = shares[0].cr_seed();

        let rlwe_n = parameters.rlwe_n().0;

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
            let rgsw_x_rgsw_decomposer =
                parameters.rgsw_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();
            let rlwe_x_rgsw_decomposer = self.pbs_info().rlwe_rgsw_decomposer();
            let rgsw_x_rgsw_dimension = (
                rgsw_x_rgsw_decomposer.a().decomposition_count().0 * 2
                    + rgsw_x_rgsw_decomposer.b().decomposition_count().0 * 2,
                rlwe_n,
            );
            let rlwe_x_rgsw_dimension = (
                rlwe_x_rgsw_decomposer.a().decomposition_count().0 * 2
                    + rlwe_x_rgsw_decomposer.b().decomposition_count().0 * 2,
                rlwe_n,
            );

            let mut rgsw_x_rgsw_scratch = M::zeros(
                rgsw_x_rgsw_scratch_rows(rlwe_x_rgsw_decomposer, &rgsw_x_rgsw_decomposer),
                rlwe_n,
            );

            let shares_in_correct_order = (0..total_users)
                .map(|i| shares.iter().find(|s| s.user_id() == i).unwrap())
                .collect_vec();

            let lwe_n = self.parameters().lwe_n().0;
            let (users_segments, users_segments_sizes): (Vec<(usize, usize)>, Vec<usize>) = (0
                ..total_users)
                .map(|user_id| {
                    let (start_index, end_index) =
                        multi_party_user_id_lwe_segment(user_id, total_users, lwe_n);
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
                                    &mut RgswCiphertextMutRef::new(
                                        rgsw_i.as_mut(),
                                        rlwe_x_rgsw_decomposer.a().decomposition_count().0,
                                        rlwe_x_rgsw_decomposer.b().decomposition_count().0,
                                    ),
                                    &RgswCiphertextRef::new(
                                        other_rgsw_i.as_ref(),
                                        rgsw_x_rgsw_decomposer.a().decomposition_count().0,
                                        rgsw_x_rgsw_decomposer.b().decomposition_count().0,
                                    ),
                                    rlwe_x_rgsw_decomposer,
                                    &rgsw_x_rgsw_decomposer,
                                    &mut RuntimeScratchMutRef::new(rgsw_x_rgsw_scratch.as_mut()),
                                    rlweq_nttop,
                                    rlweq_modop,
                                );
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

        SeededInteractiveMultiPartyServerKey::new(
            rgsw_cts,
            auto_keys,
            lwe_ksk,
            cr_seed.clone(),
            parameters,
        )
    }

    pub(super) fn aggregate_non_interactive_multi_party_server_key_shares(
        &self,
        cr_seed: &NonInteractiveMultiPartyCrs<[u8; 32]>,
        key_shares: &[CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<
            M,
            BoolParameters<M::MatElement>,
            NonInteractiveMultiPartyCrs<[u8; 32]>,
        >],
    ) -> SeededNonInteractiveMultiPartyServerKey<
        M,
        NonInteractiveMultiPartyCrs<[u8; 32]>,
        BoolParameters<M::MatElement>,
    >
    where
        M: Clone + Debug,
        <M as Matrix>::R: RowMut + Clone,
    {
        assert_eq!(
            self.parameters().variant(),
            &ParameterVariant::NonInteractiveMultiParty
        );

        let total_users = key_shares.len();
        let key_shares = (0..total_users)
            .map(|user_id| {
                // find share of user_id
                key_shares
                    .iter()
                    .find(|share| share.user_index() == user_id)
                    .expect(&format!("Key Share for user_id={user_id} missing"))
            })
            .collect_vec();

        // check parameters and cr seed are equal
        {
            key_shares.iter().for_each(|k| {
                assert!(k.parameters() == self.parameters());
                assert!(k.cr_seed() == cr_seed);
            });
        }

        let rlwe_modop = &self.pbs_info().rlwe_modop;
        let nttop = &self.pbs_info().rlwe_nttop;
        let ring_size = self.parameters().rlwe_n().0;
        let rlwe_q = self.parameters().rlwe_q();
        let lwe_modop = self.pbs_info().modop_lweq();

        // Generate Key switching key from u_j to s, where u_j user j's RLWE secret and
        // s is the ideal RLWE secret.
        //
        // User j gives [s_j * a_{i, j}  + e  + \beta^i u_j] where a_{i, j} is user j
        // specific publicly know polynomial sampled from user j's pseudo random seed
        // defined in the protocol.
        //
        // User k, k != j, gives [s_k * a_{i, j} + e].
        //
        // We set Ksk(u_j -> s) = [s_j * a_{i, j}  + e  + \beta^i u_j + \sum_{k \in P, k
        // != j} s_k * a_{i, j} + e]
        let ni_uj_to_s_decomposer = self
            .parameters()
            .non_interactive_ui_to_s_key_switch_decomposer::<DefaultDecomposer<M::MatElement>>();
        let mut uj_to_s_ksks = key_shares
            .iter()
            .map(|share| {
                let mut useri_ui_to_s_ksk = share.ui_to_s_ksk().clone();
                assert!(
                    useri_ui_to_s_ksk.dimension()
                        == (ni_uj_to_s_decomposer.decomposition_count().0, ring_size)
                );
                key_shares
                    .iter()
                    .filter(|x| x.user_index() != share.user_index())
                    .for_each(|other_share| {
                        let op2 = other_share.ui_to_s_ksk_zero_encs_for_user_i(share.user_index());
                        assert!(
                            op2.dimension()
                                == (ni_uj_to_s_decomposer.decomposition_count().0, ring_size)
                        );
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
            // Send u_j -> s ksk in evaluation domain and sample corresponding a's using
            // user j's ksk seed to prepare for upcoming key switches
            uj_to_s_ksks.iter_mut().for_each(|ksk_i| {
                ksk_i
                    .iter_rows_mut()
                    .for_each(|r| nttop.forward(r.as_mut()))
            });
            let uj_to_s_ksks_part_a_eval = key_shares
                .iter()
                .map(|share| {
                    let mut ksk_prng = DefaultSecureRng::new_seeded(
                        cr_seed.ui_to_s_ks_seed_for_user_i::<DefaultSecureRng>(share.user_index()),
                    );
                    let mut ais =
                        M::zeros(ni_uj_to_s_decomposer.decomposition_count().0, ring_size);

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

            let rgsw_x_rgsw_decomposer = self
                .parameters()
                .rgsw_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();
            let rlwe_x_rgsw_decomposer = self
                .parameters()
                .rlwe_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();

            let d_max = if rgsw_x_rgsw_decomposer.a().decomposition_count().0
                > rgsw_x_rgsw_decomposer.b().decomposition_count().0
            {
                rgsw_x_rgsw_decomposer.a().decomposition_count().0
            } else {
                rgsw_x_rgsw_decomposer.b().decomposition_count().0
            };

            let mut scratch_rgsw_x_rgsw = M::zeros(
                rgsw_x_rgsw_scratch_rows(&rlwe_x_rgsw_decomposer, &rgsw_x_rgsw_decomposer),
                self.parameters().rlwe_n().0,
            );

            // Recall that given u_j * a_{i, l} + e + \beta^i X^{s_j[l]} from user j
            //
            // We generate:
            //
            // - RLWE(-s \beta^i X^{s_j[l]}) = KS_{u_j -> s}(a_{i, l} * s + e) + (0 , u_j *
            //   a_{i, l} + e + \beta^i X^{s_j[l]}), where KS_{u_j -> s}(a_{i, l} * s + e) =
            //   RLWE_s(a_{i, l} * u_j)
            // - RLWE(\beta^i X^{s_j[l]}) = KS_{u_j -> s}(-a_{i,l}) + (u_j * a_{i, l} + e +
            //   \beta^i X^{s_j[l]}, 0), where KS_{u_j -> s}(-a_{i,l}) = RLWE_s(-a_{i,l} *
            //   u_j)
            //
            // a_{i, l} * s + e = \sum_{j \in P} a_{i, l} * s_{j} + e
            let user_segments = (0..total_users)
                .map(|user_id| {
                    multi_party_user_id_lwe_segment(
                        user_id,
                        total_users,
                        self.parameters().lwe_n().0,
                    )
                })
                .collect_vec();
            // Note: Each user is assigned a contigous LWE segement and the LWE dimension is
            // split approximately uniformly across all users. Hence, concatenation of all
            // user specific lwe segments will give LWE dimension.
            let rgsw_cts = user_segments
                .into_iter()
                .enumerate()
                .flat_map(|(user_id, lwe_segment)| {
                    (lwe_segment.0..lwe_segment.1)
                        .into_iter()
                        .map(|lwe_index| {
                            // We sample d_b `-a_i`s to key switch and generate RLWE'(m). But before
                            // we sampling we need to puncture a_prng d_max - d_b times to align
                            // a_i's. After sampling we decompose `-a_i`s and send them to
                            // evaluation domain for upcoming key switches.
                            let mut a_prng = DefaultSecureRng::new_seeded(
                                cr_seed.ni_rgsw_ct_seed_for_index::<DefaultSecureRng>(lwe_index),
                            );

                            let mut scratch = M::R::zeros(self.parameters().rlwe_n().0);
                            (0..d_max - rgsw_x_rgsw_decomposer.b().decomposition_count().0)
                                .for_each(|_| {
                                    RandomFillUniformInModulus::random_fill(
                                        &mut a_prng,
                                        rlwe_q,
                                        scratch.as_mut(),
                                    );
                                });

                            let decomp_neg_ais = (0..rgsw_x_rgsw_decomposer
                                .b()
                                .decomposition_count()
                                .0)
                                .map(|_| {
                                    RandomFillUniformInModulus::random_fill(
                                        &mut a_prng,
                                        rlwe_q,
                                        scratch.as_mut(),
                                    );
                                    rlwe_modop.elwise_neg_mut(scratch.as_mut());

                                    let mut decomp_neg_ai = M::zeros(
                                        ni_uj_to_s_decomposer.decomposition_count().0,
                                        self.parameters().rlwe_n().0,
                                    );
                                    scratch.as_ref().iter().enumerate().for_each(|(index, el)| {
                                        ni_uj_to_s_decomposer
                                            .decompose_iter(el)
                                            .enumerate()
                                            .for_each(|(row_j, d_el)| {
                                                (decomp_neg_ai.as_mut()[row_j]).as_mut()[index] =
                                                    d_el;
                                            });
                                    });

                                    decomp_neg_ai
                                        .iter_rows_mut()
                                        .for_each(|ri| nttop.forward(ri.as_mut()));

                                    decomp_neg_ai
                                })
                                .collect_vec();

                            // Aggregate zero encryptions to produce a_{i, l} * s + e =
                            // \sum_{k in P} a_{i, l} * s_{k} + e where s is the ideal RLWE
                            // secret. Aggregated  a_{i, l} * s + e are key switched using
                            // Ksk(u_j -> s) to produce RLWE_{s}(a_{i, l} * s) which are
                            // then use to produce RLWE'(-sX^{s_{lwe}[l]}).
                            // Hence, after aggregation we decompose a_{i, l} * s + e to
                            // prepare for key switching
                            let ni_rgsw_zero_encs = (0..rgsw_x_rgsw_decomposer
                                .a()
                                .decomposition_count()
                                .0)
                                .map(|i| {
                                    let mut sum = M::R::zeros(self.parameters().rlwe_n().0);
                                    key_shares.iter().for_each(|k| {
                                        let to_add_ref = k
                                            .ni_rgsw_zero_enc_for_lwe_index(lwe_index)
                                            .get_row_slice(i);
                                        assert!(to_add_ref.len() == self.parameters().rlwe_n().0);
                                        rlwe_modop.elwise_add_mut(sum.as_mut(), to_add_ref);
                                    });

                                    // decompose
                                    let mut decomp_sum = M::zeros(
                                        ni_uj_to_s_decomposer.decomposition_count().0,
                                        self.parameters().rlwe_n().0,
                                    );
                                    sum.as_ref().iter().enumerate().for_each(|(index, el)| {
                                        ni_uj_to_s_decomposer
                                            .decompose_iter(el)
                                            .enumerate()
                                            .for_each(|(row_j, d_el)| {
                                                (decomp_sum.as_mut()[row_j]).as_mut()[index] = d_el;
                                            });
                                    });

                                    decomp_sum
                                        .iter_rows_mut()
                                        .for_each(|r| nttop.forward(r.as_mut()));

                                    decomp_sum
                                })
                                .collect_vec();

                            // Produce RGSW(X^{s_{j=user_id, lwe}[l]}) for the
                            // leader, ie user's id = user_id.
                            // Recall leader's RGSW ciphertext must be constructed
                            // for RLWE x RGSW multiplication, and is then used
                            // to accumulate, using RGSW x RGSW multiplication,
                            // X^{s_{j != user_id, lwe}[l]} from other users.
                            let mut rgsw_i = produce_rgsw_ciphertext_from_ni_rgsw(
                                key_shares[user_id]
                                    .ni_rgsw_cts_for_self_leader_lwe_index(lwe_index),
                                &ni_rgsw_zero_encs[rgsw_x_rgsw_decomposer
                                    .a()
                                    .decomposition_count()
                                    .0
                                    - rlwe_x_rgsw_decomposer.a().decomposition_count().0..],
                                &decomp_neg_ais[rgsw_x_rgsw_decomposer
                                    .b()
                                    .decomposition_count()
                                    .0
                                    - rlwe_x_rgsw_decomposer.b().decomposition_count().0..],
                                &rlwe_x_rgsw_decomposer,
                                self.parameters(),
                                (&uj_to_s_ksks[user_id], &uj_to_s_ksks_part_a_eval[user_id]),
                                rlwe_modop,
                                nttop,
                                false,
                            );

                            // RGSW for lwe_index of users that are not leader.
                            //
                            // Recall that for users that are not leader for the
                            // lwe_index we require to produce RGSW ciphertext for
                            // RGSW x RGSW product
                            (0..total_users)
                                .filter(|i| *i != user_id)
                                .for_each(|other_user_id| {
                                    let mut other_rgsw_i = produce_rgsw_ciphertext_from_ni_rgsw(
                                        key_shares[other_user_id]
                                            .ni_rgsw_cts_for_self_not_leader_lwe_index(lwe_index),
                                        &ni_rgsw_zero_encs,
                                        &decomp_neg_ais,
                                        &rgsw_x_rgsw_decomposer,
                                        self.parameters(),
                                        (
                                            &uj_to_s_ksks[other_user_id],
                                            &uj_to_s_ksks_part_a_eval[other_user_id],
                                        ),
                                        rlwe_modop,
                                        nttop,
                                        true,
                                    );

                                    rgsw_by_rgsw_inplace(
                                        &mut RgswCiphertextMutRef::new(
                                            rgsw_i.as_mut(),
                                            rlwe_x_rgsw_decomposer.a().decomposition_count().0,
                                            rlwe_x_rgsw_decomposer.b().decomposition_count().0,
                                        ),
                                        &RgswCiphertextRef::new(
                                            other_rgsw_i.as_ref(),
                                            rgsw_x_rgsw_decomposer.a().decomposition_count().0,
                                            rgsw_x_rgsw_decomposer.b().decomposition_count().0,
                                        ),
                                        &rlwe_x_rgsw_decomposer,
                                        &rgsw_x_rgsw_decomposer,
                                        &mut RuntimeScratchMutRef::new(
                                            scratch_rgsw_x_rgsw.as_mut(),
                                        ),
                                        nttop,
                                        rlwe_modop,
                                    )
                                });

                            rgsw_i
                        })
                        .collect_vec()
                })
                .collect_vec();

            // put u_j to s ksk in coefficient domain
            uj_to_s_ksks.iter_mut().for_each(|ksk_i| {
                ksk_i
                    .iter_rows_mut()
                    .for_each(|r| nttop.backward(r.as_mut()))
            });

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
            uj_to_s_ksks,
            rgsw_cts,
            auto_keys,
            lwe_ksk,
            cr_seed.clone(),
            self.parameters().clone(),
        )
    }

    pub(super) fn gen_non_interactive_multi_party_key_share<
        K: NonInteractiveMultiPartyClientKey<Element = i32>,
    >(
        &self,
        cr_seed: &NonInteractiveMultiPartyCrs<[u8; 32]>,
        self_index: usize,
        total_users: usize,
        client_key: &K,
    ) -> CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<
        M,
        BoolParameters<M::MatElement>,
        NonInteractiveMultiPartyCrs<[u8; 32]>,
    > {
        assert_eq!(
            self.parameters().variant(),
            &ParameterVariant::NonInteractiveMultiParty
        );

        // TODO:  check whether parameters support `total_users`
        let nttop = self.pbs_info().nttop_rlweq();
        let rlwe_modop = self.pbs_info().modop_rlweq();
        // let ring_size = self.pbs_info().rlwe_n();
        let rlwe_q = self.parameters().rlwe_q();

        let sk_rlwe = client_key.sk_rlwe();
        let sk_u_rlwe = client_key.sk_u_rlwe();
        let sk_lwe = client_key.sk_lwe();

        let (ui_to_s_ksk, ksk_zero_encs_for_others) = DefaultSecureRng::with_local_mut(|rng| {
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

        // Non-interactive RGSW cts
        let (ni_rgsw_zero_encs, self_leader_ni_rgsw_cts, not_self_leader_rgsw_cts) = {
            let rgsw_x_rgsw_decomposer = self
                .parameters()
                .rgsw_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();
            let rlwe_x_rgsw_decomposer = self
                .parameters()
                .rlwe_rgsw_decomposer::<DefaultDecomposer<M::MatElement>>();

            // We assume that d_{a/b} for RGSW x RGSW are always < d'_{a/b} for RLWE x RGSW
            assert!(
                rlwe_x_rgsw_decomposer.a().decomposition_count().0
                    < rgsw_x_rgsw_decomposer.a().decomposition_count().0
            );
            assert!(
                rlwe_x_rgsw_decomposer.b().decomposition_count().0
                    < rgsw_x_rgsw_decomposer.b().decomposition_count().0
            );

            let sj_poly_eval = {
                let mut s = M::R::try_convert_from(&sk_rlwe, rlwe_q);
                nttop.forward(s.as_mut());
                s
            };

            let d_rgsw_a = rgsw_x_rgsw_decomposer.a().decomposition_count().0;
            let d_rgsw_b = rgsw_x_rgsw_decomposer.b().decomposition_count().0;
            let d_max = std::cmp::max(d_rgsw_a, d_rgsw_b);

            // Zero encyptions for each LWE index. We generate d_a zero encryptions for each
            // LWE index using a_{i, l} with i \in {d_max - d_a , d_max) and l = lwe_index
            let zero_encs = {
                (0..self.parameters().lwe_n().0)
                    .map(|lwe_index| {
                        let mut p_rng = DefaultSecureRng::new_seeded(
                            cr_seed.ni_rgsw_ct_seed_for_index::<DefaultSecureRng>(lwe_index),
                        );

                        let mut scratch = M::R::zeros(self.parameters().rlwe_n().0);

                        // puncture seeded prng d_max - d_a times
                        (0..(d_max - d_rgsw_a)).into_iter().for_each(|_| {
                            RandomFillUniformInModulus::random_fill(
                                &mut p_rng,
                                rlwe_q,
                                scratch.as_mut(),
                            );
                        });

                        let mut zero_enc = M::zeros(d_rgsw_a, self.parameters().rlwe_n().0);
                        zero_enc.iter_rows_mut().for_each(|out| {
                            // sample a_i
                            RandomFillUniformInModulus::random_fill(
                                &mut p_rng,
                                rlwe_q,
                                out.as_mut(),
                            );

                            // a_i * s_j
                            nttop.forward(out.as_mut());
                            rlwe_modop.elwise_mul_mut(out.as_mut(), sj_poly_eval.as_ref());
                            nttop.backward(out.as_mut());

                            // a_j * s_j + e
                            DefaultSecureRng::with_local_mut_mut(&mut |rng| {
                                RandomFillGaussianInModulus::random_fill(
                                    rng,
                                    rlwe_q,
                                    scratch.as_mut(),
                                );
                            });

                            rlwe_modop.elwise_add_mut(out.as_mut(), scratch.as_ref());
                        });

                        zero_enc
                    })
                    .collect_vec()
            };

            let uj_poly_eval = {
                let mut u = M::R::try_convert_from(&sk_u_rlwe, rlwe_q);
                nttop.forward(u.as_mut());
                u
            };

            // Generate non-interactive RGSW ciphertexts a_{i, l} u_j + e + \beta X^{s_j[l]}
            // for i \in (0, d_max]
            let (self_start_index, self_end_index) = multi_party_user_id_lwe_segment(
                self_index,
                total_users,
                self.parameters().lwe_n().0,
            );

            // For LWE indices [self_start_index, self_end_index) user generates
            // non-interactive RGSW cts for RLWE x RGSW product. We refer to
            // such indices as where user is the leader. For the rest of
            // the indices user generates non-interactive RGWS cts for RGSW x
            // RGSW multiplication. We refer to such indices as where user is
            // not the leader.
            let self_leader_ni_rgsw_cts = {
                let max_rlwe_x_rgsw_decomposer =
                    if rlwe_x_rgsw_decomposer.a().decomposition_count().0
                        > rlwe_x_rgsw_decomposer.b().decomposition_count().0
                    {
                        rlwe_x_rgsw_decomposer.a()
                    } else {
                        rlwe_x_rgsw_decomposer.b()
                    };

                let gadget_vec = max_rlwe_x_rgsw_decomposer.gadget_vector();

                (self_start_index..self_end_index)
                    .map(|lwe_index| {
                        let mut p_rng = DefaultSecureRng::new_seeded(
                            cr_seed.ni_rgsw_ct_seed_for_index::<DefaultSecureRng>(lwe_index),
                        );

                        // puncture p_rng d_max - d'_max time to align with `a_{i, l}`s used to
                        // produce RGSW cts for RGSW x RGSW
                        let mut scratch = M::R::zeros(self.parameters().rlwe_n().0);
                        (0..(d_max - max_rlwe_x_rgsw_decomposer.decomposition_count().0))
                            .into_iter()
                            .for_each(|_| {
                                RandomFillUniformInModulus::random_fill(
                                    &mut p_rng,
                                    rlwe_q,
                                    scratch.as_mut(),
                                );
                            });

                        let mut ni_rgsw_cts = M::zeros(
                            max_rlwe_x_rgsw_decomposer.decomposition_count().0,
                            self.parameters().rlwe_n().0,
                        );

                        // X^{s_{j, lwe}[l]}
                        let m_poly = encode_x_pow_si_with_emebedding_factor::<M::R, _>(
                            sk_lwe[lwe_index],
                            self.pbs_info().embedding_factor(),
                            self.parameters().rlwe_n().0,
                            rlwe_q,
                        );

                        izip!(ni_rgsw_cts.iter_rows_mut(), gadget_vec.iter()).for_each(
                            |(out, beta)| {
                                // sample a_i
                                RandomFillUniformInModulus::random_fill(
                                    &mut p_rng,
                                    rlwe_q,
                                    out.as_mut(),
                                );

                                // u_j * a_i
                                nttop.forward(out.as_mut());
                                rlwe_modop.elwise_mul_mut(out.as_mut(), uj_poly_eval.as_ref());
                                nttop.backward(out.as_mut());

                                // u_j + a_i + e
                                DefaultSecureRng::with_local_mut_mut(&mut |rng| {
                                    RandomFillGaussianInModulus::random_fill(
                                        rng,
                                        rlwe_q,
                                        scratch.as_mut(),
                                    );
                                });
                                rlwe_modop.elwise_add_mut(out.as_mut(), scratch.as_ref());

                                // u_j + a_i + e + beta m
                                rlwe_modop.elwise_scalar_mul(
                                    scratch.as_mut(),
                                    m_poly.as_ref(),
                                    beta,
                                );
                                rlwe_modop.elwise_add_mut(out.as_mut(), scratch.as_ref());
                            },
                        );

                        ni_rgsw_cts
                    })
                    .collect_vec()
            };

            let not_self_leader_rgsw_cts = {
                let max_rgsw_x_rgsw_decomposer =
                    if rgsw_x_rgsw_decomposer.a().decomposition_count().0
                        > rgsw_x_rgsw_decomposer.b().decomposition_count().0
                    {
                        rgsw_x_rgsw_decomposer.a()
                    } else {
                        rgsw_x_rgsw_decomposer.b()
                    };
                let gadget_vec = max_rgsw_x_rgsw_decomposer.gadget_vector();

                ((0..self_start_index).chain(self_end_index..self.parameters().lwe_n().0))
                    .map(|lwe_index| {
                        let mut p_rng = DefaultSecureRng::new_seeded(
                            cr_seed.ni_rgsw_ct_seed_for_index::<DefaultSecureRng>(lwe_index),
                        );
                        let mut ni_rgsw_cts = M::zeros(
                            max_rgsw_x_rgsw_decomposer.decomposition_count().0,
                            self.parameters().rlwe_n().0,
                        );
                        let mut scratch = M::R::zeros(self.parameters().rlwe_n().0);

                        // X^{s_{j, lwe}[l]}
                        let m_poly = encode_x_pow_si_with_emebedding_factor::<M::R, _>(
                            sk_lwe[lwe_index],
                            self.pbs_info().embedding_factor(),
                            self.parameters().rlwe_n().0,
                            rlwe_q,
                        );

                        izip!(ni_rgsw_cts.iter_rows_mut(), gadget_vec.iter()).for_each(
                            |(out, beta)| {
                                // sample a_i
                                RandomFillUniformInModulus::random_fill(
                                    &mut p_rng,
                                    rlwe_q,
                                    out.as_mut(),
                                );

                                // u_j * a_i
                                nttop.forward(out.as_mut());
                                rlwe_modop.elwise_mul_mut(out.as_mut(), uj_poly_eval.as_ref());
                                nttop.backward(out.as_mut());

                                // u_j + a_i + e
                                DefaultSecureRng::with_local_mut_mut(&mut |rng| {
                                    RandomFillGaussianInModulus::random_fill(
                                        rng,
                                        rlwe_q,
                                        scratch.as_mut(),
                                    );
                                });
                                rlwe_modop.elwise_add_mut(out.as_mut(), scratch.as_ref());

                                // u_j * a_i + e + beta m
                                rlwe_modop.elwise_scalar_mul(
                                    scratch.as_mut(),
                                    m_poly.as_ref(),
                                    beta,
                                );
                                rlwe_modop.elwise_add_mut(out.as_mut(), scratch.as_ref());
                            },
                        );

                        ni_rgsw_cts
                    })
                    .collect_vec()
            };

            (zero_encs, self_leader_ni_rgsw_cts, not_self_leader_rgsw_cts)
        };

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
            self_leader_ni_rgsw_cts,
            not_self_leader_rgsw_cts,
            ni_rgsw_zero_encs,
            ui_to_s_ksk,
            ksk_zero_encs_for_others,
            auto_keys_share,
            lwe_ksk_share,
            self_index,
            total_users,
            self.parameters().lwe_n().0,
            cr_seed.clone(),
            self.parameters().clone(),
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
                    self.pbs_info.auto_decomposer.decomposition_count().0,
                    ring_size,
                );
                seeded_auto_key_gen(
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
            let lwe_modop = &self.pbs_info.lwe_modop;
            let d_lwe_gadget_vec = self.pbs_info.lwe_decomposer.gadget_vector();
            seeded_lwe_ksk_keygen(
                sk_rlwe,
                sk_lwe,
                &d_lwe_gadget_vec,
                lwe_modop,
                &mut p_rng,
                rng,
            )
        })
    }

    pub(super) fn multi_party_public_key_share<K: InteractiveMultiPartyClientKey<Element = i32>>(
        &self,
        cr_seed: &InteractiveMultiPartyCrs<[u8; 32]>,
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
            encrypt_lwe(&m, &client_key.sk_rlwe(), &self.pbs_info.rlwe_modop, rng)
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

    fn not_inplace(&self, c0: &mut M::R) {
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

    fn not(&self, c: &Self::Ciphertext) -> Self::Ciphertext {
        let mut out = c.clone();
        self.not_inplace(&mut out);
        out
    }
}
