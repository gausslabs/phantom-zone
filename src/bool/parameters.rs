use num_traits::{ConstZero, FromPrimitive, PrimInt};
use serde::{Deserialize, Serialize};

use crate::{
    backend::Modulus,
    decomposer::{Decomposer, NumInfo},
    utils::log2,
};

pub(crate) trait DoubleDecomposerCount {
    type Count;
    fn a(&self) -> Self::Count;
    fn b(&self) -> Self::Count;
}

pub(crate) trait DoubleDecomposerParams {
    type Base;
    type Count;

    fn decomposition_base(&self) -> Self::Base;
    fn decomposition_count_a(&self) -> Self::Count;
    fn decomposition_count_b(&self) -> Self::Count;
}

pub(crate) trait SingleDecomposerParams {
    type Base;
    type Count;

    // fn new(base: Self::Base, count: Self::Count) -> Self;
    fn decomposition_base(&self) -> Self::Base;
    fn decomposition_count(&self) -> Self::Count;
}

impl DoubleDecomposerParams
    for (
        DecompostionLogBase,
        // Assume (Decomposition count for A, Decomposition count for B)
        (DecompositionCount, DecompositionCount),
    )
{
    type Base = DecompostionLogBase;
    type Count = DecompositionCount;

    // fn new(
    //     base: DecompostionLogBase,
    //     count_a: DecompositionCount,
    //     count_b: DecompositionCount,
    // ) -> Self {
    //     (base, (count_a, count_b))
    // }

    fn decomposition_base(&self) -> Self::Base {
        self.0
    }

    fn decomposition_count_a(&self) -> Self::Count {
        self.1 .0
    }

    fn decomposition_count_b(&self) -> Self::Count {
        self.1 .1
    }
}

impl SingleDecomposerParams for (DecompostionLogBase, DecompositionCount) {
    type Base = DecompostionLogBase;
    type Count = DecompositionCount;

    // fn new(base: DecompostionLogBase, count: DecompositionCount) -> Self {
    //     (base, count)
    // }

    fn decomposition_base(&self) -> Self::Base {
        self.0
    }

    fn decomposition_count(&self) -> Self::Count {
        self.1
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub(crate) enum SecretKeyDistribution {
    /// Elements of secret key are sample from Gaussian distribitution with
    /// \sigma = 3.19 and \mu = 0.0
    ErrorDistribution,
    /// Elements of secret key are chosen from the set {1,0,-1} with hamming
    /// weight `floor(N/2)` where `N` is the secret dimension.
    TernaryDistribution,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub(crate) enum ParameterVariant {
    SingleParty,
    InteractiveMultiParty,
    NonInteractiveMultiParty,
}
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BoolParameters<El> {
    /// RLWE secret key distribution
    rlwe_secret_key_dist: SecretKeyDistribution,
    /// LWE secret key distribtuion
    lwe_secret_key_dist: SecretKeyDistribution,
    /// RLWE ciphertext modulus Q
    rlwe_q: CiphertextModulus<El>,
    /// LWE ciphertext modulus q (usually referred to as Q_{ks})
    lwe_q: CiphertextModulus<El>,
    /// Blind rotation modulus. It is the modulus to which we switch before
    /// blind rotation.
    ///
    /// Since blind rotation decrypts LWE ciphertext in the exponent  of a ring
    /// polynomial, which is a ring mod 2N, blind rotation modulus is
    /// always <= 2N.
    br_q: usize,
    /// Ring dimension `N` for 2N^{th} cyclotomic polynomial ring
    rlwe_n: PolynomialSize,
    /// LWE dimension `n`
    lwe_n: LweDimension,
    /// LWE key switch decompositon params
    lwe_decomposer_params: (DecompostionLogBase, DecompositionCount),
    /// Decompostion parameters for RLWE x RGSW.
    ///
    /// We restrict decomposition for RLWE'(-sm) and RLWE'(m) to have same base
    /// but can have different decomposition count. We refer to this
    /// DoubleDecomposer / RlweDecomposer
    ///
    /// Decomposition count `d_a` (i.e. for SignedDecompose(RLWE_A(m)) x
    /// RLWE'(-sm)) and `d_b` (i.e. for SignedDecompose(RLWE_B(m)) x RLWE'(m))
    /// are always stored as `(d_a, d_b)`
    rlrg_decomposer_params: (
        DecompostionLogBase,
        (DecompositionCount, DecompositionCount),
    ),
    /// Decomposition parameters for RLWE automorphism
    auto_decomposer_params: (DecompostionLogBase, DecompositionCount),
    /// Decomposition parameters for RGSW0 x RGSW1
    ///
    /// `0` and `1` indicate that RGSW0 and RGSW1 may not use same decomposition
    /// parameters.
    ///
    /// In RGSW0 x RGSW1, decomposition parameters for RGSW1 are required.
    /// Hence, the parameters we store are decomposition parameters of RGSW1.
    ///
    /// Like RLWE x RGSW decomposition parameters (1) we restrict to same base
    /// but can have different decomposition counts `d_a` and `d_b` and (2)
    /// decomposition count `d_a` and `d_b` are always stored as `(d_a, d_b)`
    ///
    /// RGSW0 x RGSW1 are optional because they only necessary to be supplied in
    /// multi-party setting.
    rgrg_decomposer_params: Option<(
        DecompostionLogBase,
        (DecompositionCount, DecompositionCount),
    )>,
    /// Decomposition parameters for non-interactive key switching from u_j to
    /// s, hwere u_j is RLWE secret `u` of party `j` and `s` is the ideal RLWE
    /// secret key.
    ///
    /// Decomposition parameters for non-interactive key switching are optional
    /// and must be supplied only for non-interactive multi-party
    non_interactive_ui_to_s_key_switch_decomposer:
        Option<(DecompostionLogBase, DecompositionCount)>,
    /// Group generator for Z^*_{br_q}
    g: usize,
    /// Window size parameter for LMKC++ blind rotation
    w: usize,
    /// Parameter variant
    variant: ParameterVariant,
}

impl<El> BoolParameters<El> {
    pub(crate) fn rlwe_secret_key_dist(&self) -> &SecretKeyDistribution {
        &self.rlwe_secret_key_dist
    }

    pub(crate) fn lwe_secret_key_dist(&self) -> &SecretKeyDistribution {
        &self.lwe_secret_key_dist
    }

    pub(crate) fn rlwe_q(&self) -> &CiphertextModulus<El> {
        &self.rlwe_q
    }

    pub(crate) fn lwe_q(&self) -> &CiphertextModulus<El> {
        &self.lwe_q
    }

    pub(crate) fn br_q(&self) -> &usize {
        &self.br_q
    }

    pub(crate) fn rlwe_n(&self) -> &PolynomialSize {
        &self.rlwe_n
    }

    pub(crate) fn lwe_n(&self) -> &LweDimension {
        &self.lwe_n
    }

    pub(crate) fn g(&self) -> usize {
        self.g
    }

    pub(crate) fn w(&self) -> usize {
        self.w
    }

    pub(crate) fn rlwe_by_rgsw_decomposition_params(
        &self,
    ) -> &(
        DecompostionLogBase,
        (DecompositionCount, DecompositionCount),
    ) {
        &self.rlrg_decomposer_params
    }

    pub(crate) fn rgsw_by_rgsw_decomposition_params(
        &self,
    ) -> (
        DecompostionLogBase,
        (DecompositionCount, DecompositionCount),
    ) {
        self.rgrg_decomposer_params.expect(&format!(
            "Parameter variant {:?} does not support RGSWxRGSW",
            self.variant
        ))
    }

    pub(crate) fn rlwe_rgsw_decomposition_base(&self) -> DecompostionLogBase {
        self.rlrg_decomposer_params.0
    }

    pub(crate) fn rlwe_rgsw_decomposition_count(&self) -> (DecompositionCount, DecompositionCount) {
        self.rlrg_decomposer_params.1
    }

    pub(crate) fn rgsw_rgsw_decomposition_count(&self) -> (DecompositionCount, DecompositionCount) {
        let params = self.rgrg_decomposer_params.expect(&format!(
            "Parameter variant {:?} does not support RGSW x RGSW",
            self.variant
        ));
        params.1
    }

    pub(crate) fn auto_decomposition_param(&self) -> &(DecompostionLogBase, DecompositionCount) {
        &self.auto_decomposer_params
    }

    pub(crate) fn auto_decomposition_base(&self) -> DecompostionLogBase {
        self.auto_decomposer_params.decomposition_base()
    }

    pub(crate) fn auto_decomposition_count(&self) -> DecompositionCount {
        self.auto_decomposer_params.decomposition_count()
    }

    pub(crate) fn lwe_decomposition_base(&self) -> DecompostionLogBase {
        self.lwe_decomposer_params.decomposition_base()
    }

    pub(crate) fn lwe_decomposition_count(&self) -> DecompositionCount {
        self.lwe_decomposer_params.decomposition_count()
    }

    pub(crate) fn non_interactive_ui_to_s_key_switch_decomposition_count(
        &self,
    ) -> DecompositionCount {
        let params = self
            .non_interactive_ui_to_s_key_switch_decomposer
            .expect(&format!(
                "Parameter variant {:?} does not support non-interactive",
                self.variant
            ));
        params.decomposition_count()
    }

    pub(crate) fn rgsw_rgsw_decomposer<D: Decomposer<Element = El>>(&self) -> (D, D)
    where
        El: Copy,
    {
        let params = self.rgrg_decomposer_params.expect(&format!(
            "Parameter variant {:?} does not support RGSW x RGSW",
            self.variant
        ));
        (
            // A
            D::new(
                self.rlwe_q.0,
                params.decomposition_base().0,
                params.decomposition_count_a().0,
            ),
            // B
            D::new(
                self.rlwe_q.0,
                params.decomposition_base().0,
                params.decomposition_count_b().0,
            ),
        )
    }

    pub(crate) fn auto_decomposer<D: Decomposer<Element = El>>(&self) -> D
    where
        El: Copy,
    {
        D::new(
            self.rlwe_q.0,
            self.auto_decomposer_params.decomposition_base().0,
            self.auto_decomposer_params.decomposition_count().0,
        )
    }

    pub(crate) fn lwe_decomposer<D: Decomposer<Element = El>>(&self) -> D
    where
        El: Copy,
    {
        D::new(
            self.lwe_q.0,
            self.lwe_decomposer_params.decomposition_base().0,
            self.lwe_decomposer_params.decomposition_count().0,
        )
    }

    pub(crate) fn rlwe_rgsw_decomposer<D: Decomposer<Element = El>>(&self) -> (D, D)
    where
        El: Copy,
    {
        (
            // A
            D::new(
                self.rlwe_q.0,
                self.rlrg_decomposer_params.decomposition_base().0,
                self.rlrg_decomposer_params.decomposition_count_a().0,
            ),
            // B
            D::new(
                self.rlwe_q.0,
                self.rlrg_decomposer_params.decomposition_base().0,
                self.rlrg_decomposer_params.decomposition_count_b().0,
            ),
        )
    }

    pub(crate) fn non_interactive_ui_to_s_key_switch_decomposer<D: Decomposer<Element = El>>(
        &self,
    ) -> D
    where
        El: Copy,
    {
        let params = self
            .non_interactive_ui_to_s_key_switch_decomposer
            .expect(&format!(
                "Parameter variant {:?} does not support non-interactive",
                self.variant
            ));
        D::new(
            self.rlwe_q.0,
            params.decomposition_base().0,
            params.decomposition_count().0,
        )
    }

    /// Returns dlogs of `g` for which auto keys are required as
    /// per the parameter. Given that autos are required for [-g, g, g^2, ...,
    /// g^w] function returns the following [0, 1, 2, ..., w] where `w` is
    /// the window size. Note that although g^0 = 1, we use 0 for -g.
    pub(crate) fn auto_element_dlogs(&self) -> Vec<usize> {
        let mut els = vec![0];
        (1..self.w + 1).into_iter().for_each(|e| {
            els.push(e);
        });
        els
    }

    pub(crate) fn variant(&self) -> &ParameterVariant {
        &self.variant
    }
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecompostionLogBase(pub(crate) usize);
impl AsRef<usize> for DecompostionLogBase {
    fn as_ref(&self) -> &usize {
        &self.0
    }
}
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecompositionCount(pub(crate) usize);
impl AsRef<usize> for DecompositionCount {
    fn as_ref(&self) -> &usize {
        &self.0
    }
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct LweDimension(pub(crate) usize);
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct PolynomialSize(pub(crate) usize);
#[derive(Clone, Copy, PartialEq, Debug)]

/// T equals modulus when modulus is non-native. Otherwise T equals 0. bool is
/// true when modulus is native, false otherwise.
#[derive(Serialize, Deserialize)]
pub struct CiphertextModulus<T>(T, bool);

impl<T: ConstZero> CiphertextModulus<T> {
    const fn new_native() -> Self {
        // T::zero is stored only for convenience. It has no use when modulus
        // is native. That is, either u128,u64,u32,u16
        Self(T::ZERO, true)
    }

    const fn new_non_native(q: T) -> Self {
        Self(q, false)
    }
}

impl<T> CiphertextModulus<T>
where
    T: PrimInt + NumInfo,
{
    fn _bits() -> usize {
        T::BITS as usize
    }

    fn _native(&self) -> bool {
        self.1
    }

    fn _half_q(&self) -> T {
        if self._native() {
            T::one() << (Self::_bits() - 1)
        } else {
            self.0 >> 1
        }
    }

    fn _q(&self) -> Option<T> {
        if self._native() {
            None
        } else {
            Some(self.0)
        }
    }
}

impl<T> Modulus for CiphertextModulus<T>
where
    T: PrimInt + FromPrimitive + NumInfo,
{
    type Element = T;
    fn is_native(&self) -> bool {
        self._native()
    }
    fn largest_unsigned_value(&self) -> Self::Element {
        if self._native() {
            T::max_value()
        } else {
            self.0 - T::one()
        }
    }
    fn neg_one(&self) -> Self::Element {
        if self._native() {
            T::max_value()
        } else {
            self.0 - T::one()
        }
    }
    // fn signed_max(&self) -> Self::Element {}
    // fn signed_min(&self) -> Self::Element {}
    fn smallest_unsigned_value(&self) -> Self::Element {
        T::zero()
    }

    fn map_element_to_i64(&self, v: &Self::Element) -> i64 {
        assert!(*v <= self.largest_unsigned_value());
        if *v > self._half_q() {
            -((self.largest_unsigned_value() - *v) + T::one())
                .to_i64()
                .unwrap()
        } else {
            v.to_i64().unwrap()
        }
    }

    fn map_element_from_f64(&self, v: f64) -> Self::Element {
        let v = v.round();

        let v_el = T::from_f64(v.abs()).unwrap();
        assert!(v_el <= self.largest_unsigned_value());

        if v < 0.0 {
            self.largest_unsigned_value() - v_el + T::one()
        } else {
            v_el
        }
    }

    fn map_element_from_i64(&self, v: i64) -> Self::Element {
        let v_el = T::from_i64(v.abs()).unwrap();
        assert!(v_el <= self.largest_unsigned_value());
        if v < 0 {
            self.largest_unsigned_value() - v_el + T::one()
        } else {
            v_el
        }
    }

    fn q(&self) -> Option<Self::Element> {
        self._q()
    }

    fn q_as_f64(&self) -> Option<f64> {
        if self._native() {
            Some(T::max_value().to_f64().unwrap() + 1.0)
        } else {
            self.0.to_f64()
        }
    }

    fn log_q(&self) -> usize {
        if self.is_native() {
            Self::_bits()
        } else {
            log2(&self.q().unwrap())
        }
    }
}

pub(crate) const I_2P_LB_SR: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    lwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    rlwe_q: CiphertextModulus::new_non_native(18014398509404161),
    lwe_q: CiphertextModulus::new_non_native(1 << 15),
    br_q: 1 << 11,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(580),
    lwe_decomposer_params: (DecompostionLogBase(1), DecompositionCount(12)),
    rlrg_decomposer_params: (
        DecompostionLogBase(17),
        (DecompositionCount(1), DecompositionCount(1)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(7),
        (DecompositionCount(6), DecompositionCount(5)),
    )),
    auto_decomposer_params: (DecompostionLogBase(24), DecompositionCount(1)),
    non_interactive_ui_to_s_key_switch_decomposer: None,
    g: 5,
    w: 10,
    variant: ParameterVariant::InteractiveMultiParty,
};

pub(crate) const I_4P: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    lwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    rlwe_q: CiphertextModulus::new_non_native(18014398509404161),
    lwe_q: CiphertextModulus::new_non_native(1 << 16),
    br_q: 1 << 11,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(620),
    lwe_decomposer_params: (DecompostionLogBase(1), DecompositionCount(13)),
    rlrg_decomposer_params: (
        DecompostionLogBase(17),
        (DecompositionCount(1), DecompositionCount(1)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(6),
        (DecompositionCount(7), DecompositionCount(6)),
    )),
    auto_decomposer_params: (DecompostionLogBase(24), DecompositionCount(1)),
    non_interactive_ui_to_s_key_switch_decomposer: None,
    g: 5,
    w: 10,
    variant: ParameterVariant::InteractiveMultiParty,
};

pub(crate) const I_8P: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    lwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    rlwe_q: CiphertextModulus::new_non_native(18014398509404161),
    lwe_q: CiphertextModulus::new_non_native(1 << 17),
    br_q: 1 << 12,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(660),
    lwe_decomposer_params: (DecompostionLogBase(1), DecompositionCount(14)),
    rlrg_decomposer_params: (
        DecompostionLogBase(17),
        (DecompositionCount(1), DecompositionCount(1)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(5),
        (DecompositionCount(9), DecompositionCount(8)),
    )),
    auto_decomposer_params: (DecompostionLogBase(24), DecompositionCount(1)),
    non_interactive_ui_to_s_key_switch_decomposer: None,
    g: 5,
    w: 10,
    variant: ParameterVariant::InteractiveMultiParty,
};

pub(crate) const NI_2P: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    lwe_secret_key_dist: SecretKeyDistribution::ErrorDistribution,
    rlwe_q: CiphertextModulus::new_non_native(18014398509404161),
    lwe_q: CiphertextModulus::new_non_native(1 << 16),
    br_q: 1 << 12,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(520),
    lwe_decomposer_params: (DecompostionLogBase(1), DecompositionCount(13)),
    rlrg_decomposer_params: (
        DecompostionLogBase(17),
        (DecompositionCount(1), DecompositionCount(1)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(4),
        (DecompositionCount(10), DecompositionCount(9)),
    )),
    auto_decomposer_params: (DecompostionLogBase(24), DecompositionCount(1)),
    non_interactive_ui_to_s_key_switch_decomposer: Some((
        DecompostionLogBase(1),
        DecompositionCount(50),
    )),
    g: 5,
    w: 10,
    variant: ParameterVariant::NonInteractiveMultiParty,
};

pub(crate) const NI_4P_HB_FR: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    lwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    rlwe_q: CiphertextModulus::new_non_native(18014398509404161),
    lwe_q: CiphertextModulus::new_non_native(1 << 16),
    br_q: 1 << 11,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(620),
    lwe_decomposer_params: (DecompostionLogBase(1), DecompositionCount(13)),
    rlrg_decomposer_params: (
        DecompostionLogBase(17),
        (DecompositionCount(1), DecompositionCount(1)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(3),
        (DecompositionCount(13), DecompositionCount(12)),
    )),
    auto_decomposer_params: (DecompostionLogBase(24), DecompositionCount(1)),
    non_interactive_ui_to_s_key_switch_decomposer: Some((
        DecompostionLogBase(1),
        DecompositionCount(50),
    )),
    g: 5,
    w: 10,
    variant: ParameterVariant::NonInteractiveMultiParty,
};

pub(crate) const NI_4P_LB_SR: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    lwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    rlwe_q: CiphertextModulus::new_non_native(18014398509404161),
    lwe_q: CiphertextModulus::new_non_native(1 << 16),
    br_q: 1 << 12,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(620),
    lwe_decomposer_params: (DecompostionLogBase(1), DecompositionCount(13)),
    rlrg_decomposer_params: (
        DecompostionLogBase(17),
        (DecompositionCount(1), DecompositionCount(1)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(4),
        (DecompositionCount(10), DecompositionCount(9)),
    )),
    auto_decomposer_params: (DecompostionLogBase(24), DecompositionCount(1)),
    non_interactive_ui_to_s_key_switch_decomposer: Some((
        DecompostionLogBase(1),
        DecompositionCount(50),
    )),
    g: 5,
    w: 10,
    variant: ParameterVariant::NonInteractiveMultiParty,
};

pub(crate) const NI_8P: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    lwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    rlwe_q: CiphertextModulus::new_non_native(18014398509404161),
    lwe_q: CiphertextModulus::new_non_native(1 << 17),
    br_q: 1 << 12,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(660),
    lwe_decomposer_params: (DecompostionLogBase(1), DecompositionCount(14)),
    rlrg_decomposer_params: (
        DecompostionLogBase(17),
        (DecompositionCount(1), DecompositionCount(1)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(2),
        (DecompositionCount(20), DecompositionCount(18)),
    )),
    auto_decomposer_params: (DecompostionLogBase(24), DecompositionCount(1)),
    non_interactive_ui_to_s_key_switch_decomposer: Some((
        DecompostionLogBase(1),
        DecompositionCount(50),
    )),
    g: 5,
    w: 10,
    variant: ParameterVariant::NonInteractiveMultiParty,
};

#[cfg(test)]
pub(crate) const SP_TEST_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_secret_key_dist: SecretKeyDistribution::TernaryDistribution,
    lwe_secret_key_dist: SecretKeyDistribution::ErrorDistribution,
    rlwe_q: CiphertextModulus::new_non_native(268369921u64),
    lwe_q: CiphertextModulus::new_non_native(1 << 16),
    br_q: 1 << 9,
    rlwe_n: PolynomialSize(1 << 9),
    lwe_n: LweDimension(100),
    lwe_decomposer_params: (DecompostionLogBase(4), DecompositionCount(4)),
    rlrg_decomposer_params: (
        DecompostionLogBase(7),
        (DecompositionCount(4), DecompositionCount(4)),
    ),
    rgrg_decomposer_params: None,
    auto_decomposer_params: (DecompostionLogBase(7), DecompositionCount(4)),
    non_interactive_ui_to_s_key_switch_decomposer: None,
    g: 5,
    w: 5,
    variant: ParameterVariant::SingleParty,
};

// #[cfg(test)]
// mod tests {

//     #[test]
//     fn find_prime() {
//         let bits = 60;
//         let ring_size = 1 << 11;
//         let prime = crate::utils::generate_prime(bits, ring_size * 2, 1 <<
// bits).unwrap();         dbg!(prime);
//     }
// }
