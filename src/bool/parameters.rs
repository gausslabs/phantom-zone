use num_traits::{ConstZero, FromPrimitive, PrimInt};

use crate::{backend::Modulus, decomposer::Decomposer};

pub(super) trait DoubleDecomposerParams {
    type Base;
    type Count;

    fn new(base: Self::Base, count_a: Self::Count, count_b: Self::Count) -> Self;
    fn decomposition_base(&self) -> Self::Base;
    fn decomposition_count_a(&self) -> Self::Count;
    fn decomposition_count_b(&self) -> Self::Count;
}

trait SingleDecomposerParams {
    type Base;
    type Count;

    fn new(base: Self::Base, count: Self::Count) -> Self;
    fn decomposition_base(&self) -> Self::Base;
    fn decomposition_count(&self) -> Self::Count;
}

impl DoubleDecomposerParams
    for (
        DecompostionLogBase,
        (DecompositionCount, DecompositionCount),
    )
{
    type Base = DecompostionLogBase;
    type Count = DecompositionCount;

    fn new(
        base: DecompostionLogBase,
        count_a: DecompositionCount,
        count_b: DecompositionCount,
    ) -> Self {
        (base, (count_a, count_b))
    }

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

    fn new(base: DecompostionLogBase, count: DecompositionCount) -> Self {
        (base, count)
    }

    fn decomposition_base(&self) -> Self::Base {
        self.0
    }

    fn decomposition_count(&self) -> Self::Count {
        self.1
    }
}

// impl DecomposerParams for (DecompostionLogBase, (DecompositionCount)) {
//     type Base = DecompostionLogBase;
//     type Count = DecompositionCount;

//     fn decomposition_base(&self) -> Self::Base {
//         self.0
//     }

//     fn decomposition_count(&self) -> Self::Count {
//         self.1
//     }
// }

#[derive(Clone, PartialEq, Debug)]
pub(crate) enum ParameterVariant {
    SingleParty,
    MultiParty,
    NonInteractiveMultiParty,
}
#[derive(Clone, PartialEq)]
pub struct BoolParameters<El> {
    rlwe_q: CiphertextModulus<El>,
    lwe_q: CiphertextModulus<El>,
    br_q: usize,
    rlwe_n: PolynomialSize,
    lwe_n: LweDimension,
    lwe_decomposer_params: (DecompostionLogBase, DecompositionCount),
    /// RLWE x RGSW decomposition count for (part A, part B)
    rlrg_decomposer_params: (
        DecompostionLogBase,
        (DecompositionCount, DecompositionCount),
    ),
    auto_decomposer_params: (DecompostionLogBase, DecompositionCount),
    /// RGSW x RGSW decomposition count for (part A, part B)
    rgrg_decomposer_params: Option<(
        DecompostionLogBase,
        (DecompositionCount, DecompositionCount),
    )>,
    non_interactive_ui_to_s_key_switch_decomposer:
        Option<(DecompostionLogBase, DecompositionCount)>,
    g: usize,
    w: usize,
    variant: ParameterVariant,
}

impl<El> BoolParameters<El> {
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
    ) -> (
        DecompostionLogBase,
        (DecompositionCount, DecompositionCount),
    ) {
        self.rlrg_decomposer_params
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

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct DecompostionLogBase(pub(crate) usize);
impl AsRef<usize> for DecompostionLogBase {
    fn as_ref(&self) -> &usize {
        &self.0
    }
}
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct DecompositionCount(pub(crate) usize);
impl AsRef<usize> for DecompositionCount {
    fn as_ref(&self) -> &usize {
        &self.0
    }
}
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct LweDimension(pub(crate) usize);
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct PolynomialSize(pub(crate) usize);
#[derive(Clone, Copy, PartialEq, Debug)]

/// T equals modulus when modulus is non-native. Otherwise T equals 0. bool is
/// true when modulus is native, false otherwise.
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
    T: PrimInt,
{
    pub(crate) fn _bits() -> usize {
        std::mem::size_of::<T>() as usize * 8
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
    T: PrimInt + FromPrimitive,
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
        if v < 0.0 {
            self.largest_unsigned_value() - T::from_f64(v.abs()).unwrap() + T::one()
        } else {
            T::from_f64(v.abs()).unwrap()
        }
    }

    fn map_element_from_i64(&self, v: i64) -> Self::Element {
        if v < 0 {
            self.largest_unsigned_value() - T::from_i64(v.abs()).unwrap() + T::one()
        } else {
            T::from_i64(v.abs()).unwrap()
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
}

pub(crate) const MP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_q: CiphertextModulus::new_non_native(1152921504606830593),
    lwe_q: CiphertextModulus::new_non_native(1 << 20),
    br_q: 1 << 11,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(500),
    lwe_decomposer_params: (DecompostionLogBase(4), DecompositionCount(5)),
    rlrg_decomposer_params: (
        DecompostionLogBase(12),
        (DecompositionCount(5), DecompositionCount(5)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(12),
        (DecompositionCount(5), DecompositionCount(5)),
    )),
    auto_decomposer_params: (DecompostionLogBase(12), DecompositionCount(5)),
    non_interactive_ui_to_s_key_switch_decomposer: None,
    g: 5,
    w: 10,
    variant: ParameterVariant::MultiParty,
};

pub(crate) const SMALL_MP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_q: CiphertextModulus::new_non_native(36028797018820609),
    lwe_q: CiphertextModulus::new_non_native(1 << 20),
    br_q: 1 << 11,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(600),
    lwe_decomposer_params: (DecompostionLogBase(4), DecompositionCount(5)),
    rlrg_decomposer_params: (
        DecompostionLogBase(11),
        (DecompositionCount(2), DecompositionCount(1)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(11),
        (DecompositionCount(5), DecompositionCount(4)),
    )),
    auto_decomposer_params: (DecompostionLogBase(11), DecompositionCount(2)),
    non_interactive_ui_to_s_key_switch_decomposer: None,
    g: 5,
    w: 10,
    variant: ParameterVariant::MultiParty,
};

pub(crate) const OPTIMISED_SMALL_MP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_q: CiphertextModulus::new_non_native(18014398509404161),
    lwe_q: CiphertextModulus::new_non_native(1 << 15),
    br_q: 1 << 11,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(500),
    lwe_decomposer_params: (DecompostionLogBase(1), DecompositionCount(11)),
    rlrg_decomposer_params: (
        DecompostionLogBase(16),
        (DecompositionCount(1), DecompositionCount(1)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(8),
        (DecompositionCount(6), DecompositionCount(6)),
    )),
    auto_decomposer_params: (DecompostionLogBase(24), DecompositionCount(1)),
    non_interactive_ui_to_s_key_switch_decomposer: None,
    g: 5,
    w: 10,
    variant: ParameterVariant::MultiParty,
};

pub(crate) const NON_INTERACTIVE_SMALL_MP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_q: CiphertextModulus::new_non_native(36028797018820609),
    lwe_q: CiphertextModulus::new_non_native(1 << 20),
    br_q: 1 << 11,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(600),
    lwe_decomposer_params: (DecompostionLogBase(4), DecompositionCount(5)),
    rlrg_decomposer_params: (
        DecompostionLogBase(11),
        (DecompositionCount(2), DecompositionCount(1)),
    ),
    rgrg_decomposer_params: Some((
        DecompostionLogBase(11),
        (DecompositionCount(5), DecompositionCount(4)),
    )),
    auto_decomposer_params: (DecompostionLogBase(11), DecompositionCount(2)),
    non_interactive_ui_to_s_key_switch_decomposer: Some((
        DecompostionLogBase(1),
        DecompositionCount(55),
    )),
    g: 5,
    w: 10,
    variant: ParameterVariant::NonInteractiveMultiParty,
};

#[cfg(test)]
pub(crate) const SP_TEST_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
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

#[cfg(test)]
mod tests {

    impl BoolParameters<u64> {
        pub(crate) fn default_rlwe_modop(&self) -> ModularOpsU64<CiphertextModulus<u64>> {
            ModularOpsU64::new(self.rlwe_q)
        }
        pub(crate) fn default_rlwe_nttop(&self) -> NttBackendU64 {
            NttBackendU64::new(&self.rlwe_q, self.rlwe_n.0)
        }
    }

    use crate::{utils::generate_prime, ModInit, ModularOpsU64, Ntt, NttBackendU64, NttInit};

    use super::{BoolParameters, CiphertextModulus};

    #[test]
    fn find_prime() {
        let bits = 60;
        let ring_size = 1 << 11;
        let prime = generate_prime(bits, ring_size * 2, 1 << bits).unwrap();
        dbg!(prime);
    }
}
