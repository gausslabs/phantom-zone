use num_traits::{ConstZero, FromPrimitive, PrimInt, ToPrimitive, Zero};

use crate::{backend::Modulus, decomposer::Decomposer};

#[derive(Clone, PartialEq)]
pub(super) struct BoolParameters<El> {
    rlwe_q: CiphertextModulus<El>,
    lwe_q: CiphertextModulus<El>,
    br_q: usize,
    rlwe_n: PolynomialSize,
    lwe_n: LweDimension,
    lwe_decomposer_base: DecompostionLogBase,
    lwe_decomposer_count: DecompositionCount,
    rlrg_decomposer_base: DecompostionLogBase,
    /// RLWE x RGSW decomposition count for (part A, part B)
    rlrg_decomposer_count: (DecompositionCount, DecompositionCount),
    rgrg_decomposer_base: DecompostionLogBase,
    /// RGSW x RGSW decomposition count for (part A, part B)
    rgrg_decomposer_count: (DecompositionCount, DecompositionCount),
    auto_decomposer_base: DecompostionLogBase,
    auto_decomposer_count: DecompositionCount,
    g: usize,
    w: usize,
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

    pub(crate) fn rlwe_rgsw_decomposition_base(&self) -> DecompostionLogBase {
        self.rlrg_decomposer_base
    }

    pub(crate) fn rlwe_rgsw_decomposition_count(&self) -> (DecompositionCount, DecompositionCount) {
        self.rlrg_decomposer_count
    }

    pub(crate) fn rgsw_rgsw_decomposition_base(&self) -> DecompostionLogBase {
        self.rgrg_decomposer_base
    }

    pub(crate) fn rgsw_rgsw_decomposition_count(&self) -> (DecompositionCount, DecompositionCount) {
        self.rgrg_decomposer_count
    }

    pub(crate) fn auto_decomposition_base(&self) -> DecompostionLogBase {
        self.auto_decomposer_base
    }

    pub(crate) fn auto_decomposition_count(&self) -> DecompositionCount {
        self.auto_decomposer_count
    }

    pub(crate) fn lwe_decomposition_base(&self) -> DecompostionLogBase {
        self.lwe_decomposer_base
    }

    pub(crate) fn lwe_decomposition_count(&self) -> DecompositionCount {
        self.lwe_decomposer_count
    }

    pub(crate) fn rgsw_rgsw_decomposer<D: Decomposer<Element = El>>(&self) -> (D, D)
    where
        El: Copy,
    {
        (
            // A
            D::new(
                self.rlwe_q.0,
                self.rgrg_decomposer_base.0,
                self.rgrg_decomposer_count.0 .0,
            ),
            // B
            D::new(
                self.rlwe_q.0,
                self.rgrg_decomposer_base.0,
                self.rgrg_decomposer_count.1 .0,
            ),
        )
    }

    pub(crate) fn auto_decomposer<D: Decomposer<Element = El>>(&self) -> D
    where
        El: Copy,
    {
        D::new(
            self.rlwe_q.0,
            self.auto_decomposer_base.0,
            self.auto_decomposer_count.0,
        )
    }

    pub(crate) fn lwe_decomposer<D: Decomposer<Element = El>>(&self) -> D
    where
        El: Copy,
    {
        D::new(
            self.lwe_q.0,
            self.lwe_decomposer_base.0,
            self.lwe_decomposer_count.0,
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
                self.rlrg_decomposer_base.0,
                self.rlrg_decomposer_count.0 .0,
            ),
            // B
            D::new(
                self.rlwe_q.0,
                self.rlrg_decomposer_base.0,
                self.rlrg_decomposer_count.1 .0,
            ),
        )
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
#[derive(Clone, Copy, PartialEq)]

/// T equals modulus when modulus is non-native. Otherwise T equals 0. bool is
/// true when modulus is native, false otherwise.
pub(crate) struct CiphertextModulus<T>(T, bool);

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

pub(super) const SP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_q: CiphertextModulus::new_non_native(268369921u64),
    lwe_q: CiphertextModulus::new_non_native(1 << 16),
    br_q: 1 << 10,
    rlwe_n: PolynomialSize(1 << 10),
    lwe_n: LweDimension(493),
    lwe_decomposer_base: DecompostionLogBase(4),
    lwe_decomposer_count: DecompositionCount(4),
    rlrg_decomposer_base: DecompostionLogBase(7),
    rlrg_decomposer_count: (DecompositionCount(4), DecompositionCount(4)),
    rgrg_decomposer_base: DecompostionLogBase(7),
    rgrg_decomposer_count: (DecompositionCount(4), DecompositionCount(4)),
    auto_decomposer_base: DecompostionLogBase(7),
    auto_decomposer_count: DecompositionCount(4),
    g: 5,
    w: 1,
};

pub(super) const MP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_q: CiphertextModulus::new_non_native(1152921504606830593),
    lwe_q: CiphertextModulus::new_non_native(1 << 20),
    br_q: 1 << 10,
    rlwe_n: PolynomialSize(1 << 11),
    lwe_n: LweDimension(500),
    lwe_decomposer_base: DecompostionLogBase(4),
    lwe_decomposer_count: DecompositionCount(5),
    rlrg_decomposer_base: DecompostionLogBase(12),
    rlrg_decomposer_count: (DecompositionCount(5), DecompositionCount(5)),
    rgrg_decomposer_base: DecompostionLogBase(12),
    rgrg_decomposer_count: (DecompositionCount(5), DecompositionCount(5)),
    auto_decomposer_base: DecompostionLogBase(12),
    auto_decomposer_count: DecompositionCount(5),
    g: 5,
    w: 1,
};

#[cfg(test)]
mod tests {
    use crate::utils::generate_prime;

    #[test]
    fn find_prime() {
        let bits = 61;
        let ring_size = 1 << 11;
        let prime = generate_prime(bits, ring_size * 2, 1 << bits).unwrap();
        dbg!(prime);
    }
}
