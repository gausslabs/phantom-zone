use crate::decomposer::Decomposer;

#[derive(Clone, PartialEq)]
pub(super) struct BoolParameters<El> {
    rlwe_q: CiphertextModulus<El>,
    lwe_q: CiphertextModulus<El>,
    br_q: CiphertextModulus<El>,
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

    pub(crate) fn br_q(&self) -> &CiphertextModulus<El> {
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
pub(crate) struct CiphertextModulus<T>(T);

pub(super) const SP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_q: CiphertextModulus(268369921u64),
    lwe_q: CiphertextModulus(1 << 16),
    br_q: CiphertextModulus(1 << 10),
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
    rlwe_q: CiphertextModulus(1152921504606830593),
    lwe_q: CiphertextModulus(1 << 20),
    br_q: CiphertextModulus(1 << 11),
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
