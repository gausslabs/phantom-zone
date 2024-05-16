#[derive(Clone, PartialEq)]
pub(super) struct BoolParameters<El> {
    pub(super) rlwe_q: El,
    pub(super) rlwe_logq: usize,
    pub(super) lwe_q: El,
    pub(super) lwe_logq: usize,
    pub(super) br_q: usize,
    pub(super) rlwe_n: usize,
    pub(super) lwe_n: usize,
    pub(super) d_rgsw: usize,
    pub(super) logb_rgsw: usize,
    pub(super) d_lwe: usize,
    pub(super) logb_lwe: usize,
    pub(super) g: usize,
    pub(super) w: usize,
}

// impl<El> BoolParameters<El> {
//     fn rlwe_q(&self) -> &El {
//         &self.rlwe_q
//     }
// }

pub(super) const SP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_q: 268369921u64,
    rlwe_logq: 28,
    lwe_q: 1 << 16,
    lwe_logq: 16,
    br_q: 1 << 10,
    rlwe_n: 1 << 10,
    lwe_n: 493,
    d_rgsw: 4,
    logb_rgsw: 7,
    d_lwe: 4,
    logb_lwe: 4,
    g: 5,
    w: 1,
};

pub(super) const MP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
    rlwe_q: 1152921504606830593,
    rlwe_logq: 60,
    lwe_q: 1 << 20,
    lwe_logq: 20,
    br_q: 1 << 11,
    rlwe_n: 1 << 11,
    lwe_n: 500,
    d_rgsw: 5,
    logb_rgsw: 12,
    d_lwe: 5,
    logb_lwe: 4,
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
