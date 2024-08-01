use crate::{
    izip_eq,
    modulus::{PowerOfTwo, Prime},
};
use core::{fmt::Debug, iter::repeat_with};

#[derive(Clone, Copy, Debug)]
pub struct DecompositionParam {
    pub log_base: usize,
    pub level: usize,
}

pub trait Decomposer<T: Copy + Debug + 'static> {
    fn log_base(&self) -> usize;

    fn level(&self) -> usize;

    fn gadget_iter(&self) -> impl Iterator<Item = T>;

    fn gadget_vec(&self) -> Vec<T> {
        self.gadget_iter().collect()
    }

    fn round(&self, a: &T) -> T;

    fn decompose_next(&self, a: &mut T) -> T;

    fn decompose_iter(&self, a: &T) -> impl Iterator<Item = T> {
        let mut a = self.round(a);
        repeat_with(move || self.decompose_next(&mut a)).take(self.level())
    }

    fn decompose_vec(&self, a: &T) -> Vec<T> {
        self.decompose_iter(a).collect()
    }

    fn recompose(&self, a: impl IntoIterator<Item = T>) -> T;

    fn slice_round_assign(&self, a: &mut [T]) {
        a.iter_mut().for_each(|a| *a = self.round(a));
    }

    fn slice_decompose_next(&self, b: &mut [T], a: &mut [T]) {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.decompose_next(a));
    }

    fn slice_decompose_zip_for_each<B>(
        &self,
        a: &mut [T],
        b: impl IntoIterator<Item = B>,
        scratch: &mut [T],
        mut f: impl FnMut((&[T], B)),
    ) {
        self.slice_round_assign(a);
        izip_eq!(0..self.level(), b).for_each(|(_, b)| {
            self.slice_decompose_next(scratch, a);
            f((scratch, b));
        });
    }
}

pub struct PowerOfTwoDecomposer {
    log_base: usize,
    level: usize,
    modulus_mask: u64,
    base_mask: u64,
    ignored_bits: usize,
    ignored_half: u64,
}

impl PowerOfTwoDecomposer {
    pub fn new(modulus: PowerOfTwo, param: DecompositionParam) -> Self {
        let log_base = param.log_base;
        let level = param.level;
        let base_mask = (1 << log_base) - 1;
        let ignored_bits = modulus.bits().saturating_sub(log_base * level);
        Self {
            log_base,
            level,
            modulus_mask: modulus.mask(),
            base_mask,
            ignored_bits,
            ignored_half: (1 << ignored_bits) >> 1,
        }
    }

    fn log_gadget_iter(&self) -> impl Iterator<Item = usize> {
        (self.ignored_bits..)
            .step_by(self.log_base)
            .take(self.level)
    }
}

impl Decomposer<u64> for PowerOfTwoDecomposer {
    fn log_base(&self) -> usize {
        self.log_base
    }

    fn level(&self) -> usize {
        self.level
    }

    #[inline(always)]
    fn round(&self, a: &u64) -> u64 {
        a.wrapping_add(self.ignored_half) >> self.ignored_bits
    }

    fn gadget_iter(&self) -> impl Iterator<Item = u64> {
        self.log_gadget_iter().map(|bits| 1 << bits)
    }

    #[inline(always)]
    fn decompose_next(&self, a: &mut u64) -> u64 {
        let limb = *a & self.base_mask;
        *a >>= self.log_base;
        let carry = ((limb.wrapping_sub(1) | *a) & limb) >> (self.log_base - 1);
        *a += carry;
        limb.wrapping_sub(carry << self.log_base)
    }

    fn recompose(&self, a: impl IntoIterator<Item = u64>) -> u64 {
        izip_eq!(a, self.log_gadget_iter())
            .map(|(a, bits)| a << bits)
            .reduce(|a, b| a.wrapping_add(b))
            .unwrap_or_default()
            & self.modulus_mask
    }
}

pub struct PrimeDecomposer {}

impl PrimeDecomposer {
    pub fn new(_: Prime, _: DecompositionParam) -> Self {
        todo!()
    }
}

impl Decomposer<u64> for PrimeDecomposer {
    fn log_base(&self) -> usize {
        todo!()
    }

    fn level(&self) -> usize {
        todo!()
    }

    fn round(&self, _: &u64) -> u64 {
        todo!()
    }

    fn gadget_iter(&self) -> impl Iterator<Item = u64> {
        [].into_iter()
    }

    fn decompose_next(&self, _: &mut u64) -> u64 {
        todo!()
    }

    fn recompose(&self, _: impl IntoIterator<Item = u64>) -> u64 {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        decomposer::{Decomposer, DecompositionParam, PowerOfTwoDecomposer},
        modulus::PowerOfTwo,
    };

    #[test]
    fn power_of_two_decomposer() {
        for modulus in [PowerOfTwo::native(), PowerOfTwo::new(50)] {
            let param = DecompositionParam {
                log_base: 8,
                level: 4,
            };
            let decomposer = PowerOfTwoDecomposer::new(modulus, param);
            for (a, b) in [
                (0, [0, 0, 0, 0]),
                (decomposer.ignored_half, [1, 0, 0, 0]),
                (
                    0b_01111111_01111111_01111111_01111111 << decomposer.ignored_bits,
                    [127, 127, 127, 127],
                ),
                (
                    0b_01111111_10000000_10000000_10000000 << decomposer.ignored_bits,
                    [(-128i64 as u64), (-127i64 as u64), (-127i64 as u64), 128],
                ),
                (
                    0b_01111111_10000000_01111111_10000000 << decomposer.ignored_bits,
                    [128, 127, 128, 127],
                ),
            ] {
                let limbs = decomposer.decompose_vec(&a);
                assert_eq!(limbs, b);
                assert_eq!(
                    decomposer.recompose(limbs),
                    decomposer.round(&a) << decomposer.ignored_bits,
                );
            }
        }
    }
}
