use crate::{
    izip_eq,
    misc::scratch::Scratch,
    modulus::{add_mod, mul_mod, Modulus, PowerOfTwo, Prime},
};
use core::{borrow::Borrow, fmt::Debug, iter::repeat_with};

#[derive(Clone, Copy, Debug)]
pub struct DecompositionParam {
    pub log_base: usize,
    pub level: usize,
}

pub trait Decomposer<T: Copy + Debug + 'static> {
    fn new(modulus: Modulus, param: DecompositionParam) -> Self;

    fn modulus(&self) -> Modulus;

    fn log_base(&self) -> usize;

    fn level(&self) -> usize;

    fn ignored_bits(&self) -> usize;

    fn log_gadget_iter(&self) -> impl Iterator<Item = usize> {
        (self.ignored_bits()..)
            .step_by(self.log_base())
            .take(self.level())
    }

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

    fn slice_round(&self, b: &mut [T], a: impl IntoIterator<Item: Borrow<T>>) {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.round(a.borrow()));
    }

    fn slice_decompose_next(&self, b: &mut [T], a: &mut [T]) {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.decompose_next(a));
    }

    fn slice_decompose_zip_for_each<B>(
        &self,
        a: impl IntoIterator<Item: Borrow<T>>,
        b: impl IntoIterator<Item = B>,
        mut scratch: Scratch,
        mut f: impl FnMut(usize, &[T], B),
    ) {
        let a = a.into_iter();
        let [state, limb] = scratch.take_slice_array(a.size_hint().1.unwrap());
        self.slice_round(state, a);
        izip_eq!(0..self.level(), b).for_each(|(i, b)| {
            self.slice_decompose_next(limb, state);
            f(i, limb, b);
        });
    }

    fn allocate_scratch(&self, a: &[T]) -> Vec<T>
    where
        T: Default,
    {
        vec![T::default(); 2 * a.len()]
    }
}

pub type NativeDecomposer = PowerOfTwoDecomposer<true>;

pub type NonNativePowerOfTwoDecomposer = PowerOfTwoDecomposer<false>;

#[derive(Clone, Copy, Debug)]
pub struct PowerOfTwoDecomposer<const NATIVE: bool> {
    modulus: PowerOfTwo,
    modulus_mask: u64,
    log_base: usize,
    base_mask: u64,
    level: usize,
    ignored_bits: usize,
    ignored_half: u64,
}

impl<const NATIVE: bool> PowerOfTwoDecomposer<NATIVE> {
    pub fn new(modulus: PowerOfTwo, param: DecompositionParam) -> Self {
        let log_base = param.log_base;
        let level = param.level;
        let base_mask = (1 << log_base) - 1;
        let ignored_bits = modulus.bits().saturating_sub(log_base * level);
        Self {
            modulus,
            modulus_mask: modulus.mask(),
            log_base,
            base_mask,
            level,
            ignored_bits,
            ignored_half: (1 << ignored_bits) >> 1,
        }
    }
}

impl<const NATIVE: bool> Decomposer<u64> for PowerOfTwoDecomposer<NATIVE> {
    fn new(modulus: Modulus, param: DecompositionParam) -> Self {
        Self::new(modulus.try_into().unwrap(), param)
    }

    fn modulus(&self) -> Modulus {
        self.modulus.into()
    }

    fn log_base(&self) -> usize {
        self.log_base
    }

    fn level(&self) -> usize {
        self.level
    }

    fn ignored_bits(&self) -> usize {
        self.ignored_bits
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
        if NATIVE {
            limb.wrapping_sub(carry << self.log_base)
        } else {
            limb.wrapping_sub(carry << self.log_base) & self.modulus_mask
        }
    }

    fn recompose(&self, a: impl IntoIterator<Item = u64>) -> u64 {
        izip_eq!(a, self.log_gadget_iter())
            .map(|(a, bits)| a << bits)
            .reduce(|a, b| a.wrapping_add(b))
            .unwrap_or_default()
            & self.modulus_mask
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrimeDecomposer {
    q: u64,
    q_half: u64,
    log_base: usize,
    base_mask: u64,
    level: usize,
    ignored_bits: usize,
    ignored_half: u64,
}

impl PrimeDecomposer {
    pub fn new(q: Prime, param: DecompositionParam) -> Self {
        let log_base = param.log_base;
        let level = param.level;
        let base_mask = (1 << log_base) - 1;
        let ignored_bits = q.bits().saturating_sub(log_base * level);
        let q = u64::from(q);
        let q_half = q >> 1;
        Self {
            q,
            q_half,
            log_base,
            base_mask,
            level,
            ignored_bits,
            ignored_half: (1 << ignored_bits) >> 1,
        }
    }
}

impl Decomposer<u64> for PrimeDecomposer {
    fn new(modulus: Modulus, param: DecompositionParam) -> Self {
        Self::new(modulus.try_into().unwrap(), param)
    }

    fn modulus(&self) -> Modulus {
        Prime(self.q).into()
    }

    fn log_base(&self) -> usize {
        self.log_base
    }

    fn level(&self) -> usize {
        self.level
    }

    fn ignored_bits(&self) -> usize {
        self.ignored_bits
    }

    fn round(&self, a: &u64) -> u64 {
        let mut a = *a;
        if a >= self.q_half {
            a = a.wrapping_sub(self.q)
        }
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
        (carry.wrapping_neg() & self.q).wrapping_add(limb.wrapping_sub(carry << self.log_base))
    }

    fn recompose(&self, a: impl IntoIterator<Item = u64>) -> u64 {
        izip_eq!(a, self.gadget_iter())
            .map(|(a, b)| mul_mod(a, b, self.q))
            .reduce(|a, b| add_mod(a, b, self.q))
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        decomposer::{
            Decomposer, DecompositionParam, NativeDecomposer, NonNativePowerOfTwoDecomposer,
            PrimeDecomposer,
        },
        modulus::{Modulus, PowerOfTwo, Prime},
    };
    use rand::{
        distributions::{Distribution, Uniform},
        thread_rng,
    };

    #[test]
    fn decompose() {
        fn run<D: Decomposer<u64>>(modulus: impl Into<Modulus>) {
            let modulus = modulus.into();
            let param = DecompositionParam {
                log_base: 8,
                level: 4,
            };
            let decomposer = D::new(modulus, param);
            let neg = |a: u64| if a == 0 { 0 } else { modulus.max() - a + 1 };
            for (a, b) in [
                (0, [0, 0, 0, 0]),
                (
                    (1 << decomposer.ignored_bits()) >> 1,
                    [(decomposer.ignored_bits() > 0) as u64, 0, 0, 0],
                ),
                (
                    0b_01111111_01111111_01111111_01111111 << decomposer.ignored_bits(),
                    [127, 127, 127, 127],
                ),
                (
                    0b_01111111_10000000_10000000_10000000 << decomposer.ignored_bits(),
                    [neg(128), neg(127), neg(127), 128],
                ),
                (
                    0b_01111111_10000000_01111111_10000000 << decomposer.ignored_bits(),
                    [128, 127, 128, 127],
                ),
            ] {
                let limbs = decomposer.decompose_vec(&a);
                assert_eq!(limbs, b);
                assert_eq!(
                    decomposer.round(&decomposer.recompose(decomposer.decompose_iter(&a))),
                    decomposer.round(&a),
                );
            }
            for a in Uniform::new_inclusive(0, modulus.max())
                .sample_iter(thread_rng())
                .take(10000)
            {
                assert_eq!(
                    decomposer.round(&decomposer.recompose(decomposer.decompose_iter(&a))),
                    decomposer.round(&a),
                );
            }
        }

        run::<NativeDecomposer>(PowerOfTwo::native());
        run::<NonNativePowerOfTwoDecomposer>(PowerOfTwo::new(50));
        run::<PrimeDecomposer>(Prime::gen(50, 0));
    }
}
