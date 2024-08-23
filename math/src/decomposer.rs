use crate::{
    izip_eq,
    modulus::{power_of_two::PowerOfTwo, Modulus, ModulusOps, Prime},
    util::scratch::Scratch,
};
use core::{borrow::Borrow, fmt::Debug};

#[derive(Clone, Copy, Debug)]
pub struct DecompositionParam {
    pub log_base: usize,
    pub level: usize,
}

pub trait Decomposer<T: 'static + Copy + Debug + Send + Sync>: Clone + Debug + Send + Sync {
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
        (0..self.level()).map(move |_| self.decompose_next(&mut a))
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
}

pub type NativeDecomposer = PowerOfTwoDecomposer<true>;

pub type NonNativePowerOfTwoDecomposer = PowerOfTwoDecomposer<false>;

#[derive(Clone, Copy, Debug)]
pub struct PowerOfTwoDecomposer<const NATIVE: bool> {
    q: PowerOfTwo<NATIVE>,
    log_base: usize,
    base_mask: u64,
    level: usize,
    ignored_bits: usize,
    ignored_half: u64,
}

impl<const NATIVE: bool> PowerOfTwoDecomposer<NATIVE> {
    pub fn new(q: PowerOfTwo<NATIVE>, param: DecompositionParam) -> Self {
        let log_base = param.log_base;
        let level = param.level;
        let base_mask = (1 << log_base) - 1;
        let ignored_bits = q.bits().saturating_sub(log_base * level);
        Self {
            q,
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
        self.q.into()
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
        self.q.reduce(limb.wrapping_sub(carry << self.log_base))
    }

    fn recompose(&self, a: impl IntoIterator<Item = u64>) -> u64 {
        self.q.reduce(
            izip_eq!(a, self.log_gadget_iter())
                .map(|(a, bits)| a << bits)
                .reduce(|a, b| a.wrapping_add(b))
                .unwrap_or_default(),
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrimeDecomposer {
    q: Prime,
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
        Self {
            q,
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
        self.q.into()
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
        let mut a = *a;
        if a >= self.q.half() {
            a = a.wrapping_sub(*self.q)
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
        (carry.wrapping_neg() & *self.q).wrapping_add(limb.wrapping_sub(carry << self.log_base))
    }

    fn recompose(&self, a: impl IntoIterator<Item = u64>) -> u64 {
        izip_eq!(a, self.gadget_iter())
            .map(|(a, b)| self.q.mul(&a, &b))
            .reduce(|a, b| self.q.add(&a, &b))
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
        modulus::{ModulusOps, Native, NonNativePowerOfTwo, Prime},
    };
    use core::iter::Sum;
    use num_traits::{Signed, ToPrimitive};
    use rand::thread_rng;

    #[derive(Clone)]
    struct Stats<T> {
        samples: Vec<T>,
    }

    impl<T> Default for Stats<T> {
        fn default() -> Self {
            Stats { samples: vec![] }
        }
    }

    impl<T: Copy + ToPrimitive + Signed + Sum<T>> Stats<T>
    where
        T: for<'a> Sum<&'a T>,
    {
        fn mean(&self) -> f64 {
            self.samples.iter().sum::<T>().to_f64().unwrap() / (self.samples.len() as f64)
        }

        fn variance(&self) -> f64 {
            let mean = self.mean();

            let diff_sq = self
                .samples
                .iter()
                .map(|v| {
                    let t = v.to_f64().unwrap() - mean;
                    t * t
                })
                .sum::<f64>();

            diff_sq / (self.samples.len() as f64 - 1.0)
        }

        fn std_dev(&self) -> f64 {
            self.variance().sqrt()
        }

        fn add_many_samples(&mut self, values: impl IntoIterator<Item = T>) {
            self.samples.extend(values);
        }
    }

    #[test]
    fn decompose() {
        fn run_po2<D: Decomposer<u64>>(modulus: impl ModulusOps<Elem = u64>) {
            let param = DecompositionParam {
                log_base: 8,
                level: 4,
            };
            let decomposer = D::new(modulus.modulus(), param);
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
                    [modulus.neg(&128), modulus.neg(&127), modulus.neg(&127), 128],
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
        }
        fn run_common<D: Decomposer<u64>>(modulus: impl ModulusOps<Elem = u64>) {
            let param = DecompositionParam {
                log_base: 8,
                level: 4,
            };
            let decomposer = D::new(modulus.modulus(), param);
            for a in modulus.sample_uniform_iter(thread_rng()).take(10000) {
                assert_eq!(
                    decomposer.round(&decomposer.recompose(decomposer.decompose_iter(&a))),
                    decomposer.round(&a),
                );
            }
        }

        run_po2::<NativeDecomposer>(Native::native());
        run_po2::<NonNativePowerOfTwoDecomposer>(NonNativePowerOfTwo::new(50));

        run_common::<NativeDecomposer>(Native::native());
        run_common::<NonNativePowerOfTwoDecomposer>(NonNativePowerOfTwo::new(50));
        run_common::<PrimeDecomposer>(Prime::gen(50, 0));
    }

    #[test]
    fn decompose_stats() {
        fn run<D: Decomposer<u64>>(modulus: impl ModulusOps<Elem = u64>) {
            let mut stats = Stats::default();

            let param = DecompositionParam {
                log_base: 10,
                level: 5,
            };
            let decomposer = D::new(modulus.modulus(), param);

            for a in modulus.sample_uniform_iter(thread_rng()).take(10000000) {
                stats.add_many_samples(decomposer.decompose_iter(&a).map(|v| modulus.to_i64(v)));
            }

            // Signed decomposition outputs limbs uniformly distributed in range [-B/2, B/2). The distribution must have mean nearly 0 and stanadrd deviation \sqrt{B^2 / 12}.
            assert!(stats.mean().abs() < 0.5);
            assert!(
                stats.std_dev().log2()
                    - ((1 << (param.log_base * 2)).to_f64().unwrap() / 12.0)
                        .sqrt()
                        .log2()
                    < 0.01
            );
        }

        run::<NativeDecomposer>(Native::native());
        run::<NonNativePowerOfTwoDecomposer>(NonNativePowerOfTwo::new(50));
        run::<PrimeDecomposer>(Prime::gen(50, 0));
    }
}
