use itertools::Itertools;
use num_traits::{AsPrimitive, FromPrimitive, Num, One, PrimInt, ToPrimitive, WrappingSub, Zero};
use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::Rem,
};

use crate::backend::{ArithmeticOps, ModularOpsU64};

fn gadget_vector<T: PrimInt>(logq: usize, logb: usize, d: usize) -> Vec<T> {
    let d_ideal = (logq as f64 / logb as f64).ceil().to_usize().unwrap();
    let ignored_limbs = d_ideal - d;
    (ignored_limbs..ignored_limbs + d)
        .into_iter()
        .map(|i| T::one() << (logb * i))
        .collect_vec()
}

pub trait RlweDecomposer {
    type Element;
    type D: Decomposer<Element = Self::Element>;

    /// Decomposer for RLWE Part A
    fn a(&self) -> &Self::D;
    /// Decomposer for RLWE Part B
    fn b(&self) -> &Self::D;
}

impl<D> RlweDecomposer for (D, D)
where
    D: Decomposer,
{
    type D = D;
    type Element = D::Element;
    fn a(&self) -> &Self::D {
        &self.0
    }
    fn b(&self) -> &Self::D {
        &self.1
    }
}

pub trait Decomposer {
    type Element;
    type Iter: Iterator<Item = Self::Element>;
    fn new(q: Self::Element, logb: usize, d: usize) -> Self;

    fn decompose_to_vec(&self, v: &Self::Element) -> Vec<Self::Element>;
    fn decompose_iter(&self, v: &Self::Element) -> Self::Iter;
    fn decomposition_count(&self) -> usize;
}

pub struct DefaultDecomposer<T> {
    /// Ciphertext modulus
    q: T,
    /// Log of ciphertext modulus
    logq: usize,
    /// Log of base B
    logb: usize,
    /// base B
    b: T,
    /// (B - 1). To simulate (% B) as &(B-1), that is extract least significant
    /// logb bits
    b_mask: T,
    /// B/2
    bby2: T,
    /// Decomposition count
    d: usize,
    /// No. of bits to ignore in rounding
    ignore_bits: usize,
    /// No. of limbs to ignore in rounding. Set to ceil(logq / logb) - d
    ignore_limbs: usize,
}

pub trait NumInfo {
    const BITS: u32;
}

impl NumInfo for u64 {
    const BITS: u32 = u64::BITS;
}
impl NumInfo for u32 {
    const BITS: u32 = u32::BITS;
}
impl NumInfo for u128 {
    const BITS: u32 = u128::BITS;
}

impl<T: PrimInt + NumInfo + Debug> DefaultDecomposer<T> {
    fn recompose<Op>(&self, limbs: &[T], modq_op: &Op) -> T
    where
        Op: ArithmeticOps<Element = T>,
    {
        let mut value = T::zero();
        for i in 0..self.d {
            value = modq_op.add(
                &value,
                &(modq_op.mul(
                    &limbs[i],
                    &(T::one() << (self.logb * (i + self.ignore_limbs))),
                )),
            )
        }
        value
    }

    pub(crate) fn gadget_vector(&self) -> Vec<T> {
        return gadget_vector(self.logq, self.logb, self.d);
    }
}

impl<T: PrimInt + ToPrimitive + FromPrimitive + WrappingSub + NumInfo + From<bool> + Display>
    Decomposer for DefaultDecomposer<T>
{
    type Element = T;
    type Iter = DecomposerIter<T>;

    fn new(q: T, logb: usize, d: usize) -> DefaultDecomposer<T> {
        // if q is power of 2, then `BITS - leading_zeros` outputs logq + 1.
        let logq = if q & (q - T::one()) == T::zero() {
            (T::BITS - q.leading_zeros() - 1) as usize
        } else {
            (T::BITS - q.leading_zeros()) as usize
        };

        let d_ideal = (logq as f64 / logb as f64).ceil().to_usize().unwrap();
        let ignore_limbs = (d_ideal - d);
        let ignore_bits = (d_ideal - d) * logb;

        DefaultDecomposer {
            q,
            logq,
            logb,
            b: T::one() << logb,
            b_mask: (T::one() << logb) - T::one(),
            bby2: T::one() << (logb - 1),
            d,
            ignore_bits,
            ignore_limbs,
        }
    }

    // TODO(Jay): Outline the caveat
    fn decompose_to_vec(&self, value: &T) -> Vec<T> {
        let mut value = round_value(*value, self.ignore_bits);

        let q = self.q;
        let logb = self.logb;
        let b = T::one() << logb;
        let full_mask = b - T::one();
        let bby2 = b >> 1;

        if value >= (q >> 1) {
            value = !(q - value) + T::one()
        }

        let mut out = Vec::with_capacity(self.d);
        for _ in 0..self.d {
            let k_i = value & full_mask;

            value = (value - k_i) >> logb;

            if k_i > bby2 || (k_i == bby2 && ((value & T::one()) == T::one())) {
                out.push(q - (b - k_i));
                value = value + T::one();
            } else {
                out.push(k_i);
            }
        }

        return out;
    }

    fn decomposition_count(&self) -> usize {
        self.d
    }

    fn decompose_iter(&self, value: &T) -> DecomposerIter<T> {
        let mut value = round_value(*value, self.ignore_bits);

        if value >= (self.q >> 1) {
            value = !(self.q - value) + T::one()
        }

        DecomposerIter {
            value,
            q: self.q,
            logq: self.logq,
            logb: self.logb,
            b: self.b,
            bby2: self.bby2,
            b_mask: self.b_mask,
            steps_left: self.d,
        }
    }
}

impl<T: PrimInt> DefaultDecomposer<T> {}

pub struct DecomposerIter<T> {
    /// Value to decompose
    value: T,
    steps_left: usize,
    /// (1 << logb) - 1 (for % (1<<logb); i.e. to extract least signiciant logb
    /// bits)
    b_mask: T,
    logb: usize,
    // b/2 = 1 << (logb-1)
    bby2: T,
    /// Ciphertext modulus
    q: T,
    /// Log of ciphertext modulus
    logq: usize,
    /// b = 1 << logb
    b: T,
}

impl<T: PrimInt + From<bool> + WrappingSub + Display> Iterator for DecomposerIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.steps_left != 0 {
            self.steps_left -= 1;
            let k_i = self.value & self.b_mask;

            self.value = (self.value - k_i) >> self.logb;

            // if k_i > self.bby2 || (k_i == self.bby2 && ((self.value &
            // T::one()) == T::one())) {     self.value = self.value
            // + T::one();     Some(self.q + k_i - self.b)
            // } else {
            //     Some(k_i)
            // }

            // Following is without branching impl of the commented version above. It
            // happens to speed up bootstrapping for `SMALL_MP_BOOL_PARAMS` (& other
            // parameters as well but I haven't tested) by roughly 15ms.
            // Suprisingly the improvement does not show up when I benchmark
            // `decomposer_iter` in isolation. Putting this remark here as a
            // future task to investiage (TODO).
            let carry_bool =
                k_i > self.bby2 || (k_i == self.bby2 && ((self.value & T::one()) == T::one()));
            let carry = <T as From<bool>>::from(carry_bool);
            let neg_carry = (T::zero().wrapping_sub(&carry));
            self.value = self.value + carry;
            Some((neg_carry & self.q) + k_i - (carry << self.logb))

            // Some(
            //     (self.q & ((carry << self.logq) - (T::one() & carry))) + k_i
            // - (carry << self.logb), )

            // Some(k_i)
        } else {
            None
        }
    }
}

fn round_value<T: PrimInt>(value: T, ignore_bits: usize) -> T {
    if ignore_bits == 0 {
        return value;
    }

    let ignored_msb = (value & ((T::one() << ignore_bits) - T::one())) >> (ignore_bits - 1);
    (value >> ignore_bits) + ignored_msb
}

#[cfg(test)]
mod tests {

    use itertools::Itertools;
    use rand::{thread_rng, Rng};

    use crate::{
        backend::{ModInit, ModularOpsU64},
        decomposer::round_value,
        utils::{generate_prime, tests::Stats, TryConvertFrom1},
    };

    use super::{Decomposer, DefaultDecomposer};

    #[test]
    fn decomposition_works() {
        let logq = 55;
        let logb = 11;
        let d = 5;
        let ring_size = 1 << 11;

        let mut rng = thread_rng();
        let mut stats = Stats::new();

        for i in [true] {
            let q = if i {
                generate_prime(logq, 2 * ring_size, 1u64 << logq).unwrap()
            } else {
                1u64 << logq
            };
            let decomposer = DefaultDecomposer::new(q, logb, d);
            let modq_op = ModularOpsU64::new(q);
            for _ in 0..100000 {
                let value = rng.gen_range(0..q);
                let limbs = decomposer.decompose_to_vec(&value);
                let limbs_from_iter = decomposer.decompose_iter(&value).collect_vec();
                assert_eq!(limbs, limbs_from_iter);
                let value_back = round_value(
                    decomposer.recompose(&limbs, &modq_op),
                    decomposer.ignore_bits,
                );
                let rounded_value = round_value(value, decomposer.ignore_bits);
                assert_eq!(
                    rounded_value, value_back,
                    "Expected {rounded_value} got {value_back} for q={q}"
                );

                stats.add_more(&Vec::<i64>::try_convert_from(&limbs, &q));
            }
        }
        println!("Mean: {}", stats.mean());
        println!("Std: {}", stats.std_dev().abs().log2());
    }
}
