use itertools::Itertools;
use num_traits::{AsPrimitive, One, PrimInt, ToPrimitive, WrappingSub, Zero};
use std::{fmt::Debug, marker::PhantomData, ops::Rem};

use crate::backend::{ArithmeticOps, ModularOpsU64};

pub fn gadget_vector<T: PrimInt>(logq: usize, logb: usize, d: usize) -> Vec<T> {
    let d_ideal = (logq as f64 / logb as f64).ceil().to_usize().unwrap();
    let ignored_limbs = d_ideal - d;
    (ignored_limbs..ignored_limbs + d)
        .into_iter()
        .map(|i| T::one() << (logb * i))
        .collect_vec()
}

pub trait Decomposer {
    type Element;
    //FIXME(Jay): there's no reason why it returns a vec instead of an iterator
    fn decompose(&self, v: &Self::Element) -> Vec<Self::Element>;
    fn d(&self) -> usize;
}

pub struct DefaultDecomposer<T> {
    q: T,
    logq: usize,
    logb: usize,
    d: usize,
    ignore_bits: usize,
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
    pub fn new(q: T, logb: usize, d: usize) -> DefaultDecomposer<T> {
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
            d,
            ignore_bits,
            ignore_limbs,
        }
    }

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
}

impl<T: PrimInt + WrappingSub + Debug> Decomposer for DefaultDecomposer<T> {
    type Element = T;
    fn decompose(&self, value: &T) -> Vec<T> {
        let mut value = round_value(*value, self.ignore_bits);

        let q = self.q;
        if value >= (q >> 1) {
            value = value.wrapping_sub(&q);
        }

        let logb = self.logb;
        let b = T::one() << logb; // base
        let b_by2 = T::one() << (logb - 1);
        // let neg_b_by2_modq = q - b_by2;
        let full_mask = (T::one() << logb) - T::one();
        // let half_mask = b_by2 - T::one();
        let mut carry = T::zero();
        let mut out = Vec::<T>::with_capacity(self.d);
        for i in 0..self.d {
            let mut limb = ((value >> (logb * i)) & full_mask) + carry;
            carry = T::zero();
            if limb > b_by2 {
                limb = (q + limb) - b;
                carry = T::one();
            }

            // carry = ((q + g - limb) % q) >> logb;

            // carry = limb & b_by2;
            // limb = (q + limb) - (carry << 1);
            // if limb > q {
            //     limb = limb - q;
            // }
            out.push(limb);

            // carry = carry >> (logb - 1);
        }

        return out;
    }

    fn d(&self) -> usize {
        self.d
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
    use rand::{thread_rng, Rng};

    use crate::{
        backend::{ModInit, ModularOpsU64},
        decomposer::round_value,
        utils::generate_prime,
    };

    use super::{Decomposer, DefaultDecomposer};

    #[test]
    fn decomposition_works() {
        let logq = 15;
        let logb = 3;
        let d = 5;

        let mut rng = thread_rng();

        // q is prime of bits logq and i is true, other q = 1<<logq
        // FIXME: Test fails when q is prime, albeit the difference is minute
        for i in [true] {
            let q = if i {
                generate_prime(logq, 1 << 4, 1u64 << logq).unwrap()
            } else {
                1u64 << logq
            };
            let decomposer = DefaultDecomposer::new(q, logb, d);
            let modq_op = ModularOpsU64::new(q);
            for _ in 0..100 {
                let value = rng.gen_range(0..q);
                let limbs = decomposer.decompose(&value);
                let value_back = decomposer.recompose(&limbs, &modq_op);
                let rounded_value =
                    round_value(value, decomposer.ignore_bits) << decomposer.ignore_bits;
                // dbg!(rounded_value, value, value_back);
                assert_eq!(
                    rounded_value, value_back,
                    "Expected {rounded_value} got {value_back} for q={q}"
                );
            }
        }
    }
}
