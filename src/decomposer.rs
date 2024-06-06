use itertools::Itertools;
use num_traits::{AsPrimitive, FromPrimitive, Num, One, PrimInt, ToPrimitive, WrappingSub, Zero};
use std::{fmt::Debug, marker::PhantomData, ops::Rem};

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
    fn new(q: Self::Element, logb: usize, d: usize) -> Self;
    //FIXME(Jay): there's no reason why it returns a vec instead of an iterator
    fn decompose(&self, v: &Self::Element) -> Vec<Self::Element>;
    fn decomposition_count(&self) -> usize;
}

// TODO(Jay): Shouldn't Decompose also return corresponding gadget vector ?
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

impl<T: PrimInt + ToPrimitive + FromPrimitive + WrappingSub + NumInfo> Decomposer
    for DefaultDecomposer<T>
{
    type Element = T;

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
            d,
            ignore_bits,
            ignore_limbs,
        }
    }

    /// Signed BNAF decomposition. Only returns most significant `d`
    /// decomposition limbs
    ///
    /// Implements algorithm 3 of https://eprint.iacr.org/2021/1161.pdf

    // fn decompose(q){
    //     if value > q/ 2 {
    //         let out = decompose(q - value)
    //         // all values in out mod Q
    //     }else {
    //         let value = value
    //         for i in range(0..d) {
    //             let k = value mod B;
    //             value = value / B
    //             if k > B/2 and (i < d-1) {
    //                 out.push(Q+k-B)
    //                 value += 1
    //             }else {
    //                 out.push(k)
    //             }
    //         }
    //     }
    // }

    // fn decompose(q){
    // let neg_flag = False
    //     if value > q/ 2 {
    //            value = !(q - value) + 1
    //            neg_flag = True
    //     }
    //         let value = value
    //         for i in range(0..d) {
    //             let k = value mod B;
    //             value = value / B
    //             if (k > B/2 and i < d-1) or ((i == d-1) and neg_flag == True) {
    //                 out.push(Q+k-B)
    //                 value += 1
    //             }else {
    //                 out.push(k)
    //             }
    //         }
    //     }
    // }

    fn decompose(&self, value: &T) -> Vec<T> {
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
        utils::{generate_prime, Stats, TryConvertFrom1},
    };

    use super::{Decomposer, DefaultDecomposer};

    #[test]
    fn decomposition_works() {
        let logq = 55;
        let logb = 11;
        let d = 5;
        let ring_size = 1 << 11;

        let mut rng = thread_rng();
        let mut stats = Stats { samples: vec![] };

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
                let limbs = decomposer.decompose(&value);
                let value_back = decomposer.recompose(&limbs, &modq_op);
                let rounded_value =
                    round_value(value, decomposer.ignore_bits) << decomposer.ignore_bits;
                stats.add_more(&Vec::<i64>::try_convert_from(&limbs, &q));
                assert_eq!(
                    rounded_value, value_back,
                    "Expected {rounded_value} got {value_back} for q={q}"
                );
            }
        }
        println!("Mean: {}", stats.mean());
        println!("Std: {}", stats.std_dev().abs().log2());
    }
}
