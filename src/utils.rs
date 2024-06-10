use std::{fmt::Debug, usize};

use itertools::Itertools;
use num_traits::{FromPrimitive, PrimInt, Signed, Unsigned};

use crate::{
    backend::Modulus,
    random::{RandomElement, RandomElementInModulus, RandomFill},
};
pub trait WithLocal {
    fn with_local<F, R>(func: F) -> R
    where
        F: Fn(&Self) -> R;

    fn with_local_mut<F, R>(func: F) -> R
    where
        F: Fn(&mut Self) -> R;

    fn with_local_mut_mut<F, R>(func: &mut F) -> R
    where
        F: FnMut(&mut Self) -> R;
}

pub trait Global {
    fn global() -> &'static Self;
}

pub trait ShoupMul {
    fn representation(value: Self, q: Self) -> Self;
    fn mul(a: Self, b: Self, b_shoup: Self, q: Self) -> Self;
}

impl ShoupMul for u64 {
    #[inline]
    fn representation(value: Self, q: Self) -> Self {
        ((value as u128 * (1u128 << 64)) / q as u128) as u64
    }

    #[inline]
    /// Returns a * b % q
    fn mul(a: Self, b: Self, b_shoup: Self, q: Self) -> Self {
        (b.wrapping_mul(a))
            .wrapping_sub(q.wrapping_mul(((b_shoup as u128 * a as u128) >> 64) as u64))
    }
}

pub fn fill_random_ternary_secret_with_hamming_weight<
    T: Signed,
    R: RandomFill<[u8]> + RandomElementInModulus<usize, usize>,
>(
    out: &mut [T],
    hamming_weight: usize,
    rng: &mut R,
) {
    let mut bytes = vec![0u8; hamming_weight.div_ceil(8)];
    RandomFill::<[u8]>::random_fill(rng, &mut bytes);

    let size = out.len();
    let mut secret_indices = (0..size).into_iter().map(|i| i).collect_vec();
    let mut bit_index = 0;
    let mut byte_index = 0;
    for _ in 0..hamming_weight {
        let s_index = RandomElementInModulus::<usize, usize>::random(rng, &secret_indices.len());

        let curr_bit = (bytes[byte_index] >> bit_index) & 1;
        if curr_bit == 1 {
            out[secret_indices[s_index]] = T::one();
        } else {
            out[secret_indices[s_index]] = -T::one();
        }

        secret_indices[s_index] = *secret_indices.last().unwrap();
        secret_indices.truncate(secret_indices.len());

        if bit_index == 7 {
            bit_index = 0;
            byte_index += 1;
        } else {
            bit_index += 1;
        }
    }
}

// TODO (Jay): this is only a workaround. Add a propoer way to perform primality
// tests.
fn is_probably_prime(candidate: u64) -> bool {
    num_bigint_dig::prime::probably_prime(&num_bigint_dig::BigUint::from(candidate), 0)
}

/// Finds prime that satisfy
/// - $prime \lt upper_bound$
/// - $\log{prime} = num_bits$
/// - `prime % modulo == 1`
pub fn generate_prime(num_bits: usize, modulo: u64, upper_bound: u64) -> Option<u64> {
    let leading_zeros = (64 - num_bits) as u32;

    let mut tentative_prime = upper_bound - 1;
    while tentative_prime % modulo != 1 && tentative_prime.leading_zeros() == leading_zeros {
        tentative_prime -= 1;
    }

    while !is_probably_prime(tentative_prime)
        && tentative_prime.leading_zeros() == leading_zeros
        && tentative_prime >= modulo
    {
        tentative_prime -= modulo;
    }

    if is_probably_prime(tentative_prime) && tentative_prime.leading_zeros() == leading_zeros {
        Some(tentative_prime)
    } else {
        None
    }
}

/// Returns a^b mod q
pub fn mod_exponent(a: u64, mut b: u64, q: u64) -> u64 {
    let mod_mul = |v1: &u64, v2: &u64| {
        let tmp = *v1 as u128 * *v2 as u128;
        (tmp % q as u128) as u64
    };

    let mut acc = a;
    let mut out = 1;
    while b != 0 {
        let flag = b & 1;

        if flag == 1 {
            out = mod_mul(&acc, &out);
        }
        acc = mod_mul(&acc, &acc);

        b >>= 1;
    }

    out
}

pub fn mod_inverse(a: u64, q: u64) -> u64 {
    mod_exponent(a, q - 2, q)
}

pub fn negacyclic_mul<T: PrimInt, F: Fn(&T, &T) -> T>(
    a: &[T],
    b: &[T],
    mul: F,
    modulus: T,
) -> Vec<T> {
    let mut r = vec![T::zero(); a.len()];
    for i in 0..a.len() {
        for j in 0..i + 1 {
            // println!("i: {j} {}", i - j);
            r[i] = (r[i] + mul(&a[j], &b[i - j])) % modulus;
        }

        for j in i + 1..a.len() {
            // println!("i: {j} {}", a.len() - j + i);
            r[i] = (r[i] + modulus - mul(&a[j], &b[a.len() - j + i])) % modulus;
        }
        // println!("")
    }

    return r;
}

pub trait TryConvertFrom1<T: ?Sized, P> {
    fn try_convert_from(value: &T, parameters: &P) -> Self;
}

impl<P: Modulus<Element = u64>> TryConvertFrom1<[i64], P> for Vec<u64> {
    fn try_convert_from(value: &[i64], parameters: &P) -> Self {
        value
            .iter()
            .map(|v| parameters.map_element_from_i64(*v))
            .collect_vec()
    }
}

impl<P: Modulus<Element = u64>> TryConvertFrom1<[i32], P> for Vec<u64> {
    fn try_convert_from(value: &[i32], parameters: &P) -> Self {
        value
            .iter()
            .map(|v| parameters.map_element_from_i64(*v as i64))
            .collect_vec()
    }
}

impl<P: Modulus> TryConvertFrom1<[P::Element], P> for Vec<i64> {
    fn try_convert_from(value: &[P::Element], parameters: &P) -> Self {
        value
            .iter()
            .map(|v| parameters.map_element_to_i64(v))
            .collect_vec()
    }
}

// pub trait TryConvertFrom<T: ?Sized> {
//     type Parameters: ?Sized;

//     fn try_convert_from(value: &T, parameters: &Self::Parameters) -> Self;
// }

// impl TryConvertFrom1<[i32]> for Vec<Vec<u32>> {
//     type Parameters = u32;
//     fn try_convert_from(value: &[i32], parameters: &Self::Parameters) -> Self
// {         let row0 = value
//             .iter()
//             .map(|v| {
//                 let is_neg = v.is_negative();
//                 let v_u32 = v.abs() as u32;

//                 assert!(v_u32 < *parameters);

//                 if is_neg {
//                     parameters - v_u32
//                 } else {
//                     v_u32
//                 }
//             })
//             .collect_vec();

//         vec![row0]
//     }
// }

// impl TryConvertFrom1<[i32]> for Vec<Vec<u64>> {
//     type Parameters = u64;
//     fn try_convert_from(value: &[i32], parameters: &Self::Parameters) -> Self
// {         let row0 = value
//             .iter()
//             .map(|v| {
//                 let is_neg = v.is_negative();
//                 let v_u64 = v.abs() as u64;

//                 assert!(v_u64 < *parameters);

//                 if is_neg {
//                     parameters - v_u64
//                 } else {
//                     v_u64
//                 }
//             })
//             .collect_vec();

//         vec![row0]
//     }
// }

// impl TryConvertFrom1<[i32]> for Vec<u64> {
//     type Parameters = u64;
//     fn try_convert_from(value: &[i32], parameters: &Self::Parameters) -> Self
// {         value
//             .iter()
//             .map(|v| {
//                 let is_neg = v.is_negative();
//                 let v_u64 = v.abs() as u64;

//                 assert!(v_u64 < *parameters);

//                 if is_neg {
//                     parameters - v_u64
//                 } else {
//                     v_u64
//                 }
//             })
//             .collect_vec()
//     }
// }

// impl TryConvertFrom1<[u64]> for Vec<i64> {
//     type Parameters = u64;
//     fn try_convert_from(value: &[u64], parameters: &Self::Parameters) -> Self
// {         let q = *parameters;
//         let qby2 = q / 2;
//         value
//             .iter()
//             .map(|v| {
//                 if *v > qby2 {
//                     -((q - v) as i64)
//                 } else {
//                     *v as i64
//                 }
//             })
//             .collect_vec()
//     }
// }

pub(crate) struct Stats<T> {
    pub(crate) samples: Vec<T>,
}

impl<T: PrimInt + FromPrimitive + Debug> Stats<T>
where
    // T: for<'a> Sum<&'a T>,
    T: for<'a> std::iter::Sum<&'a T> + std::iter::Sum<T>,
{
    pub(crate) fn new() -> Self {
        Self { samples: vec![] }
    }

    pub(crate) fn mean(&self) -> f64 {
        self.samples.iter().sum::<T>().to_f64().unwrap() / (self.samples.len() as f64)
    }

    pub(crate) fn std_dev(&self) -> f64 {
        let mean = self.mean();

        // diff
        let diff_sq = self
            .samples
            .iter()
            .map(|v| {
                let t = v.to_f64().unwrap() - mean;
                t * t
            })
            .into_iter()
            .sum::<f64>();

        (diff_sq / (self.samples.len() as f64)).sqrt()
    }

    pub(crate) fn add_more(&mut self, values: &[T]) {
        self.samples.extend(values.iter());
    }
}
