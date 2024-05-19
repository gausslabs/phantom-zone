use std::{fmt::Debug, usize};

use itertools::Itertools;
use num_traits::{FromPrimitive, PrimInt, Signed};

use crate::RandomUniformDist;
pub trait WithLocal {
    fn with_local<F, R>(func: F) -> R
    where
        F: Fn(&Self) -> R;

    fn with_local_mut<F, R>(func: F) -> R
    where
        F: Fn(&mut Self) -> R;
}

pub fn fill_random_ternary_secret_with_hamming_weight<
    T: Signed,
    R: RandomUniformDist<[u8], Parameters = u8> + RandomUniformDist<usize, Parameters = usize>,
>(
    out: &mut [T],
    hamming_weight: usize,
    rng: &mut R,
) {
    let mut bytes = vec![0u8; hamming_weight.div_ceil(8)];
    RandomUniformDist::<[u8]>::random_fill(rng, &0, &mut bytes);

    let size = out.len();
    let mut secret_indices = (0..size).into_iter().map(|i| i).collect_vec();
    let mut bit_index = 0;
    let mut byte_index = 0;
    for _ in 0..hamming_weight {
        let mut s_index = 0usize;
        RandomUniformDist::<usize>::random_fill(rng, &secret_indices.len(), &mut s_index);

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

pub fn shoup_representation_fq(v: u64, q: u64) -> u64 {
    ((v as u128 * (1u128 << 64)) / q as u128) as u64
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

pub trait TryConvertFrom<T: ?Sized> {
    type Parameters: ?Sized;

    fn try_convert_from(value: &T, parameters: &Self::Parameters) -> Self;
}

impl TryConvertFrom<[i32]> for Vec<Vec<u32>> {
    type Parameters = u32;
    fn try_convert_from(value: &[i32], parameters: &Self::Parameters) -> Self {
        let row0 = value
            .iter()
            .map(|v| {
                let is_neg = v.is_negative();
                let v_u32 = v.abs() as u32;

                assert!(v_u32 < *parameters);

                if is_neg {
                    parameters - v_u32
                } else {
                    v_u32
                }
            })
            .collect_vec();

        vec![row0]
    }
}

impl TryConvertFrom<[i32]> for Vec<Vec<u64>> {
    type Parameters = u64;
    fn try_convert_from(value: &[i32], parameters: &Self::Parameters) -> Self {
        let row0 = value
            .iter()
            .map(|v| {
                let is_neg = v.is_negative();
                let v_u64 = v.abs() as u64;

                assert!(v_u64 < *parameters);

                if is_neg {
                    parameters - v_u64
                } else {
                    v_u64
                }
            })
            .collect_vec();

        vec![row0]
    }
}

impl TryConvertFrom<[i32]> for Vec<u64> {
    type Parameters = u64;
    fn try_convert_from(value: &[i32], parameters: &Self::Parameters) -> Self {
        value
            .iter()
            .map(|v| {
                let is_neg = v.is_negative();
                let v_u64 = v.abs() as u64;

                assert!(v_u64 < *parameters);

                if is_neg {
                    parameters - v_u64
                } else {
                    v_u64
                }
            })
            .collect_vec()
    }
}

impl TryConvertFrom<[u64]> for Vec<i64> {
    type Parameters = u64;
    fn try_convert_from(value: &[u64], parameters: &Self::Parameters) -> Self {
        let q = *parameters;
        let qby2 = q / 2;
        value
            .iter()
            .map(|v| {
                if *v > qby2 {
                    -((q - v) as i64)
                } else {
                    *v as i64
                }
            })
            .collect_vec()
    }
}

pub(crate) struct Stats<T> {
    pub(crate) samples: Vec<T>,
}

impl<T: PrimInt + FromPrimitive + Debug> Stats<T>
where
    // T: for<'a> Sum<&'a T>,
    T: for<'a> std::iter::Sum<&'a T> + std::iter::Sum<T>,
{
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
