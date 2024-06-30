use std::{usize, vec};

use itertools::{izip, Itertools};
use num_traits::{One, PrimInt, Signed};

use crate::{
    backend::Modulus,
    decomposer::NumInfo,
    random::{RandomElementInModulus, RandomFill},
    Matrix, RowEntity, RowMut,
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

pub(crate) trait ShoupMul {
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

pub(crate) trait ToShoup {
    type Modulus;
    fn to_shoup(value: &Self, modulus: Self::Modulus) -> Self;
}

impl ToShoup for u64 {
    type Modulus = u64;
    fn to_shoup(value: &Self, modulus: Self) -> Self {
        ((*value as u128 * (1u128 << 64)) / modulus as u128) as u64
    }
}

impl ToShoup for Vec<Vec<u64>> {
    type Modulus = u64;
    fn to_shoup(value: &Self, modulus: Self::Modulus) -> Self {
        let (row, col) = value.dimension();
        let mut shoup_value = vec![vec![0u64; col]; row];
        izip!(shoup_value.iter_mut(), value.iter()).for_each(|(shoup_r, r)| {
            izip!(shoup_r.iter_mut(), r.iter()).for_each(|(s, e)| {
                *s = u64::to_shoup(e, modulus);
            })
        });
        shoup_value
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
    for i in 0..hamming_weight {
        let s_index = RandomElementInModulus::<usize, usize>::random(rng, &secret_indices.len());

        let curr_bit = (bytes[byte_index] >> bit_index) & 1;
        if curr_bit == 1 {
            out[secret_indices[s_index]] = T::one();
        } else {
            out[secret_indices[s_index]] = -T::one();
        }

        secret_indices[s_index] = *secret_indices.last().unwrap();
        secret_indices.truncate(secret_indices.len() - 1);

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
pub(crate) fn generate_prime(num_bits: usize, modulo: u64, upper_bound: u64) -> Option<u64> {
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

pub(crate) fn mod_inverse(a: u64, q: u64) -> u64 {
    mod_exponent(a, q - 2, q)
}

pub(crate) fn negacyclic_mul<T: PrimInt, F: Fn(&T, &T) -> T>(
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

/// Returns a polynomial X^{emebedding_factor * si} \mod {Z_Q / X^{N}+1}
pub(crate) fn encode_x_pow_si_with_emebedding_factor<
    R: RowEntity + RowMut,
    M: Modulus<Element = R::Element>,
>(
    si: i32,
    embedding_factor: usize,
    ring_size: usize,
    modulus: &M,
) -> R
where
    R::Element: One,
{
    assert!((si.abs() as usize) < ring_size);
    let mut m = R::zeros(ring_size);
    let si = si * (embedding_factor as i32);
    if si < 0 {
        // X^{-si} = X^{2N-si} = -X^{N-si}, assuming abs(si) < N
        m.as_mut()[ring_size - (si.abs() as usize)] = modulus.neg_one();
    } else {
        m.as_mut()[si as usize] = R::Element::one();
    }
    m
}

pub(crate) fn puncture_p_rng<S: Default + Copy, R: RandomFill<S>>(
    p_rng: &mut R,
    times: usize,
) -> S {
    let mut out = S::default();
    for _ in 0..times {
        RandomFill::<S>::random_fill(p_rng, &mut out);
    }
    return out;
}

pub(crate) fn log2<T: PrimInt + NumInfo>(v: &T) -> usize {
    if (*v & (*v - T::one())) == T::zero() {
        // value is power of 2
        (T::BITS - v.leading_zeros() - 1) as usize
    } else {
        (T::BITS - v.leading_zeros()) as usize
    }
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

#[cfg(test)]
pub(crate) mod tests {
    use std::fmt::Debug;

    use num_traits::{FromPrimitive, PrimInt};

    use crate::random::DefaultSecureRng;

    use super::fill_random_ternary_secret_with_hamming_weight;

    #[derive(Clone)]
    pub(crate) struct Stats<T> {
        pub(crate) samples: Vec<T>,
    }

    impl<T> Default for Stats<T> {
        fn default() -> Self {
            Stats { samples: vec![] }
        }
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

    #[test]
    fn ternary_secret_has_correct_hw() {
        let mut rng = DefaultSecureRng::new();
        for n in 4..15 {
            let ring_size = 1 << n;
            let mut out = vec![0i32; ring_size];
            fill_random_ternary_secret_with_hamming_weight(&mut out, ring_size >> 1, &mut rng);

            // check hamming weight of out equals ring_size/2
            let mut non_zeros = 0;
            out.iter().for_each(|i| {
                if *i != 0 {
                    non_zeros += 1;
                }
            });

            assert_eq!(ring_size >> 1, non_zeros);
        }
    }
}
