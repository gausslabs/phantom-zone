use core::iter::successors;
use num_bigint_dig::{prime::probably_prime, BigUint};
use num_traits::ToPrimitive;

pub fn two_adic_primes(bits: usize, two_adicity: usize) -> impl Iterator<Item = u64> {
    assert!(bits > two_adicity);
    let (min, max) = (1 << (bits - two_adicity - 1), 1 << (bits - two_adicity));
    primes((min..max).rev().map(move |v| (v << two_adicity) + 1))
}

fn primes(candidates: impl IntoIterator<Item = u64>) -> impl Iterator<Item = u64> {
    candidates
        .into_iter()
        .filter(|candidate| probably_prime(&(*candidate).into(), 20))
}

pub fn generator(q: u64) -> u64 {
    let order = q - 1;
    (1..order)
        .find(|g| mod_pow(*g, order >> 1, q) == order)
        .unwrap()
}

pub fn two_adic_generator(q: u64, two_adicity: usize) -> u64 {
    debug_assert_eq!((q - 1) % (1 << two_adicity) as u64, 0);
    mod_pow(generator(q), (q - 1) >> two_adicity, q)
}

pub fn mod_pow(b: u64, e: u64, q: u64) -> u64 {
    BigUint::from(b)
        .modpow(&e.into(), &q.into())
        .to_u64()
        .unwrap()
}

pub fn mod_inv(v: u64, q: u64) -> u64 {
    mod_pow(v, q - 2, q)
}

pub fn mod_powers(b: u64, q: u64) -> impl Iterator<Item = u64> {
    successors(Some(1), move |v| {
        (((*v as u128 * b as u128) % q as u128) as u64).into()
    })
}
