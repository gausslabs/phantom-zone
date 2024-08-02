use core::iter::successors;
use num_bigint_dig::{prime::probably_prime, BigUint};
use num_traits::ToPrimitive;

#[derive(Clone, Copy, Debug)]
pub enum Modulus {
    PowerOfTwo(PowerOfTwo),
    Prime(Prime),
}

impl Modulus {
    pub fn native() -> Self {
        PowerOfTwo::native().into()
    }

    pub fn bits(&self) -> usize {
        match self {
            Self::PowerOfTwo(power_of_two) => power_of_two.bits(),
            Self::Prime(prime) => prime.bits(),
        }
    }

    pub fn max(&self) -> u64 {
        match self {
            Self::PowerOfTwo(power_of_two) => power_of_two.max(),
            Self::Prime(prime) => prime.max(),
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            Self::PowerOfTwo(power_of_two) => power_of_two.to_f64(),
            Self::Prime(prime) => prime.to_f64(),
        }
    }
}

impl From<Prime> for Modulus {
    fn from(value: Prime) -> Self {
        Self::Prime(value)
    }
}

impl TryFrom<Modulus> for Prime {
    type Error = ();

    fn try_from(value: Modulus) -> Result<Self, Self::Error> {
        if let Modulus::Prime(prime) = value {
            Ok(prime)
        } else {
            Err(())
        }
    }
}

impl From<PowerOfTwo> for Modulus {
    fn from(value: PowerOfTwo) -> Self {
        Self::PowerOfTwo(value)
    }
}

impl TryFrom<Modulus> for PowerOfTwo {
    type Error = ();

    fn try_from(value: Modulus) -> Result<Self, Self::Error> {
        if let Modulus::PowerOfTwo(prime) = value {
            Ok(prime)
        } else {
            Err(())
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PowerOfTwo(pub(crate) usize);

impl PowerOfTwo {
    pub fn new(log_q: usize) -> Self {
        assert!(log_q <= 64);
        Self(log_q)
    }

    pub fn native() -> Self {
        Self(64)
    }

    pub fn bits(&self) -> usize {
        self.0
    }

    pub fn max(&self) -> u64 {
        if self.0 == 64 {
            u64::MAX
        } else {
            (1 << self.0) - 1
        }
    }

    pub fn mask(&self) -> u64 {
        self.max()
    }

    pub fn to_f64(&self) -> f64 {
        2f64.powi(self.0 as _)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Prime(pub(crate) u64);

impl Prime {
    pub fn gen(bits: usize, two_adicity: usize) -> Self {
        Self::gen_iter(bits, two_adicity).next().unwrap()
    }

    pub fn gen_iter(bits: usize, two_adicity: usize) -> impl Iterator<Item = Self> {
        assert!(bits > two_adicity);
        let min = 1 << (bits - two_adicity - 1);
        let max = min << 1;
        let candidates = (min..max).rev().map(move |hi| (hi << two_adicity) + 1);
        candidates.into_iter().filter(|v| is_prime(*v)).map(Self)
    }

    pub fn bits(&self) -> usize {
        self.0.next_power_of_two().ilog2() as _
    }

    pub fn max(&self) -> u64 {
        self.0 - 1
    }

    pub fn to_f64(&self) -> f64 {
        self.0 as _
    }
}

impl From<Prime> for u64 {
    fn from(Prime(value): Prime) -> Self {
        value
    }
}

pub(crate) fn is_prime(q: u64) -> bool {
    probably_prime(&(q).into(), 20)
}

pub(crate) fn pow_mod(b: u64, e: u64, q: u64) -> u64 {
    BigUint::from(b)
        .modpow(&e.into(), &q.into())
        .to_u64()
        .unwrap()
}

pub(crate) fn inv_mod(a: u64, q: u64) -> Option<u64> {
    let inv = pow_mod(a, q - 2, q);
    (mul_mod(a, inv, q) == 1).then_some(inv)
}

pub(crate) fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 + b as u128) % q as u128) as _
}

pub(crate) fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as _
}

pub(crate) fn powers_mod(b: u64, q: u64) -> impl Iterator<Item = u64> {
    successors(Some(1), move |v| mul_mod(*v, b, q).into())
}
