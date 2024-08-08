use core::iter::successors;
use num_bigint_dig::{prime::probably_prime, BigUint};
use num_traits::{AsPrimitive, PrimInt, ToPrimitive};

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

    #[inline(always)]
    pub fn to_i64(&self, v: u64) -> i64 {
        match self {
            Self::PowerOfTwo(po2) => {
                if po2.0 == 64 {
                    v as _
                } else {
                    assert!(v < (1u64 << po2.0));
                    v.wrapping_sub((v >> (po2.0 - 1)) << po2.0) as _
                }
            }
            Modulus::Prime(prime) => {
                assert!(v < prime.0);
                if v > (prime.0 >> 1) {
                    -((prime.0 - v) as i64)
                } else {
                    v as _
                }
            }
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
    pub fn new(q: u64) -> Self {
        debug_assert!(is_prime(q));
        Self(q)
    }

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

pub fn pow_mod<T: PrimInt + AsPrimitive<u64>>(b: T, e: T, q: T) -> T {
    let [b, e, q] = [b, e, q].map(|v| BigUint::from(v.as_()));
    T::from(b.modpow(&e, &q).to_u64().unwrap()).unwrap()
}

pub fn inv_mod<T: PrimInt + AsPrimitive<u64>>(a: T, q: T) -> Option<T> {
    let [a, q] = [a, q].map(|v| v.as_());
    let inv = pow_mod(a, q - 2, q);
    (mul_mod(a, inv, q) == 1).then(|| T::from(inv).unwrap())
}

pub fn neg_mod<T: PrimInt + AsPrimitive<u64>>(a: T, q: T) -> T {
    let [a, q] = [a, q].map(|v| v.as_());
    (a == 0)
        .then(T::zero)
        .unwrap_or_else(|| T::from(q - (a % q)).unwrap())
}

pub fn add_mod<T: PrimInt + AsPrimitive<u128>>(a: T, b: T, q: T) -> T {
    let [a, b, q] = [a, b, q].map(|v| v.as_());
    T::from((a + b) % q).unwrap()
}

pub fn mul_mod<T: PrimInt + AsPrimitive<u128>>(a: T, b: T, q: T) -> T {
    let [a, b, q] = [a, b, q].map(|v| v.as_());
    T::from((a * b) % q).unwrap()
}

pub fn powers_mod<T: PrimInt + AsPrimitive<u128>>(b: T, q: T) -> impl Iterator<Item = T> {
    successors(Some(T::one()), move |v| mul_mod(*v, b, q).into())
}
