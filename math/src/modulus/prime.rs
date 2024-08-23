use crate::{
    decomposer::PrimeDecomposer,
    distribution::{DistributionSized, Sampler},
    modulus::{ElemFrom, ElemOps, ElemTo, Modulus, ModulusOps},
};
use core::ops::Deref;
use num_bigint_dig::BigUint;
use num_traits::ToPrimitive;
use rand::distributions::{Distribution, Uniform};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(into = "SerdePrime", from = "SerdePrime")
)]
pub struct Prime {
    q: u64,
    q_half: u64,
    q_twice: u64,
    log_q: usize,
    barrett_mu: u128,
    barrett_alpha: usize,
}

impl Prime {
    pub const fn new(q: u64) -> Self {
        assert!(q > 1);
        let log_q = q.next_power_of_two().ilog2() as usize;
        let barrett_mu = (1u128 << (log_q * 2 + 3)) / (q as u128);
        let barrett_alpha = log_q + 3;
        Self {
            q,
            q_half: q >> 1,
            q_twice: q << 1,
            log_q,
            barrett_mu,
            barrett_alpha,
        }
    }

    #[inline(always)]
    pub fn bits(&self) -> usize {
        self.log_q
    }

    #[inline(always)]
    pub fn max(&self) -> u64 {
        self.q - 1
    }

    #[inline(always)]
    pub fn half(&self) -> u64 {
        self.q_half
    }

    #[inline(always)]
    pub fn as_f64(&self) -> f64 {
        self.q as _
    }

    pub fn pow(&self, b: u64, e: u64) -> u64 {
        BigUint::from(b)
            .modpow(&BigUint::from(e), &BigUint::from(self.q))
            .to_u64()
            .unwrap()
    }

    pub fn two_adic_generator(&self, two_adicity: usize) -> u64 {
        assert_eq!((self.q - 1) % (1 << two_adicity) as u64, 0);
        self.pow(self.multiplicative_generator(), (self.q - 1) >> two_adicity)
    }

    pub fn multiplicative_generator(&self) -> u64 {
        let order = self.q - 1;
        (1..order)
            .find(|g| self.pow(*g, order >> 1) == order)
            .unwrap()
    }

    #[inline(always)]
    pub fn center(&self, v: u64) -> i64 {
        if v >= self.q_half {
            -((self.q - v) as i64)
        } else {
            v as _
        }
    }

    #[inline(always)]
    pub(crate) fn reduce_i128(&self, c: i128) -> u64 {
        // c / (2^{n + \beta})
        // note: \beta is assumed to -2
        let tmp = c >> (self.log_q - 2);
        // k = ((c / (2^{n + \beta})) * \mu) / 2^{\alpha - (-2)}
        let k = (tmp * self.barrett_mu as i128) >> (self.barrett_alpha + 2);
        // c - k*p
        let tmp = k * (self.q as i128);

        let mut c = (c - tmp) as u64;
        self.reduce_once_assign(&mut c);
        c
    }

    #[inline(always)]
    pub(crate) fn reduce_u128(&self, c: u128) -> u64 {
        // c / (2^{n + \beta})
        // note: \beta is assumed to -2
        let tmp = c >> (self.log_q - 2);
        // k = ((c / (2^{n + \beta})) * \mu) / 2^{\alpha - (-2)}
        let k = (tmp * self.barrett_mu) >> (self.barrett_alpha + 2);
        // c - k*p
        let tmp = k * (self.q as u128);

        let mut c = (c - tmp) as u64;
        self.reduce_once_assign(&mut c);
        c
    }

    #[inline(always)]
    pub(crate) fn reduce_once_assign(&self, a: &mut u64) {
        if *a >= self.q {
            *a -= self.q
        }
    }
}

#[cfg(any(test, feature = "dev"))]
impl Prime {
    pub fn gen(bits: usize, two_adicity: usize) -> Self {
        Self::gen_iter(bits, two_adicity).next().unwrap()
    }

    pub fn gen_iter(bits: usize, two_adicity: usize) -> impl Iterator<Item = Self> {
        assert!(bits > two_adicity);
        let min = 1 << (bits - two_adicity - 1);
        let max = min << 1;
        let candidates = (min..max).rev().map(move |hi| (hi << two_adicity) + 1);
        candidates
            .into_iter()
            .filter(|v| is_prime(*v))
            .map(Self::new)
    }
}

impl Deref for Prime {
    type Target = u64;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.q
    }
}

impl ElemOps for Prime {
    type Elem = u64;
}

impl ModulusOps for Prime {
    type ElemPrep = Shoup;
    type Decomposer = PrimeDecomposer;

    fn new(modulus: Modulus) -> Self {
        modulus.try_into().unwrap()
    }

    #[inline(always)]
    fn modulus(&self) -> Modulus {
        (*self).into()
    }

    #[inline(always)]
    fn uniform_distribution(
        &self,
    ) -> impl Distribution<Self::Elem> + DistributionSized<Self::Elem> {
        Uniform::new_inclusive(0, self.max())
    }

    #[inline(always)]
    fn zero(&self) -> Self::Elem {
        0
    }

    #[inline(always)]
    fn one(&self) -> Self::Elem {
        1
    }

    #[inline(always)]
    fn neg_one(&self) -> Self::Elem {
        self.max()
    }

    #[inline(always)]
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        debug_assert!(*a < self.q);
        if *a != 0 {
            self.q - a
        } else {
            0
        }
    }

    #[inline(always)]
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        debug_assert!(*a < self.q);
        debug_assert!(*b < self.q);
        let mut c = a + b;
        self.reduce_once_assign(&mut c);
        c
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        debug_assert!(*a < self.q);
        debug_assert!(*b < self.q);
        if a >= b {
            a - b
        } else {
            self.q + a - b
        }
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        debug_assert!(*a < self.q_twice);
        debug_assert!(*b < self.q_twice);

        self.reduce_u128(*a as u128 * *b as u128)
    }

    fn inv(&self, a: &Self::Elem) -> Option<Self::Elem> {
        (*a != 0).then(|| self.pow(*a, self.q - 2))
    }

    #[inline(always)]
    fn prepare(&self, a: &Self::Elem) -> Self::ElemPrep {
        Shoup::new(*a, self.q)
    }

    #[inline(always)]
    fn mul_prep(&self, a: &Self::Elem, b: &Self::ElemPrep) -> Self::Elem {
        b.mul(*a, self.q)
    }
}

impl ElemFrom<u64> for Prime {
    #[inline(always)]
    fn elem_from(&self, v: u64) -> Self::Elem {
        v % self.q
    }
}

impl ElemFrom<i64> for Prime {
    #[inline(always)]
    fn elem_from(&self, v: i64) -> Self::Elem {
        v.rem_euclid(self.q as _) as _
    }
}

impl ElemFrom<u32> for Prime {
    #[inline(always)]
    fn elem_from(&self, v: u32) -> Self::Elem {
        self.elem_from(v as u64)
    }
}

impl ElemFrom<i32> for Prime {
    #[inline(always)]
    fn elem_from(&self, v: i32) -> Self::Elem {
        self.elem_from(v as i64)
    }
}

impl ElemFrom<f64> for Prime {
    #[inline(always)]
    fn elem_from(&self, v: f64) -> Self::Elem {
        self.reduce_i128(v.round() as i128)
    }
}

impl ElemTo<u64> for Prime {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> u64 {
        v
    }
}

impl ElemTo<i64> for Prime {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> i64 {
        self.center(v)
    }
}

impl ElemTo<f64> for Prime {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> f64 {
        self.center(v) as f64
    }
}

impl Sampler for Prime {}

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

#[derive(Clone, Copy, Debug, Default)]
pub struct Shoup(u64, u64);

impl Shoup {
    #[inline(always)]
    pub fn new(v: u64, q: u64) -> Self {
        debug_assert!(v < q);
        let quotient = (((v as u128) << 64) / q as u128) as _;
        Self(v, quotient)
    }

    #[inline(always)]
    pub fn value(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    pub fn quotient(&self) -> u64 {
        self.1
    }

    #[inline(always)]
    pub fn mul(&self, a: u64, q: u64) -> u64 {
        let t = ((self.quotient() as u128 * a as u128) >> 64) as _;
        (a.wrapping_mul(self.value())).wrapping_sub(q.wrapping_mul(t))
    }
}

#[cfg(any(test, feature = "dev"))]
pub(crate) fn is_prime(q: u64) -> bool {
    num_bigint_dig::prime::probably_prime(&(q).into(), 20)
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct SerdePrime {
    q: u64,
}

#[cfg(feature = "serde")]
impl From<SerdePrime> for Prime {
    fn from(value: SerdePrime) -> Self {
        Self::new(value.q)
    }
}

#[cfg(feature = "serde")]
impl From<Prime> for SerdePrime {
    fn from(value: Prime) -> Self {
        Self { q: value.q }
    }
}
