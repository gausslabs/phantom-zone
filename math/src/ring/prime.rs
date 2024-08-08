use crate::{
    distribution::{DistributionSized, Sampler},
    modulus::{Modulus, Prime},
    ring::{ElemFrom, ElemTo, ModulusOps, SliceOps},
};
use rand::distributions::{Distribution, Uniform};

pub mod noisy;
pub mod precise;

#[derive(Clone, Debug)]
pub struct PrimeRing<T> {
    q: u64,
    q_half: u64,
    q_twice: u64,
    log_q: usize,
    barrett_mu: u128,
    barrett_alpha: usize,
    fft: T,
}

impl<T> PrimeRing<T> {
    fn new(Prime(q): Prime, fft: T) -> Self {
        let log_q = q.next_power_of_two().ilog2() as usize;
        let barrett_mu = (1u128 << (log_q * 2 + 3)) / (q as u128);
        let barrett_alpha = log_q + 3;
        PrimeRing {
            q,
            q_half: q >> 1,
            q_twice: q << 1,
            log_q,
            barrett_mu,
            barrett_alpha,
            fft,
        }
    }

    #[inline(always)]
    pub fn to_i64(&self, v: u64) -> i64 {
        if v < self.q_half {
            v as _
        } else {
            -((self.q - v) as i64)
        }
    }

    #[inline(always)]
    fn reduce_i128(&self, c: i128) -> u64 {
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
    fn reduce_u128(&self, c: u128) -> u64 {
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
    fn reduce_once_assign(&self, a: &mut u64) {
        if *a >= self.q {
            *a -= self.q
        }
    }
}

impl<T> ModulusOps for PrimeRing<T> {
    type Elem = u64;
    type Prep = Shoup;

    fn modulus(&self) -> Modulus {
        Prime(self.q).into()
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
        self.q - 1
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

    #[inline(always)]
    fn prepare(&self, a: &Self::Elem) -> Self::Prep {
        Shoup::new(*a, self.q)
    }

    #[inline(always)]
    fn mul_prep(&self, a: &Self::Elem, b: &Self::Prep) -> Self::Elem {
        b.mul(*a, self.q)
    }
}

impl<T> ElemFrom<u64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_from(&self, v: u64) -> Self::Elem {
        v % self.q
    }
}

impl<T> ElemFrom<i64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_from(&self, v: i64) -> Self::Elem {
        v.rem_euclid(self.q as _) as _
    }
}

impl<T> ElemFrom<u32> for PrimeRing<T> {
    #[inline(always)]
    fn elem_from(&self, v: u32) -> Self::Elem {
        self.elem_from(v as u64)
    }
}

impl<T> ElemFrom<i32> for PrimeRing<T> {
    #[inline(always)]
    fn elem_from(&self, v: i32) -> Self::Elem {
        self.elem_from(v as i64)
    }
}

impl<T> ElemFrom<f64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_from(&self, v: f64) -> Self::Elem {
        self.reduce_i128(v.round() as i128)
    }
}

impl<T> ElemTo<u64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> u64 {
        v
    }
}

impl<T> ElemTo<i64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> i64 {
        self.to_i64(v)
    }
}

impl<T> ElemTo<f64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> f64 {
        self.to_i64(v) as f64
    }
}

impl<T> SliceOps for PrimeRing<T> {}

impl<T> Sampler for PrimeRing<T> {
    fn uniform_distribution(
        &self,
    ) -> impl Distribution<Self::Elem> + DistributionSized<Self::Elem> {
        Uniform::new(0, self.q)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Shoup(u64, u64);

impl Shoup {
    #[inline(always)]
    pub fn new(v: u64, q: u64) -> Self {
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
