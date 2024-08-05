use crate::{
    distribution::{DistributionSized, Sampler},
    modulus::PowerOfTwo,
    ring::{ArithmeticOps, ElemFrom, ElemTo, SliceOps},
};
use rand::distributions::{Distribution, Uniform};

pub mod noisy;
pub mod precise;

#[derive(Clone, Copy, Debug)]
pub struct PowerOfTwoRing<T, const NATIVE: bool> {
    modulus: PowerOfTwo,
    mask: u64,
    fft: T,
}

impl<T, const NATIVE: bool> PowerOfTwoRing<T, NATIVE> {
    fn new(modulus: PowerOfTwo, fft: T) -> Self {
        if NATIVE {
            assert_eq!(modulus.bits(), 64);
        } else {
            assert!(modulus.bits() < 64);
        }
        Self {
            modulus,
            mask: modulus.mask(),
            fft,
        }
    }

    #[inline(always)]
    fn reduce(&self, v: u64) -> u64 {
        if NATIVE {
            v
        } else {
            v & self.mask
        }
    }

    #[inline(always)]
    fn to_i64(&self, v: u64) -> i64 {
        if NATIVE {
            v as _
        } else {
            v.wrapping_sub((v >> (self.modulus.bits() - 1)) << self.modulus.bits()) as i64
        }
    }
}

impl<T, const NATIVE: bool> ArithmeticOps for PowerOfTwoRing<T, NATIVE> {
    type Elem = u64;
    type Prep = u64;

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
        self.mask
    }

    #[inline(always)]
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        self.reduce(a.wrapping_neg())
    }

    #[inline(always)]
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.reduce(a.wrapping_add(*b))
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.reduce(a.wrapping_sub(*b))
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.reduce(a.wrapping_mul(*b))
    }

    #[inline(always)]
    fn prepare(&self, a: &Self::Elem) -> Self::Prep {
        *a
    }

    #[inline(always)]
    fn mul_prep(&self, a: &Self::Elem, b: &Self::Prep) -> Self::Elem {
        self.mul(a, b)
    }
}

impl<T, const NATIVE: bool> ElemFrom<u64> for PowerOfTwoRing<T, NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: u64) -> Self::Elem {
        self.reduce(v)
    }
}

impl<T, const NATIVE: bool> ElemFrom<i64> for PowerOfTwoRing<T, NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: i64) -> Self::Elem {
        self.reduce(v as u64)
    }
}

impl<T, const NATIVE: bool> ElemFrom<u32> for PowerOfTwoRing<T, NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: u32) -> Self::Elem {
        self.elem_from(v as u64)
    }
}

impl<T, const NATIVE: bool> ElemFrom<i32> for PowerOfTwoRing<T, NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: i32) -> Self::Elem {
        self.elem_from(v as i64)
    }
}

impl<T, const NATIVE: bool> ElemTo<u64> for PowerOfTwoRing<T, NATIVE> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> u64 {
        v
    }
}

impl<T, const NATIVE: bool> ElemTo<i64> for PowerOfTwoRing<T, NATIVE> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> i64 {
        self.to_i64(v)
    }
}

impl<T, const NATIVE: bool> SliceOps for PowerOfTwoRing<T, NATIVE> {}

impl<T, const NATIVE: bool> Sampler for PowerOfTwoRing<T, NATIVE> {
    fn uniform_distribution(
        &self,
    ) -> impl Distribution<Self::Elem> + DistributionSized<Self::Elem> {
        Uniform::new_inclusive(0, self.modulus.max())
    }
}
