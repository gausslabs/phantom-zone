use crate::{
    distribution::{DistributionSized, Sampler},
    modulus::{Modulus, Prime},
    ring::{ElemFrom, ElemTo, ModulusOps, SliceOps},
};
use core::fmt::Debug;
use rand::distributions::Distribution;

pub mod noisy;
pub mod precise;

#[derive(Clone, Debug)]
pub struct PrimeRing<T> {
    q: Prime,
    fft: T,
}

impl<T> PrimeRing<T> {
    fn new(q: Prime, fft: T) -> Self {
        PrimeRing { q, fft }
    }
}

impl<T: Clone + Debug> ModulusOps for PrimeRing<T> {
    type Elem = <Prime as ModulusOps>::Elem;
    type Prep = <Prime as ModulusOps>::Prep;

    #[inline(always)]
    fn modulus(&self) -> Modulus {
        self.q.modulus()
    }

    #[inline(always)]
    fn uniform_distribution(
        &self,
    ) -> impl Distribution<Self::Elem> + DistributionSized<Self::Elem> {
        self.q.uniform_distribution()
    }

    #[inline(always)]
    fn zero(&self) -> Self::Elem {
        self.q.zero()
    }

    #[inline(always)]
    fn one(&self) -> Self::Elem {
        self.q.one()
    }

    #[inline(always)]
    fn neg_one(&self) -> Self::Elem {
        self.q.neg_one()
    }

    #[inline(always)]
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        self.q.neg(a)
    }

    #[inline(always)]
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.q.add(a, b)
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.q.sub(a, b)
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.q.mul(a, b)
    }

    #[inline(always)]
    fn powers(&self, a: &Self::Elem) -> impl Iterator<Item = Self::Elem> {
        self.q.powers(a)
    }

    fn inv(&self, a: &Self::Elem) -> Option<Self::Elem> {
        self.q.inv(a)
    }

    #[inline(always)]
    fn prepare(&self, a: &Self::Elem) -> Self::Prep {
        self.q.prepare(a)
    }

    #[inline(always)]
    fn mul_prep(&self, a: &Self::Elem, b: &Self::Prep) -> Self::Elem {
        self.q.mul_prep(a, b)
    }
}

impl<T: Clone + Debug> ElemFrom<u64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_from(&self, v: u64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<T: Clone + Debug> ElemFrom<i64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_from(&self, v: i64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<T: Clone + Debug> ElemFrom<u32> for PrimeRing<T> {
    #[inline(always)]
    fn elem_from(&self, v: u32) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<T: Clone + Debug> ElemFrom<i32> for PrimeRing<T> {
    #[inline(always)]
    fn elem_from(&self, v: i32) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<T: Clone + Debug> ElemFrom<f64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_from(&self, v: f64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<T: Clone + Debug> ElemTo<u64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> u64 {
        self.q.elem_to(v)
    }
}

impl<T: Clone + Debug> ElemTo<i64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> i64 {
        self.q.elem_to(v)
    }
}

impl<T: Clone + Debug> ElemTo<f64> for PrimeRing<T> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> f64 {
        self.q.elem_to(v)
    }
}

impl<T: Clone + Debug> SliceOps for PrimeRing<T> {}

impl<T: Clone + Debug> Sampler for PrimeRing<T> {}
