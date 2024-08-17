use crate::{
    distribution::{DistributionSized, Sampler},
    modulus::{power_of_two::PowerOfTwo, Modulus},
    ring::{ElemFrom, ElemTo, ModulusOps, SliceOps},
};
use core::fmt::Debug;
use rand::distributions::Distribution;

pub mod noisy;
pub mod precise;

#[derive(Clone, Copy, Debug)]
pub struct PowerOfTwoRing<T, const NATIVE: bool> {
    q: PowerOfTwo<NATIVE>,
    fft: T,
}

impl<T, const NATIVE: bool> PowerOfTwoRing<T, NATIVE> {
    fn new(q: PowerOfTwo<NATIVE>, fft: T) -> Self {
        Self { q, fft }
    }
}

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> ModulusOps for PowerOfTwoRing<T, NATIVE> {
    type Elem = u64;
    type Prep = u64;

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

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> ElemFrom<u64>
    for PowerOfTwoRing<T, NATIVE>
{
    #[inline(always)]
    fn elem_from(&self, v: u64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> ElemFrom<i64>
    for PowerOfTwoRing<T, NATIVE>
{
    #[inline(always)]
    fn elem_from(&self, v: i64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> ElemFrom<u32>
    for PowerOfTwoRing<T, NATIVE>
{
    #[inline(always)]
    fn elem_from(&self, v: u32) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> ElemFrom<i32>
    for PowerOfTwoRing<T, NATIVE>
{
    #[inline(always)]
    fn elem_from(&self, v: i32) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> ElemFrom<f64>
    for PowerOfTwoRing<T, NATIVE>
{
    #[inline(always)]
    fn elem_from(&self, v: f64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> ElemTo<u64> for PowerOfTwoRing<T, NATIVE> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> u64 {
        self.q.elem_to(v)
    }
}

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> ElemTo<i64> for PowerOfTwoRing<T, NATIVE> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> i64 {
        self.q.elem_to(v)
    }
}

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> ElemTo<f64> for PowerOfTwoRing<T, NATIVE> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> f64 {
        self.q.elem_to(v)
    }
}

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> SliceOps for PowerOfTwoRing<T, NATIVE> {}

impl<T: Clone + Debug + Send + Sync, const NATIVE: bool> Sampler for PowerOfTwoRing<T, NATIVE> {}
