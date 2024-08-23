use crate::{
    distribution::{DistributionSized, Sampler},
    modulus::{power_of_two::PowerOfTwo, ElemFrom, ElemOps, ElemTo, Modulus, ModulusOps},
    poly::ffnt::Ffnt,
};
use core::fmt::Debug;
use rand::distributions::Distribution;

pub mod noisy;
pub mod precise;

#[derive(Clone, Debug)]
pub struct PowerOfTwoRing<const NATIVE: bool, const LIMBS: usize> {
    q: PowerOfTwo<NATIVE>,
    fft: Ffnt,
}

impl<const NATIVE: bool, const LIMBS: usize> PowerOfTwoRing<NATIVE, LIMBS> {
    fn new(q: PowerOfTwo<NATIVE>, fft: Ffnt) -> Self {
        Self { q, fft }
    }
}

impl<const NATIVE: bool, const LIMBS: usize> ElemOps for PowerOfTwoRing<NATIVE, LIMBS> {
    type Elem = <PowerOfTwo<NATIVE> as ElemOps>::Elem;
}

impl<const NATIVE: bool, const LIMBS: usize> ModulusOps for PowerOfTwoRing<NATIVE, LIMBS> {
    type ElemPrep = <PowerOfTwo<NATIVE> as ModulusOps>::ElemPrep;
    type Decomposer = <PowerOfTwo<NATIVE> as ModulusOps>::Decomposer;

    fn new(modulus: Modulus) -> Self {
        Self::new(modulus.try_into().unwrap(), Ffnt::default())
    }

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
    fn prepare(&self, a: &Self::Elem) -> Self::ElemPrep {
        self.q.prepare(a)
    }

    #[inline(always)]
    fn mul_prep(&self, a: &Self::Elem, b: &Self::ElemPrep) -> Self::Elem {
        self.q.mul_prep(a, b)
    }
}

impl<const NATIVE: bool, const LIMBS: usize> ElemFrom<u64> for PowerOfTwoRing<NATIVE, LIMBS> {
    #[inline(always)]
    fn elem_from(&self, v: u64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<const NATIVE: bool, const LIMBS: usize> ElemFrom<i64> for PowerOfTwoRing<NATIVE, LIMBS> {
    #[inline(always)]
    fn elem_from(&self, v: i64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<const NATIVE: bool, const LIMBS: usize> ElemFrom<u32> for PowerOfTwoRing<NATIVE, LIMBS> {
    #[inline(always)]
    fn elem_from(&self, v: u32) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<const NATIVE: bool, const LIMBS: usize> ElemFrom<i32> for PowerOfTwoRing<NATIVE, LIMBS> {
    #[inline(always)]
    fn elem_from(&self, v: i32) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<const NATIVE: bool, const LIMBS: usize> ElemFrom<f64> for PowerOfTwoRing<NATIVE, LIMBS> {
    #[inline(always)]
    fn elem_from(&self, v: f64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<const NATIVE: bool, const LIMBS: usize> ElemTo<u64> for PowerOfTwoRing<NATIVE, LIMBS> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> u64 {
        self.q.elem_to(v)
    }
}

impl<const NATIVE: bool, const LIMBS: usize> ElemTo<i64> for PowerOfTwoRing<NATIVE, LIMBS> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> i64 {
        self.q.elem_to(v)
    }
}

impl<const NATIVE: bool, const LIMBS: usize> ElemTo<f64> for PowerOfTwoRing<NATIVE, LIMBS> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> f64 {
        self.q.elem_to(v)
    }
}

impl<const NATIVE: bool, const LIMBS: usize> Sampler for PowerOfTwoRing<NATIVE, LIMBS> {}
