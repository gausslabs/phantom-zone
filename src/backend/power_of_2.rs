use itertools::izip;

use crate::{ArithmeticOps, ModInit, VectorOps};

use super::{GetModulus, Modulus};

pub(crate) struct ModulusPowerOf2<T> {
    modulus: T,
    /// Modulus mask: (1 << q) - 1
    mask: u64,
}

impl<T> ArithmeticOps for ModulusPowerOf2<T> {
    type Element = u64;
    #[inline]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a.wrapping_add(*b)) & self.mask
    }
    #[inline]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a.wrapping_sub(*b)) & self.mask
    }
    #[inline]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a.wrapping_mul(*b)) & self.mask
    }
    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        (0u64.wrapping_sub(*a)) & self.mask
    }
}

impl<T> VectorOps for ModulusPowerOf2<T> {
    type Element = u64;

    #[inline]
    fn elwise_add_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(a0, b0)| *a0 = (a0.wrapping_add(*b0)) & self.mask);
    }

    #[inline]
    fn elwise_mul_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(a0, b0)| *a0 = (a0.wrapping_mul(*b0)) & self.mask);
    }

    #[inline]
    fn elwise_neg_mut(&self, a: &mut [Self::Element]) {
        a.iter_mut()
            .for_each(|a0| *a0 = 0u64.wrapping_sub(*a0) & self.mask);
    }
    #[inline]
    fn elwise_sub_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(a0, b0)| *a0 = (a0.wrapping_sub(*b0)) & self.mask);
    }

    #[inline]
    fn elwise_fma_mut(&self, a: &mut [Self::Element], b: &[Self::Element], c: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter(), c.iter()).for_each(|(a0, b0, c0)| {
            *a0 = a0.wrapping_add(b0.wrapping_mul(*c0)) & self.mask;
        });
    }

    #[inline]
    fn elwise_fma_scalar_mut(
        &self,
        a: &mut [Self::Element],
        b: &[Self::Element],
        c: &Self::Element,
    ) {
        izip!(a.iter_mut(), b.iter()).for_each(|(a0, b0)| {
            *a0 = a0.wrapping_add(b0.wrapping_mul(*c)) & self.mask;
        });
    }
    #[inline]
    fn elwise_scalar_mul_mut(&self, a: &mut [Self::Element], b: &Self::Element) {
        a.iter_mut()
            .for_each(|a0| *a0 = a0.wrapping_mul(*b) & self.mask)
    }

    #[inline]
    fn elwise_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &[Self::Element]) {
        izip!(out.iter_mut(), a.iter(), b.iter()).for_each(|(o0, a0, b0)| {
            *o0 = a0.wrapping_mul(*b0) & self.mask;
        });
    }

    #[inline]
    fn elwise_scalar_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &Self::Element) {
        izip!(out.iter_mut(), a.iter()).for_each(|(o0, a0)| {
            *o0 = a0.wrapping_mul(*b) & self.mask;
        });
    }
}

impl<T: Modulus<Element = u64>> ModInit for ModulusPowerOf2<T> {
    type M = T;
    fn new(modulus: Self::M) -> Self {
        assert!(!modulus.is_native());
        assert!(modulus.q().unwrap().is_power_of_two());
        let q = modulus.q().unwrap();
        let mask = q - 1;
        Self { modulus, mask }
    }
}

impl<T: Modulus<Element = u64>> GetModulus for ModulusPowerOf2<T> {
    type Element = u64;
    type M = T;
    fn modulus(&self) -> &Self::M {
        &self.modulus
    }
}
