use itertools::izip;
use num_traits::{PrimInt, Signed, ToPrimitive, WrappingAdd, WrappingMul, WrappingSub, Zero};

use super::{
    ArithmeticLazyOps, ArithmeticOps, GetModulus, ModInit, Modulus, ShoupMatrixFMA, VectorOps,
};
use crate::{utils::ShoupMul, Matrix, RowMut};

pub struct WordSizeModulus<T> {
    modulus: T,
}

impl<T> ModInit for WordSizeModulus<T>
where
    T: Modulus,
{
    type M = T;
    fn new(modulus: T) -> Self {
        assert!(modulus.is_native());
        // For now assume ModulusOpsU64 is only used for u64
        Self { modulus: modulus }
    }
}

impl<T> ArithmeticOps for WordSizeModulus<T>
where
    T: Modulus,
    T::Element: WrappingAdd + WrappingSub + WrappingMul + Zero,
{
    type Element = T::Element;
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        T::Element::wrapping_add(a, b)
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        T::Element::wrapping_mul(a, b)
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        T::Element::wrapping_sub(&T::Element::zero(), a)
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        T::Element::wrapping_sub(a, b)
    }
}

impl<T> VectorOps for WordSizeModulus<T>
where
    T: Modulus,
    T::Element: WrappingAdd + WrappingSub + WrappingMul + Zero,
{
    type Element = T::Element;

    fn elwise_add_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| {
            *ai = T::Element::wrapping_add(ai, bi);
        });
    }

    fn elwise_sub_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| {
            *ai = T::Element::wrapping_sub(ai, bi);
        });
    }

    fn elwise_mul_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| {
            *ai = T::Element::wrapping_mul(ai, bi);
        });
    }

    fn elwise_neg_mut(&self, a: &mut [Self::Element]) {
        a.iter_mut()
            .for_each(|ai| *ai = T::Element::wrapping_sub(&T::Element::zero(), ai));
    }

    fn elwise_scalar_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &Self::Element) {
        izip!(out.iter_mut(), a.iter()).for_each(|(oi, ai)| {
            *oi = T::Element::wrapping_mul(ai, b);
        });
    }

    fn elwise_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &[Self::Element]) {
        izip!(out.iter_mut(), a.iter(), b.iter()).for_each(|(oi, ai, bi)| {
            *oi = T::Element::wrapping_mul(ai, bi);
        });
    }

    fn elwise_scalar_mul_mut(&self, a: &mut [Self::Element], b: &Self::Element) {
        a.iter_mut().for_each(|ai| {
            *ai = T::Element::wrapping_mul(ai, b);
        });
    }

    fn elwise_fma_mut(&self, a: &mut [Self::Element], b: &[Self::Element], c: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter(), c.iter()).for_each(|(ai, bi, ci)| {
            *ai = T::Element::wrapping_add(ai, &T::Element::wrapping_mul(bi, ci));
        });
    }

    fn elwise_fma_scalar_mut(
        &self,
        a: &mut [Self::Element],
        b: &[Self::Element],
        c: &Self::Element,
    ) {
        izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| {
            *ai = T::Element::wrapping_add(ai, &T::Element::wrapping_mul(bi, c));
        });
    }

    // fn modulus(&self) -> &T {
    //     &self.modulus
    // }
}

impl<T> GetModulus for WordSizeModulus<T>
where
    T: Modulus,
{
    type Element = T::Element;
    type M = T;
    fn modulus(&self) -> &Self::M {
        &self.modulus
    }
}
