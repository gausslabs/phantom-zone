use num_traits::ToPrimitive;

use crate::{utils::log2, Row};

mod modulus_u64;
mod power_of_2;
mod word_size;

pub use modulus_u64::ModularOpsU64;
pub(crate) use power_of_2::ModulusPowerOf2;

pub trait Modulus {
    type Element;
    /// Modulus value if it fits in Element
    fn q(&self) -> Option<Self::Element>;
    /// Log2 of `q`
    fn log_q(&self) -> usize;
    /// Modulus value as f64 if it fits in f64
    fn q_as_f64(&self) -> Option<f64>;
    /// Is modulus native?
    fn is_native(&self) -> bool;
    /// -1 in signed representaiton
    fn neg_one(&self) -> Self::Element;
    /// Largest unsigned value that fits in the modulus. That is, q - 1.
    fn largest_unsigned_value(&self) -> Self::Element;
    /// Smallest unsigned value that fits in the modulus
    /// Always assmed to be 0.
    fn smallest_unsigned_value(&self) -> Self::Element;
    /// Convert unsigned value in signed represetation to i64
    fn map_element_to_i64(&self, v: &Self::Element) -> i64;
    /// Convert f64 to signed represented in modulus
    fn map_element_from_f64(&self, v: f64) -> Self::Element;
    /// Convert i64 to signed represented in modulus
    fn map_element_from_i64(&self, v: i64) -> Self::Element;
}

impl Modulus for u64 {
    type Element = u64;
    fn is_native(&self) -> bool {
        // q that fits in u64 can never be a native modulus
        false
    }
    fn largest_unsigned_value(&self) -> Self::Element {
        self - 1
    }
    fn neg_one(&self) -> Self::Element {
        self - 1
    }
    fn smallest_unsigned_value(&self) -> Self::Element {
        0
    }
    fn map_element_to_i64(&self, v: &Self::Element) -> i64 {
        assert!(v <= self, "{v} must be <= {self}");
        if *v >= (self >> 1) {
            -ToPrimitive::to_i64(&(self - v)).unwrap()
        } else {
            ToPrimitive::to_i64(v).unwrap()
        }
    }
    fn map_element_from_f64(&self, v: f64) -> Self::Element {
        let v = v.round();
        let v_u64 = v.abs().to_u64().unwrap();
        assert!(v_u64 <= self.largest_unsigned_value());
        if v < 0.0 {
            self - v_u64
        } else {
            v_u64
        }
    }
    fn map_element_from_i64(&self, v: i64) -> Self::Element {
        let v_u64 = v.abs().to_u64().unwrap();
        assert!(v_u64 <= self.largest_unsigned_value());
        if v < 0 {
            self - v_u64
        } else {
            v_u64
        }
    }
    fn q(&self) -> Option<Self::Element> {
        Some(*self)
    }
    fn q_as_f64(&self) -> Option<f64> {
        self.to_f64()
    }
    fn log_q(&self) -> usize {
        log2(&self.q().unwrap())
    }
}

pub trait ModInit {
    type M;
    fn new(modulus: Self::M) -> Self;
}

pub trait GetModulus {
    type Element;
    type M: Modulus<Element = Self::Element>;
    fn modulus(&self) -> &Self::M;
}

pub trait VectorOps {
    type Element;

    /// Sets out as `out[i] = a[i] * b`
    fn elwise_scalar_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &Self::Element);
    fn elwise_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &[Self::Element]);

    fn elwise_add_mut(&self, a: &mut [Self::Element], b: &[Self::Element]);
    fn elwise_sub_mut(&self, a: &mut [Self::Element], b: &[Self::Element]);
    fn elwise_mul_mut(&self, a: &mut [Self::Element], b: &[Self::Element]);
    fn elwise_scalar_mul_mut(&self, a: &mut [Self::Element], b: &Self::Element);
    fn elwise_neg_mut(&self, a: &mut [Self::Element]);
    /// inplace mutates `a`: a = a + b*c
    fn elwise_fma_mut(&self, a: &mut [Self::Element], b: &[Self::Element], c: &[Self::Element]);
    fn elwise_fma_scalar_mut(
        &self,
        a: &mut [Self::Element],
        b: &[Self::Element],
        c: &Self::Element,
    );
}

pub trait ArithmeticOps {
    type Element;
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn neg(&self, a: &Self::Element) -> Self::Element;
}

pub trait ArithmeticLazyOps {
    type Element;
    fn mul_lazy(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn add_lazy(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
}

pub trait ShoupMatrixFMA<R: Row> {
    /// Returns summation of `row-wise product of matrix a and b` + out where
    /// each element is in range [0, 2q)
    fn shoup_matrix_fma(&self, out: &mut [R::Element], a: &[R], a_shoup: &[R], b: &[R]);
}
