use std::marker::PhantomData;

use itertools::izip;
use num_traits::{PrimInt, Signed, ToPrimitive, WrappingAdd, WrappingMul, WrappingSub, Zero};

pub trait Modulus {
    type Element;
    /// Modulus value if it fits in Element
    fn q(&self) -> Option<Self::Element>;
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
        // q of size u64 can never be a naitve modulus
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
        //FIXME (Jay): Before I check whether v is smaller than 0 with `let is_neg =
        // o.is_sign_negative() && o != 0.0; I'm ocnfused why didn't I simply check <
        // 0.0?
        let v = v.round();
        if v < 0.0 {
            self - v.abs().to_u64().unwrap()
        } else {
            v.to_u64().unwrap()
        }
    }
    fn map_element_from_i64(&self, v: i64) -> Self::Element {
        if v < 0 {
            self - v.abs().to_u64().unwrap()
        } else {
            v.to_u64().unwrap()
        }
    }
    fn q(&self) -> Option<Self::Element> {
        Some(*self)
    }
    fn q_as_f64(&self) -> Option<f64> {
        self.to_f64()
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

    // fn modulus(&self) -> Self::Element;
}

pub trait ArithmeticOps {
    type Element;

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn neg(&self, a: &Self::Element) -> Self::Element;

    // fn modulus(&self) -> Self::Element;
}

pub struct ModularOpsU64<T> {
    q: u64,
    logq: usize,
    barrett_mu: u128,
    barrett_alpha: usize,
    modulus: T,
}

impl<T> ModInit for ModularOpsU64<T>
where
    T: Modulus<Element = u64>,
{
    type M = T;
    fn new(modulus: Self::M) -> ModularOpsU64<T> {
        assert!(!modulus.is_native());

        // largest unsigned value modulus fits is modulus-1
        let q = modulus.largest_unsigned_value() + 1;
        let logq = 64 - (q + 1u64).leading_zeros();

        // barrett calculation
        let mu = (1u128 << (logq * 2 + 3)) / (q as u128);
        let alpha = logq + 3;

        ModularOpsU64 {
            q,
            logq: logq as usize,
            barrett_alpha: alpha as usize,
            barrett_mu: mu,
            modulus,
        }
    }
}

impl<T> ModularOpsU64<T> {
    fn add_mod_fast(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q);
        debug_assert!(b < self.q);

        let mut o = a + b;
        if o >= self.q {
            o -= self.q;
        }
        o
    }

    fn sub_mod_fast(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q);
        debug_assert!(b < self.q);

        if a >= b {
            a - b
        } else {
            (self.q + a) - b
        }
    }

    /// returns (a * b)  % q
    ///
    /// - both a and b must be in range [0, 2q)
    /// - output is in range [0 , q)
    fn mul_mod_fast(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < 2 * self.q);
        debug_assert!(b < 2 * self.q);

        let ab = a as u128 * b as u128;

        // ab / (2^{n + \beta})
        // note: \beta is assumed to -2
        let tmp = ab >> (self.logq - 2);

        // k = ((ab / (2^{n + \beta})) * \mu) / 2^{\alpha - (-2)}
        let k = (tmp * self.barrett_mu) >> (self.barrett_alpha + 2);

        // ab - k*p
        let tmp = k * (self.q as u128);

        let mut out = (ab - tmp) as u64;

        if out >= self.q {
            out -= self.q;
        }

        return out;
    }
}

impl<T> ArithmeticOps for ModularOpsU64<T> {
    type Element = u64;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.add_mod_fast(*a, *b)
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul_mod_fast(*a, *b)
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.sub_mod_fast(*a, *b)
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        self.q - *a
    }

    // fn modulus(&self) -> Self::Element {
    //     self.q
    // }
}

impl<T> VectorOps for ModularOpsU64<T> {
    type Element = u64;

    fn elwise_add_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| {
            *ai = self.add_mod_fast(*ai, *bi);
        });
    }

    fn elwise_sub_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| {
            *ai = self.sub_mod_fast(*ai, *bi);
        });
    }

    fn elwise_mul_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| {
            *ai = self.mul_mod_fast(*ai, *bi);
        });
    }

    fn elwise_neg_mut(&self, a: &mut [Self::Element]) {
        a.iter_mut().for_each(|ai| *ai = self.q - *ai);
    }

    fn elwise_scalar_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &Self::Element) {
        izip!(out.iter_mut(), a.iter()).for_each(|(oi, ai)| {
            *oi = self.mul_mod_fast(*ai, *b);
        });
    }

    fn elwise_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &[Self::Element]) {
        izip!(out.iter_mut(), a.iter(), b.iter()).for_each(|(oi, ai, bi)| {
            *oi = self.mul_mod_fast(*ai, *bi);
        });
    }

    fn elwise_scalar_mul_mut(&self, a: &mut [Self::Element], b: &Self::Element) {
        a.iter_mut().for_each(|ai| {
            *ai = self.mul_mod_fast(*ai, *b);
        });
    }

    fn elwise_fma_mut(&self, a: &mut [Self::Element], b: &[Self::Element], c: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter(), c.iter()).for_each(|(ai, bi, ci)| {
            *ai = self.add_mod_fast(*ai, self.mul_mod_fast(*bi, *ci));
        });
    }

    fn elwise_fma_scalar_mut(
        &self,
        a: &mut [Self::Element],
        b: &[Self::Element],
        c: &Self::Element,
    ) {
        izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| {
            *ai = self.add_mod_fast(*ai, self.mul_mod_fast(*bi, *c));
        });
    }

    // fn modulus(&self) -> Self::Element {
    //     self.q
    // }
}

impl<T> GetModulus for ModularOpsU64<T>
where
    T: Modulus,
{
    type Element = T::Element;
    type M = T;
    fn modulus(&self) -> &Self::M {
        &self.modulus
    }
}

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
