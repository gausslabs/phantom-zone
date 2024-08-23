use crate::{
    decomposer::PowerOfTwoDecomposer,
    distribution::{DistributionSized, Sampler},
    modulus::{ElemFrom, ElemOps, ElemTo, Modulus, ModulusOps},
};
use rand::distributions::{Distribution, Uniform};

pub type Native = PowerOfTwo<true>;
pub type NonNativePowerOfTwo = PowerOfTwo<false>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PowerOfTwo<const NATIVE: bool> {
    bits: usize,
    mask: u64,
}

impl<const NATIVE: bool> PowerOfTwo<NATIVE> {
    #[inline(always)]
    pub const fn new(bits: usize) -> Self {
        if NATIVE {
            debug_assert!(bits == 64);
        } else {
            debug_assert!(bits < 64);
        }
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1 << bits) - 1
        };
        Self { bits, mask }
    }

    #[inline(always)]
    pub fn bits(&self) -> usize {
        self.bits
    }

    #[inline(always)]
    pub fn max(&self) -> u64 {
        self.mask
    }

    #[inline(always)]
    pub fn mask(&self) -> u64 {
        self.mask
    }

    #[inline(always)]
    pub fn as_f64(&self) -> f64 {
        2f64.powi(self.bits as _)
    }

    #[inline(always)]
    pub(crate) fn center(&self, v: u64) -> i64 {
        if NATIVE {
            v as _
        } else {
            v.wrapping_sub((v >> (self.bits - 1)) << self.bits) as _
        }
    }

    #[inline(always)]
    pub(crate) fn reduce(&self, v: u64) -> u64 {
        if NATIVE {
            v
        } else {
            v & self.mask
        }
    }
}

impl Native {
    #[inline(always)]
    pub const fn native() -> Self {
        Self::new(64)
    }
}

impl<const NATIVE: bool> ElemOps for PowerOfTwo<NATIVE> {
    type Elem = u64;
}

impl<const NATIVE: bool> ModulusOps for PowerOfTwo<NATIVE> {
    type ElemPrep = u64;
    type Decomposer = PowerOfTwoDecomposer<NATIVE>;

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
    fn prepare(&self, a: &Self::Elem) -> Self::ElemPrep {
        *a
    }

    #[inline(always)]
    fn mul_prep(&self, a: &Self::Elem, b: &Self::ElemPrep) -> Self::Elem {
        self.mul(a, b)
    }
}

impl<const NATIVE: bool> ElemFrom<u64> for PowerOfTwo<NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: u64) -> Self::Elem {
        self.reduce(v)
    }
}

impl<const NATIVE: bool> ElemFrom<i64> for PowerOfTwo<NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: i64) -> Self::Elem {
        self.reduce(v as u64)
    }
}

impl<const NATIVE: bool> ElemFrom<u32> for PowerOfTwo<NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: u32) -> Self::Elem {
        self.reduce(v as u64)
    }
}

impl<const NATIVE: bool> ElemFrom<i32> for PowerOfTwo<NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: i32) -> Self::Elem {
        self.reduce(v as u64)
    }
}

impl<const NATIVE: bool> ElemFrom<f64> for PowerOfTwo<NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: f64) -> Self::Elem {
        self.reduce(f64_mod_u64(v))
    }
}

impl<const NATIVE: bool> ElemTo<u64> for PowerOfTwo<NATIVE> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> u64 {
        v
    }
}

impl<const NATIVE: bool> ElemTo<i64> for PowerOfTwo<NATIVE> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> i64 {
        self.center(v)
    }
}

impl<const NATIVE: bool> ElemTo<f64> for PowerOfTwo<NATIVE> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> f64 {
        self.center(v) as f64
    }
}

impl<const NATIVE: bool> Sampler for PowerOfTwo<NATIVE> {}

impl<const NATIVE: bool> From<PowerOfTwo<NATIVE>> for Modulus {
    fn from(value: PowerOfTwo<NATIVE>) -> Self {
        if NATIVE {
            Self::Native(PowerOfTwo {
                bits: value.bits,
                mask: value.mask,
            })
        } else {
            Self::NonNativePowerOfTwo(PowerOfTwo {
                bits: value.bits,
                mask: value.mask,
            })
        }
    }
}

impl<const NATIVE: bool> TryFrom<Modulus> for PowerOfTwo<NATIVE> {
    type Error = ();

    fn try_from(value: Modulus) -> Result<Self, Self::Error> {
        match value {
            Modulus::Native(value) if NATIVE => Ok(PowerOfTwo {
                bits: value.bits,
                mask: value.mask,
            }),
            Modulus::NonNativePowerOfTwo(value) if !NATIVE => Ok(PowerOfTwo {
                bits: value.bits,
                mask: value.mask,
            }),
            _ => Err(()),
        }
    }
}

/// Round f64 to u64 modulo 2^64.
#[inline(always)]
pub(crate) fn f64_mod_u64(v: f64) -> u64 {
    let bits = v.to_bits();
    let sign = bits >> 63;
    let exponent = (bits >> 52) & 0x7ff;
    // Shift 52-bits mantissa to the top and add back the 1.0 part.
    let mantissa = (bits << 11) | 0x8000000000000000;
    // Now the actual shift is (52 + 11) + (1023 - exponent) = 1086 - exponent.
    let value = match 1086 - exponent as i64 {
        // If exponent is larger than 63, shift left and overflowing bits are discarded as in modulus 2^64.
        shift @ -63..=0 => mantissa << -shift,
        // If exponent is less than 63, rounding-shift right.
        shift @ 1..=64 => ((mantissa >> (shift - 1)).wrapping_add(1)) >> 1,
        // If exponent is too large or too small, the value is modulo or round to 0.
        _ => 0,
    };
    // Negate if sign bit is set.
    if sign == 0 {
        value
    } else {
        value.wrapping_neg()
    }
}

#[cfg(test)]
mod test {
    use crate::modulus::power_of_two;
    use num_bigint_dig::BigInt;
    use num_traits::{FromPrimitive, ToPrimitive};

    #[test]
    fn f64_mod_u64() {
        let expected = |a: f64| {
            let a = BigInt::from_f64(a.round()).unwrap();
            (a % (1i128 << 64)).to_i128().unwrap() as u64
        };
        for exp in -1023..1024 {
            let a = 2f64.powi(exp);
            assert_eq!(power_of_two::f64_mod_u64(a), expected(a));
            assert_eq!(power_of_two::f64_mod_u64(a + 0.5), expected(a + 0.5));
            assert_eq!(power_of_two::f64_mod_u64(a - 0.5), expected(a - 0.5));
            assert_eq!(power_of_two::f64_mod_u64(-a), expected(-a));
            assert_eq!(power_of_two::f64_mod_u64(-a + 0.5), expected(-a + 0.5));
            assert_eq!(power_of_two::f64_mod_u64(-a - 0.5), expected(-a - 0.5));
        }
    }
}
