mod enc_dec;
mod ops;

pub type FheUint8 = enc_dec::FheUint8<Vec<u64>>;

#[cfg(feature = "non_interactive_mp")]
pub type EncFheUint8 = enc_dec::SeededBatchedFheUint8<Vec<u64>, [u8; 32]>;

#[cfg(feature = "interactive_mp")]
pub type EncFheUint8 = enc_dec::BatchedFheUint8<std::vec::Vec<std::vec::Vec<u64>>>;

use std::cell::RefCell;

use crate::bool::{BoolEvaluator, BooleanGates, FheBool, RuntimeServerKey};

thread_local! {
     static DIV_ZERO_ERROR: RefCell<Option<FheBool>> = RefCell::new(None);
}

/// Returns Boolean ciphertext indicating whether last division was attempeted
/// with decnomiantor set to 0.
pub fn div_zero_error_flag() -> Option<FheBool> {
    DIV_ZERO_ERROR.with_borrow(|c| c.clone())
}

/// Reset all error flags
///
/// Error flags are thread local. When running multiple circuits in sequence
/// within a single program you must prevent error flags set during the
/// execution of previous circuit to affect error flags set during execution of
/// the next circuit. To do so call `reset_error_flags()`.
pub fn reset_error_flags() {
    DIV_ZERO_ERROR.with_borrow_mut(|c| *c = None);
}

mod frontend {
    use super::ops::{
        arbitrary_bit_adder, arbitrary_bit_division_for_quotient_and_rem, arbitrary_bit_subtractor,
        eight_bit_mul, is_zero,
    };
    use crate::utils::{Global, WithLocal};

    use super::*;

    /// Set Div by Zero flag after each divison. Div by zero flag is set to true
    /// if either 1 of the division executed in circuit evaluation has
    /// denominator set to 0.
    fn set_div_by_zero_flag(denominator: &FheUint8) {
        {
            BoolEvaluator::with_local_mut(|e| {
                let key = RuntimeServerKey::global();
                let is_zero = is_zero(e, denominator.data(), key);
                DIV_ZERO_ERROR.with_borrow_mut(|before_is_zero| {
                    if before_is_zero.is_none() {
                        *before_is_zero = Some(FheBool { data: is_zero });
                    } else {
                        e.or_inplace(before_is_zero.as_mut().unwrap().data_mut(), &is_zero, key);
                    }
                });
            })
        }
    }

    mod arithetic {

        use super::*;
        use std::ops::{Add, AddAssign, Div, Mul, Rem, Sub};

        impl AddAssign<&FheUint8> for FheUint8 {
            fn add_assign(&mut self, rhs: &FheUint8) {
                BoolEvaluator::with_local_mut_mut(&mut |e| {
                    let key = RuntimeServerKey::global();
                    arbitrary_bit_adder(e, self.data_mut(), rhs.data(), false, key);
                });
            }
        }

        impl Add<&FheUint8> for &FheUint8 {
            type Output = FheUint8;
            fn add(self, rhs: &FheUint8) -> Self::Output {
                let mut a = self.clone();
                a += rhs;
                a
            }
        }

        impl Sub<&FheUint8> for &FheUint8 {
            type Output = FheUint8;
            fn sub(self, rhs: &FheUint8) -> Self::Output {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let (out, _, _) = arbitrary_bit_subtractor(e, self.data(), rhs.data(), key);
                    FheUint8 { data: out }
                })
            }
        }

        impl Mul<&FheUint8> for &FheUint8 {
            type Output = FheUint8;
            fn mul(self, rhs: &FheUint8) -> Self::Output {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let out = eight_bit_mul(e, self.data(), rhs.data(), key);
                    FheUint8 { data: out }
                })
            }
        }

        impl Div<&FheUint8> for &FheUint8 {
            type Output = FheUint8;
            fn div(self, rhs: &FheUint8) -> Self::Output {
                // set div by 0 error flag
                set_div_by_zero_flag(rhs);

                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();

                    let (quotient, _) = arbitrary_bit_division_for_quotient_and_rem(
                        e,
                        self.data(),
                        rhs.data(),
                        key,
                    );
                    FheUint8 { data: quotient }
                })
            }
        }

        impl Rem<&FheUint8> for &FheUint8 {
            type Output = FheUint8;
            fn rem(self, rhs: &FheUint8) -> Self::Output {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let (_, remainder) = arbitrary_bit_division_for_quotient_and_rem(
                        e,
                        self.data(),
                        rhs.data(),
                        key,
                    );
                    FheUint8 { data: remainder }
                })
            }
        }

        impl FheUint8 {
            /// Calculates `Self += rhs` and returns `overflow`
            ///
            /// `overflow` is set to `True` if `Self += rhs` overflowed,
            /// otherwise it is set to `False`
            pub fn overflowing_add_assign(&mut self, rhs: &FheUint8) -> FheBool {
                BoolEvaluator::with_local_mut_mut(&mut |e| {
                    let key = RuntimeServerKey::global();
                    let (overflow, _) =
                        arbitrary_bit_adder(e, self.data_mut(), rhs.data(), false, key);
                    FheBool { data: overflow }
                })
            }

            /// Returns (Self + rhs, overflow).
            ///
            /// `overflow` is set to `True` if `Self + rhs` overflowed,
            /// otherwise it is set to `False`
            pub fn overflowing_add(self, rhs: &FheUint8) -> (FheUint8, FheBool) {
                BoolEvaluator::with_local_mut(|e| {
                    let mut lhs = self.clone();
                    let key = RuntimeServerKey::global();
                    let (overflow, _) =
                        arbitrary_bit_adder(e, lhs.data_mut(), rhs.data(), false, key);
                    (lhs, FheBool { data: overflow })
                })
            }

            /// Returns (Self - rhs, overflow).
            ///
            /// `overflow` is set to `True` if `Self - rhs` overflowed,
            /// otherwise it is set to `False`
            pub fn overflowing_sub(&self, rhs: &FheUint8) -> (FheUint8, FheBool) {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let (out, mut overflow, _) =
                        arbitrary_bit_subtractor(e, self.data(), rhs.data(), key);
                    e.not_inplace(&mut overflow);
                    (FheUint8 { data: out }, FheBool { data: overflow })
                })
            }

            /// Returns (quotient, remainder) s.t. self = rhs x quotient +
            /// remainder.
            ///
            /// If rhs is 0, then quotient = 255, remainder = self, and Div by
            /// Zero error flag (accessible via `div_zero_error_flag`) is set to
            /// `True`
            pub fn div_rem(&self, rhs: &FheUint8) -> (FheUint8, FheUint8) {
                // set div by 0 error flag
                set_div_by_zero_flag(rhs);

                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();

                    let (quotient, remainder) = arbitrary_bit_division_for_quotient_and_rem(
                        e,
                        self.data(),
                        rhs.data(),
                        key,
                    );
                    (FheUint8 { data: quotient }, FheUint8 { data: remainder })
                })
            }
        }
    }

    mod booleans {
        use crate::shortint::ops::{
            arbitrary_bit_comparator, arbitrary_bit_equality, arbitrary_bit_mux,
        };

        use super::*;

        impl FheUint8 {
            /// Returns `FheBool` indicating `Self == other`
            pub fn eq(&self, other: &FheUint8) -> FheBool {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let out = arbitrary_bit_equality(e, self.data(), other.data(), key);
                    FheBool { data: out }
                })
            }

            /// Returns `FheBool` indicating `Self != other`
            pub fn neq(&self, other: &FheUint8) -> FheBool {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let mut is_equal = arbitrary_bit_equality(e, self.data(), other.data(), key);
                    e.not_inplace(&mut is_equal);
                    FheBool { data: is_equal }
                })
            }

            /// Returns `FheBool` indicating `Self < other`
            pub fn lt(&self, other: &FheUint8) -> FheBool {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let out = arbitrary_bit_comparator(e, other.data(), self.data(), key);
                    FheBool { data: out }
                })
            }

            /// Returns `FheBool` indicating `Self > other`
            pub fn gt(&self, other: &FheUint8) -> FheBool {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let out = arbitrary_bit_comparator(e, self.data(), other.data(), key);
                    FheBool { data: out }
                })
            }

            /// Returns `FheBool` indicating `Self <= other`
            pub fn le(&self, other: &FheUint8) -> FheBool {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let mut a_greater_b =
                        arbitrary_bit_comparator(e, self.data(), other.data(), key);
                    e.not_inplace(&mut a_greater_b);
                    FheBool { data: a_greater_b }
                })
            }

            /// Returns `FheBool` indicating `Self >= other`
            pub fn ge(&self, other: &FheUint8) -> FheBool {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let mut a_less_b = arbitrary_bit_comparator(e, other.data(), self.data(), key);
                    e.not_inplace(&mut a_less_b);
                    FheBool { data: a_less_b }
                })
            }

            /// Returns `Self` if `selector = True` else returns `other`
            pub fn mux(&self, other: &FheUint8, selector: &FheBool) -> FheUint8 {
                BoolEvaluator::with_local_mut(|e| {
                    let key = RuntimeServerKey::global();
                    let out = arbitrary_bit_mux(e, selector.data(), self.data(), other.data(), key);
                    FheUint8 { data: out }
                })
            }

            /// Returns max(`Self`, `other`)
            pub fn max(&self, other: &FheUint8) -> FheUint8 {
                let self_gt = self.gt(other);
                self.mux(other, &self_gt)
            }

            /// Returns min(`Self`, `other`)
            pub fn min(&self, other: &FheUint8) -> FheUint8 {
                let self_lt = self.lt(other);
                self.mux(other, &self_lt)
            }
        }
    }
}
