use core::fmt::Debug;
use phantom_zone_math::util::serde::Serde;

pub mod fhew;

/// A trait to perform boolean operation on associated
/// [`BoolEvaluator::Ciphertext`].
///
/// To use Rust core operators `&`, `|` and `^` on
/// [`BoolEvaluator::Ciphertext`] directly, one can wrap it with reference to
/// its corresponding [`BoolEvaluator`] by [`FheBool`][FheBool].
///
/// [FheBool]: crate::boolean::FheBool
pub trait BoolEvaluator: Send + Sync {
    type Ciphertext: Clone + Debug + Serde + Send + Sync;

    /// Performs bitwise NOT assignment.
    fn bitnot_assign(&self, a: &mut Self::Ciphertext);

    /// Performs bitwise AND assignment.
    fn bitand_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    /// Performs bitwise NAND assignment.
    fn bitnand_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    /// Performs bitwise OR assignment.
    fn bitor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    /// Performs bitwise NOR assignment.
    fn bitnor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    /// Performs bitwise XOR assignment.
    fn bitxor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    /// Performs bitwise XNOR assignment.
    fn bitxnor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    /// Performs half adder assignment and returns the carry.
    fn overflowing_add_assign(
        &self,
        a: &mut Self::Ciphertext,
        b: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        let mut carry = a.clone();
        self.bitxor_assign(a, b);
        self.bitand_assign(&mut carry, b);
        carry
    }

    /// Performs half subtractor assignment and returns the borrow.
    fn overflowing_sub_assign(
        &self,
        a: &mut Self::Ciphertext,
        b: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        let mut borrow = a.clone();
        self.bitxor_assign(a, b);
        self.bitnot_assign(&mut borrow);
        self.bitand_assign(&mut borrow, b);
        borrow
    }

    /// Performs full adder assignment and returns the carry.
    fn carrying_add_assign(
        &self,
        a: &mut Self::Ciphertext,
        b: &Self::Ciphertext,
        carry_in: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        let mut carry_out = a.clone(); // carry_out = a
        self.bitxor_assign(a, b); // a' = a ^ b
        let mut t = a.clone(); // t = a ^ b
        self.bitxor_assign(a, carry_in); // a' = a ^ b ^ carry_in
        self.bitand_assign(&mut t, carry_in); // t = (a ^ b) & carry_in
        self.bitand_assign(&mut carry_out, b); // carry_out = a & b
        self.bitor_assign(&mut carry_out, &t); // carry_out = (a & b) | ((a ^ b) & carry_in)
        carry_out
    }

    /// Performs full subtractor assignment and returns the borrow.
    fn borrowing_sub_assign(
        &self,
        a: &mut Self::Ciphertext,
        b: &Self::Ciphertext,
        borrow_in: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        let mut borrow_out = a.clone(); // borrow_out = a
        self.bitxor_assign(a, b); // a' = a ^ b
        let mut t = a.clone(); // t = a ^ b
        self.bitxor_assign(a, borrow_in); // a' = a ^ b ^ borrow_in
        self.bitnot_assign(&mut t); // t = !(a ^ b)
        self.bitand_assign(&mut t, borrow_in); // t = (!(a ^ b)) & borrow_in
        self.bitnot_assign(&mut borrow_out); // borrow_out = !a
        self.bitand_assign(&mut borrow_out, b); // borrow_out = !a & b
        self.bitor_assign(&mut borrow_out, &t); // borrow_out = (!a & b) | ((!(a ^ b)) & borrow_in)
        borrow_out
    }
}
