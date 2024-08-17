use core::fmt::Debug;

pub mod fhew;

pub trait BoolEvaluator: Send + Sync {
    type Ciphertext: Clone + Debug;

    fn bitnot_assign(&self, a: &mut Self::Ciphertext);

    fn bitand_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    fn bitnand_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    fn bitor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    fn bitnor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    fn bitxor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

    fn bitxnor_assign(&self, a: &mut Self::Ciphertext, b: &Self::Ciphertext);

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
