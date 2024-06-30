use itertools::{izip, Itertools};

use crate::bool::evaluator::BooleanGates;

pub(super) fn half_adder<E: BooleanGates>(
    evaluator: &mut E,
    a: &mut E::Ciphertext,
    b: &E::Ciphertext,
    key: &E::Key,
) -> E::Ciphertext {
    let carry = evaluator.and(a, b, key);
    evaluator.xor_inplace(a, b, key);
    carry
}

pub(super) fn full_adder_plain_carry_in<E: BooleanGates>(
    evaluator: &mut E,
    a: &mut E::Ciphertext,
    b: &E::Ciphertext,
    carry_in: bool,
    key: &E::Key,
) -> E::Ciphertext {
    let mut a_and_b = evaluator.and(a, b, key);
    evaluator.xor_inplace(a, b, key); //a = a ^ b
    if carry_in {
        // a_and_b = A & B | ((A^B) & C_in={True})
        evaluator.or_inplace(&mut a_and_b, &a, key);
    } else {
        // a_and_b = A & B | ((A^B) & C_in={False})
        // a_and_b = A & B
        // noop
    }

    // In xor if a input is 0, output equals the firt variable. If input is 1 then
    // output equals !(first variable)
    if carry_in {
        // (A^B)^1 = !(A^B)
        evaluator.not_inplace(a);
    } else {
        // (A^B)^0
        // no-op
    }
    a_and_b
}

pub(super) fn full_adder<E: BooleanGates>(
    evaluator: &mut E,
    a: &mut E::Ciphertext,
    b: &E::Ciphertext,
    carry_in: &E::Ciphertext,
    key: &E::Key,
) -> E::Ciphertext {
    let mut a_and_b = evaluator.and(a, b, key);
    evaluator.xor_inplace(a, b, key); //a = a ^ b
    let a_xor_b_and_c = evaluator.and(&a, carry_in, key);
    evaluator.or_inplace(&mut a_and_b, &a_xor_b_and_c, key); // a_and_b = A & B | ((A^B) & C_in)
    evaluator.xor_inplace(a, &carry_in, key);
    a_and_b
}

pub(super) fn arbitrary_bit_adder<E: BooleanGates>(
    evaluator: &mut E,
    a: &mut [E::Ciphertext],
    b: &[E::Ciphertext],
    carry_in: bool,
    key: &E::Key,
) -> (E::Ciphertext, E::Ciphertext)
where
    E::Ciphertext: Clone,
{
    assert!(a.len() == b.len());
    let n = a.len();

    let mut carry = if !carry_in {
        half_adder(evaluator, &mut a[0], &b[0], key)
    } else {
        full_adder_plain_carry_in(evaluator, &mut a[0], &b[0], true, key)
    };

    izip!(a.iter_mut(), b.iter())
        .skip(1)
        .take(n - 3)
        .for_each(|(a_bit, b_bit)| {
            carry = full_adder(evaluator, a_bit, b_bit, &carry, key);
        });

    let carry_last_last = full_adder(evaluator, &mut a[n - 2], &b[n - 2], &carry, key);
    let carry_last = full_adder(evaluator, &mut a[n - 1], &b[n - 1], &carry_last_last, key);

    (carry_last, carry_last_last)
}

pub(super) fn arbitrary_bit_subtractor<E: BooleanGates>(
    evaluator: &mut E,
    a: &[E::Ciphertext],
    b: &[E::Ciphertext],
    key: &E::Key,
) -> (Vec<E::Ciphertext>, E::Ciphertext, E::Ciphertext)
where
    E::Ciphertext: Clone,
{
    let mut neg_b: Vec<E::Ciphertext> = b.iter().map(|v| evaluator.not(v)).collect();
    let (carry_last, carry_last_last) = arbitrary_bit_adder(evaluator, &mut neg_b, &a, true, key);
    return (neg_b, carry_last, carry_last_last);
}

pub(super) fn bit_mux<E: BooleanGates>(
    evaluator: &mut E,
    selector: E::Ciphertext,
    if_true: &E::Ciphertext,
    if_false: &E::Ciphertext,
    key: &E::Key,
) -> E::Ciphertext {
    // (s&a) | ((1-s)^b)
    let not_selector = evaluator.not(&selector);

    let s_and_a = evaluator.and(&selector, if_true, key);
    let s_and_b = evaluator.and(&not_selector, if_false, key);
    evaluator.or(&s_and_a, &s_and_b, key)
}

pub(super) fn arbitrary_bit_mux<E: BooleanGates>(
    evaluator: &mut E,
    selector: &E::Ciphertext,
    if_true: &[E::Ciphertext],
    if_false: &[E::Ciphertext],
    key: &E::Key,
) -> Vec<E::Ciphertext> {
    // (s&a) | ((1-s)^b)
    let not_selector = evaluator.not(&selector);

    izip!(if_true.iter(), if_false.iter())
        .map(|(a, b)| {
            let s_and_a = evaluator.and(&selector, a, key);
            let s_and_b = evaluator.and(&not_selector, b, key);
            evaluator.or(&s_and_a, &s_and_b, key)
        })
        .collect()
}

pub(super) fn eight_bit_mul<E: BooleanGates>(
    evaluator: &mut E,
    a: &[E::Ciphertext],
    b: &[E::Ciphertext],
    key: &E::Key,
) -> Vec<E::Ciphertext> {
    assert!(a.len() == 8);
    assert!(b.len() == 8);
    let mut carries = Vec::with_capacity(7);
    let mut out = Vec::with_capacity(8);

    for i in (0..8) {
        if i == 0 {
            let s = evaluator.and(&a[0], &b[0], key);
            out.push(s);
        } else if i == 1 {
            let mut tmp0 = evaluator.and(&a[1], &b[0], key);
            let tmp1 = evaluator.and(&a[0], &b[1], key);
            let carry = half_adder(evaluator, &mut tmp0, &tmp1, key);
            carries.push(carry);
            out.push(tmp0);
        } else {
            let mut sum = {
                let mut sum = evaluator.and(&a[i], &b[0], key);
                let tmp = evaluator.and(&a[i - 1], &b[1], key);
                carries[0] = full_adder(evaluator, &mut sum, &tmp, &carries[0], key);
                sum
            };

            for j in 2..i {
                let tmp = evaluator.and(&a[i - j], &b[j], key);
                carries[j - 1] = full_adder(evaluator, &mut sum, &tmp, &carries[j - 1], key);
            }

            let tmp = evaluator.and(&a[0], &b[i], key);
            let carry = half_adder(evaluator, &mut sum, &tmp, key);
            carries.push(carry);

            out.push(sum)
        }
        debug_assert!(carries.len() <= 7);
    }

    out
}

pub(super) fn arbitrary_bit_division_for_quotient_and_rem<E: BooleanGates>(
    evaluator: &mut E,
    a: &[E::Ciphertext],
    b: &[E::Ciphertext],
    key: &E::Key,
) -> (Vec<E::Ciphertext>, Vec<E::Ciphertext>)
where
    E::Ciphertext: Clone,
{
    let n = a.len();
    let neg_b = b.iter().map(|v| evaluator.not(v)).collect_vec();

    // Both remainder and quotient are initially stored in Big-endian in contrast to
    // the usual little endian we use. This is more friendly to vec pushes in
    // division. After computing remainder and quotient, we simply reverse the
    // vectors.
    let mut remainder = vec![];
    let mut quotient = vec![];
    for i in 0..n {
        // left shift
        remainder.push(a[n - 1 - i].clone());

        let mut subtract = remainder.clone();

        // subtraction
        // At i^th iteration remainder is only filled with i bits and the rest of the
        // bits are zero. For example, at i = 1
        //            0 0 0 0 0 0 X X => remainder
        //          - Y Y Y Y Y Y Y Y => divisor                                      .
        //            ---------------                                                 .
        //            Z Z Z Z Z Z Z Z => result
        // For the next iteration we only care about result if divisor is <= remainder
        // (which implies result <= remainder). Otherwise we care about remainder
        // (recall re-storing division). Hence we optimise subtraction and
        // ignore full adders for places where remainder bits are known to be false
        // bits. We instead use `ANDs` to compute the carry overs, since the
        // last carry over indicates whether the value has overflown (i.e. divisor <=
        // remainder). Last carry out is `true` if value has not overflown, otherwise
        // false.
        let mut carry =
            full_adder_plain_carry_in(evaluator, &mut subtract[i], &neg_b[0], true, key);
        for j in 1..i + 1 {
            carry = full_adder(evaluator, &mut subtract[i - j], &neg_b[j], &carry, key);
        }
        for j in i + 1..n {
            // All I care about are the carries
            evaluator.and_inplace(&mut carry, &neg_b[j], key);
        }

        let not_carry = evaluator.not(&carry);
        // Choose `remainder` if subtraction has overflown (i.e. carry = false).
        // Otherwise choose `subtractor`.
        //
        // mux k^a | !(k)^b, where k is the selector.
        izip!(remainder.iter_mut(), subtract.iter_mut()).for_each(|(r, s)| {
            // choose `s` when carry is true, otherwise choose r
            evaluator.and_inplace(s, &carry, key);
            evaluator.and_inplace(r, &not_carry, key);
            evaluator.or_inplace(r, s, key);
        });

        // Set i^th MSB of quotient to 1 if carry = true, otherwise set it to 0.
        // X&1 | X&0 => X&1 => X
        quotient.push(carry);
    }

    remainder.reverse();
    quotient.reverse();

    (quotient, remainder)
}

fn is_zero<E: BooleanGates>(evaluator: &mut E, a: &[E::Ciphertext], key: &E::Key) -> E::Ciphertext {
    let mut a = a.iter().map(|v| evaluator.not(v)).collect_vec();
    let (out, rest_a) = a.split_at_mut(1);
    rest_a.iter().for_each(|c| {
        evaluator.and_inplace(&mut out[0], c, key);
    });
    return a.remove(0);
}

pub(super) fn arbitrary_bit_equality<E: BooleanGates>(
    evaluator: &mut E,
    a: &[E::Ciphertext],
    b: &[E::Ciphertext],
    key: &E::Key,
) -> E::Ciphertext {
    assert!(a.len() == b.len());
    let mut out = evaluator.xnor(&a[0], &b[0], key);
    izip!(a.iter(), b.iter()).skip(1).for_each(|(abit, bbit)| {
        let e = evaluator.xnor(abit, bbit, key);
        evaluator.and_inplace(&mut out, &e, key);
    });
    return out;
}

/// Comparator handle computes comparator result 2ns MSB onwards. It is
/// separated because comparator subroutine for signed and unsgind integers
/// differs only for 1st MSB and is common second MSB onwards
fn _comparator_handler_from_second_msb<E: BooleanGates>(
    evaluator: &mut E,
    a: &[E::Ciphertext],
    b: &[E::Ciphertext],
    mut comp: E::Ciphertext,
    mut casc: E::Ciphertext,
    key: &E::Key,
) -> E::Ciphertext {
    let n = a.len();

    // handle MSB - 1
    let mut tmp = evaluator.not(&b[n - 2]);
    evaluator.and_inplace(&mut tmp, &a[n - 2], key);
    evaluator.and_inplace(&mut tmp, &casc, key);
    evaluator.or_inplace(&mut comp, &tmp, key);

    for i in 2..n {
        // calculate cascading bit
        let tmp_casc = evaluator.xnor(&a[n - i], &b[n - i], key);
        evaluator.and_inplace(&mut casc, &tmp_casc, key);

        // calculate computate bit
        let mut tmp = evaluator.not(&b[n - 1 - i]);
        evaluator.and_inplace(&mut tmp, &a[n - 1 - i], key);
        evaluator.and_inplace(&mut tmp, &casc, key);
        evaluator.or_inplace(&mut comp, &tmp, key);
    }

    return comp;
}

/// Signed integer comparison is same as unsigned integer with MSB flipped.
pub(super) fn arbitrary_signed_bit_comparator<E: BooleanGates>(
    evaluator: &mut E,
    a: &[E::Ciphertext],
    b: &[E::Ciphertext],
    key: &E::Key,
) -> E::Ciphertext {
    assert!(a.len() == b.len());
    let n = a.len();

    // handle MSB
    let mut comp = evaluator.not(&a[n - 1]);
    evaluator.and_inplace(&mut comp, &b[n - 1], key); // comp
    let casc = evaluator.xnor(&a[n - 1], &b[n - 1], key); // casc

    return _comparator_handler_from_second_msb(evaluator, a, b, comp, casc, key);
}

pub(super) fn arbitrary_bit_comparator<E: BooleanGates>(
    evaluator: &mut E,
    a: &[E::Ciphertext],
    b: &[E::Ciphertext],
    key: &E::Key,
) -> E::Ciphertext {
    assert!(a.len() == b.len());
    let n = a.len();

    // handle MSB
    let mut comp = evaluator.not(&b[n - 1]);
    evaluator.and_inplace(&mut comp, &a[n - 1], key);
    let casc = evaluator.xnor(&a[n - 1], &b[n - 1], key);

    return _comparator_handler_from_second_msb(evaluator, a, b, comp, casc, key);
}
