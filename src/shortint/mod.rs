use itertools::Itertools;

use crate::{
    bool::evaluator::{BoolEvaluator, ClientKey, ServerKeyEvaluationDomain, BOOL_SERVER_KEY},
    utils::{Global, WithLocal},
    Decryptor, Encryptor,
};
use ops::{
    arbitrary_bit_adder, arbitrary_bit_division_for_quotient_and_rem, arbitrary_bit_subtractor,
    eight_bit_mul,
};

mod ops;
mod types;

type FheUint8 = types::FheUint8<Vec<u64>>;

fn add_mut(a: &mut FheUint8, b: &FheUint8) {
    BoolEvaluator::with_local_mut_mut(&mut |e| {
        let key = ServerKeyEvaluationDomain::global();
        arbitrary_bit_adder(e, a.data_mut(), b.data(), false, key);
    });
}

fn sub(a: &FheUint8, b: &FheUint8) -> FheUint8 {
    BoolEvaluator::with_local_mut(|e| {
        let key = ServerKeyEvaluationDomain::global();
        let (out, _, _) = arbitrary_bit_subtractor(e, a.data(), b.data(), key);
        FheUint8 { data: out }
    })
}

fn mul(a: &FheUint8, b: &FheUint8) -> FheUint8 {
    BoolEvaluator::with_local_mut(|e| {
        let key = ServerKeyEvaluationDomain::global();
        let out = eight_bit_mul(e, a.data(), b.data(), key);
        FheUint8 { data: out }
    })
}

fn div(a: &FheUint8, b: &FheUint8) -> (FheUint8, FheUint8) {
    BoolEvaluator::with_local_mut(|e| {
        let key = ServerKeyEvaluationDomain::global();
        let (quotient, remainder) =
            arbitrary_bit_division_for_quotient_and_rem(e, a.data(), b.data(), key);

        (FheUint8 { data: quotient }, FheUint8 { data: remainder })
    })
}

impl Encryptor<u8, FheUint8> for ClientKey {
    fn encrypt(&self, m: &u8) -> FheUint8 {
        let cts = (0..8)
            .into_iter()
            .map(|i| {
                let bit = ((m >> i) & 1) == 1;
                Encryptor::<bool, Vec<u64>>::encrypt(self, &bit)
            })
            .collect_vec();
        FheUint8 { data: cts }
    }
}

impl Decryptor<u8, FheUint8> for ClientKey {
    fn decrypt(&self, c: &FheUint8) -> u8 {
        let mut out = 0u8;
        c.data().iter().enumerate().for_each(|(index, bit_c)| {
            let bool = Decryptor::<bool, Vec<u64>>::decrypt(self, bit_c);
            if bool {
                out += 1 << index;
            }
        });
        out
    }
}

mod frontend {
    use super::ops::{
        arbitrary_bit_adder, arbitrary_bit_division_for_quotient_and_rem, arbitrary_bit_subtractor,
        eight_bit_mul,
    };
    use crate::{
        bool::evaluator::{BoolEvaluator, ServerKeyEvaluationDomain},
        utils::{Global, WithLocal},
    };

    use super::{add_mut, div, mul, FheUint8};

    mod arithetic {
        use super::*;
        use std::ops::{Add, AddAssign, Div, Mul, Rem, Sub};

        impl AddAssign<&FheUint8> for FheUint8 {
            fn add_assign(&mut self, rhs: &FheUint8) {
                BoolEvaluator::with_local_mut_mut(&mut |e| {
                    let key = ServerKeyEvaluationDomain::global();
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
                    let key = ServerKeyEvaluationDomain::global();
                    let (out, _, _) = arbitrary_bit_subtractor(e, self.data(), self.data(), key);
                    FheUint8 { data: out }
                })
            }
        }

        impl Mul<&FheUint8> for &FheUint8 {
            type Output = FheUint8;
            fn mul(self, rhs: &FheUint8) -> Self::Output {
                BoolEvaluator::with_local_mut(|e| {
                    let key = ServerKeyEvaluationDomain::global();
                    let out = eight_bit_mul(e, self.data(), rhs.data(), key);
                    FheUint8 { data: out }
                })
            }
        }

        impl Div<&FheUint8> for &FheUint8 {
            type Output = FheUint8;
            fn div(self, rhs: &FheUint8) -> Self::Output {
                BoolEvaluator::with_local_mut(|e| {
                    let key = ServerKeyEvaluationDomain::global();
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
                    let key = ServerKeyEvaluationDomain::global();
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
    }

    mod booleans {}
}

#[cfg(test)]
mod tests {
    use num_traits::Euclid;

    use crate::{
        bool::{
            evaluator::{gen_keys, set_parameter_set, BoolEvaluator},
            parameters::SP_BOOL_PARAMS,
        },
        shortint::{add_mut, div, mul, sub, types::FheUint8},
        Decryptor, Encryptor,
    };

    #[test]
    fn qwerty() {
        set_parameter_set(&SP_BOOL_PARAMS);

        let (ck, sk) = gen_keys();
        sk.set_server_key();

        for i in 1..=255 {
            for j in 0..=255 {
                let m0 = i;
                let m1 = j;
                let c0 = ck.encrypt(&m0);
                let c1 = ck.encrypt(&m1);

                assert!(ck.decrypt(&c0) == m0);
                assert!(ck.decrypt(&c1) == m1);

                // Add
                // let mut c_m0_plus_m1 = FheUint8 {
                //     data: c0.data().to_vec(),
                // };
                // add_mut(&mut c_m0_plus_m1, &c1);
                // let m0_plus_m1 = ck.decrypt(&c_m0_plus_m1);
                // assert_eq!(
                //     m0_plus_m1,
                //     m0.wrapping_add(m1),
                //     "Expected {} but got {m0_plus_m1} for {i}+{j}",
                //     m0.wrapping_add(m1)
                // );

                // Sub
                // let c_sub = sub(&c0, &c1);
                // let m0_sub_m1 = ck.decrypt(&c_sub);
                // dbg!(m0, m1, m0_sub_m1);
                // assert_eq!(
                //     m0_sub_m1,
                //     m0.wrapping_sub(m1),
                //     "Expected {} but got {m0_sub_m1} for {i}-{j}",
                //     m0.wrapping_sub(m1)
                // );

                // Mul
                // let c_m0m1 = mul(&c0, &c1);
                // let m0m1 = ck.decrypt(&c_m0m1);
                // assert_eq!(
                //     m0m1,
                //     m0.wrapping_mul(m1),
                //     "Expected {} but got {m0m1} for {i}x{j}",
                //     m0.wrapping_mul(m1)
                // );

                // Div
                // let (c_quotient, c_rem) = div(&c0, &c1);
                // let m_quotient = ck.decrypt(&c_quotient);
                // let m_remainder = ck.decrypt(&c_rem);
                // if j != 0 {
                //     let (q, r) = i.div_rem_euclid(&j);
                //     assert_eq!(
                //         m_quotient, q,
                //         "Expected {} but got {m_quotient} for {i}/{j}",
                //         q
                //     );
                //     assert_eq!(
                //         m_remainder, r,
                //         "Expected {} but got {m_quotient} for {i}%{j}",
                //         r
                //     );
                // } else {
                //     assert_eq!(
                //         m_quotient, 255,
                //         "Expected 255 but got {m_quotient}. Case div by zero"
                //     );
                //     assert_eq!(
                //         m_remainder, i,
                //         "Expected {i} but got {m_quotient}. Case div by zero"
                //     )
                // }
            }
        }
    }
}
