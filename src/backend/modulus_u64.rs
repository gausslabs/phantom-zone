use itertools::izip;
use num_traits::WrappingMul;

use super::{
    ArithmeticLazyOps, ArithmeticOps, GetModulus, ModInit, Modulus, ShoupMatrixFMA, VectorOps,
};
use crate::RowMut;

pub struct ModularOpsU64<T> {
    q: u64,
    q_twice: u64,
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
            q_twice: q << 1,
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

    fn add_mod_fast_lazy(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q_twice);
        debug_assert!(b < self.q_twice);

        let mut o = a + b;
        if o >= self.q_twice {
            o -= self.q_twice;
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

    // returns (a * b)  % q
    ///
    /// - both a and b must be in range [0, 2q)
    /// - output is in range [0 , 2q)
    fn mul_mod_fast_lazy(&self, a: u64, b: u64) -> u64 {
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

        (ab - tmp) as u64
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

impl<T> ArithmeticLazyOps for ModularOpsU64<T> {
    type Element = u64;
    fn add_lazy(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.add_mod_fast_lazy(*a, *b)
    }
    fn mul_lazy(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul_mod_fast_lazy(*a, *b)
    }
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

impl<R: RowMut<Element = u64>, T> ShoupMatrixFMA<R> for ModularOpsU64<T> {
    fn shoup_matrix_fma(&self, out: &mut [R::Element], a: &[R], a_shoup: &[R], b: &[R]) {
        assert!(a.len() == a_shoup.len());
        assert!(a.len() == b.len());

        let q = self.q;
        let q_twice = self.q << 1;

        izip!(a.iter(), a_shoup.iter(), b.iter()).for_each(|(a_row, a_shoup_row, b_row)| {
            izip!(
                out.as_mut().iter_mut(),
                a_row.as_ref().iter(),
                a_shoup_row.as_ref().iter(),
                b_row.as_ref().iter()
            )
            .for_each(|(o, a0, a0_shoup, b0)| {
                let quotient = ((*a0_shoup as u128 * *b0 as u128) >> 64) as u64;
                let mut v = (a0.wrapping_mul(b0)).wrapping_add(*o);
                v = v.wrapping_sub(q.wrapping_mul(quotient));

                if v >= q_twice {
                    v -= q_twice;
                }

                *o = v;
            });
        });
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use rand::{thread_rng, Rng};
    use rand_distr::Uniform;

    #[test]
    fn fma() {
        let mut rng = thread_rng();
        let prime = 36028797017456641;
        let ring_size = 1 << 3;

        let dist = Uniform::new(0, prime);
        let d = 2;
        let a0_matrix = (0..d)
            .into_iter()
            .map(|_| (&mut rng).sample_iter(dist).take(ring_size).collect_vec())
            .collect_vec();
        // a0 in shoup representation
        let a0_shoup_matrix = a0_matrix
            .iter()
            .map(|r| {
                r.iter()
                    .map(|v| {
                        // $(v * 2^{\beta}) / p$
                        ((*v as u128 * (1u128 << 64)) / prime as u128) as u64
                    })
                    .collect_vec()
            })
            .collect_vec();
        let a1_matrix = (0..d)
            .into_iter()
            .map(|_| (&mut rng).sample_iter(dist).take(ring_size).collect_vec())
            .collect_vec();

        let modop = ModularOpsU64::new(prime);

        let mut out_shoup_fma_lazy = vec![0u64; ring_size];
        modop.shoup_matrix_fma(
            &mut out_shoup_fma_lazy,
            &a0_matrix,
            &a0_shoup_matrix,
            &a1_matrix,
        );
        let out_shoup_fma = out_shoup_fma_lazy
            .iter()
            .map(|v| if *v >= prime { v - prime } else { *v })
            .collect_vec();

        // expected
        let mut out_expected = vec![0u64; ring_size];
        izip!(a0_matrix.iter(), a1_matrix.iter()).for_each(|(a_r, b_r)| {
            izip!(out_expected.iter_mut(), a_r.iter(), b_r.iter()).for_each(|(o, a0, a1)| {
                *o = (*o + ((*a0 as u128 * *a1 as u128) % prime as u128) as u64) % prime;
            });
        });

        assert_eq!(out_expected, out_shoup_fma);
    }
}
