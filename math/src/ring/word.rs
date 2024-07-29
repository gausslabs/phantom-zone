use crate::{
    fft::{bit_reverse, fft64_in_place, ifft64_in_place},
    ring::{ArithmeticOps, RingOps, SliceOps},
};
use core::f64::consts::PI;
use itertools::{chain, izip, Itertools};
use num_complex::Complex64;
use rand::RngCore;

#[derive(Clone, Debug)]
pub struct U64Ring {
    ring_size: usize,
    twiddle: Vec<Complex64>,
    twiddle_inv: Vec<Complex64>,
    twiddle_bo: Vec<Complex64>,
    twiddle_bo_inv: Vec<Complex64>,
}

impl U64Ring {
    pub fn new(ring_size: usize) -> Self {
        assert!(ring_size.is_power_of_two());

        let twiddle = (0..ring_size)
            .map(|i| Complex64::cis((i as f64 * PI) / ring_size as f64))
            .collect_vec();
        let twiddle_inv = twiddle.iter().map(Complex64::conj).collect_vec();

        Self {
            ring_size,
            twiddle: twiddle.clone(),
            twiddle_inv: twiddle_inv.clone(),
            twiddle_bo: bit_reverse(twiddle),
            twiddle_bo_inv: bit_reverse(twiddle_inv),
        }
    }
}

impl ArithmeticOps for U64Ring {
    type Element = u64;
    type Prepared = u64;

    #[inline]
    fn zero(&self) -> Self::Element {
        0
    }
    #[inline]
    fn one(&self) -> Self::Element {
        1
    }
    #[inline]
    fn neg_one(&self) -> Self::Element {
        u64::MAX
    }

    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.wrapping_neg()
    }
    #[inline]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.wrapping_add(*b)
    }
    #[inline]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.wrapping_sub(*b)
    }
    #[inline]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.wrapping_mul(*b)
    }

    #[inline]
    fn prepare(&self, a: &Self::Element) -> Self::Prepared {
        *a
    }
    #[inline]
    fn mul_prepared(&self, a: &Self::Element, b: &Self::Prepared) -> Self::Element {
        self.mul(a, b)
    }

    #[inline]
    fn from_f64(&self, v: f64) -> Self::Element {
        let bits = v.to_bits();
        let sign = bits >> 63;
        let exponent = ((bits >> 52) & 0x7ff) as i64;
        let mantissa = (bits << 11) | 0x8000000000000000;
        let value = match 1086 - exponent {
            shift @ -64..=-1 => mantissa << -shift,
            shift @ 0..=63 => mantissa >> shift,
            _ => 0,
        };
        if sign == 0 {
            value
        } else {
            value.wrapping_neg()
        }
    }
    #[inline]
    fn from_i64(&self, v: i64) -> Self::Element {
        v as _
    }
    #[inline]
    fn to_i64(&self, v: &Self::Element) -> i64 {
        *v as _
    }

    fn sample_uniform(&self, mut rng: impl RngCore) -> Self::Element {
        rng.next_u64()
    }
}

impl SliceOps for U64Ring {}

impl RingOps for U64Ring {
    type Evaluation = Complex64;
    type Twiddle = Complex64;

    fn ring_size(&self) -> usize {
        self.ring_size
    }

    fn evaluation_size(&self) -> usize {
        self.ring_size / 2
    }

    fn forward(&self, a: Vec<Self::Element>) -> Vec<Self::Evaluation> {
        if a.len() == 1 {
            return vec![(self.to_i64(&a[0]) as f64).into()];
        }
        let (lo, hi) = a.split_at(a.len() / 2);
        let mut a = izip!(lo, hi, &self.twiddle)
            .map(|(lo, hi, t)| Complex64::new(self.to_i64(lo) as _, self.to_i64(hi) as _) * t)
            .collect_vec();
        fft64_in_place(&mut a, &self.twiddle_bo);
        a
    }

    fn backward(&self, mut a: Vec<Self::Evaluation>) -> Vec<Self::Element> {
        if a.len() == 1 {
            return vec![self.from_f64(a[0].re)];
        }
        ifft64_in_place(&mut a, &self.twiddle_bo_inv);
        let (lo, hi) = izip!(a, &self.twiddle_inv)
            .map(|(a, t)| a * t)
            .map(|a| (self.from_f64(a.re), self.from_f64(a.im)))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        chain![lo, hi].collect()
    }

    fn evaluation_mul_assign(&self, a: &mut [Self::Evaluation], b: &[Self::Evaluation]) {
        izip!(a, b).for_each(|(a, b)| *a *= b);
    }
}

#[cfg(test)]
mod test {
    use crate::{
        fft::test::{assert_precision, nega_cyclic_schoolbook_mul},
        ring::{word::U64Ring, ArithmeticOps, RingOps},
    };
    use rand::{distributions::Uniform, thread_rng};

    #[test]
    fn ring_ops() {
        let mut rng = thread_rng();
        for log_ring_size in 0..12 {
            for log_b in 12..18 {
                let ring_size = 1 << log_ring_size;
                let ring = U64Ring::new(ring_size);
                for _ in 0..100 {
                    let a = ring.sample_uniform_vec(ring_size, &mut rng);
                    let b = ring.sample_vec(ring_size, Uniform::new(0, 1 << log_b), &mut rng);
                    assert_precision!(
                        a,
                        ring.backward(ring.forward(a.clone())),
                        u64::BITS + log_ring_size - 1 - f64::MANTISSA_DIGITS,
                    );
                    assert_precision!(
                        ring.ring_mul(a.clone(), b.clone()),
                        nega_cyclic_schoolbook_mul(&ring, &a, &b),
                        u64::BITS + log_b + log_ring_size + 1 - f64::MANTISSA_DIGITS,
                    );
                }
            }
        }
    }
}
