use crate::{
    distribution::Sampler,
    ring::{ffnt::Ffnt, ArithmeticOps, ElemFrom, RingOps, SliceOps},
};
use num_complex::Complex64;
use rand::RngCore;

use super::Modulus;

#[derive(Clone, Debug)]
pub struct U64Ring {
    ffnt: Ffnt,
}

impl U64Ring {
    pub fn new(ring_size: usize) -> Self {
        let ffnt = Ffnt::new(ring_size);
        Self { ffnt }
    }
}

impl ArithmeticOps for U64Ring {
    type Elem = u64;
    type Prep = u64;

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
        u64::MAX
    }

    #[inline(always)]
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        a.wrapping_neg()
    }

    #[inline(always)]
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a.wrapping_add(*b)
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a.wrapping_sub(*b)
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a.wrapping_mul(*b)
    }

    #[inline(always)]
    fn prepare(&self, a: &Self::Elem) -> Self::Prep {
        *a
    }

    #[inline(always)]
    fn mul_prep(&self, a: &Self::Elem, b: &Self::Prep) -> Self::Elem {
        self.mul(a, b)
    }
}

impl ElemFrom<u64> for U64Ring {
    fn elem_from(&self, v: u64) -> Self::Elem {
        v
    }
}

impl ElemFrom<i64> for U64Ring {
    fn elem_from(&self, v: i64) -> Self::Elem {
        v as _
    }
}

impl SliceOps for U64Ring {}

impl RingOps for U64Ring {
    type Eval = Complex64;

    fn new(_: Modulus, _: usize) -> Self {
        todo!()
    }

    fn eval_ops(&self) -> &impl SliceOps<Elem = Self::Eval> {
        &self.ffnt
    }

    fn ring_size(&self) -> usize {
        self.ffnt.ring_size()
    }

    fn eval_size(&self) -> usize {
        self.ffnt.fft_size()
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        self.ffnt.forward(b, a, |a| *a as i64 as _);
    }

    fn forward_normalized(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        self.ffnt.forward_normalized(b, a, |a| *a as i64 as _);
    }

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.ffnt.backward(b, a, |a| f64_mod_u64(a) as _);
    }

    fn backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.ffnt.backward_normalized(b, a, |a| f64_mod_u64(a) as _);
    }

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.ffnt
            .add_backward(b, a, |b, a| *b = self.add(b, &f64_mod_u64(a)));
    }
}

impl Sampler for U64Ring {
    fn sample_uniform(&self, mut rng: impl RngCore) -> Self::Elem {
        rng.next_u64()
    }
}

pub(crate) fn f64_mod_u64(v: f64) -> u64 {
    let bits = v.to_bits();
    let sign = bits >> 63;
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let mantissa = (bits << 11) | 0x8000000000000000;
    let value = match 1086 - exponent {
        shift @ -64..=0 => mantissa << -shift,
        shift @ 1..=63 => mantissa.wrapping_add(1 << (shift - 1)) >> shift,
        _ => 0,
    };
    if sign == 0 {
        value
    } else {
        value.wrapping_neg()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        distribution::Sampler,
        misc::test::assert_precision,
        ring::{
            test::{test_ring_mul, test_round_trip},
            word::U64Ring,
            RingOps,
        },
    };
    use rand::{distributions::Uniform, thread_rng};

    fn round_trip_prec_loss(log_ring_size: usize) -> usize {
        log_ring_size + u64::BITS as usize - (f64::MANTISSA_DIGITS - 1) as usize
    }

    #[test]
    fn round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..12 {
            let prec_loss = round_trip_prec_loss(log_ring_size);
            let ring = U64Ring::new(1 << log_ring_size);
            for _ in 0..100 {
                let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                test_round_trip(&ring, &a, |a, b| assert_precision!(a, b, prec_loss));
            }
        }
    }

    fn ring_mul_prec_loss(log_ring_size: usize, log_b: usize) -> usize {
        log_ring_size + u64::BITS as usize + log_b - (f64::MANTISSA_DIGITS - 1) as usize
    }

    #[test]
    fn ring_mul() {
        let mut rng = thread_rng();
        for log_ring_size in 0..12 {
            let ring = U64Ring::new(1 << log_ring_size);
            for log_b in 12..18 {
                let prec_loss = ring_mul_prec_loss(log_b, log_ring_size);
                let uniform_b = Uniform::new(0, 1u64 << log_b);
                for _ in 0..100 {
                    let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                    let b = ring.sample_vec(ring.ring_size(), uniform_b, &mut rng);
                    test_ring_mul(&ring, &a, &b, |a, b| assert_precision!(a, b, prec_loss));
                }
            }
        }
    }
}
