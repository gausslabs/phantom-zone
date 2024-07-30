use crate::{
    distribution::Sampler,
    ring::{ffnt::Ffnt, word::f64_mod_u64, ArithmeticOps, RingOps, SliceOps},
};
use num_complex::Complex64;
use rand::RngCore;

#[derive(Clone, Debug)]
pub struct PowerOfTwoRing {
    log_q: usize,
    mask: u64,
    ffnt: Ffnt,
}

impl PowerOfTwoRing {
    pub fn new(log_q: usize, ring_size: usize) -> Self {
        assert!(log_q < 64);
        let mask = (1 << log_q) - 1;
        let ffnt = Ffnt::new(ring_size);
        Self { log_q, mask, ffnt }
    }

    fn to_i64(&self, v: u64) -> i64 {
        v.wrapping_sub((v >> (self.log_q - 1)) << self.log_q) as i64
    }
}

impl ArithmeticOps for PowerOfTwoRing {
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
        self.mask
    }

    #[inline(always)]
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        a.wrapping_neg() & self.mask
    }

    #[inline(always)]
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a.wrapping_add(*b) & self.mask
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a.wrapping_sub(*b) & self.mask
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a.wrapping_mul(*b) & self.mask
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

impl SliceOps for PowerOfTwoRing {}

impl RingOps for PowerOfTwoRing {
    type Eval = Complex64;

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
        self.ffnt.forward(b, a, |a| self.to_i64(*a) as _);
    }

    fn forward_normalized(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        self.ffnt.forward_normalized(b, a, |a| self.to_i64(*a) as _);
    }

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.ffnt.backward(b, a, |a| f64_mod_u64(a) & self.mask);
    }

    fn backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.ffnt
            .backward_normalized(b, a, |a| f64_mod_u64(a) & self.mask);
    }

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.ffnt.add_backward(b, a, |b, a| {
            *b = (b.wrapping_add(f64_mod_u64(a))) & self.mask
        });
    }
}

impl Sampler<u64> for PowerOfTwoRing {
    fn from_i64(&self, v: i64) -> u64 {
        (v as u64) & self.mask
    }

    fn sample_uniform(&self, mut rng: impl RngCore) -> u64 {
        rng.next_u64() & self.mask
    }
}

#[cfg(test)]
mod test {
    use crate::{
        distribution::Sampler,
        misc::test::assert_precision,
        ring::{
            po2::PowerOfTwoRing,
            test::{test_ring_mul, test_round_trip},
            RingOps,
        },
    };
    use rand::{distributions::Uniform, thread_rng};

    fn round_trip_prec_loss(log_ring_size: usize, log_q: usize) -> usize {
        (log_ring_size + log_q).saturating_sub((f64::MANTISSA_DIGITS - 1) as usize)
    }

    #[test]
    fn round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..12 {
            for log_q in 50..56 {
                let prec_loss = round_trip_prec_loss(log_ring_size, log_q);
                let ring = PowerOfTwoRing::new(log_q as _, 1 << log_ring_size);
                for _ in 0..100 {
                    let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                    test_round_trip(&ring, &a, |a, b| assert_precision!(a, b, prec_loss));
                }
            }
        }
    }

    fn ring_mul_prec_loss(log_ring_size: usize, log_q: usize, log_b: usize) -> usize {
        (log_ring_size + log_q + log_b).saturating_sub((f64::MANTISSA_DIGITS - 1) as usize)
    }

    #[test]
    fn ring_mul() {
        let mut rng = thread_rng();
        for log_ring_size in 0..12 {
            for log_q in 50..56 {
                let ring = PowerOfTwoRing::new(log_q as _, 1 << log_ring_size);
                for log_b in 12..18 {
                    let prec_loss = ring_mul_prec_loss(log_ring_size, log_q, log_b);
                    let uniform_b = Uniform::new(0, 1 << log_b);
                    for _ in 0..100 {
                        let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                        let b = ring.sample_vec(ring.ring_size(), uniform_b, &mut rng);
                        test_ring_mul(&ring, &a, &b, |a, b| assert_precision!(a, b, prec_loss));
                    }
                }
            }
        }
    }
}
