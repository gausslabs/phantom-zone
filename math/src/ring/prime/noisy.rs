use crate::{
    modulus::{ElemFrom, Modulus},
    poly::ffnt::Ffnt,
    ring::{prime, RingOps},
};
use num_complex::Complex64;

/// A `RingOps` implementation that supports small prime modulus (less than
/// `1 << 62`) .
///
/// The ring multiplication is implemented with complex FFT, hence slightly
/// faster than [`PrimeRing`](crate::ring::prime::precise::PrimeRing) but noisy.
///
/// It panics in [`RingOps::new`] if `modulus` is not in range `1..1 << 62`.
pub type NoisyPrimeRing = prime::PrimeRing<Ffnt>;

impl RingOps for NoisyPrimeRing {
    type Eval = Complex64;
    type EvalPrep = Complex64;

    fn new(modulus: Modulus, ring_size: usize) -> Self {
        let q = modulus.try_into().unwrap();
        Self::new(q, Ffnt::new(ring_size))
    }

    fn ring_size(&self) -> usize {
        self.fft.ring_size()
    }

    fn eval_size(&self) -> usize {
        self.fft.fft_size()
    }

    fn eval_scratch_size(&self) -> usize {
        self.fft.fft_scratch_size()
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem], scratch: &mut [Self::Eval]) {
        self.fft.forward(b, a, |a| self.q.center(*a) as _, scratch);
    }

    fn forward_elem_from<T: Copy>(&self, b: &mut [Self::Eval], a: &[T], scratch: &mut [Self::Eval])
    where
        Self: ElemFrom<T>,
    {
        self.fft
            .forward(b, a, |a| self.q.center(self.elem_from(*a)) as _, scratch);
    }

    fn forward_normalized(
        &self,
        b: &mut [Self::Eval],
        a: &[Self::Elem],
        scratch: &mut [Self::Eval],
    ) {
        self.fft
            .forward_normalized(b, a, |a| self.q.center(*a) as _, scratch);
    }

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval], scratch: &mut [Self::Eval]) {
        self.fft.backward(b, a, |a| self.elem_from(a), scratch);
    }

    fn backward_normalized(
        &self,
        b: &mut [Self::Elem],
        a: &mut [Self::Eval],
        scratch: &mut [Self::Eval],
    ) {
        self.fft
            .backward_normalized(b, a, |a| self.elem_from(a), scratch);
    }

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval], scratch: &mut [Self::Eval]) {
        let add_from_f64 =
            |b: &mut _, a: f64| *b = self.q.reduce_i128(*b as i128 + a.round() as i128);
        self.fft.add_backward(b, a, add_from_f64, scratch);
    }

    fn add_backward_normalized(
        &self,
        b: &mut [Self::Elem],
        a: &mut [Self::Eval],
        scratch: &mut [Self::Eval],
    ) {
        self.fft.normalize(a);
        self.add_backward(b, a, scratch);
    }

    fn eval_prepare(&self, b: &mut [Self::EvalPrep], a: &[Self::Eval]) {
        b.copy_from_slice(a)
    }

    fn eval_mul(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        self.fft.eval_mul(c, a, b)
    }

    fn eval_mul_assign(&self, b: &mut [Self::Eval], a: &[Self::Eval]) {
        self.fft.eval_mul_assign(b, a)
    }

    fn eval_fma(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        self.fft.eval_fma(c, a, b)
    }

    fn eval_mul_prep(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::EvalPrep]) {
        self.eval_mul(c, a, b)
    }

    fn eval_mul_assign_prep(&self, b: &mut [Self::Eval], a: &[Self::EvalPrep]) {
        self.eval_mul_assign(b, a)
    }

    fn eval_fma_prep(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::EvalPrep]) {
        self.eval_fma(c, a, b)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        distribution::Sampler,
        modulus::Prime,
        poly::ffnt::test::{poly_mul_prec_loss, round_trip_prec_loss},
        ring::{
            prime::noisy::NoisyPrimeRing,
            test::{test_poly_mul, test_round_trip},
            RingOps,
        },
        util::test::assert_precision,
    };
    use rand::{distributions::Uniform, thread_rng};

    #[test]
    fn round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for log_q in 50..56 {
                let prec_loss = round_trip_prec_loss(log_ring_size, log_q);
                let modulus = Prime::gen(log_q, log_ring_size + 1).into();
                let ring: NoisyPrimeRing = RingOps::new(modulus, 1 << log_ring_size);
                for _ in 0..100 {
                    let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                    test_round_trip(&ring, &a, |a, b| assert_precision!(a, b, prec_loss));
                }
            }
        }
    }

    #[test]
    fn poly_mul() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for log_q in 50..54 {
                let modulus = Prime::gen(log_q, log_ring_size + 1).into();
                let ring: NoisyPrimeRing = RingOps::new(modulus, 1 << log_ring_size);
                for log_b in 12..16 {
                    let prec_loss = poly_mul_prec_loss(log_ring_size, log_q, log_b);
                    let uniform_b = Uniform::new(0, 1u64 << log_b);
                    for _ in 0..100 {
                        let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                        let b = ring.sample_vec(ring.ring_size(), uniform_b, &mut rng);
                        test_poly_mul(&ring, &a, &b, |a, b| assert_precision!(a, b, prec_loss));
                    }
                }
            }
        }
    }
}
