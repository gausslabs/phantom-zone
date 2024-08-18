use crate::{
    decomposer::PrimeDecomposer,
    izip_eq,
    modulus::{prime::Shoup, Modulus},
    poly::ntt::Ntt,
    ring::{ElemFrom, ModulusOps, RingOps, SliceOps},
};

pub type PrimeRing = crate::ring::prime::PrimeRing<Ntt>;

impl RingOps for PrimeRing {
    type Eval = u64;
    type EvalPrep = Shoup;
    type Decomposer = PrimeDecomposer;

    fn new(modulus: Modulus, ring_size: usize) -> Self {
        let q = modulus.try_into().unwrap();
        Self::new(q, Ntt::new(q, ring_size))
    }

    fn ring_size(&self) -> usize {
        self.fft.ring_size()
    }

    fn eval_size(&self) -> usize {
        self.fft.eval_size()
    }

    fn eval_scratch_size(&self) -> usize {
        0
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem], _: &mut [Self::Eval]) {
        b.copy_from_slice(a);
        self.fft.forward(b);
    }

    fn forward_elem_from<T: Copy>(&self, b: &mut [Self::Eval], a: &[T], _: &mut [Self::Eval])
    where
        Self: ElemFrom<T>,
    {
        self.slice_elem_from(b, a);
        self.fft.forward(b);
    }

    fn forward_normalized(
        &self,
        b: &mut [Self::Eval],
        a: &[Self::Elem],
        scratch: &mut [Self::Eval],
    ) {
        self.forward(b, a, scratch);
        self.fft.normalize(b);
    }

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval], _: &mut [Self::Eval]) {
        b.copy_from_slice(a);
        self.fft.backward(b);
        b.iter_mut().for_each(|b| self.q.reduce_once_assign(b));
    }

    fn backward_normalized(
        &self,
        b: &mut [Self::Elem],
        a: &mut [Self::Eval],
        _: &mut [Self::Eval],
    ) {
        b.copy_from_slice(a);
        self.fft.backward(b);
        self.fft.normalize(b);
    }

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval], _: &mut [Self::Eval]) {
        self.fft.backward(a);
        izip_eq!(b, a).for_each(|(b, a)| {
            self.q.reduce_once_assign(a);
            *b = self.add(a, b);
        })
    }

    fn add_backward_normalized(
        &self,
        b: &mut [Self::Elem],
        a: &mut [Self::Eval],
        _: &mut [Self::Eval],
    ) {
        self.fft.backward(a);
        izip_eq!(b, a).for_each(|(b, a)| *b = self.add(&self.q.mul_prep(a, self.fft.n_inv()), b))
    }

    fn eval_prepare(&self, b: &mut [Self::EvalPrep], a: &[Self::Eval]) {
        self.slice_prepare(b, a)
    }

    fn eval_mul(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        self.slice_mul(c, a, b)
    }

    fn eval_mul_assign(&self, b: &mut [Self::Eval], a: &[Self::Eval]) {
        self.slice_mul_assign(b, a)
    }

    fn eval_fma(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        self.slice_fma(c, a, b);
    }

    fn eval_mul_prep(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::EvalPrep]) {
        self.slice_mul_prep(c, a, b)
    }

    fn eval_mul_assign_prep(&self, b: &mut [Self::Eval], a: &[Self::EvalPrep]) {
        self.slice_mul_assign_prep(b, a)
    }

    fn eval_fma_prep(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::EvalPrep]) {
        self.slice_op(c, a, b, |c, a, b| {
            *c = c.wrapping_add(self.mul_prep(a, b));
            self.q.reduce_once_assign(c)
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{
        distribution::Sampler,
        modulus::Prime,
        ring::{
            prime::precise::PrimeRing,
            test::{test_poly_mul, test_round_trip},
            RingOps,
        },
    };
    use rand::thread_rng;

    #[test]
    fn round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for q in Prime::gen_iter(50, log_ring_size + 1).take(10) {
                let ring: PrimeRing = RingOps::new(q.into(), 1 << log_ring_size);
                let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                test_round_trip(&ring, &a, |a, b| assert_eq!(a, b));
            }
        }
    }

    #[test]
    fn poly_mul() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for q in Prime::gen_iter(50, log_ring_size + 1).take(10) {
                let ring: PrimeRing = RingOps::new(q.into(), 1 << log_ring_size);
                let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                let b = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                test_poly_mul(&ring, &a, &b, |a, b| assert_eq!(a, b));
            }
        }
    }
}
