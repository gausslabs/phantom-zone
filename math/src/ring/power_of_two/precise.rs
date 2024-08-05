use crate::{
    decomposer::PowerOfTwoDecomposer,
    misc::poly_mul::{
        nega_cyclic_karatsuba_fma, nega_cyclic_karatsuba_mul, nega_cyclic_karatsuba_mul_assign,
    },
    modulus::Modulus,
    ring::{power_of_two, ElemFrom, RingOps, SliceOps},
};

pub type PowerOfTwoRing<const NATIVE: bool> = power_of_two::PowerOfTwoRing<usize, NATIVE>;

pub type NativeRing = PowerOfTwoRing<true>;

pub type NonNativePowerOfTwoRing = PowerOfTwoRing<false>;

impl<const NATIVE: bool> RingOps for PowerOfTwoRing<NATIVE> {
    type Eval = Self::Elem;
    type EvalPrep = Self::Elem;
    type Decomposer = PowerOfTwoDecomposer<NATIVE>;

    fn new(modulus: Modulus, ring_size: usize) -> Self {
        Self::new(modulus.try_into().unwrap(), ring_size)
    }

    fn modulus(&self) -> Modulus {
        self.modulus.into()
    }

    fn ring_size(&self) -> usize {
        self.ffnt
    }

    fn eval_size(&self) -> usize {
        self.ffnt
    }

    fn eval_prep_size(&self) -> usize {
        self.ffnt
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        b.copy_from_slice(a)
    }

    fn forward_elem_from<T: Copy>(&self, b: &mut [Self::Eval], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.slice_elem_from(b, a)
    }

    fn forward_normalized(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        b.copy_from_slice(a)
    }

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        b.copy_from_slice(a);
    }

    fn backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        b.copy_from_slice(a)
    }

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.slice_add_assign(b, a)
    }

    fn add_backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.slice_add_assign(b, a)
    }

    fn eval_prepare(&self, b: &mut [Self::EvalPrep], a: &[Self::Eval]) {
        b.copy_from_slice(a)
    }

    fn eval_mul(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        nega_cyclic_karatsuba_mul(self, c, a, b)
    }

    fn eval_mul_assign(&self, b: &mut [Self::Eval], a: &[Self::Eval]) {
        nega_cyclic_karatsuba_mul_assign(self, b, a)
    }

    fn eval_fma(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        nega_cyclic_karatsuba_fma(self, c, a, b)
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
        modulus::{Modulus, PowerOfTwo},
        ring::{
            power_of_two::{NativeRing, NonNativePowerOfTwoRing},
            test::{test_poly_mul, test_round_trip},
            RingOps,
        },
    };
    use rand::thread_rng;

    #[test]
    fn non_native_round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for log_q in 50..56 {
                let ring: NonNativePowerOfTwoRing =
                    RingOps::new(PowerOfTwo(log_q).into(), 1 << log_ring_size);
                let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                test_round_trip(&ring, &a, |a, b| assert_eq!(a, b));
            }
        }
    }

    #[test]
    fn native_round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            let ring: NativeRing = RingOps::new(Modulus::native(), 1 << log_ring_size);
            let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
            test_round_trip(&ring, &a, |a, b| assert_eq!(a, b));
        }
    }

    #[test]
    fn non_native_poly_mul() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for log_q in 50..54 {
                let ring: NonNativePowerOfTwoRing =
                    RingOps::new(PowerOfTwo(log_q).into(), 1 << log_ring_size);
                let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                let b = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                test_poly_mul(&ring, &a, &b, |a, b| assert_eq!(a, b));
            }
        }
    }

    #[test]
    fn native_poly_mul() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            let ring: NativeRing = RingOps::new(Modulus::native(), 1 << log_ring_size);
            let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
            let b = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
            test_poly_mul(&ring, &a, &b, |a, b| assert_eq!(a, b));
        }
    }
}
