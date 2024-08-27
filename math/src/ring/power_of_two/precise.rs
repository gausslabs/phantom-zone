use crate::{
    modulus::{ElemFrom, Modulus, ModulusOps},
    poly::ffnt::Ffnt,
    ring::{power_of_two, RingOps},
};
use num_complex::Complex64;

const LIMBS: usize = 4;
const LIMB_BITS: usize = u64::BITS as usize / LIMBS;
const LIMB_MASK: u64 = (1 << LIMB_BITS) - 1;

pub type NativeRing = PowerOfTwoRing<true>;
pub type NonNativePowerOfTwoRing = PowerOfTwoRing<false>;
pub type PowerOfTwoRing<const NATIVE: bool> = power_of_two::PowerOfTwoRing<NATIVE, LIMBS>;

impl<const NATIVE: bool> PowerOfTwoRing<NATIVE> {
    #[inline(always)]
    fn split_limbs<'a, T>(&self, a: &'a [T]) -> [&'a [T]; LIMBS] {
        debug_assert_eq!(a.len(), LIMBS * self.fft.fft_size());
        let (a, b) = a.split_at(2 * self.fft.fft_size());
        let (aa, ab) = a.split_at(self.fft.fft_size());
        let (ba, bb) = b.split_at(self.fft.fft_size());
        [aa, ab, ba, bb]
    }

    #[inline(always)]
    fn split_limbs_mut<'a, T>(&self, a: &'a mut [T]) -> [&'a mut [T]; LIMBS] {
        debug_assert_eq!(a.len(), LIMBS * self.fft.fft_size());
        let (a, b) = a.split_at_mut(2 * self.fft.fft_size());
        let (aa, ab) = a.split_at_mut(self.fft.fft_size());
        let (ba, bb) = b.split_at_mut(self.fft.fft_size());
        [aa, ab, ba, bb]
    }

    #[inline(always)]
    fn forward_limb<T: Copy, const I: usize>(
        &self,
        b: &mut [Complex64],
        a: &[T],
        scratch: &mut [Complex64],
    ) where
        Self: ElemFrom<T, Elem = u64>,
    {
        let to_f64 = |a: &_| ((self.elem_from(*a) >> (I * LIMB_BITS)) & LIMB_MASK) as _;
        self.fft.forward(b, a, to_f64, scratch);
    }

    #[inline(always)]
    fn add_backward_limb<const I: usize>(
        &self,
        b: &mut [u64],
        a: &mut [Complex64],
        scratch: &mut [Complex64],
    ) {
        let add_from_f64 = |b: &mut u64, a| {
            *b = if I == LIMBS - 1 {
                self.q
                    .reduce(b.wrapping_add(from_f64(a) << (I * LIMB_BITS)))
            } else {
                b.wrapping_add(from_f64(a) << (I * LIMB_BITS))
            }
        };
        self.fft.add_backward(b, a, add_from_f64, scratch);
    }
}

impl<const NATIVE: bool> RingOps for PowerOfTwoRing<NATIVE>
where
    Self: ModulusOps<Elem = u64>,
{
    type Eval = Complex64;
    type EvalPrep = Complex64;

    fn new(modulus: Modulus, ring_size: usize) -> Self {
        Self::new(modulus.try_into().unwrap(), Ffnt::new(ring_size))
    }

    fn ring_size(&self) -> usize {
        self.fft.ring_size()
    }

    fn eval_size(&self) -> usize {
        LIMBS * self.fft.fft_size()
    }

    fn eval_scratch_size(&self) -> usize {
        self.fft.fft_scratch_size()
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem], scratch: &mut [Self::Eval]) {
        let [b0, b1, b2, b3] = self.split_limbs_mut(b);
        self.forward_limb::<_, 0>(b0, a, scratch);
        self.forward_limb::<_, 1>(b1, a, scratch);
        self.forward_limb::<_, 2>(b2, a, scratch);
        self.forward_limb::<_, 3>(b3, a, scratch);
    }

    fn forward_elem_from<T: Copy>(&self, b: &mut [Self::Eval], a: &[T], scratch: &mut [Self::Eval])
    where
        Self: ElemFrom<T>,
    {
        let [b0, b1, b2, b3] = self.split_limbs_mut(b);
        self.forward_limb::<_, 0>(b0, a, scratch);
        self.forward_limb::<_, 1>(b1, a, scratch);
        self.forward_limb::<_, 2>(b2, a, scratch);
        self.forward_limb::<_, 3>(b3, a, scratch);
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

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval], scratch: &mut [Self::Eval]) {
        let [a0, a1, a2, a3] = self.split_limbs_mut(a);
        self.fft.backward(b, a0, from_f64, scratch);
        self.add_backward_limb::<1>(b, a1, scratch);
        self.add_backward_limb::<2>(b, a2, scratch);
        self.add_backward_limb::<3>(b, a3, scratch);
    }

    fn backward_normalized(
        &self,
        b: &mut [Self::Elem],
        a: &mut [Self::Eval],
        scratch: &mut [Self::Eval],
    ) {
        self.fft.normalize(a);
        self.backward(b, a, scratch);
    }

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval], scratch: &mut [Self::Eval]) {
        let [a0, a1, a2, a3] = self.split_limbs_mut(a);
        self.add_backward_limb::<0>(b, a0, scratch);
        self.add_backward_limb::<1>(b, a1, scratch);
        self.add_backward_limb::<2>(b, a2, scratch);
        self.add_backward_limb::<3>(b, a3, scratch);
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
        let [c0, c1, c2, c3] = self.split_limbs_mut(c);
        let [a0, a1, a2, a3] = self.split_limbs(a);
        let [b0, b1, b2, b3] = self.split_limbs(b);

        self.fft.eval_mul(&mut *c3, a0, b3);
        self.fft.eval_fma(&mut *c3, a1, b2);
        self.fft.eval_fma(&mut *c3, a2, b1);
        self.fft.eval_fma(&mut *c3, a3, b0);

        self.fft.eval_mul(&mut *c2, a0, b2);
        self.fft.eval_fma(&mut *c2, a1, b1);
        self.fft.eval_fma(&mut *c2, a2, b0);

        self.fft.eval_mul(&mut *c1, a0, b1);
        self.fft.eval_fma(&mut *c1, a1, b0);

        self.fft.eval_mul(&mut *c0, a0, b0);
    }

    fn eval_mul_assign(&self, b: &mut [Self::Eval], a: &[Self::Eval]) {
        let [b0, b1, b2, b3] = self.split_limbs_mut(b);
        let [a0, a1, a2, a3] = self.split_limbs(a);

        self.fft.eval_mul_assign(&mut *b3, a0);
        self.fft.eval_fma(&mut *b3, a1, &*b2);
        self.fft.eval_fma(&mut *b3, a2, &*b1);
        self.fft.eval_fma(&mut *b3, a3, &*b0);

        self.fft.eval_mul_assign(&mut *b2, a0);
        self.fft.eval_fma(&mut *b2, a1, &*b1);
        self.fft.eval_fma(&mut *b2, a2, &*b0);

        self.fft.eval_mul_assign(&mut *b1, a0);
        self.fft.eval_fma(&mut *b1, a1, &*b0);

        self.fft.eval_mul_assign(&mut *b0, a0);
    }

    fn eval_fma(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        let [c0, c1, c2, c3] = self.split_limbs_mut(c);
        let [a0, a1, a2, a3] = self.split_limbs(a);
        let [b0, b1, b2, b3] = self.split_limbs(b);

        self.fft.eval_fma(&mut *c3, a0, b3);
        self.fft.eval_fma(&mut *c3, a1, b2);
        self.fft.eval_fma(&mut *c3, a2, b1);
        self.fft.eval_fma(&mut *c3, a3, b0);

        self.fft.eval_fma(&mut *c2, a0, b2);
        self.fft.eval_fma(&mut *c2, a1, b1);
        self.fft.eval_fma(&mut *c2, a2, b0);

        self.fft.eval_fma(&mut *c1, a0, b1);
        self.fft.eval_fma(&mut *c1, a1, b0);

        self.fft.eval_fma(&mut *c0, a0, b0);
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

#[inline(always)]
fn from_f64(a: f64) -> u64 {
    a.round() as i64 as u64
}

#[cfg(test)]
mod test {
    use crate::{
        distribution::Sampler,
        modulus::{Native, NonNativePowerOfTwo},
        ring::{
            power_of_two::precise::{NativeRing, NonNativePowerOfTwoRing},
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
                    RingOps::new(NonNativePowerOfTwo::new(log_q).into(), 1 << log_ring_size);
                let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                test_round_trip(&ring, &a, |a, b| assert_eq!(a, b));
            }
        }
    }

    #[test]
    fn native_round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            let ring: NativeRing = RingOps::new(Native::native().into(), 1 << log_ring_size);
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
                    RingOps::new(NonNativePowerOfTwo::new(log_q).into(), 1 << log_ring_size);
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
            let ring: NativeRing = RingOps::new(Native::native().into(), 1 << log_ring_size);
            let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
            let b = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
            test_poly_mul(&ring, &a, &b, |a, b| assert_eq!(a, b));
        }
    }
}
