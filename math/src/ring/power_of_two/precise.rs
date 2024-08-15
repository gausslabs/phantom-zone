use crate::{
    decomposer::PowerOfTwoDecomposer,
    modulus::{Modulus, ModulusOps},
    poly::ffnt::Ffnt,
    ring::{power_of_two, ElemFrom, RingOps},
};
use itertools::izip;
use num_complex::Complex64;

const LIMBS: usize = 4;
const LIMB_BITS: usize = u64::BITS as usize / LIMBS;
const LIMB_MASK: u64 = (1 << LIMB_BITS) - 1;

pub type NativeRing = PowerOfTwoRing<true>;

pub type NonNativePowerOfTwoRing = PowerOfTwoRing<false>;

pub type PowerOfTwoRing<const NATIVE: bool> = power_of_two::PowerOfTwoRing<NATIVE, LIMBS>;

impl<const NATIVE: bool> PowerOfTwoRing<NATIVE> {
    fn split_limbs<'a, T>(&self, a: &'a [T]) -> [&'a [T]; LIMBS] {
        let (a, b) = a.split_at(2 * self.fft.fft_size());
        let (aa, ab) = a.split_at(self.fft.fft_size());
        let (ba, bb) = b.split_at(self.fft.fft_size());
        [aa, ab, ba, bb]
    }

    fn split_limbs_mut<'a, T>(&self, a: &'a mut [T]) -> [&'a mut [T]; LIMBS] {
        let (a, b) = a.split_at_mut(2 * self.fft.fft_size());
        let (aa, ab) = a.split_at_mut(self.fft.fft_size());
        let (ba, bb) = b.split_at_mut(self.fft.fft_size());
        [aa, ab, ba, bb]
    }
}

impl<const NATIVE: bool> RingOps for PowerOfTwoRing<NATIVE>
where
    Self: ModulusOps<Elem = u64>,
{
    type Eval = Complex64;
    type EvalPrep = Complex64;
    type Decomposer = PowerOfTwoDecomposer<NATIVE>;

    fn new(modulus: Modulus, ring_size: usize) -> Self {
        Self::new(modulus.try_into().unwrap(), Ffnt::new(ring_size))
    }

    fn ring_size(&self) -> usize {
        self.fft.ring_size()
    }

    fn eval_size(&self) -> usize {
        LIMBS * self.fft.fft_size()
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        let [b0, b1, b2, b3] = self.split_limbs_mut(b);
        self.fft.forward(b0, a, to_f64::<0>);
        self.fft.forward(b1, a, to_f64::<1>);
        self.fft.forward(b2, a, to_f64::<2>);
        self.fft.forward(b3, a, to_f64::<3>);
    }

    fn forward_elem_from<T: Copy>(&self, b: &mut [Self::Eval], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        let [b0, b1, b2, b3] = self.split_limbs_mut(b);
        let to_u64 = |a: &_| self.elem_from(*a);
        self.fft.forward(b0, a, |a| to_f64::<0>(&to_u64(a)));
        self.fft.forward(b1, a, |a| to_f64::<1>(&to_u64(a)));
        self.fft.forward(b2, a, |a| to_f64::<2>(&to_u64(a)));
        self.fft.forward(b3, a, |a| to_f64::<3>(&to_u64(a)));
    }

    fn forward_normalized(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        self.forward(b, a);
        self.fft.normalize(b);
    }

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        let [a0, a1, a2, a3] = self.split_limbs_mut(a);
        let reduce_add_from_f64 = |b: &mut _, a| *b = self.q.reduce(add_f64::<3>(b, a));
        self.fft.backward(b, a0, from_f64);
        self.fft.add_backward(b, a1, add_from_f64::<1>);
        self.fft.add_backward(b, a2, add_from_f64::<2>);
        self.fft.add_backward(b, a3, reduce_add_from_f64);
    }

    fn backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.fft.normalize(a);
        self.backward(b, a);
    }

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        let [a0, a1, a2, a3] = self.split_limbs_mut(a);
        let reduce_add_from_f64 = |b: &mut _, a| *b = self.q.reduce(add_f64::<3>(b, a));
        self.fft.add_backward(b, a0, add_from_f64::<0>);
        self.fft.add_backward(b, a1, add_from_f64::<1>);
        self.fft.add_backward(b, a2, add_from_f64::<2>);
        self.fft.add_backward(b, a3, reduce_add_from_f64);
    }

    fn add_backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.fft.normalize(a);
        self.add_backward(b, a);
    }

    fn eval_prepare(&self, b: &mut [Self::EvalPrep], a: &[Self::Eval]) {
        b.copy_from_slice(a)
    }

    fn eval_mul(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        let [c0, c1, c2, c3] = self.split_limbs_mut(c);
        let [a0, a1, a2, a3] = self.split_limbs(a);
        let [b0, b1, b2, b3] = self.split_limbs(b);

        izip!(&mut *c3, a0, b3).for_each(|(c, a, b)| *c = *a * b);
        izip!(&mut *c3, a1, b2).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *c3, a2, b1).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *c3, a3, b0).for_each(|(c, a, b)| *c += *a * b);

        izip!(&mut *c2, a0, b2).for_each(|(c, a, b)| *c = *a * b);
        izip!(&mut *c2, a1, b1).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *c2, a2, b0).for_each(|(c, a, b)| *c += *a * b);

        izip!(&mut *c1, a0, b1).for_each(|(c, a, b)| *c = *a * b);
        izip!(&mut *c1, a1, b0).for_each(|(c, a, b)| *c += *a * b);

        izip!(&mut *c0, a0, b0).for_each(|(c, a, b)| *c = *a * b);
    }

    fn eval_mul_assign(&self, b: &mut [Self::Eval], a: &[Self::Eval]) {
        let [b0, b1, b2, b3] = self.split_limbs_mut(b);
        let [a0, a1, a2, a3] = self.split_limbs(a);

        izip!(&mut *b3, a0).for_each(|(c, a)| *c *= *a);
        izip!(&mut *b3, a1, &*b2).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *b3, a2, &*b1).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *b3, a3, &*b0).for_each(|(c, a, b)| *c += *a * b);

        izip!(&mut *b2, a0).for_each(|(c, a)| *c *= *a);
        izip!(&mut *b2, a1, &*b1).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *b2, a2, &*b0).for_each(|(c, a, b)| *c += *a * b);

        izip!(&mut *b1, a0).for_each(|(c, a)| *c *= *a);
        izip!(&mut *b1, a1, &*b0).for_each(|(c, a, b)| *c += *a * b);

        izip!(&mut *b0, a0).for_each(|(c, a)| *c *= *a);
    }

    fn eval_fma(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        let [c0, c1, c2, c3] = self.split_limbs_mut(c);
        let [a0, a1, a2, a3] = self.split_limbs(a);
        let [b0, b1, b2, b3] = self.split_limbs(b);

        izip!(&mut *c3, a0, b3).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *c3, a1, b2).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *c3, a2, b1).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *c3, a3, b0).for_each(|(c, a, b)| *c += *a * b);

        izip!(&mut *c2, a0, b2).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *c2, a1, b1).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *c2, a2, b0).for_each(|(c, a, b)| *c += *a * b);

        izip!(&mut *c1, a0, b1).for_each(|(c, a, b)| *c += *a * b);
        izip!(&mut *c1, a1, b0).for_each(|(c, a, b)| *c += *a * b);

        izip!(&mut *c0, a0, b0).for_each(|(c, a, b)| *c += *a * b);
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
fn to_f64<const I: usize>(a: &u64) -> f64 {
    ((a >> (I * LIMB_BITS)) & LIMB_MASK) as _
}

#[inline(always)]
fn from_f64(a: f64) -> u64 {
    a.round() as i64 as u64
}

#[inline(always)]
fn add_f64<const I: usize>(b: &u64, a: f64) -> u64 {
    b.wrapping_add(from_f64(a) << (I * LIMB_BITS))
}

#[inline(always)]
fn add_from_f64<const I: usize>(b: &mut u64, a: f64) {
    *b = add_f64::<I>(b, a)
}

#[cfg(test)]
mod test {
    use crate::{
        distribution::Sampler,
        modulus::{Modulus, NonNativePowerOfTwo},
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
            let ring: NativeRing = RingOps::new(Modulus::native(), 1 << log_ring_size);
            let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
            let b = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
            test_poly_mul(&ring, &a, &b, |a, b| assert_eq!(a, b));
        }
    }
}
