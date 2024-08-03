use crate::{
    decomposer::PowerOfTwoDecomposer,
    distribution::{DistributionSized, Sampler},
    modulus::{Modulus, PowerOfTwo},
    ring::{ffnt::Ffnt, ArithmeticOps, ElemFrom, ElemTo, RingOps, SliceOps},
};
use num_complex::Complex64;
use rand::distributions::{Distribution, Uniform};

pub type NativeRing = PowerOfTwoRing<true>;

pub type NonNativePowerOfTwoRing = PowerOfTwoRing<false>;

#[derive(Clone, Debug)]
pub struct PowerOfTwoRing<const NATIVE: bool> {
    modulus: PowerOfTwo,
    mask: u64,
    ffnt: Ffnt,
}

impl<const NATIVE: bool> PowerOfTwoRing<NATIVE> {
    fn new(modulus: PowerOfTwo, ring_size: usize) -> Self {
        if NATIVE {
            assert_eq!(modulus.bits(), 64);
        } else {
            assert!(modulus.bits() < 64);
        }
        let ffnt = Ffnt::new(ring_size);
        Self {
            modulus,
            mask: modulus.mask(),
            ffnt,
        }
    }

    #[inline(always)]
    fn reduce(&self, v: u64) -> u64 {
        if NATIVE {
            v
        } else {
            v & self.mask
        }
    }

    #[inline(always)]
    fn to_i64(&self, v: u64) -> i64 {
        if NATIVE {
            v as _
        } else {
            v.wrapping_sub((v >> (self.modulus.bits() - 1)) << self.modulus.bits()) as i64
        }
    }
}

impl<const NATIVE: bool> ArithmeticOps for PowerOfTwoRing<NATIVE> {
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
        self.reduce(a.wrapping_neg())
    }

    #[inline(always)]
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.reduce(a.wrapping_add(*b))
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.reduce(a.wrapping_sub(*b))
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.reduce(a.wrapping_mul(*b))
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

impl<const NATIVE: bool> ElemFrom<u64> for PowerOfTwoRing<NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: u64) -> Self::Elem {
        self.reduce(v)
    }
}

impl<const NATIVE: bool> ElemFrom<i64> for PowerOfTwoRing<NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: i64) -> Self::Elem {
        self.reduce(v as u64)
    }
}

impl<const NATIVE: bool> ElemFrom<u32> for PowerOfTwoRing<NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: u32) -> Self::Elem {
        self.elem_from(v as u64)
    }
}

impl<const NATIVE: bool> ElemFrom<i32> for PowerOfTwoRing<NATIVE> {
    #[inline(always)]
    fn elem_from(&self, v: i32) -> Self::Elem {
        self.elem_from(v as i64)
    }
}

impl<const NATIVE: bool> ElemTo<u64> for PowerOfTwoRing<NATIVE> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> u64 {
        v
    }
}

impl<const NATIVE: bool> ElemTo<i64> for PowerOfTwoRing<NATIVE> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> i64 {
        self.to_i64(v)
    }
}

impl<const NATIVE: bool> SliceOps for PowerOfTwoRing<NATIVE> {}

impl<const NATIVE: bool> Sampler for PowerOfTwoRing<NATIVE> {
    fn uniform_distribution(
        &self,
    ) -> impl Distribution<Self::Elem> + DistributionSized<Self::Elem> {
        Uniform::new_inclusive(0, self.modulus.max())
    }
}

impl<const NATIVE: bool> RingOps for PowerOfTwoRing<NATIVE> {
    type Eval = Complex64;
    type Decomposer = PowerOfTwoDecomposer<NATIVE>;

    fn new(modulus: Modulus, ring_size: usize) -> Self {
        Self::new(modulus.try_into().unwrap(), ring_size)
    }

    fn modulus(&self) -> Modulus {
        self.modulus.into()
    }

    fn ring_size(&self) -> usize {
        self.ffnt.ring_size()
    }

    fn eval_size(&self) -> usize {
        self.ffnt.fft_size()
    }

    fn eval(&self) -> &impl SliceOps<Elem = Self::Eval> {
        &self.ffnt
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        self.ffnt.forward(b, a, |a| self.to_i64(*a) as _);
    }

    fn forward_elem_from<T: Copy>(&self, b: &mut [Self::Eval], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.ffnt.forward(b, a, |a| {
            let a: i64 = self.elem_to(self.elem_from(*a));
            a as _
        });
    }

    fn forward_normalized(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        self.ffnt.forward_normalized(b, a, |a| self.to_i64(*a) as _);
    }

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.ffnt.backward(b, a, |a| self.reduce(f64_mod_u64(a)));
    }

    fn backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.ffnt
            .backward_normalized(b, a, |a| self.reduce(f64_mod_u64(a)));
    }

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.ffnt.add_backward(b, a, |b, a| {
            *b = self.reduce(b.wrapping_add(f64_mod_u64(a)))
        });
    }

    fn add_backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.ffnt.add_backward_normalized(b, a, |b, a| {
            *b = self.reduce(b.wrapping_add(f64_mod_u64(a)))
        });
    }
}

fn f64_mod_u64(v: f64) -> u64 {
    let bits = v.to_bits();
    let sign = bits >> 63;
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let mantissa = (bits << 11) | 0x8000000000000000;
    let value = match 1086 - exponent {
        shift @ -63..=0 => mantissa << -shift,
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
        modulus::PowerOfTwo,
        ring::{
            power_of_two::{NativeRing, NonNativePowerOfTwoRing},
            test::{test_poly_mul, test_round_trip},
            RingOps,
        },
    };
    use rand::{distributions::Uniform, thread_rng};

    fn round_trip_prec_loss(log_ring_size: usize, log_q: usize) -> usize {
        (log_ring_size + log_q).saturating_sub((f64::MANTISSA_DIGITS - 1) as usize)
    }

    #[test]
    fn non_native_round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for log_q in 50..56 {
                let prec_loss = round_trip_prec_loss(log_ring_size, log_q);
                let ring = NonNativePowerOfTwoRing::new(PowerOfTwo(log_q), 1 << log_ring_size);
                for _ in 0..100 {
                    let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                    test_round_trip(&ring, &a, |a, b| assert_precision!(a, b, prec_loss));
                }
            }
        }
    }

    #[test]
    fn native_round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            let prec_loss = round_trip_prec_loss(log_ring_size, 64);
            let ring = NativeRing::new(PowerOfTwo::native(), 1 << log_ring_size);
            for _ in 0..100 {
                let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                test_round_trip(&ring, &a, |a, b| assert_precision!(a, b, prec_loss));
            }
        }
    }

    fn poly_mul_prec_loss(log_ring_size: usize, log_q: usize, log_b: usize) -> usize {
        (log_ring_size + log_q + log_b).saturating_sub((f64::MANTISSA_DIGITS - 1) as usize)
    }

    #[test]
    fn non_native_poly_mul() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for log_q in 50..54 {
                let ring = NonNativePowerOfTwoRing::new(PowerOfTwo(log_q), 1 << log_ring_size);
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

    #[test]
    fn native_poly_mul() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            let ring = NativeRing::new(PowerOfTwo::native(), 1 << log_ring_size);
            for log_b in 12..16 {
                let prec_loss = poly_mul_prec_loss(log_ring_size, 64, log_b);
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
