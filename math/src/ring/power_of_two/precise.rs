use crate::ring::power_of_two::PowerOfTwoRing;

pub type NativeRing<const LIMB_BITS: usize = 16> = PowerOfTwoRing<true, LIMB_BITS>;
pub type NonNativePowerOfTwoRing<const LIMB_BITS: usize = 16> = PowerOfTwoRing<false, LIMB_BITS>;

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
