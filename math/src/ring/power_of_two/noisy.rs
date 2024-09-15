use crate::ring::power_of_two::PowerOfTwoRing;

pub type NoisyNativeRing = PowerOfTwoRing<true, 64>;
pub type NoisyNonNativePowerOfTwoRing = PowerOfTwoRing<false, 64>;

#[cfg(test)]
mod test {
    use crate::{
        distribution::Sampler,
        modulus::{Native, NonNativePowerOfTwo},
        poly::ffnt::test::{poly_mul_prec_loss, round_trip_prec_loss},
        ring::{
            power_of_two::noisy::{NoisyNativeRing, NoisyNonNativePowerOfTwoRing},
            test::{test_poly_mul, test_round_trip},
            RingOps,
        },
        util::test::assert_precision,
    };
    use rand::{distributions::Uniform, thread_rng};

    #[test]
    fn non_native_round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for log_q in 50..56 {
                let prec_loss = round_trip_prec_loss(log_ring_size, log_q);
                let ring: NoisyNonNativePowerOfTwoRing =
                    RingOps::new(NonNativePowerOfTwo::new(log_q).into(), 1 << log_ring_size);
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
            let ring: NoisyNativeRing = RingOps::new(Native::native().into(), 1 << log_ring_size);
            for _ in 0..100 {
                let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                test_round_trip(&ring, &a, |a, b| assert_precision!(a, b, prec_loss));
            }
        }
    }

    #[test]
    fn non_native_poly_mul() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for log_q in 50..54 {
                let ring: NoisyNonNativePowerOfTwoRing =
                    RingOps::new(NonNativePowerOfTwo::new(log_q).into(), 1 << log_ring_size);
                for log_b in 12..16 {
                    let prec_loss = poly_mul_prec_loss(log_ring_size, log_q, log_b);
                    let uniform_b = Uniform::new(-(1i64 << (log_b - 1)), 1i64 << (log_b - 1));
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
            let ring: NoisyNativeRing = RingOps::new(Native::native().into(), 1 << log_ring_size);
            for log_b in 12..16 {
                let prec_loss = poly_mul_prec_loss(log_ring_size, 64, log_b);
                let uniform_b = Uniform::new(-(1i64 << (log_b - 1)), 1i64 << (log_b - 1));
                for _ in 0..100 {
                    let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                    let b = ring.sample_vec(ring.ring_size(), uniform_b, &mut rng);
                    test_poly_mul(&ring, &a, &b, |a, b| assert_precision!(a, b, prec_loss));
                }
            }
        }
    }
}
