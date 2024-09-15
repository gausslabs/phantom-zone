use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use phantom_zone_math::{
    modulus::{Modulus, Native, NonNativePowerOfTwo, Prime},
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
};
use rand::thread_rng;

fn forward(c: &mut Criterion) {
    fn runner<R: RingOps + 'static>(modulus: Modulus, ring_size: usize) -> Box<dyn FnMut()> {
        let ring = <R as RingOps>::new(modulus, ring_size);
        let mut rng = thread_rng();
        let mut scratch = ring.allocate_scratch(0, 1, 0);
        let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
        Box::new(move || {
            let mut scratch = scratch.borrow_mut();
            let b = ring.take_eval(&mut scratch);
            let eval_scratch = ring.take_eval_scratch(&mut scratch);
            ring.forward(b, &a, eval_scratch)
        })
    }

    let mut b = c.benchmark_group("forward");
    for log_ring_size in 11..13 {
        let ring_size = 1 << log_ring_size;
        let runners = [
            ("noisy_native", {
                let modulus = Native::native().into();
                runner::<NoisyNativeRing>(modulus, ring_size)
            }),
            ("noisy_non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(50).into();
                runner::<NoisyNonNativePowerOfTwoRing>(modulus, ring_size)
            }),
            ("native", {
                let modulus = Native::native().into();
                runner::<NativeRing>(modulus, ring_size)
            }),
            ("non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(50).into();
                runner::<NonNativePowerOfTwoRing>(modulus, ring_size)
            }),
            ("noisy_prime", {
                let modulus = Prime::gen(50, log_ring_size + 1).into();
                runner::<NoisyPrimeRing>(modulus, ring_size)
            }),
            ("prime", {
                let modulus = Prime::gen(50, log_ring_size + 1).into();
                runner::<PrimeRing>(modulus, ring_size)
            }),
        ];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, ring_size);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn poly_mul(c: &mut Criterion) {
    fn runner<R: RingOps + 'static>(modulus: Modulus, ring_size: usize) -> Box<dyn FnMut()> {
        let ring = <R as RingOps>::new(modulus, ring_size);
        let mut rng = thread_rng();
        let mut scratch = black_box(ring.allocate_scratch(1, 2, 0));
        let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
        let b = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
        Box::new(move || {
            let mut scratch = scratch.borrow_mut();
            let c = ring.take_poly(&mut scratch);
            ring.poly_mul(c, &a, &b, scratch.reborrow())
        })
    }

    let mut b = c.benchmark_group("poly_mul");
    for log_ring_size in 11..13 {
        let ring_size = 1 << log_ring_size;
        let runners = [
            ("noisy_native", {
                let modulus = Native::native().into();
                runner::<NoisyNativeRing>(modulus, ring_size)
            }),
            ("noisy_non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(50).into();
                runner::<NoisyNonNativePowerOfTwoRing>(modulus, ring_size)
            }),
            ("native", {
                let modulus = Native::native().into();
                runner::<NativeRing>(modulus, ring_size)
            }),
            ("non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(50).into();
                runner::<NonNativePowerOfTwoRing>(modulus, ring_size)
            }),
            ("noisy_prime", {
                let modulus = Prime::gen(50, log_ring_size + 1).into();
                runner::<NoisyPrimeRing>(modulus, ring_size)
            }),
            ("prime", {
                let modulus = Prime::gen(50, log_ring_size + 1).into();
                runner::<PrimeRing>(modulus, ring_size)
            }),
        ];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, ring_size);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

criterion_group!(benches, forward, poly_mul);
criterion_main!(benches);
