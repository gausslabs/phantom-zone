use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use phantom_zone_math::{
    modulus::{Modulus, NonNativePowerOfTwo, Prime},
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
};

fn poly_mul(c: &mut Criterion) {
    fn runner<R: RingOps + 'static>(ring: R) -> Box<dyn FnMut()> {
        let mut scratch = black_box(ring.allocate_scratch(3, 2, 0));
        Box::new(move || {
            let scratch = &mut scratch.borrow_mut();
            let [a, b, c] = ring.take_polys(scratch);
            ring.poly_mul(c, a, b, scratch.reborrow())
        })
    }

    let mut b = c.benchmark_group("poly_mul");
    for log_ring_size in 11..15 {
        let ring_size = 1 << log_ring_size;
        let runners = [
            ("noisy_native", {
                let modulus = Modulus::native();
                runner(NoisyNativeRing::new(modulus, ring_size))
            }),
            ("noisy_non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(50).into();
                runner(NoisyNonNativePowerOfTwoRing::new(modulus, ring_size))
            }),
            ("native", {
                let modulus = Modulus::native();
                runner(NativeRing::new(modulus, ring_size))
            }),
            ("non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(50).into();
                runner(NonNativePowerOfTwoRing::new(modulus, ring_size))
            }),
            ("noisy_prime", {
                let modulus = Prime::gen(50, log_ring_size + 1).into();
                runner(NoisyPrimeRing::new(modulus, ring_size))
            }),
            ("prime", {
                let modulus = Prime::gen(50, log_ring_size + 1).into();
                runner(PrimeRing::new(modulus, ring_size))
            }),
        ];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, ring_size);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

criterion_group!(benches, poly_mul);
criterion_main!(benches);
