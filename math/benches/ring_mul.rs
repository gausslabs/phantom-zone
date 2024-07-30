use core::array::from_fn;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use phantom_zone_math::{
    prime::two_adic_primes,
    ring::{po2::PowerOfTwoRing, prime::PrimeRing, word::U64Ring, RingOps},
};

fn ring_mul(c: &mut Criterion) {
    struct RingMulRunner<R: RingOps> {
        ring: R,
        abc: [Vec<R::Elem>; 3],
        scratch: Vec<R::Eval>,
    }

    impl<R: RingOps> RingMulRunner<R> {
        fn new(ring: R) -> Self {
            let abc = from_fn(|_| vec![R::Elem::default(); ring.ring_size()]);
            let scratch = vec![R::Eval::default(); ring.scratch_size()];
            Self { ring, abc, scratch }
        }

        fn run(&mut self) {
            let [a, b, c] = &mut self.abc;
            self.ring.ring_mul(c, a, b, &mut self.scratch)
        }
    }

    let mut b = c.benchmark_group("ring_mul");
    for log_ring_size in 11..15 {
        let ring_size = 1 << log_ring_size;

        let mut u64_ring_runner = black_box(RingMulRunner::new(U64Ring::new(ring_size)));
        b.bench_with_input(BenchmarkId::new("u64_ring", ring_size), &(), |b, _| {
            b.iter(|| u64_ring_runner.run())
        });

        let mut po2_ring_runner = black_box(RingMulRunner::new(PowerOfTwoRing::new(50, ring_size)));
        b.bench_with_input(BenchmarkId::new("po2_ring", ring_size), &(), |b, _| {
            b.iter(|| po2_ring_runner.run())
        });

        let q = two_adic_primes(50, log_ring_size + 1).next().unwrap();
        let mut prime_ring_runner = black_box(RingMulRunner::new(PrimeRing::new(q, ring_size)));
        b.bench_with_input(BenchmarkId::new("prime_ring", ring_size), &(), |b, _| {
            b.iter(|| prime_ring_runner.run())
        });
    }
}

criterion_group!(benches, ring_mul);
criterion_main!(benches);
