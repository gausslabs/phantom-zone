use bin_rs::{ModInit, ModularOpsU64, VectorOps};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rand_distr::Uniform;

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("modulus");
    // 55
    for prime in [36028797017456641] {
        for ring_size in [1 << 11, 1 << 15] {
            let modop = ModularOpsU64::new(prime);

            let mut rng = thread_rng();
            let dist = Uniform::new(0, prime);

            let a0 = (&mut rng).sample_iter(dist).take(ring_size).collect_vec();
            let a1 = (&mut rng).sample_iter(dist).take(ring_size).collect_vec();
            let a2 = (&mut rng).sample_iter(dist).take(ring_size).collect_vec();

            group.bench_function(
                BenchmarkId::new("elwise_fma", format!("q={prime}/{ring_size}")),
                |b| {
                    b.iter_batched_ref(
                        || (a0.clone(), a1.clone(), a2.clone()),
                        |(a0, a1, a2)| black_box(modop.elwise_fma_mut(a0, a1, a2)),
                        criterion::BatchSize::PerIteration,
                    )
                },
            );
        }
    }

    group.finish();
}

criterion_group!(modulus, benchmark);
criterion_main!(modulus);
