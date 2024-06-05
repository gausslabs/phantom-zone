use bin_rs::{Ntt, NttBackendU64, NttInit};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rand_distr::Uniform;

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt");
    // 55
    for prime in [36028797017456641] {
        for ring_size in [1 << 11, 1 << 15] {
            let ntt = NttBackendU64::new(&prime, ring_size);
            let mut rng = thread_rng();

            let a = rng
                .sample_iter(Uniform::new(0, prime))
                .take(ring_size)
                .collect_vec();
            group.bench_function(
                BenchmarkId::new("forward", format!("q={prime}/{ring_size}")),
                |b| {
                    b.iter_batched_ref(
                        || a.clone(),
                        |mut a| black_box(ntt.forward(&mut a)),
                        criterion::BatchSize::PerIteration,
                    )
                },
            );
        }
    }

    group.finish();
}

criterion_group!(ntt, benchmark);
criterion_main!(ntt);
