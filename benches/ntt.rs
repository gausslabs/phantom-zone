use bin_rs::{Ntt, NttBackendU64, NttInit};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rand_distr::Uniform;

fn forward_matrix(a: &mut [Vec<u64>], nttop: &NttBackendU64) {
    a.iter_mut().for_each(|r| nttop.forward(r.as_mut_slice()));
}

fn forward_lazy_matrix(a: &mut [Vec<u64>], nttop: &NttBackendU64) {
    a.iter_mut()
        .for_each(|r| nttop.forward_lazy(r.as_mut_slice()));
}

fn backward_matrix(a: &mut [Vec<u64>], nttop: &NttBackendU64) {
    a.iter_mut().for_each(|r| nttop.backward(r.as_mut_slice()));
}

fn backward_lazy_matrix(a: &mut [Vec<u64>], nttop: &NttBackendU64) {
    a.iter_mut()
        .for_each(|r| nttop.backward_lazy(r.as_mut_slice()));
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt");
    // 55
    for prime in [36028797017456641] {
        for ring_size in [1 << 11] {
            let ntt = NttBackendU64::new(&prime, ring_size);
            let mut rng = thread_rng();

            let a = (&mut rng)
                .sample_iter(Uniform::new(0, prime))
                .take(ring_size)
                .collect_vec();
            let d = 2;
            let a_matrix = (0..d)
                .map(|_| {
                    (&mut rng)
                        .sample_iter(Uniform::new(0, prime))
                        .take(ring_size)
                        .collect_vec()
                })
                .collect_vec();

            {
                group.bench_function(
                    BenchmarkId::new("forward", format!("q={prime}/N={ring_size}")),
                    |b| {
                        b.iter_batched_ref(
                            || a.clone(),
                            |mut a| black_box(ntt.forward(&mut a)),
                            criterion::BatchSize::PerIteration,
                        )
                    },
                );

                group.bench_function(
                    BenchmarkId::new("forward_lazy", format!("q={prime}/N={ring_size}")),
                    |b| {
                        b.iter_batched_ref(
                            || a.clone(),
                            |mut a| black_box(ntt.forward_lazy(&mut a)),
                            criterion::BatchSize::PerIteration,
                        )
                    },
                );

                group.bench_function(
                    BenchmarkId::new("forward_matrix", format!("q={prime}/N={ring_size}/d={d}")),
                    |b| {
                        b.iter_batched_ref(
                            || a_matrix.clone(),
                            |a_matrix| black_box(forward_matrix(a_matrix, &ntt)),
                            criterion::BatchSize::PerIteration,
                        )
                    },
                );

                group.bench_function(
                    BenchmarkId::new(
                        "forward_lazy_matrix",
                        format!("q={prime}/N={ring_size}/d={d}"),
                    ),
                    |b| {
                        b.iter_batched_ref(
                            || a_matrix.clone(),
                            |a_matrix| black_box(forward_lazy_matrix(a_matrix, &ntt)),
                            criterion::BatchSize::PerIteration,
                        )
                    },
                );
            }

            {
                group.bench_function(
                    BenchmarkId::new("backward", format!("q={prime}/N={ring_size}")),
                    |b| {
                        b.iter_batched_ref(
                            || a.clone(),
                            |mut a| black_box(ntt.backward(&mut a)),
                            criterion::BatchSize::PerIteration,
                        )
                    },
                );

                group.bench_function(
                    BenchmarkId::new("backward_lazy", format!("q={prime}/N={ring_size}")),
                    |b| {
                        b.iter_batched_ref(
                            || a.clone(),
                            |mut a| black_box(ntt.backward_lazy(&mut a)),
                            criterion::BatchSize::PerIteration,
                        )
                    },
                );

                group.bench_function(
                    BenchmarkId::new("backward_matrix", format!("q={prime}/N={ring_size}")),
                    |b| {
                        b.iter_batched_ref(
                            || a_matrix.clone(),
                            |a_matrix| black_box(backward_matrix(a_matrix, &ntt)),
                            criterion::BatchSize::PerIteration,
                        )
                    },
                );

                group.bench_function(
                    BenchmarkId::new(
                        "backward_lazy_matrix",
                        format!("q={prime}/N={ring_size}/d={d}"),
                    ),
                    |b| {
                        b.iter_batched_ref(
                            || a_matrix.clone(),
                            |a_matrix| black_box(backward_lazy_matrix(a_matrix, &ntt)),
                            criterion::BatchSize::PerIteration,
                        )
                    },
                );
            }
        }
    }

    group.finish();
}

criterion_group!(ntt, benchmark);
criterion_main!(ntt);
