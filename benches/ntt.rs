use bin_rs::{Ntt, NttBackendU64, NttInit};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::{izip, Itertools};
use rand::{thread_rng, Rng};
use rand_distr::Uniform;

pub(crate) fn forward_matrix(matrix_a: &mut [Vec<u64>], ntt_op: &NttBackendU64) {
    matrix_a.iter_mut().for_each(|r| {
        ntt_op.forward(r.as_mut_slice());
    });
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt");
    // 55
    for prime in [36028797017456641] {
        for ring_size in [1 << 11] {
            let ntt = NttBackendU64::new(&prime, ring_size);
            let mut rng = thread_rng();

            let distr = Uniform::new(0, prime);
            let a = (&mut rng).sample_iter(distr).take(ring_size).collect_vec();
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

            let d = 2;
            let a_mat = (0..d)
                .into_iter()
                .map(|_| (&mut rng).sample_iter(distr).take(ring_size).collect_vec())
                .collect_vec();
            group.bench_function(
                BenchmarkId::new("forward_matrix", format!("q={prime}/{ring_size}")),
                |b| {
                    b.iter_batched_ref(
                        || a_mat.clone(),
                        |a_mat| black_box(forward_matrix(a_mat, &ntt)),
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
