use bin_rs::{Decomposer, DefaultDecomposer, ModInit, ModularOpsU64, VectorOps};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::{izip, Itertools};
use rand::{thread_rng, Rng};
use rand_distr::Uniform;

pub(crate) fn decompose_r(
    r: &[u64],
    decomp_r: &mut [Vec<u64>],
    decomposer: &DefaultDecomposer<u64>,
) {
    let ring_size = r.len();
    // let d = decomposer.decomposition_count();
    // let mut count = 0;
    for ri in 0..ring_size {
        // let el_decomposed = decomposer.decompose(&r[ri]);
        decomposer
            .decompose_iter(&r[ri])
            .enumerate()
            .into_iter()
            .for_each(|(j, el)| {
                decomp_r[j][ri] = el;
            });
    }
}

fn benchmark_decomposer(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomposer");

    // let decomposers = vec![];
    // 55
    for prime in [36028797017456641] {
        for ring_size in [1 << 11] {
            let logb = 11;
            let decomposer = DefaultDecomposer::new(prime, logb, 2);

            let mut rng = thread_rng();
            let dist = Uniform::new(0, prime);
            let a = (&mut rng).sample_iter(dist).take(ring_size).collect_vec();

            group.bench_function(
                BenchmarkId::new(
                    "decompose",
                    format!(
                        "q={prime}/N={ring_size}/logB={logb}/d={}",
                        decomposer.decomposition_count()
                    ),
                ),
                |b| {
                    b.iter_batched_ref(
                        || {
                            (
                                a.clone(),
                                vec![vec![0u64; ring_size]; decomposer.decomposition_count()],
                            )
                        },
                        |(r, decomp_r)| (decompose_r(r, decomp_r, &decomposer)),
                        criterion::BatchSize::PerIteration,
                    )
                },
            );
        }
    }

    group.finish();
}

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

criterion_group!(decomposer, benchmark_decomposer);
criterion_group!(modulus, benchmark);
criterion_main!(modulus, decomposer);
