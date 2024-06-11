use bin_rs::{
    ArithmeticLazyOps, ArithmeticOps, Decomposer, DefaultDecomposer, ModInit, ModularOpsU64,
    ShoupMatrixFMA, VectorOps,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::{izip, Itertools};
use rand::{thread_rng, Rng};
use rand_distr::Uniform;

fn decompose_r(r: &[u64], decomp_r: &mut [Vec<u64>], decomposer: &DefaultDecomposer<u64>) {
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

fn matrix_fma(out: &mut [u64], a: &Vec<Vec<u64>>, b: &Vec<Vec<u64>>, modop: &ModularOpsU64<u64>) {
    izip!(a.iter(), b.iter()).for_each(|(a_r, b_r)| {
        izip!(out.iter_mut(), a_r.iter(), b_r.iter())
            .for_each(|(o, ai, bi)| *o = modop.add_lazy(o, &modop.mul_lazy(ai, bi)));
    });
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
        for ring_size in [1 << 11] {
            let modop = ModularOpsU64::new(prime);

            let mut rng = thread_rng();
            let dist = Uniform::new(0, prime);

            let a0 = (&mut rng).sample_iter(dist).take(ring_size).collect_vec();
            let a1 = (&mut rng).sample_iter(dist).take(ring_size).collect_vec();
            let a2 = (&mut rng).sample_iter(dist).take(ring_size).collect_vec();

            let d = 2;
            let a0_matrix = (0..d)
                .into_iter()
                .map(|_| (&mut rng).sample_iter(dist).take(ring_size).collect_vec())
                .collect_vec();
            // a0 in shoup representation
            let a0_shoup_matrix = a0_matrix
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|v| {
                            // $(v * 2^{\beta}) / p$
                            ((*v as u128 * (1u128 << 64)) / prime as u128) as u64
                        })
                        .collect_vec()
                })
                .collect_vec();
            let a1_matrix = (0..d)
                .into_iter()
                .map(|_| (&mut rng).sample_iter(dist).take(ring_size).collect_vec())
                .collect_vec();

            group.bench_function(
                BenchmarkId::new("matrix_fma_lazy", format!("q={prime}/N={ring_size}/d={d}")),
                |b| {
                    b.iter_batched_ref(
                        || (vec![0u64; ring_size]),
                        |(out)| black_box(matrix_fma(out, &a0_matrix, &a1_matrix, &modop)),
                        criterion::BatchSize::PerIteration,
                    )
                },
            );

            group.bench_function(
                BenchmarkId::new(
                    "matrix_shoup_fma_lazy",
                    format!("q={prime}/N={ring_size}/d={d}"),
                ),
                |b| {
                    b.iter_batched_ref(
                        || (vec![0u64; ring_size]),
                        |(out)| {
                            black_box(modop.shoup_matrix_fma(
                                out,
                                &a0_matrix,
                                &a0_shoup_matrix,
                                &a1_matrix,
                            ))
                        },
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
