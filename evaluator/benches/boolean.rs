use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use phantom_zone_crypto::{
    core::{lwe::LweSecretKey, rgsw::RgswDecompositionParam},
    util::rng::StdLweRng,
};
use phantom_zone_evaluator::boolean::evaluator::{
    fhew::{self, FhewBoolCiphertext, FhewBoolParam},
    BoolEvaluator,
};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::Gaussian,
    modulus::{Modulus, NonNativePowerOfTwo, Prime},
    ring::{
        NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing, NonNativePowerOfTwoRing,
        PrimeRing, RingOps,
    },
};
use rand::{Rng, SeedableRng};

fn fhew(c: &mut Criterion) {
    type FhewBoolEvaluator<R> = fhew::FhewBoolEvaluator<R, NonNativePowerOfTwoRing>;

    fn runner<R: RingOps + 'static>(param: FhewBoolParam) -> Box<dyn FnMut()> {
        let mut rng = StdLweRng::from_entropy();
        let ring = R::new(param.modulus, param.ring_size);
        let sk = LweSecretKey::sample(param.ring_size, Gaussian(3.2).into(), &mut rng);
        let evaluator = FhewBoolEvaluator::<R>::sample(param, &sk, &mut rng);
        let cts = (0..2000)
            .map(|_| {
                let m = rng.gen_bool(0.5);
                FhewBoolCiphertext::encrypt(&ring, &sk, m, Gaussian(3.2).into(), &mut rng)
            })
            .collect_vec();
        let mut cts = cts.into_iter();
        Box::new(move || evaluator.bitnand_assign(&mut cts.next().unwrap(), &cts.next().unwrap()))
    }

    fn test_param(big_q: impl Into<Modulus>, ring_size: usize, q: usize) -> FhewBoolParam {
        FhewBoolParam {
            modulus: big_q.into(),
            ring_size,
            auto_decomposition_param: DecompositionParam {
                log_base: 24,
                level: 1,
            },
            rgsw_decomposition_param: RgswDecompositionParam {
                log_base: 17,
                level_a: 1,
                level_b: 1,
            },
            lwe_modulus: NonNativePowerOfTwo::new(16).into(),
            lwe_dimension: 600,
            lwe_ks_decomposition_param: DecompositionParam {
                log_base: 1,
                level: 13,
            },
            q,
            g: 5,
            w: 10,
        }
    }

    let mut b = c.benchmark_group("fhew");
    for (log_ring_size, embedding_factor) in (11..12).cartesian_product([1, 2]) {
        let ring_size = 1 << log_ring_size;
        let q = 2 * ring_size / embedding_factor;
        let runners = [
            ("noisy_native", {
                let modulus = Modulus::native();
                runner::<NoisyNativeRing>(test_param(modulus, ring_size, q))
            }),
            ("noisy_non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(54);
                runner::<NoisyNonNativePowerOfTwoRing>(test_param(modulus, ring_size, q))
            }),
            ("noisy_prime", {
                let modulus = Prime::gen(54, log_ring_size + 1);
                runner::<NoisyPrimeRing>(test_param(modulus, ring_size, q))
            }),
            ("prime", {
                let modulus = Prime::gen(54, log_ring_size + 1);
                runner::<PrimeRing>(test_param(modulus, ring_size, q))
            }),
        ];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, format!("N={ring_size}/q={q}"));
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

criterion_group!(benches, fhew);
criterion_main!(benches);
