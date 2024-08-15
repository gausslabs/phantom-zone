use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phantom_zone_crypto::core::{
    rgsw::{self, RgswCiphertext, RgswDecompositionParam},
    rlwe::{self, RlweAutoKey, RlweCiphertext},
};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    modulus::{Modulus, NonNativePowerOfTwo, Prime},
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
};

fn automorphism(c: &mut Criterion) {
    fn runner<R: RingOps + 'static>(
        modulus: Modulus,
        ring_size: usize,
        decomposition_param: DecompositionParam,
    ) -> Box<dyn FnMut()> {
        let ring = R::new(modulus, ring_size);
        let auto_key_prep =
            RlweAutoKey::allocate_eval(ring.ring_size(), ring.eval_size(), decomposition_param, 5);
        let mut ct = RlweCiphertext::allocate(ring.ring_size());
        let mut scratch = ring.allocate_scratch(2, 3, 0);
        Box::new(move || {
            let mut scratch = scratch.borrow_mut();
            rlwe::automorphism_prep_in_place(&ring, &mut ct, &auto_key_prep, scratch.reborrow());
        })
    }

    let mut b = c.benchmark_group("automorphism");
    let decomposition_param = DecompositionParam {
        log_base: 24,
        level: 1,
    };
    for log_ring_size in 11..13 {
        let ring_size = 1 << log_ring_size;
        let runners = [
            ("noisy_native", {
                let modulus = Modulus::native();
                runner::<NoisyNativeRing>(modulus, ring_size, decomposition_param)
            }),
            ("noisy_non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(54).into();
                runner::<NoisyNonNativePowerOfTwoRing>(modulus, ring_size, decomposition_param)
            }),
            ("native", {
                let modulus = Modulus::native();
                runner::<NativeRing>(modulus, ring_size, decomposition_param)
            }),
            ("non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(54).into();
                runner::<NonNativePowerOfTwoRing>(modulus, ring_size, decomposition_param)
            }),
            ("noisy_prime", {
                let modulus = Prime::gen(54, log_ring_size + 1).into();
                runner::<NoisyPrimeRing>(modulus, ring_size, decomposition_param)
            }),
            ("prime", {
                let modulus = Prime::gen(54, log_ring_size + 1).into();
                runner::<PrimeRing>(modulus, ring_size, decomposition_param)
            }),
        ];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, ring_size);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn rlwe_by_rgsw(c: &mut Criterion) {
    fn runner<R: RingOps + 'static>(
        modulus: Modulus,
        ring_size: usize,
        decomposition_param: RgswDecompositionParam,
    ) -> Box<dyn FnMut()> {
        let ring = R::new(modulus, ring_size);
        let ct_rgsw_prep =
            RgswCiphertext::allocate_eval(ring.ring_size(), ring.eval_size(), decomposition_param);
        let mut ct_rlwe = RlweCiphertext::allocate(ring.ring_size());
        let mut scratch = ring.allocate_scratch(2, 3, 0);
        Box::new(move || {
            let mut scratch = scratch.borrow_mut();
            rgsw::rlwe_by_rgsw_prep_in_place(
                &ring,
                &mut ct_rlwe,
                &ct_rgsw_prep,
                scratch.reborrow(),
            );
        })
    }

    let mut b = c.benchmark_group("rlwe_by_rgsw");
    let decomposition_param = RgswDecompositionParam {
        log_base: 17,
        level_a: 1,
        level_b: 1,
    };
    for log_ring_size in 11..13 {
        let ring_size = 1 << log_ring_size;
        let runners = [
            ("noisy_native", {
                let modulus = Modulus::native();
                runner::<NoisyNativeRing>(modulus, ring_size, decomposition_param)
            }),
            ("noisy_non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(54).into();
                runner::<NoisyNonNativePowerOfTwoRing>(modulus, ring_size, decomposition_param)
            }),
            ("native", {
                let modulus = Modulus::native();
                runner::<NativeRing>(modulus, ring_size, decomposition_param)
            }),
            ("non_native_power_of_two", {
                let modulus = NonNativePowerOfTwo::new(54).into();
                runner::<NonNativePowerOfTwoRing>(modulus, ring_size, decomposition_param)
            }),
            ("noisy_prime", {
                let modulus = Prime::gen(54, log_ring_size + 1).into();
                runner::<NoisyPrimeRing>(modulus, ring_size, decomposition_param)
            }),
            ("prime", {
                let modulus = Prime::gen(54, log_ring_size + 1).into();
                runner::<PrimeRing>(modulus, ring_size, decomposition_param)
            }),
        ];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, ring_size);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

criterion_group!(benches, automorphism, rlwe_by_rgsw);
criterion_main!(benches);
