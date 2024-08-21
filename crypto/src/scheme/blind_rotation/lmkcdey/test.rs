use crate::{
    core::{
        lwe::{
            test::{Lwe, LweParam},
            LweSecretKey,
        },
        rgsw::{
            test::{Rgsw, RgswParam},
            RgswDecompositionParam,
        },
        rlwe::{test::RlweParam, RlwePlaintextOwned, RlwePublicKey},
    },
    scheme::blind_rotation::lmkcdey::{
        self, LmkcdeyInteractiveParam, LmkcdeyKey, LmkcdeyKeyShare, LmkcdeyParam,
    },
    util::rng::StdLweRng,
};
use core::{array::from_fn, iter::repeat_with};
use itertools::{izip, Itertools};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::{Gaussian, Ternary},
    izip_eq,
    modulus::{Modulus, NonNativePowerOfTwo, Prime},
    poly::automorphism::AutomorphismMap,
    ring::{
        NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
        NonNativePowerOfTwoRing, PrimeRing, RingOps,
    },
    util::scratch::ScratchOwned,
};
use rand::{thread_rng, RngCore, SeedableRng};

#[derive(Clone, Copy, Debug)]
struct FhewBootstrappingParam {
    rgsw: RgswParam,
    lwe_ks: LweParam,
    q: usize,
    g: usize,
    w: usize,
    rgsw_by_rgsw_decomposition_param: RgswDecompositionParam,
}

impl FhewBootstrappingParam {
    fn build<R: RingOps>(&self) -> (Rgsw<R>, Lwe<R>, Lwe<NonNativePowerOfTwoRing>) {
        (
            self.rgsw.build(),
            self.rgsw.rlwe.to_lwe().build(),
            self.lwe_ks.build(),
        )
    }

    fn lmkcdey(&self) -> LmkcdeyParam {
        LmkcdeyParam {
            modulus: self.rgsw.rlwe.ciphertext_modulus,
            ring_size: self.rgsw.rlwe.ring_size,
            sk_distribution: self.rgsw.rlwe.sk_distribution,
            noise_distribution: self.rgsw.rlwe.noise_distribution,
            auto_decomposition_param: self.rgsw.rlwe.ks_decomposition_param,
            rlwe_by_rgsw_decomposition_param: self.rgsw.decomposition_param,
            lwe_modulus: self.lwe_ks.ciphertext_modulus,
            lwe_dimension: self.lwe_ks.dimension,
            lwe_sk_distribution: self.lwe_ks.sk_distribution,
            lwe_ks_decomposition_param: self.lwe_ks.ks_decomposition_param,
            q: self.q,
            g: self.g,
            w: self.w,
        }
    }

    fn lmkcdey_interactive(&self) -> LmkcdeyInteractiveParam {
        LmkcdeyInteractiveParam {
            param: self.lmkcdey(),
            rgsw_by_rgsw_decomposition_param: self.rgsw_by_rgsw_decomposition_param,
        }
    }
}

fn testing_param(big_q: impl Into<Modulus>, embedding_factor: usize) -> FhewBootstrappingParam {
    let message_modulus = 4;
    let ring_size = 1024;
    FhewBootstrappingParam {
        rgsw: RgswParam {
            rlwe: RlweParam {
                message_modulus,
                ciphertext_modulus: big_q.into(),
                ring_size,
                sk_distribution: Gaussian(3.2).into(),
                noise_distribution: Gaussian(3.2).into(),
                u_distribution: Ternary(ring_size / 2).into(),
                ks_decomposition_param: DecompositionParam {
                    log_base: 24,
                    level: 1,
                },
            },
            decomposition_param: RgswDecompositionParam {
                log_base: 17,
                level_a: 1,
                level_b: 1,
            },
        },
        lwe_ks: LweParam {
            message_modulus,
            ciphertext_modulus: NonNativePowerOfTwo::new(16).into(),
            dimension: 100,
            sk_distribution: Gaussian(3.2).into(),
            noise_distribution: Gaussian(3.2).into(),
            ks_decomposition_param: DecompositionParam {
                log_base: 1,
                level: 13,
            },
        },
        q: 2 * ring_size / embedding_factor,
        g: 5,
        w: 10,
        rgsw_by_rgsw_decomposition_param: RgswDecompositionParam {
            log_base: 6,
            level_a: 7,
            level_b: 6,
        },
    }
}

#[test]
fn bootstrap() {
    fn run<R: RingOps>(big_q: impl Into<Modulus>, embedding_factor: usize) {
        let mut rng = StdLweRng::from_entropy();
        let param = testing_param(big_q, embedding_factor);
        let (rgsw, lwe, lwe_ks) = param.build::<R>();
        let rlwe = rgsw.rlwe();
        let ring = rlwe.ring();
        let mod_ks = lwe_ks.modulus();

        let sk = rlwe.sk_gen();
        let sk_ks = lwe_ks.sk_gen();
        let bs_key = {
            let mut scratch = ring.allocate_scratch(0, 3, 0);
            let mut bs_key = LmkcdeyKey::allocate(param.lmkcdey());
            lmkcdey::bs_key_gen(
                ring,
                mod_ks,
                &mut bs_key,
                &sk,
                &sk_ks,
                scratch.borrow_mut(),
                &mut rng,
            );
            let mut bs_key_prep = LmkcdeyKey::allocate_eval(*bs_key.param(), ring.eval_size());
            lmkcdey::prepare_bs_key(ring, &mut bs_key_prep, &bs_key, scratch.borrow_mut());
            bs_key_prep
        };
        let sk = sk.into();

        let big_q_by_8 = ring.elem_from(ring.modulus().as_f64() / 8f64);
        let nand_lut = {
            let auto_map = AutomorphismMap::new(param.q / 2, -(param.g as i64));
            let lut = [0, 0, 0, 1];
            let lut_value = [big_q_by_8, ring.neg(&big_q_by_8)];
            let log_q_by_8 = (param.q / 8).ilog2() as usize;
            let f = |(sign, idx)| lut_value[sign as usize ^ lut[idx >> log_q_by_8]];
            RlwePlaintextOwned::new(auto_map.iter().map(f).collect(), param.q / 2)
        };

        let mut scratch = ScratchOwned::allocate(bs_key.param().scratch_bytes(ring, mod_ks));
        for m in 0..1 << 2 {
            let [a, b] = from_fn(|i| (m >> i) & 1 == 1);
            let ct_a = lwe.sk_encrypt(&sk, lwe.encode(a as _), &mut rng);
            let ct_b = lwe.sk_encrypt(&sk, lwe.encode(b as _), &mut rng);
            let mut ct = lwe.add(&ct_a, &ct_b);
            lmkcdey::bootstrap(
                ring,
                mod_ks,
                &mut ct,
                &bs_key,
                &nand_lut,
                scratch.borrow_mut(),
            );
            *ct.b_mut() = ring.add(ct.b(), &big_q_by_8);
            assert_eq!(!(a & b) as u64, lwe.decode(lwe.decrypt(&sk, &ct)));
        }
    }

    for embedding_factor in [1, 2] {
        run::<NoisyNativeRing>(Modulus::native(), embedding_factor);
        run::<NoisyNonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(50), embedding_factor);
        run::<NativeRing>(Modulus::native(), embedding_factor);
        run::<NonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(50), embedding_factor);
        run::<NoisyPrimeRing>(Prime::gen(50, 12), embedding_factor);
        run::<PrimeRing>(Prime::gen(50, 12), embedding_factor);
    }
}

#[test]
fn interactive() {
    fn run<R: RingOps>(big_q: impl Into<Modulus>) {
        let param = testing_param(big_q, 1);
        let (rgsw, lwe, lwe_ks) = param.build::<R>();
        let rlwe = rgsw.rlwe();
        let ring = rlwe.ring();
        let mod_ks = lwe_ks.modulus();
        let seed = from_fn(|_| thread_rng().next_u64() as u8);
        let total_shares = 3;

        let mut rngs = repeat_with(|| StdLweRng::from_seed(seed))
            .take(total_shares + 1)
            .collect_vec();
        let sk_shares = repeat_with(|| rlwe.sk_gen())
            .take(total_shares)
            .collect_vec();
        let pk_shares = izip!(&sk_shares, &mut rngs)
            .map(|(sk, rng)| rlwe.seeded_pk_gen(sk, rng))
            .collect_vec();
        let pk = {
            let mut pk = RlwePublicKey::allocate(ring.ring_size());
            lmkcdey::aggregate_pk_shares(ring, &mut pk, &pk_shares, rngs.last_mut().unwrap());
            pk
        };
        let bs_key_shares = izip!(0.., &sk_shares, &mut rngs)
            .map(|(share_idx, sk, rng)| {
                let sk_ks = lwe_ks.sk_gen();
                let mut bs_key_share =
                    LmkcdeyKeyShare::allocate(param.lmkcdey_interactive(), share_idx, total_shares);
                let mut scratch = ring.allocate_scratch(2, 3, 0);
                lmkcdey::bs_key_share_gen(
                    ring,
                    mod_ks,
                    &mut bs_key_share,
                    sk,
                    &pk,
                    &sk_ks,
                    scratch.borrow_mut(),
                    rng,
                );
                bs_key_share
            })
            .collect_vec();
        let bs_key = {
            let mut bs_key = LmkcdeyKey::allocate(param.lmkcdey());
            let mut scratch = ring.allocate_scratch(
                2,
                3,
                2 * (param.rgsw_by_rgsw_decomposition_param.level_a
                    + param.rgsw_by_rgsw_decomposition_param.level_b),
            );
            lmkcdey::aggregate_bs_key_shares(
                ring,
                mod_ks,
                &mut bs_key,
                &bs_key_shares,
                scratch.borrow_mut(),
                rngs.last_mut().unwrap(),
            );
            let mut bs_key_prep = LmkcdeyKey::allocate_eval(*bs_key.param(), ring.eval_size());
            lmkcdey::prepare_bs_key(ring, &mut bs_key_prep, &bs_key, scratch.borrow_mut());
            bs_key_prep
        };
        let sk = sk_shares
            .into_iter()
            .map(LweSecretKey::from)
            .reduce(|mut sk, sk_share| {
                izip_eq!(sk.as_mut(), sk_share.as_ref()).for_each(|(a, b)| *a += b);
                sk
            })
            .unwrap();

        let big_q_by_8 = ring.elem_from(ring.modulus().as_f64() / 8f64);
        let nand_lut = {
            let auto_map = AutomorphismMap::new(param.q / 2, -(param.g as i64));
            let lut = [0, 0, 0, 1];
            let lut_value = [big_q_by_8, ring.neg(&big_q_by_8)];
            let log_q_by_8 = (param.q / 8).ilog2() as usize;
            let f = |(sign, idx)| lut_value[sign as usize ^ lut[idx >> log_q_by_8]];
            RlwePlaintextOwned::new(auto_map.iter().map(f).collect(), param.q / 2)
        };

        let mut scratch = ScratchOwned::allocate(bs_key.param().scratch_bytes(ring, mod_ks));
        let mut rng = StdLweRng::from_entropy();
        for m in 0..1 << 2 {
            let [a, b] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct_a, ct_b] = [a, b].map(|m| {
                let pt = rlwe.encode(&vec![m as u64; ring.ring_size()]);
                rlwe.sample_extract(&rlwe.pk_encrypt(&pk, &pt, &mut rng), 0)
            });
            let mut ct = lwe.add(&ct_a, &ct_b);
            lmkcdey::bootstrap(
                ring,
                mod_ks,
                &mut ct,
                &bs_key,
                &nand_lut,
                scratch.borrow_mut(),
            );
            *ct.b_mut() = ring.add(ct.b(), &big_q_by_8);
            assert_eq!(!(a & b) as u64, lwe.decode(lwe.decrypt(&sk, &ct)));
        }
    }

    // run::<NoisyNativeRing>(Modulus::native());
    // run::<NoisyNonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(54));
    run::<NativeRing>(Modulus::native());
    run::<NonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(54));
    // run::<NoisyPrimeRing>(Prime::gen(54, 12));
    run::<PrimeRing>(Prime::gen(54, 12));
}
