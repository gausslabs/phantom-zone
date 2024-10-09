use crate::{
    core::{
        lwe::test::LweParam,
        rlwe::{test::RlweParam, RlweCiphertext},
    },
    scheme::ring_packing::cdks::{
        self, aggregate_rp_key_shares, prepare_rp_key, rp_key_gen, rp_key_share_gen, CdksCrs,
        CdksKey, CdksKeyShare, CdksParam,
    },
    util::rng::StdLweRng,
};
use core::iter::repeat_with;
use itertools::{izip, Itertools};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::{Gaussian, Sampler, Ternary},
    izip_eq,
    modulus::{Modulus, ModulusOps, Native, Prime},
    ring::{PrimeRing, RingOps},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn test_param(modulus: impl Into<Modulus>) -> CdksParam {
    let ring_size = 512;
    CdksParam {
        modulus: modulus.into(),
        ring_size,
        sk_distribution: Ternary.into(),
        noise_distribution: Gaussian(3.19).into(),
        auto_decomposition_param: DecompositionParam {
            log_base: 17,
            level: 1,
        },
    }
}

impl From<CdksParam> for RlweParam {
    fn from(param: CdksParam) -> Self {
        RlweParam {
            message_modulus: 4,
            ciphertext_modulus: param.modulus,
            ring_size: param.ring_size,
            sk_distribution: param.sk_distribution,
            noise_distribution: param.noise_distribution,
            u_distribution: Ternary.into(),
            ks_decomposition_param: param.auto_decomposition_param,
        }
    }
}

impl From<CdksParam> for LweParam {
    fn from(param: CdksParam) -> Self {
        RlweParam::from(param).to_lwe()
    }
}

#[test]
fn pack_lwes() {
    fn run<R: RingOps>(modulus: impl Into<Modulus>, total_shares: usize) {
        let mut rng = StdLweRng::from_entropy();
        let param = test_param(modulus);
        let lwe = LweParam::from(param).build::<R>();
        let rlwe = RlweParam::from(param).build::<R>();
        let ring = rlwe.ring();
        let ring_size = ring.ring_size();

        let (sk, rp_key) = if total_shares == 1 {
            let sk = rlwe.sk_gen(&mut rng);
            let rp_key = {
                let mut rp_key = CdksKey::allocate(param);
                rp_key_gen(ring, &mut rp_key, &sk, &mut rng);
                let mut rp_key_prep = CdksKey::allocate_eval(param, ring.eval_size());
                prepare_rp_key(ring, &mut rp_key_prep, &rp_key);
                rp_key_prep
            };
            (sk, rp_key)
        } else {
            let crs = CdksCrs::<StdRng>::new(rng.gen());
            let sk_shares = repeat_with(|| rlwe.sk_gen(&mut rng)).take(4).collect_vec();
            let sk = sk_shares
                .iter()
                .cloned()
                .reduce(|mut sk, sk_share| {
                    izip_eq!(sk.as_mut(), sk_share.as_ref()).for_each(|(a, b)| *a += b);
                    sk
                })
                .unwrap();
            let rp_key = {
                let rp_key_shares = sk_shares
                    .iter()
                    .map(|sk| {
                        let mut rp_key_share = CdksKeyShare::allocate(param);
                        rp_key_share_gen(ring, &mut rp_key_share, &crs, sk, &mut rng);
                        rp_key_share
                    })
                    .collect_vec();
                let mut rp_key = CdksKey::allocate(param);
                aggregate_rp_key_shares(ring, &mut rp_key, &crs, &rp_key_shares);
                let mut rp_key_prep = CdksKey::allocate_eval(param, ring.eval_size());
                prepare_rp_key(ring, &mut rp_key_prep, &rp_key);
                rp_key_prep
            };
            (sk, rp_key)
        };

        let ms = rlwe.message_ring().sample_uniform_vec(ring_size, &mut rng);
        let encrypt = |m: &_| lwe.sk_encrypt(&sk.clone().into(), lwe.encode(*m), &mut rng);
        let decrypt = |ct: &_| rlwe.decode(&rlwe.decrypt(&sk, ct));
        let cts = ms.iter().map(encrypt).collect_vec();

        let mut ct = RlweCiphertext::allocate(ring_size);
        for k in 0..=ring_size {
            let ell = k.next_power_of_two().ilog2();
            cdks::pack_lwes(ring, &mut ct, &rp_key, &cts[..k]);
            izip!(&ms[..k], decrypt(&ct).into_iter().step_by(ring_size >> ell))
                .for_each(|(a, b)| assert!(*a == b));
        }
    }

    run::<PrimeRing>(Prime::gen(54, 11), 1);
    run::<PrimeRing>(Prime::gen(54, 11), 4);
}

#[test]
fn pack_lwes_ms() {
    fn run<M: ModulusOps, R: RingOps>(
        lwe_modulus: impl Into<Modulus>,
        modulus: impl Into<Modulus>,
    ) {
        let mut rng = StdLweRng::from_entropy();
        let lwe_param = test_param(lwe_modulus);
        let lwe = LweParam::from(lwe_param).build::<M>();
        let param = test_param(modulus);
        let rlwe = RlweParam::from(param).build::<R>();
        let ring = rlwe.ring();
        let ring_size = ring.ring_size();

        let sk = rlwe.sk_gen(&mut rng);
        let rp_key = {
            let mut rp_key = CdksKey::allocate(param);
            rp_key_gen(ring, &mut rp_key, &sk, &mut rng);
            let mut rp_key_prep = CdksKey::allocate_eval(param, ring.eval_size());
            prepare_rp_key(ring, &mut rp_key_prep, &rp_key);
            rp_key_prep
        };

        let ms = rlwe.message_ring().sample_uniform_vec(ring_size, &mut rng);
        let encrypt = |m: &_| lwe.sk_encrypt(&sk.clone().into(), lwe.encode(*m), &mut rng);
        let decrypt = |ct: &_| rlwe.decode(&rlwe.decrypt(&sk, ct));
        let cts = ms.iter().map(encrypt).collect_vec();

        let mut ct = RlweCiphertext::allocate(ring_size);
        for k in 0..=ring_size {
            let ell = k.next_power_of_two().ilog2();
            cdks::pack_lwes_ms(lwe.modulus(), ring, &mut ct, &rp_key, &cts[..k]);
            izip!(&ms[..k], decrypt(&ct).into_iter().step_by(ring_size >> ell))
                .for_each(|(a, b)| assert!(*a == b));
        }
    }

    run::<Native, PrimeRing>(Native::native(), Prime::gen(61, 11));
}
