use std::cell::RefCell;

use itertools::izip;
use rand::{distributions::Uniform, thread_rng, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::Distribution;

use crate::utils::WithLocal;

thread_local! {
    pub(crate) static DEFAULT_RNG: RefCell<DefaultSecureRng> = RefCell::new(DefaultSecureRng::new());
}

pub(crate) trait NewWithSeed {
    type Seed;
    fn new_with_seed(seed: Self::Seed) -> Self;
}

pub trait RandomGaussianDist<M>
where
    M: ?Sized,
{
    type Parameters: ?Sized;
    fn random_fill(&mut self, parameters: &Self::Parameters, container: &mut M);
}

pub trait RandomUniformDist<M>
where
    M: ?Sized,
{
    type Parameters: ?Sized;
    fn random_fill(&mut self, parameters: &Self::Parameters, container: &mut M);
}

pub(crate) struct DefaultSecureRng {
    rng: ChaCha8Rng,
}

impl DefaultSecureRng {
    pub fn new_seeded(seed: <ChaCha8Rng as SeedableRng>::Seed) -> DefaultSecureRng {
        let rng = ChaCha8Rng::from_seed(seed);
        DefaultSecureRng { rng }
    }

    pub fn new() -> DefaultSecureRng {
        let rng = ChaCha8Rng::from_entropy();
        DefaultSecureRng { rng }
    }

    pub fn fill_bytes(&mut self, a: &mut [u8; 32]) {
        self.rng.fill_bytes(a);
    }
}

impl NewWithSeed for DefaultSecureRng {
    type Seed = <ChaCha8Rng as SeedableRng>::Seed;
    fn new_with_seed(seed: Self::Seed) -> Self {
        DefaultSecureRng::new_seeded(seed)
    }
}

impl RandomUniformDist<usize> for DefaultSecureRng {
    type Parameters = usize;
    fn random_fill(&mut self, parameters: &Self::Parameters, container: &mut usize) {
        *container = self.rng.gen_range(0..*parameters);
    }
}

impl RandomUniformDist<[u8]> for DefaultSecureRng {
    type Parameters = u8;
    fn random_fill(&mut self, parameters: &Self::Parameters, container: &mut [u8]) {
        self.rng.fill_bytes(container);
    }
}

impl RandomUniformDist<[u32]> for DefaultSecureRng {
    type Parameters = u32;
    fn random_fill(&mut self, parameters: &Self::Parameters, container: &mut [u32]) {
        izip!(
            (&mut self.rng).sample_iter(Uniform::new(0, parameters)),
            container.iter_mut()
        )
        .for_each(|(from, to)| {
            *to = from;
        });
    }
}

impl RandomUniformDist<[u64]> for DefaultSecureRng {
    type Parameters = u64;
    fn random_fill(&mut self, parameters: &Self::Parameters, container: &mut [u64]) {
        izip!(
            (&mut self.rng).sample_iter(Uniform::new(0, parameters)),
            container.iter_mut()
        )
        .for_each(|(from, to)| {
            *to = from;
        });
    }
}

impl RandomGaussianDist<u64> for DefaultSecureRng {
    type Parameters = u64;
    fn random_fill(&mut self, parameters: &Self::Parameters, container: &mut u64) {
        // let o = rand_distr::Normal::new(0.0, 3.2f64)
        //     .unwrap()
        //     .sample(&mut self.rng)
        //     .round();

        // // let o = 0.0f64;

        // let is_neg = o.is_sign_negative() && o != 0.0;
        // if is_neg {
        //     *container = parameters - (o.abs() as u64);
        // } else {
        //     *container = o as u64;
        // }
    }
}

impl RandomGaussianDist<u32> for DefaultSecureRng {
    type Parameters = u32;
    fn random_fill(&mut self, parameters: &Self::Parameters, container: &mut u32) {
        let o = rand_distr::Normal::new(0.0, 3.2f32)
            .unwrap()
            .sample(&mut self.rng)
            .round();

        // let o = 0.0f32;
        let is_neg = o.is_sign_negative() && o != 0.0;

        if is_neg {
            *container = parameters - (o.abs() as u32);
        } else {
            *container = o as u32;
        }
    }
}

impl RandomGaussianDist<[u64]> for DefaultSecureRng {
    type Parameters = u64;
    fn random_fill(&mut self, parameters: &Self::Parameters, container: &mut [u64]) {
        // izip!(
        //     rand_distr::Normal::new(0.0, 3.2f64)
        //         .unwrap()
        //         .sample_iter(&mut self.rng),
        //     container.iter_mut()
        // )
        // .for_each(|(oi, v)| {
        //     let oi = oi.round();
        //     let is_neg = oi.is_sign_negative() && oi != 0.0;
        //     if is_neg {
        //         *v = parameters - (oi.abs() as u64);
        //     } else {
        //         *v = oi as u64;
        //     }
        // });
    }
}

impl RandomGaussianDist<[u32]> for DefaultSecureRng {
    type Parameters = u32;
    fn random_fill(&mut self, parameters: &Self::Parameters, container: &mut [u32]) {
        izip!(
            rand_distr::Normal::new(0.0, 3.2f32)
                .unwrap()
                .sample_iter(&mut self.rng),
            container.iter_mut()
        )
        .for_each(|(oi, v)| {
            let oi = oi.round();
            let is_neg = oi.is_sign_negative() && oi != 0.0;
            if is_neg {
                *v = parameters - (oi.abs() as u32);
            } else {
                *v = oi as u32;
            }
        });
    }
}

impl WithLocal for DefaultSecureRng {
    fn with_local<F, R>(func: F) -> R
    where
        F: Fn(&Self) -> R,
    {
        DEFAULT_RNG.with_borrow(|r| func(r))
    }

    fn with_local_mut<F, R>(func: F) -> R
    where
        F: Fn(&mut Self) -> R,
    {
        DEFAULT_RNG.with_borrow_mut(|r| func(r))
    }
}
