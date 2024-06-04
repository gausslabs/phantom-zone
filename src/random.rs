use std::cell::RefCell;

use itertools::izip;
use num_traits::{PrimInt, Zero};
use rand::{distributions::Uniform, thread_rng, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{uniform::SampleUniform, Distribution};

use crate::{backend::Modulus, utils::WithLocal};

thread_local! {
    pub(crate) static DEFAULT_RNG: RefCell<DefaultSecureRng> = RefCell::new(DefaultSecureRng::new_seeded([0u8;32]));
}

pub(crate) trait NewWithSeed {
    type Seed;
    fn new_with_seed(seed: Self::Seed) -> Self;
}

pub trait RandomElement<T> {
    /// Sample Random element of type T
    fn random(&mut self) -> T;
}

pub trait RandomElementInModulus<T, M> {
    /// Sample Random element of type T in range [0, modulus)
    fn random(&mut self, modulus: &M) -> T;
}

pub trait RandomGaussianElementInModulus<T, M> {
    /// Sample Random gaussian element from \mu = 0.0 and \sigma = 3.19. Sampled
    /// element is converted to signed representation in modulus.
    fn random(&mut self, modulus: &M) -> T;
}

pub trait RandomFill<M>
where
    M: ?Sized,
{
    /// Fill container with random elements of type of its elements
    fn random_fill(&mut self, container: &mut M);
}

pub trait RandomFillUniformInModulus<M, P>
where
    M: ?Sized,
{
    /// Fill container with random elements in range [0, modulus)
    fn random_fill(&mut self, modulus: &P, container: &mut M);
}

pub trait RandomFillGaussianInModulus<M, P>
where
    M: ?Sized,
{
    /// Fill container with gaussian elements sampled from normal distribution
    /// with \mu = 0.0 and \sigma = 3.19. Elements are converted to signed
    /// represented in the modulus.
    fn random_fill(&mut self, modulus: &P, container: &mut M);
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

impl<T, C> RandomFillUniformInModulus<[T], C> for DefaultSecureRng
where
    T: PrimInt + SampleUniform,
    C: Modulus<Element = T>,
{
    fn random_fill(&mut self, modulus: &C, container: &mut [T]) {
        izip!(
            (&mut self.rng).sample_iter(Uniform::new_inclusive(
                T::zero(),
                modulus.largest_unsigned_value()
            )),
            container.iter_mut()
        )
        .for_each(|(from, to)| {
            *to = from;
        });
    }
}

impl<T, C> RandomFillGaussianInModulus<[T], C> for DefaultSecureRng
where
    T: PrimInt + SampleUniform,
    C: Modulus<Element = T>,
{
    fn random_fill(&mut self, modulus: &C, container: &mut [T]) {
        // izip!(
        //     rand_distr::Normal::new(0.0, 3.19f64)
        //         .unwrap()
        //         .sample_iter(&mut self.rng),
        //     container.iter_mut()
        // )
        // .for_each(|(from, to)| {
        //     *to = modulus.map_element_from_f64(from);
        // });
    }
}

impl<T> RandomFill<[T]> for DefaultSecureRng
where
    T: PrimInt + SampleUniform,
{
    fn random_fill(&mut self, container: &mut [T]) {
        izip!(
            (&mut self.rng).sample_iter(Uniform::new_inclusive(T::zero(), T::max_value())),
            container.iter_mut()
        )
        .for_each(|(from, to)| {
            *to = from;
        });
    }
}

impl<T> RandomFill<[T; 32]> for DefaultSecureRng
where
    T: PrimInt + SampleUniform,
{
    fn random_fill(&mut self, container: &mut [T; 32]) {
        izip!(
            (&mut self.rng).sample_iter(Uniform::new_inclusive(T::zero(), T::max_value())),
            container.iter_mut()
        )
        .for_each(|(from, to)| {
            *to = from;
        });
    }
}

impl<T> RandomElement<T> for DefaultSecureRng
where
    T: PrimInt + SampleUniform,
{
    fn random(&mut self) -> T {
        Uniform::new_inclusive(T::zero(), T::max_value()).sample(&mut self.rng)
    }
}

impl<T> RandomElementInModulus<T, T> for DefaultSecureRng
where
    T: Zero + SampleUniform,
{
    fn random(&mut self, modulus: &T) -> T {
        Uniform::new(T::zero(), modulus).sample(&mut self.rng)
    }
}

impl<T, M: Modulus<Element = T>> RandomGaussianElementInModulus<T, M> for DefaultSecureRng {
    fn random(&mut self, modulus: &M) -> T {
        // modulus.map_element_from_f64(
        //     rand_distr::Normal::new(0.0, 3.19f64)
        //         .unwrap()
        //         .sample(&mut self.rng),
        // )
        modulus.map_element_from_f64(0.0)
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

    fn with_local_mut_mut<F, R>(func: &mut F) -> R
    where
        F: FnMut(&mut Self) -> R,
    {
        DEFAULT_RNG.with_borrow_mut(|r| func(r))
    }
}
