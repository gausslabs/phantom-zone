use core::fmt::Debug;
use num_traits::FromPrimitive;
use phantom_zone_math::distribution::{DistributionSized, Gaussian, Ternary};
use rand::{distributions::Distribution, Rng};

#[derive(Clone, Copy, Debug)]
pub enum SecretKeyDistribution {
    Gaussian(Gaussian),
    Ternary(Ternary),
}

impl From<Gaussian> for SecretKeyDistribution {
    fn from(inner: Gaussian) -> Self {
        Self::Gaussian(inner)
    }
}

impl From<Ternary> for SecretKeyDistribution {
    fn from(inner: Ternary) -> Self {
        Self::Ternary(inner)
    }
}

impl<T> DistributionSized<T> for SecretKeyDistribution
where
    Gaussian: DistributionSized<T>,
    Ternary: DistributionSized<T>,
{
    fn sample_map_into<R: Rng, O>(self, out: &mut [O], f: impl Fn(T) -> O, rng: R) {
        match self {
            Self::Gaussian(inner) => inner.sample_map_into(out, f, rng),
            Self::Ternary(inner) => inner.sample_map_into(out, f, rng),
        }
    }

    fn sample_vec<R: Rng>(self, n: usize, rng: R) -> Vec<T> {
        match self {
            Self::Gaussian(inner) => inner.sample_vec(n, rng),
            Self::Ternary(inner) => inner.sample_vec(n, rng),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NoiseDistribution {
    Gaussian(Gaussian),
    Ternary(Ternary),
}

impl From<Gaussian> for NoiseDistribution {
    fn from(inner: Gaussian) -> Self {
        Self::Gaussian(inner)
    }
}

impl From<Ternary> for NoiseDistribution {
    fn from(inner: Ternary) -> Self {
        Self::Ternary(inner)
    }
}

impl<T: FromPrimitive> Distribution<T> for NoiseDistribution
where
    Gaussian: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        match self {
            Self::Gaussian(inner) => inner.sample(rng),
            Self::Ternary(_) => T::from_i64([-1, 0, 0, 1][rng.next_u64() as usize & 2]).unwrap(),
        }
    }
}

impl<T> DistributionSized<T> for NoiseDistribution
where
    Gaussian: DistributionSized<T>,
    Ternary: DistributionSized<T>,
{
    fn sample_map_into<R: Rng, O>(self, out: &mut [O], f: impl Fn(T) -> O, rng: R) {
        match self {
            Self::Gaussian(inner) => inner.sample_map_into(out, f, rng),
            Self::Ternary(inner) => inner.sample_map_into(out, f, rng),
        }
    }

    fn sample_vec<R: Rng>(self, n: usize, rng: R) -> Vec<T> {
        match self {
            Self::Gaussian(inner) => inner.sample_vec(n, rng),
            Self::Ternary(inner) => inner.sample_vec(n, rng),
        }
    }
}
