use num_traits::FromPrimitive;
use phantom_zone_math::distribution::{DistributionSized, DistributionVariance, Gaussian, Ternary};
use rand::{distributions::Distribution, Rng};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SecretDistribution {
    Gaussian(Gaussian),
    Ternary(Ternary),
}

impl From<Gaussian> for SecretDistribution {
    fn from(inner: Gaussian) -> Self {
        Self::Gaussian(inner)
    }
}

impl From<Ternary> for SecretDistribution {
    fn from(inner: Ternary) -> Self {
        Self::Ternary(inner)
    }
}

impl<T> DistributionSized<T> for SecretDistribution
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

    fn sample_into<R: Rng>(self, out: &mut [T], rng: R) {
        match self {
            Self::Gaussian(inner) => inner.sample_into(out, rng),
            Self::Ternary(inner) => inner.sample_into(out, rng),
        }
    }

    fn sample_vec<R: Rng>(self, n: usize, rng: R) -> Vec<T> {
        match self {
            Self::Gaussian(inner) => inner.sample_vec(n, rng),
            Self::Ternary(inner) => inner.sample_vec(n, rng),
        }
    }
}

impl DistributionVariance for SecretDistribution {
    fn variance(self) -> f64 {
        match self {
            Self::Gaussian(inner) => inner.variance(),
            Self::Ternary(inner) => inner.variance(),
        }
    }

    fn std_dev(self) -> f64 {
        match self {
            Self::Gaussian(inner) => inner.std_dev(),
            Self::Ternary(inner) => inner.std_dev(),
        }
    }

    fn log2_std_dev(self) -> f64 {
        match self {
            Self::Gaussian(inner) => inner.log2_std_dev(),
            Self::Ternary(inner) => inner.log2_std_dev(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseDistribution {
    Gaussian(Gaussian),
}

impl From<Gaussian> for NoiseDistribution {
    fn from(inner: Gaussian) -> Self {
        Self::Gaussian(inner)
    }
}

impl<T: FromPrimitive> Distribution<T> for NoiseDistribution
where
    Gaussian: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        match self {
            Self::Gaussian(inner) => inner.sample(rng),
        }
    }
}

impl<T> DistributionSized<T> for NoiseDistribution
where
    Gaussian: DistributionSized<T>,
{
    fn sample_map_into<R: Rng, O>(self, out: &mut [O], f: impl Fn(T) -> O, rng: R) {
        match self {
            Self::Gaussian(inner) => inner.sample_map_into(out, f, rng),
        }
    }

    fn sample_into<R: Rng>(self, out: &mut [T], rng: R) {
        match self {
            Self::Gaussian(inner) => inner.sample_into(out, rng),
        }
    }

    fn sample_vec<R: Rng>(self, n: usize, rng: R) -> Vec<T> {
        match self {
            Self::Gaussian(inner) => inner.sample_vec(n, rng),
        }
    }
}

impl DistributionVariance for NoiseDistribution {
    fn variance(self) -> f64 {
        match self {
            Self::Gaussian(inner) => inner.variance(),
        }
    }

    fn std_dev(self) -> f64 {
        match self {
            Self::Gaussian(inner) => inner.std_dev(),
        }
    }

    fn log2_std_dev(self) -> f64 {
        match self {
            Self::Gaussian(inner) => inner.log2_std_dev(),
        }
    }
}
