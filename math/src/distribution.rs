use core::iter::repeat_with;
use rand::{distributions::Distribution, RngCore};

pub trait Sampler<T> {
    #[allow(clippy::wrong_self_convention)]
    fn from_i64(&self, v: i64) -> T;

    fn sample(&self, dist: impl Distribution<i64>, mut rng: impl RngCore) -> T {
        self.from_i64(dist.sample(&mut rng))
    }

    fn sample_iter(
        &self,
        dist: impl Distribution<i64>,
        rng: impl RngCore,
    ) -> impl Iterator<Item = T> {
        dist.sample_iter(rng).map(|v| self.from_i64(v))
    }

    fn sample_vec(&self, n: usize, dist: impl Distribution<i64>, mut rng: impl RngCore) -> Vec<T> {
        self.sample_iter(dist, &mut rng).take(n).collect()
    }

    fn sample_uniform(&self, rng: impl RngCore) -> T;

    fn sample_uniform_iter(&self, mut rng: impl RngCore) -> impl Iterator<Item = T> {
        repeat_with(move || self.sample_uniform(&mut rng))
    }

    fn sample_uniform_vec(&self, n: usize, rng: impl RngCore) -> Vec<T> {
        self.sample_uniform_iter(rng).take(n).collect()
    }
}
