use crate::ring::{ArithmeticOps, ElemFrom};
use core::{convert::identity, iter::repeat_with};
use itertools::{izip, Itertools};
use num_traits::{AsPrimitive, One, PrimInt, Signed, Zero};
use rand::{
    distributions::{Distribution, Uniform},
    RngCore,
};
use rand_distr::Normal;

pub trait Sampler: ArithmeticOps {
    fn sample<T>(&self, dist: impl Distribution<T>, mut rng: impl RngCore) -> Self::Elem
    where
        Self: ElemFrom<T>,
    {
        self.elem_from(dist.sample(&mut rng))
    }

    fn sample_iter<T>(
        &self,
        dist: impl Distribution<T>,
        rng: impl RngCore,
    ) -> impl Iterator<Item = Self::Elem>
    where
        Self: ElemFrom<T>,
    {
        dist.sample_iter(rng).map(|v| self.elem_from(v))
    }

    fn sample_vec<T>(
        &self,
        n: usize,
        dist: impl Distribution<T>,
        mut rng: impl RngCore,
    ) -> Vec<Self::Elem>
    where
        Self: ElemFrom<T>,
    {
        self.sample_iter(dist, &mut rng).take(n).collect()
    }

    fn sample_uniform(&self, rng: impl RngCore) -> Self::Elem;

    fn sample_uniform_iter(&self, mut rng: impl RngCore) -> impl Iterator<Item = Self::Elem> {
        repeat_with(move || self.sample_uniform(&mut rng))
    }

    fn sample_uniform_vec(&self, n: usize, rng: impl RngCore) -> Vec<Self::Elem> {
        self.sample_uniform_iter(rng).take(n).collect()
    }

    fn sample_gauss(&self, std_dev: f64, rng: impl RngCore) -> Self::Elem
    where
        Self: ElemFrom<i64>,
    {
        self.elem_from(sample_gaussian(std_dev, rng))
    }

    fn sample_gauss_iter(&self, std_dev: f64, rng: impl RngCore) -> impl Iterator<Item = Self::Elem>
    where
        Self: ElemFrom<i64>,
    {
        sample_gaussian_iter(std_dev, rng).map(|v| self.elem_from(v))
    }

    fn sample_gauss_vec(&self, std_dev: f64, n: usize, rng: impl RngCore) -> Vec<Self::Elem>
    where
        Self: ElemFrom<i64>,
    {
        self.sample_gauss_iter(std_dev, rng).take(n).collect()
    }
}

pub fn sample_vec<T>(n: usize, dist: impl Distribution<T>, rng: impl RngCore) -> Vec<T> {
    dist.sample_iter(rng).take(n).collect()
}

pub fn gaussian_dist<T>(std_dev: f64) -> impl Distribution<T>
where
    T: Copy + Signed + 'static,
    f64: AsPrimitive<T>,
{
    Normal::new(0.0, std_dev).unwrap().map(|a| a.round().as_())
}

pub fn sample_gaussian<T>(std_dev: f64, mut rng: impl RngCore) -> T
where
    T: Copy + Signed + 'static,
    f64: AsPrimitive<T>,
{
    gaussian_dist(std_dev).sample(&mut rng)
}

pub fn sample_gaussian_iter<T>(std_dev: f64, rng: impl RngCore) -> impl Iterator<Item = T>
where
    T: Copy + Signed + 'static,
    f64: AsPrimitive<T>,
{
    gaussian_dist(std_dev).sample_iter(rng)
}

pub fn sample_gaussian_vec<T>(std_dev: f64, n: usize, rng: impl RngCore) -> Vec<T>
where
    T: Copy + Signed + 'static,
    f64: AsPrimitive<T>,
{
    sample_gaussian_iter(std_dev, rng).take(n).collect()
}

pub fn sample_binary_iter<T: Zero + One>(mut rng: impl RngCore) -> impl Iterator<Item = T> {
    repeat_with(move || rng.next_u64())
        .flat_map(into_bits)
        .map(|bit| if bit { T::one() } else { T::zero() })
}

pub fn sample_binary_vec<T: Zero + One>(n: usize, rng: impl RngCore) -> Vec<T> {
    sample_binary_iter(rng).take(n).collect()
}

pub fn sample_ternary_vec<T: Clone + Signed>(
    hamming_weight: usize,
    n: usize,
    mut rng: impl RngCore,
) -> Vec<T> {
    assert_ne!(hamming_weight, 0);
    assert!(hamming_weight <= n);
    let indices = {
        let insert = |set: &mut [u8], idx: usize| {
            let is_none = (set[idx / 8] & 1 << (idx % 8)) == 0;
            set[idx / 8] |= 1 << (idx % 8);
            is_none
        };
        let mut set = vec![0; n.div_ceil(8)];
        let mut count = 0;
        for idx in Uniform::new(0, n).sample_iter(&mut rng) {
            count += insert(&mut set, idx) as usize;
            if count == hamming_weight {
                break;
            }
        }
        set.into_iter().flat_map(into_bits).positions(identity)
    };
    izip!(indices, repeat_with(|| rng.next_u64()).flat_map(into_bits),)
        .take(n)
        .fold(vec![T::zero(); n], |mut v, (idx, bit)| {
            v[idx] = if bit { T::one() } else { -T::one() };
            v
        })
}

fn into_bits<T: PrimInt>(byte: T) -> impl Iterator<Item = bool> {
    (0..T::zero().count_zeros() as usize).map(move |i| (byte >> i) & T::one() == T::one())
}

#[cfg(test)]
mod test {
    use crate::distribution::sample_ternary_vec;
    use num_traits::Zero;
    use rand::{
        distributions::{Distribution, Uniform},
        thread_rng,
    };

    #[test]
    fn ternary() {
        let mut rng = thread_rng();
        for n in (0..12).map(|log_n| 1 << log_n) {
            for _ in 0..n.min(100) {
                let hamming_weight = Uniform::new_inclusive(1, n).sample(&mut rng);
                let v = sample_ternary_vec::<i64>(hamming_weight, n, &mut rng);
                assert_eq!(v.iter().filter(|v| !v.is_zero()).count(), hamming_weight);
            }
        }
    }
}
