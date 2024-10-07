use core::fmt::Debug;
use rand::{rngs::StdRng, Error, RngCore, SeedableRng};

pub type StdLweRng = LweRng<StdRng, StdRng>;

#[derive(Clone, Debug)]
pub struct LweRng<R, S> {
    private: R,
    seedable: S,
}

impl<R, S> LweRng<R, S> {
    pub fn new(private: R, seedable: S) -> Self {
        Self { private, seedable }
    }

    pub fn from_rng(mut rng: impl RngCore) -> Result<Self, Error>
    where
        R: SeedableRng,
        S: SeedableRng,
    {
        Ok(Self::new(R::from_rng(&mut rng)?, S::from_rng(&mut rng)?))
    }
}

impl<R, S> LweRng<R, S> {
    pub fn seedable(&mut self) -> &mut S {
        &mut self.seedable
    }
}

impl<R: RngCore, S> RngCore for LweRng<R, S> {
    fn next_u32(&mut self) -> u32 {
        self.private.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.private.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.private.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.private.try_fill_bytes(dest)
    }
}

#[cfg(any(test, feature = "dev"))]
impl<R: RngCore + SeedableRng, S: SeedableRng> SeedableRng for LweRng<R, S> {
    type Seed = R::Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        let mut private = R::from_seed(seed);
        let seedable = S::from_rng(&mut private).unwrap();
        Self::new(private, seedable)
    }
}

pub trait HierarchicalSeedableRng:
    RngCore + SeedableRng<Seed: Copy + Debug + PartialEq + AsRef<[u8]>>
{
    fn from_hierarchical_seed(seed: Self::Seed, path: &[usize]) -> Self {
        path.iter().fold(Self::from_seed(seed), |mut rng, idx| {
            let mut seed = Self::Seed::default();
            for _ in 0..idx + 1 {
                rng.fill_bytes(seed.as_mut());
            }
            Self::from_seed(seed)
        })
    }
}

impl<R: RngCore + SeedableRng<Seed: Copy + Debug + PartialEq + AsRef<[u8]>>> HierarchicalSeedableRng
    for R
{
}
