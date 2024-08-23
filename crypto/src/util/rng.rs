use rand::{rngs::StdRng, Error, RngCore};

pub type StdLweRng = LweRng<StdRng, StdRng>;

pub struct LweRng<R, S> {
    private: R,
    seedable: S,
}

impl<R, S> LweRng<R, S> {
    pub fn new(private: R, seedable: S) -> Self {
        Self { private, seedable }
    }
}

impl<R: RngCore, S: RngCore> LweRng<R, S> {
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

#[cfg(any(test, feature = "getrandom"))]
impl<R: rand::SeedableRng, S: rand::SeedableRng> rand::SeedableRng for LweRng<R, S> {
    type Seed = S::Seed;

    fn seed_from_u64(state: u64) -> Self {
        Self::new(R::from_entropy(), S::seed_from_u64(state))
    }

    fn from_rng<T: RngCore>(rng: T) -> Result<Self, Error> {
        Ok(Self::new(R::from_entropy(), S::from_rng(rng)?))
    }

    fn from_entropy() -> Self {
        Self::new(R::from_entropy(), S::from_entropy())
    }

    fn from_seed(seed: Self::Seed) -> Self {
        Self::new(R::from_entropy(), S::from_seed(seed))
    }
}
