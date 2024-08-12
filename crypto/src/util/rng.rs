use rand::{RngCore, SeedableRng};

pub struct LweRng<R1, R2> {
    noise: R1,
    a: R2,
}

impl<R1, R2> LweRng<R1, R2> {
    pub fn new(noise: R1, a: R2) -> Self {
        Self { noise, a }
    }

    pub fn from_entropy() -> LweRng<R1, R2>
    where
        R1: SeedableRng,
        R2: SeedableRng,
    {
        LweRng::new(R1::from_entropy(), R2::from_entropy())
    }

    pub fn from_seed(a_seed: R2::Seed) -> LweRng<R1, R2>
    where
        R1: SeedableRng,
        R2: SeedableRng,
    {
        LweRng::new(R1::from_entropy(), R2::from_seed(a_seed))
    }
}

impl<R1: RngCore, R2: RngCore> LweRng<R1, R2> {
    pub fn noise(&mut self) -> &mut R1 {
        &mut self.noise
    }

    pub fn a(&mut self) -> &mut R2 {
        &mut self.a
    }
}

impl<R1: Default, R2: Default> Default for LweRng<R1, R2> {
    fn default() -> Self {
        Self::new(R1::default(), R2::default())
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::util::rng::LweRng;
    use rand::rngs::StdRng;

    pub type StdLweRng = LweRng<StdRng, StdRng>;
}
