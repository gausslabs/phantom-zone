#[derive(Clone, Copy, Debug)]
pub enum SecretKeyDistribution {
    Gaussian(f64),
    Ternary(usize),
}
