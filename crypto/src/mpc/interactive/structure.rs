use itertools::Itertools;
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    modulus::Modulus,
    util::as_slice::{self, AsMutSlice, AsSlice},
};
use rand::{RngCore, SeedableRng};
use std::iter::repeat_with;

use crate::{
    core::{
        lwe::{SeededLweKeySwitchKey, SeededLweKeySwitchKeyMutView},
        rgsw::{RgswCiphertext, RgswCiphertextMutView, RgswDecompositionParam},
        rlwe::{RlweAutoKey, SeededRlweAutoKey, SeededRlweAutoKeyMutView},
    },
    scheme::blind_rotation::lmkcdey::power_g_mod_q,
    util::{
        distribution::{NoiseDistribution, SecretKeyDistribution},
        rng::LweRng,
    },
};

// trait Parameters {
//     fn lwe_n(&self) -> usize;
//     fn rlwe_n(&self) -> usize;
//     fn lwe_q(&self) -> usize;
//     fn rlwe_q(&self) -> usize;
//     fn lwe_ksk_decomposer(&self) -> DecompositionParam;
//     fn rlwe_x_rgsw_decomposer(&self) -> RgswDecompositionParam;
//     fn auto_decomposer(&self) -> DecompositionParam;
//     fn w(&self) -> usize;
// }

pub struct InteractiveMpcParameters {
    lwe_n: usize,
    rlwe_n: usize,
    lwe_q: Modulus,
    rlwe_q: Modulus,
    br_q: usize,
    lwe_ksk_decomposer: DecompositionParam,
    rlwe_x_rgsw_decomposer: RgswDecompositionParam,
    auto_decomposer: DecompositionParam,
    rgsw_x_rgsw_decomposer: RgswDecompositionParam,
    noise_distribution: NoiseDistribution,
    lwe_secret_key_distribution: SecretKeyDistribution,
    rlwe_secret_key_distribution: SecretKeyDistribution,
    w: usize,
    g: usize,
}

impl InteractiveMpcParameters {
    pub fn auto_decomposer(&self) -> DecompositionParam {
        self.auto_decomposer.clone()
    }
    pub fn lwe_n(&self) -> usize {
        self.lwe_n
    }
    pub fn rlwe_n(&self) -> usize {
        self.rlwe_n
    }
    pub fn rlwe_x_rgsw_decomposer(&self) -> RgswDecompositionParam {
        self.rlwe_x_rgsw_decomposer.clone()
    }
    pub fn lwe_ksk_decomposer(&self) -> DecompositionParam {
        self.lwe_ksk_decomposer.clone()
    }
    pub fn w(&self) -> usize {
        self.w
    }
    pub fn lwe_q(&self) -> Modulus {
        self.lwe_q
    }
    pub fn rlwe_q(&self) -> Modulus {
        self.rlwe_q
    }
    pub fn noise_distribution(&self) -> NoiseDistribution {
        self.noise_distribution
    }
    pub fn g(&self) -> usize {
        self.g
    }
    pub fn br_q(&self) -> usize {
        self.br_q
    }
    pub fn rgsw_x_rgsw_decomposer(&self) -> RgswDecompositionParam {
        self.rgsw_x_rgsw_decomposer.clone()
    }
    pub fn embedding_factor(&self) -> usize {
        (self.rlwe_n() * 2) / self.br_q()
    }
    pub fn lwe_secret_key_distribution(&self) -> SecretKeyDistribution {
        self.lwe_secret_key_distribution.clone()
    }
    pub fn rlwe_secret_key_distribution(&self) -> SecretKeyDistribution {
        self.rlwe_secret_key_distribution.clone()
    }
}

/// Divides LWE dimension into two chunks. The first contains indices for which `self` generates RGSW ciphertexts for RLWE x RGSW and second contains indices for which `self`  generates RGSW for RGSW x RGSW
pub(crate) fn split_lwe_dimension_for_rgsws(
    lwe_n: usize,
    user_id: usize,
    total_users: usize,
) -> (Vec<usize>, Vec<usize>) {
    assert!(user_id < total_users);
    let chunk_size = lwe_n / total_users;
    let self_indices =
        ((user_id * chunk_size)..std::cmp::min((user_id + 1) * chunk_size, lwe_n)).collect_vec();
    let not_self_indices = (0..(user_id * chunk_size))
        .chain(((user_id + 1) * chunk_size)..lwe_n)
        .collect_vec();
    (self_indices, not_self_indices)
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlwePublicKeyShare<S> {
    #[as_slice]
    data: S,
}

impl<S: AsSlice> RlwePublicKeyShare<S> {
    pub fn new(data: S) -> Self {
        Self { data }
    }
    pub fn ring_size(&self) -> usize {
        self.data.len()
    }
}

impl<T: Default> RlwePublicKeyShare<Vec<T>> {
    pub fn allocate(n: usize) -> Self {
        Self::new(repeat_with(T::default).take(n).collect())
    }
}

/// Common reference seed known to all parties participating in the interactive MPC.
///
///
#[derive(Clone, Copy, Debug)]
pub struct InteractiveCrs<S> {
    seed: S,
}

impl<S: Copy + Default + AsMut<[u8]>> InteractiveCrs<S> {
    /// Random number generator for LWE Key switching key
    ///
    /// Random elements `a`s must be derived from CRS. We puncture main RNG, seeded with self.seed, once to derive the seed of lwe_ksk_rng
    pub(crate) fn lwe_ksk_rng<R: RngCore + SeedableRng<Seed = S>>(
        &self,
        noise_rng: &mut R,
    ) -> LweRng<R, R> {
        // Puncture once
        let mut main_rng = R::from_seed(self.seed);
        let mut a_seed = <R as SeedableRng>::Seed::default();
        main_rng.fill_bytes(a_seed.as_mut());

        let mut e_seed = <R as SeedableRng>::Seed::default();
        noise_rng.fill_bytes(e_seed.as_mut());

        LweRng::new(R::from_seed(e_seed), R::from_seed(a_seed))
    }

    /// Random number generator for auto keys
    ///
    /// Random polynomials `a` of auto keys must derive from CRS. We puncture main RNG, seeded with self.seed, twice to derive the seed of auto_rng
    pub(crate) fn auto_keys_rng<R: RngCore + SeedableRng<Seed = S>>(
        &self,
        noise_rng: &mut R,
    ) -> LweRng<R, R> {
        // Puncture twice
        let mut main_rng = R::from_seed(self.seed);
        let mut a_seed = <R as SeedableRng>::Seed::default();
        main_rng.fill_bytes(a_seed.as_mut());
        main_rng.fill_bytes(a_seed.as_mut());

        let mut e_seed = <R as SeedableRng>::Seed::default();
        noise_rng.fill_bytes(e_seed.as_mut());

        LweRng::new(R::from_seed(e_seed), R::from_seed(a_seed))
    }

    /// Random number generator for RGSW ciphertexts
    ///
    /// RGSW ciphertexts are encrypted with RLWE pulic key. Hence, both random polynomials `a` and noise polynomials `e` are not drived from CRS.
    pub(crate) fn rgsw_cts_rng<R: RngCore + SeedableRng<Seed = S>>(
        &self,
        noise_rng: &mut R,
    ) -> LweRng<R, R> {
        let mut a_seed = S::default();
        noise_rng.fill_bytes(a_seed.as_mut());

        let mut e_seed = S::default();
        noise_rng.fill_bytes(e_seed.as_mut());

        LweRng::new(R::from_seed(e_seed), R::from_seed(a_seed))
    }
}

#[derive(Clone, Debug)]
pub struct CommonReferenceSeededServerKeyShare<S1, S2: AsSlice<Elem = usize>, Seed> {
    seeded_lwe_ksk: SeededLweKeySwitchKey<S1>,
    seeded_auto_keys: Vec<SeededRlweAutoKey<S1, S2>>,
    self_rgsw_cts: Vec<RgswCiphertext<S1>>,
    not_self_rgsw_cts: Vec<RgswCiphertext<S1>>,
    crs: InteractiveCrs<Seed>,
}

impl<S1: AsSlice, S2: AsSlice<Elem = usize>, Seed: Clone>
    CommonReferenceSeededServerKeyShare<S1, S2, Seed>
{
    pub fn new(
        lwe_ksk: SeededLweKeySwitchKey<S1>,
        auto_keys: Vec<SeededRlweAutoKey<S1, S2>>,
        self_rgsw_cts: Vec<RgswCiphertext<S1>>,
        not_self_rgsw_cts: Vec<RgswCiphertext<S1>>,
        crs: InteractiveCrs<Seed>,
    ) -> Self {
        Self {
            seeded_lwe_ksk: lwe_ksk,
            seeded_auto_keys: auto_keys,
            self_rgsw_cts,
            not_self_rgsw_cts,
            crs,
        }
    }

    pub fn crs(&self) -> InteractiveCrs<Seed> {
        self.crs.clone()
    }
}

impl<S1: AsMutSlice, S2: AsMutSlice<Elem = usize>, Seed>
    CommonReferenceSeededServerKeyShare<S1, S2, Seed>
{
    pub fn as_mut_lwe_ksk(&mut self) -> SeededLweKeySwitchKeyMutView<S1::Elem> {
        self.seeded_lwe_ksk.as_mut_view()
    }

    pub fn auto_keys_mut_iter(
        &mut self,
    ) -> impl Iterator<Item = SeededRlweAutoKeyMutView<S1::Elem>> {
        self.seeded_auto_keys
            .iter_mut()
            .map(|auto_key| auto_key.as_mut_view())
    }

    pub fn self_rgsw_cts_mut_iter(
        &mut self,
    ) -> impl Iterator<Item = RgswCiphertextMutView<S1::Elem>> {
        self.self_rgsw_cts.iter_mut().map(|ct| ct.as_mut_view())
    }

    pub fn not_self_rgsw_cts_mut_iter(
        &mut self,
    ) -> impl Iterator<Item = RgswCiphertextMutView<S1::Elem>> {
        self.not_self_rgsw_cts.iter_mut().map(|ct| ct.as_mut_view())
    }
}

impl<T: Default, Seed: Clone> CommonReferenceSeededServerKeyShare<Vec<T>, Vec<usize>, Seed> {
    pub fn allocate(
        parameters: InteractiveMpcParameters,
        user_id: usize,
        total_users: usize,
        crs: InteractiveCrs<Seed>,
    ) -> Self {
        let (self_count, not_self_count) =
            split_lwe_dimension_for_rgsws(parameters.lwe_n(), user_id, total_users);

        Self::new(
            SeededLweKeySwitchKey::allocate(
                parameters.rlwe_n(),
                parameters.lwe_n(),
                parameters.lwe_ksk_decomposer(),
            ),
            // Auto keys for k = [-g, g^1, ..., g^w]
            (([(parameters.br_q() - parameters.g()) as i64].into_iter()).chain(
                power_g_mod_q(parameters.g(), parameters.br_q())
                    .take(parameters.w() + 1)
                    .skip(1)
                    .map(|k| k as i64),
            ))
            .map(|k| {
                SeededRlweAutoKey::allocate(parameters.rlwe_n(), parameters.auto_decomposer(), k)
            })
            .collect(),
            (0..self_count.len())
                .map(|_| {
                    RgswCiphertext::allocate(
                        parameters.rlwe_n(),
                        parameters.rlwe_x_rgsw_decomposer(),
                    )
                })
                .collect_vec(),
            (0..not_self_count.len())
                .map(|_| {
                    RgswCiphertext::allocate(
                        parameters.rlwe_n(),
                        parameters.rgsw_x_rgsw_decomposer(),
                    )
                })
                .collect_vec(),
            crs,
        )
    }
}

pub struct CommonReferenceSeededServerKey {
    seeded_lwe_ksk: SeededLweKeySwitchKey<Vec<u8>>,
    seeded_auto_keys: Vec<SeededRlweAutoKey<Vec<u8>, Vec<usize>>>,
    rgsw_cts: Vec<RgswCiphertext<Vec<u8>>>,
    crs: InteractiveCrs<Vec<u8>>,
}
