use crate::{
    core::{
        lwe::{
            SeededLweKeySwitchKey, SeededLweKeySwitchKeyMutView, SeededLweKeySwitchKeyOwned,
            SeededLweKeySwitchKeyView,
        },
        rgsw::{
            RgswCiphertext, RgswCiphertextMutView, RgswCiphertextOwned, RgswCiphertextView,
            RgswDecompositionParam,
        },
        rlwe::{
            SeededRlweAutoKey, SeededRlweAutoKeyMutView, SeededRlweAutoKeyOwned,
            SeededRlweAutoKeyView,
        },
    },
    scheme::blind_rotation::lmkcdey::LmkcdeyParam,
    util::{distribution::SecretDistribution, rng::LweRng},
};
use core::{
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    ops::Deref,
};
use rand::{RngCore, SeedableRng};

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LmkcdeyInteractiveParam {
    pub param: LmkcdeyParam,
    pub u_distribution: SecretDistribution,
    pub rgsw_by_rgsw_decomposition_param: RgswDecompositionParam,
    pub total_shares: usize,
}

impl Deref for LmkcdeyInteractiveParam {
    type Target = LmkcdeyParam;

    fn deref(&self) -> &Self::Target {
        &self.param
    }
}

#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "S::Seed: serde::Serialize",
        deserialize = "S::Seed: serde::Deserialize<'de>"
    ))
)]
pub struct LmkcdeyInteractiveCrs<S: SeedableRng> {
    seed: S::Seed,
    #[cfg_attr(feature = "serde", serde(skip))]
    _marker: PhantomData<S>,
}

impl<S: RngCore + SeedableRng<Seed: Clone>> LmkcdeyInteractiveCrs<S> {
    pub fn sample(mut rng: impl RngCore) -> Self {
        let mut seed = S::Seed::default();
        rng.fill_bytes(seed.as_mut());
        Self {
            seed,
            _marker: PhantomData,
        }
    }

    pub fn pk_rng<R: RngCore + SeedableRng>(&self, rng: &mut R) -> LweRng<R, S> {
        let private = R::from_rng(rng).unwrap();
        let seedable = self.hierarchical_rng(&[0]);
        LweRng::new(private, seedable)
    }

    pub fn unseed_pk_rng(&self) -> LweRng<(), S> {
        let seedable = self.hierarchical_rng(&[0]);
        LweRng::new((), seedable)
    }

    pub(crate) fn ks_key_rng<R: RngCore + SeedableRng>(&self, rng: &mut R) -> LweRng<R, S> {
        let private = R::from_rng(rng).unwrap();
        let seedable = self.hierarchical_rng(&[1, 0]);
        LweRng::new(private, seedable)
    }

    pub(crate) fn unseed_ks_key_rng(&self) -> LweRng<(), S> {
        let seedable = self.hierarchical_rng(&[1, 0]);
        LweRng::new((), seedable)
    }

    pub(crate) fn ak_rng<R: RngCore + SeedableRng>(&self, rng: &mut R) -> LweRng<R, S> {
        let private = R::from_rng(rng).unwrap();
        let seedable = self.hierarchical_rng(&[1, 1]);
        LweRng::new(private, seedable)
    }

    pub(crate) fn unseed_ak_rng(&self) -> LweRng<(), S> {
        let seedable = self.hierarchical_rng(&[1, 1]);
        LweRng::new((), seedable)
    }

    pub(crate) fn brk_rng<R: RngCore + SeedableRng>(&self, rng: &mut R) -> LweRng<R, R> {
        LweRng::from_rng(rng).unwrap()
    }

    fn hierarchical_rng(&self, path: &[usize]) -> S {
        S::from_seed(self.hierarchical_seed(path))
    }

    fn hierarchical_seed(&self, path: &[usize]) -> S::Seed {
        let mut seed = self.seed.clone();
        for idx in path {
            let mut rng = S::from_seed(seed.clone());
            for _ in 0..idx + 1 {
                rng.fill_bytes(seed.as_mut());
            }
        }
        seed
    }
}

impl<S: SeedableRng<Seed: Clone>> Clone for LmkcdeyInteractiveCrs<S> {
    fn clone(&self) -> Self {
        Self {
            seed: self.seed.clone(),
            _marker: PhantomData,
        }
    }
}

impl<S: SeedableRng<Seed: Copy>> Copy for LmkcdeyInteractiveCrs<S> {}

impl<S: SeedableRng<Seed: Debug>> Debug for LmkcdeyInteractiveCrs<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LmkcdeyInteractiveCrs")
            .field("seed", &self.seed)
            .finish()
    }
}

impl<S: SeedableRng<Seed: PartialEq>> PartialEq for LmkcdeyInteractiveCrs<S> {
    fn eq(&self, other: &Self) -> bool {
        self.seed.eq(&other.seed)
    }
}

#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T1: serde::Serialize, T2: serde::Serialize, S::Seed: serde::Serialize",
        deserialize = "T1: serde::Deserialize<'de>, T2: serde::Deserialize<'de>, S::Seed: serde::Deserialize<'de>"
    ))
)]
pub struct LmkcdeyKeyShare<T1, T2, S: SeedableRng> {
    param: LmkcdeyInteractiveParam,
    crs: LmkcdeyInteractiveCrs<S>,
    share_idx: usize,
    ks_key: SeededLweKeySwitchKeyOwned<T2>,
    brks: Vec<RgswCiphertextOwned<T1>>,
    aks: Vec<SeededRlweAutoKeyOwned<T1>>,
}

impl<T1, T2, S: SeedableRng> LmkcdeyKeyShare<T1, T2, S> {
    pub fn param(&self) -> &LmkcdeyInteractiveParam {
        &self.param
    }

    pub fn crs(&self) -> &LmkcdeyInteractiveCrs<S> {
        &self.crs
    }

    pub fn share_idx(&self) -> usize {
        self.share_idx
    }

    pub fn ks_key(&self) -> SeededLweKeySwitchKeyView<T2> {
        self.ks_key.as_view()
    }

    pub fn brks(&self) -> impl Iterator<Item = RgswCiphertextView<T1>> {
        self.brks.iter().map(RgswCiphertext::as_view)
    }

    pub fn aks(&self) -> impl Iterator<Item = SeededRlweAutoKeyView<T1>> {
        self.aks.iter().map(SeededRlweAutoKey::as_view)
    }

    pub(crate) fn ks_key_mut(&mut self) -> SeededLweKeySwitchKeyMutView<T2> {
        self.ks_key.as_mut_view()
    }

    pub(crate) fn brks_mut(&mut self) -> impl Iterator<Item = RgswCiphertextMutView<T1>> {
        self.brks.iter_mut().map(RgswCiphertext::as_mut_view)
    }

    pub(crate) fn aks_mut(&mut self) -> impl Iterator<Item = SeededRlweAutoKeyMutView<T1>> {
        self.aks.iter_mut().map(SeededRlweAutoKey::as_mut_view)
    }
}

impl<T1: Default + Clone, T2: Default, S: SeedableRng> LmkcdeyKeyShare<T1, T2, S> {
    fn new(
        param: LmkcdeyInteractiveParam,
        crs: LmkcdeyInteractiveCrs<S>,
        share_idx: usize,
        ks_key: SeededLweKeySwitchKeyOwned<T2>,
        brks: Vec<RgswCiphertextOwned<T1>>,
        aks: Vec<SeededRlweAutoKeyOwned<T1>>,
    ) -> Self {
        debug_assert!(param.total_shares > 0);
        debug_assert!(share_idx < param.total_shares);
        Self {
            param,
            crs,
            share_idx,
            ks_key,
            brks,
            aks,
        }
    }

    pub fn allocate(
        param: LmkcdeyInteractiveParam,
        crs: LmkcdeyInteractiveCrs<S>,
        share_idx: usize,
    ) -> Self {
        let ks_key = SeededLweKeySwitchKey::allocate(
            param.ring_size,
            param.lwe_dimension,
            param.lwe_ks_decomposition_param,
        );
        let brks = {
            let chunk_size = param.lwe_dimension.div_ceil(param.total_shares);
            let init_range = chunk_size * share_idx..chunk_size * (share_idx + 1);
            (0..param.lwe_dimension)
                .map(|idx| {
                    let rlwe_by_rgsw_decomposition_param = if init_range.contains(&idx) {
                        param.rlwe_by_rgsw_decomposition_param
                    } else {
                        param.rgsw_by_rgsw_decomposition_param
                    };
                    RgswCiphertext::allocate(param.ring_size, rlwe_by_rgsw_decomposition_param)
                })
                .collect()
        };
        let aks = param
            .aks()
            .map(|k| {
                SeededRlweAutoKey::allocate(param.ring_size, param.auto_decomposition_param, k)
            })
            .collect();
        Self::new(param, crs, share_idx, ks_key, brks, aks)
    }
}
