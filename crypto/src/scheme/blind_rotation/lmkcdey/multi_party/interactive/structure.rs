use crate::{
    core::{
        lwe::{SeededLweKeySwitchKey, SeededLweKeySwitchKeyMutView, SeededLweKeySwitchKeyView},
        rgsw::{RgswCiphertext, RgswCiphertextMutView, RgswCiphertextView, RgswDecompositionParam},
        rlwe::{SeededRlweAutoKey, SeededRlweAutoKeyMutView, SeededRlweAutoKeyView},
    },
    scheme::blind_rotation::lmkcdey::LmkcdeyParam,
    util::rng::{HierarchicalSeedableRng, LweRng},
};
use core::{
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    ops::Deref,
};
use phantom_zone_math::{
    modulus::ModulusOps,
    util::{
        as_slice::{AsMutSlice, AsSlice},
        compact::Compact,
    },
};
use rand::{RngCore, SeedableRng};

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LmkcdeyMpiParam {
    pub param: LmkcdeyParam,
    /// RGSW by RGSW decomposition parameter.
    pub rgsw_by_rgsw_decomposition_param: RgswDecompositionParam,
    /// Total shares of multi-party.
    pub total_shares: usize,
}

impl Deref for LmkcdeyMpiParam {
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
pub struct LmkcdeyMpiCrs<S: HierarchicalSeedableRng> {
    seed: S::Seed,
    #[cfg_attr(feature = "serde", serde(skip))]
    _marker: PhantomData<S>,
}

impl<S: HierarchicalSeedableRng> LmkcdeyMpiCrs<S> {
    pub fn new(seed: S::Seed) -> Self {
        Self {
            seed,
            _marker: PhantomData,
        }
    }

    pub fn sample(mut rng: impl RngCore) -> Self {
        let mut seed = S::Seed::default();
        rng.fill_bytes(seed.as_mut());
        Self::new(seed)
    }
}

impl<S: HierarchicalSeedableRng> LmkcdeyMpiCrs<S> {
    pub(crate) fn pk_rng<R: RngCore + SeedableRng>(&self, rng: &mut R) -> LweRng<R, S> {
        let private = R::from_rng(rng).unwrap();
        let seedable = self.hierarchical_rng(&[0]);
        LweRng::new(private, seedable)
    }

    pub(crate) fn unseed_pk_rng(&self) -> LweRng<(), S> {
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
        S::from_hierarchical_seed(self.seed, path)
    }
}

impl<S: HierarchicalSeedableRng> Clone for LmkcdeyMpiCrs<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: HierarchicalSeedableRng> Copy for LmkcdeyMpiCrs<S> {}

impl<S: HierarchicalSeedableRng> Debug for LmkcdeyMpiCrs<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LmkcdeyMpiCrs")
            .field("seed", &self.seed)
            .finish()
    }
}

impl<S: HierarchicalSeedableRng> PartialEq for LmkcdeyMpiCrs<S> {
    fn eq(&self, other: &Self) -> bool {
        self.seed.eq(&other.seed)
    }
}

pub type LmkcdeyMpiKeyShareOwned<T1, T2> = LmkcdeyMpiKeyShare<Vec<T1>, Vec<T2>>;
pub type LmkcdeyMpiKeyShareCompact = LmkcdeyMpiKeyShare<Compact, Compact>;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "S1: serde::Serialize, S2: serde::Serialize",
        deserialize = "S1: serde::Deserialize<'de>, S2: serde::Deserialize<'de>"
    ))
)]
pub struct LmkcdeyMpiKeyShare<S1, S2> {
    param: LmkcdeyMpiParam,
    share_idx: usize,
    ks_key: SeededLweKeySwitchKey<S2>,
    brks: Vec<RgswCiphertext<S1>>,
    aks: Vec<SeededRlweAutoKey<S1>>,
}

impl<S1, S2> LmkcdeyMpiKeyShare<S1, S2> {
    fn new(
        param: LmkcdeyMpiParam,
        share_idx: usize,
        ks_key: SeededLweKeySwitchKey<S2>,
        brks: Vec<RgswCiphertext<S1>>,
        aks: Vec<SeededRlweAutoKey<S1>>,
    ) -> Self {
        debug_assert!(param.total_shares > 0);
        debug_assert!(share_idx < param.total_shares);
        Self {
            param,
            share_idx,
            ks_key,
            brks,
            aks,
        }
    }

    pub fn param(&self) -> &LmkcdeyMpiParam {
        &self.param
    }

    pub fn share_idx(&self) -> usize {
        self.share_idx
    }
}

impl<S1: AsSlice, S2: AsSlice> LmkcdeyMpiKeyShare<S1, S2> {
    pub fn ks_key(&self) -> SeededLweKeySwitchKeyView<S2::Elem> {
        self.ks_key.as_view()
    }

    pub fn brks(&self) -> impl Iterator<Item = RgswCiphertextView<S1::Elem>> {
        self.brks.iter().map(RgswCiphertext::as_view)
    }

    pub fn aks(&self) -> impl Iterator<Item = SeededRlweAutoKeyView<S1::Elem>> {
        self.aks.iter().map(SeededRlweAutoKey::as_view)
    }

    pub fn compact(
        &self,
        ring: &impl ModulusOps<Elem = S1::Elem>,
        mod_ks: &impl ModulusOps<Elem = S2::Elem>,
    ) -> LmkcdeyMpiKeyShareCompact {
        LmkcdeyMpiKeyShare::new(
            self.param,
            self.share_idx,
            self.ks_key.compact(mod_ks),
            self.brks.iter().map(|brk| brk.compact(ring)).collect(),
            self.aks.iter().map(|ak| ak.compact(ring)).collect(),
        )
    }
}

impl<S1: AsMutSlice, S2: AsMutSlice> LmkcdeyMpiKeyShare<S1, S2> {
    pub(crate) fn ks_key_mut(&mut self) -> SeededLweKeySwitchKeyMutView<S2::Elem> {
        self.ks_key.as_mut_view()
    }

    pub(crate) fn brks_mut(&mut self) -> impl Iterator<Item = RgswCiphertextMutView<S1::Elem>> {
        self.brks.iter_mut().map(RgswCiphertext::as_mut_view)
    }

    pub(crate) fn aks_mut(&mut self) -> impl Iterator<Item = SeededRlweAutoKeyMutView<S1::Elem>> {
        self.aks.iter_mut().map(SeededRlweAutoKey::as_mut_view)
    }
}

impl<T1: Default, T2: Default> LmkcdeyMpiKeyShareOwned<T1, T2> {
    pub fn allocate(param: LmkcdeyMpiParam, share_idx: usize) -> Self {
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
                    let decomposition_param = if init_range.contains(&idx) {
                        param.rlwe_by_rgsw_decomposition_param
                    } else {
                        param.rgsw_by_rgsw_decomposition_param
                    };
                    RgswCiphertext::allocate(param.ring_size, decomposition_param)
                })
                .collect()
        };
        let aks = param
            .aks()
            .map(|k| {
                SeededRlweAutoKey::allocate(param.ring_size, param.auto_decomposition_param, k)
            })
            .collect();
        Self::new(param, share_idx, ks_key, brks, aks)
    }
}

impl LmkcdeyMpiKeyShareCompact {
    pub fn uncompact<M1, M2>(
        &self,
        ring: &M1,
        mod_ks: &M2,
    ) -> LmkcdeyMpiKeyShareOwned<M1::Elem, M2::Elem>
    where
        M1: ModulusOps,
        M2: ModulusOps,
    {
        LmkcdeyMpiKeyShare::new(
            self.param,
            self.share_idx,
            self.ks_key.uncompact(mod_ks),
            self.brks.iter().map(|brk| brk.uncompact(ring)).collect(),
            self.aks.iter().map(|ak| ak.uncompact(ring)).collect(),
        )
    }
}
