use crate::{
    core::{
        lwe::{SeededLweKeySwitchKey, SeededLweKeySwitchKeyMutView, SeededLweKeySwitchKeyView},
        rgsw::{RgswCiphertext, RgswCiphertextMutView, RgswCiphertextView, RgswDecompositionParam},
        rlwe::{SeededRlweAutoKey, SeededRlweAutoKeyMutView, SeededRlweAutoKeyView},
    },
    scheme::blind_rotation::lmkcdey::LmkcdeyParam,
    util::rng::LweRng,
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
pub struct LmkcdeyInteractiveParam {
    pub param: LmkcdeyParam,
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

impl<S: SeedableRng> LmkcdeyInteractiveCrs<S> {
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

impl<S: RngCore + SeedableRng<Seed: Clone>> LmkcdeyInteractiveCrs<S> {
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

pub type LmkcdeyInteractiveKeyShareOwned<T1, T2, S> =
    LmkcdeyInteractiveKeyShare<Vec<T1>, Vec<T2>, S>;
pub type LmkcdeyInteractiveKeyShareCompact<S> = LmkcdeyInteractiveKeyShare<Compact, Compact, S>;

#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "S1: serde::Serialize, S2: serde::Serialize, S::Seed: serde::Serialize",
        deserialize = "S1: serde::Deserialize<'de>, S2: serde::Deserialize<'de>, S::Seed: serde::Deserialize<'de>"
    ))
)]
pub struct LmkcdeyInteractiveKeyShare<S1, S2, S: SeedableRng> {
    param: LmkcdeyInteractiveParam,
    crs: LmkcdeyInteractiveCrs<S>,
    share_idx: usize,
    ks_key: SeededLweKeySwitchKey<S2>,
    brks: Vec<RgswCiphertext<S1>>,
    aks: Vec<SeededRlweAutoKey<S1>>,
}

impl<S1, S2, S: SeedableRng> LmkcdeyInteractiveKeyShare<S1, S2, S> {
    fn new(
        param: LmkcdeyInteractiveParam,
        crs: LmkcdeyInteractiveCrs<S>,
        share_idx: usize,
        ks_key: SeededLweKeySwitchKey<S2>,
        brks: Vec<RgswCiphertext<S1>>,
        aks: Vec<SeededRlweAutoKey<S1>>,
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

    pub fn param(&self) -> &LmkcdeyInteractiveParam {
        &self.param
    }

    pub fn crs(&self) -> &LmkcdeyInteractiveCrs<S> {
        &self.crs
    }

    pub fn share_idx(&self) -> usize {
        self.share_idx
    }
}

impl<S1: AsSlice, S2: AsSlice, S: SeedableRng> LmkcdeyInteractiveKeyShare<S1, S2, S> {
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
    ) -> LmkcdeyInteractiveKeyShareCompact<S>
    where
        S::Seed: Clone,
    {
        LmkcdeyInteractiveKeyShare::new(
            self.param,
            self.crs.clone(),
            self.share_idx,
            self.ks_key.compact(mod_ks),
            self.brks.iter().map(|brk| brk.compact(ring)).collect(),
            self.aks.iter().map(|ak| ak.compact(ring)).collect(),
        )
    }
}

impl<S1: AsMutSlice, S2: AsMutSlice, S: SeedableRng> LmkcdeyInteractiveKeyShare<S1, S2, S> {
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

impl<T1: Default, T2: Default, S: SeedableRng> LmkcdeyInteractiveKeyShareOwned<T1, T2, S> {
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

impl<S: SeedableRng> LmkcdeyInteractiveKeyShareCompact<S> {
    pub fn uncompact<M1, M2>(
        &self,
        ring: &M1,
        mod_ks: &M2,
    ) -> LmkcdeyInteractiveKeyShareOwned<M1::Elem, M2::Elem, S>
    where
        M1: ModulusOps,
        M2: ModulusOps,
        S::Seed: Clone,
    {
        LmkcdeyInteractiveKeyShare::new(
            self.param,
            self.crs.clone(),
            self.share_idx,
            self.ks_key.uncompact(mod_ks),
            self.brks.iter().map(|brk| brk.uncompact(ring)).collect(),
            self.aks.iter().map(|ak| ak.uncompact(ring)).collect(),
        )
    }
}

impl<T1: Clone, T2: Clone, S: SeedableRng<Seed: Clone>> Clone
    for LmkcdeyInteractiveKeyShare<T1, T2, S>
{
    fn clone(&self) -> Self {
        LmkcdeyInteractiveKeyShare::new(
            self.param,
            self.crs.clone(),
            self.share_idx,
            self.ks_key.clone(),
            self.brks.clone(),
            self.aks.clone(),
        )
    }
}

impl<T1: Debug, T2: Debug, S: SeedableRng<Seed: Debug>> Debug
    for LmkcdeyInteractiveKeyShare<T1, T2, S>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LmkcdeyInteractiveKeyShare")
            .field("param", &self.param)
            .field("crs", &self.crs)
            .field("share_idx", &self.share_idx)
            .field("ks_key", &self.ks_key)
            .field("brks", &self.brks)
            .field("aks", &self.aks)
            .finish()
    }
}

impl<T1: PartialEq, T2: PartialEq, S: SeedableRng<Seed: PartialEq>> PartialEq
    for LmkcdeyInteractiveKeyShare<T1, T2, S>
{
    fn eq(&self, other: &Self) -> bool {
        (
            &self.param,
            &self.crs,
            &self.share_idx,
            &self.ks_key,
            &self.brks,
            &self.aks,
        )
            .eq(&(
                &other.param,
                &other.crs,
                &other.share_idx,
                &other.ks_key,
                &other.brks,
                &other.aks,
            ))
    }
}
