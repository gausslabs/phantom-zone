use crate::{
    core::rlwe::{
        RlweAutoKey, RlweAutoKeyMutView, RlweAutoKeyView, SeededRlweAutoKey,
        SeededRlweAutoKeyMutView, SeededRlweAutoKeyView,
    },
    util::{
        distribution::{NoiseDistribution, SecretDistribution},
        rng::{HierarchicalSeedableRng, LweRng},
    },
};
use core::{borrow::Borrow, fmt};
use phantom_zone_math::{
    poly::automorphism::AutomorphismMap,
    prelude::{Compact, DecompositionParam, Modulus, ModulusOps},
    ring::RingOps,
    util::as_slice::{AsMutSlice, AsSlice},
};
use rand::{RngCore, SeedableRng};
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CdksParam {
    /// RLWE ciphertext modulus.
    pub modulus: Modulus,
    /// RLWE ring size.
    pub ring_size: usize,
    /// RLWE secret key distribution.
    pub sk_distribution: SecretDistribution,
    /// RLWE noise distribution for sk/pk encryption.
    pub noise_distribution: NoiseDistribution,
    /// RLWE automorphism decomposition parameter.
    pub auto_decomposition_param: DecompositionParam,
}

impl CdksParam {
    pub fn aks(&self) -> impl Iterator<Item = usize> {
        (1..)
            .map(|ell| (1 << ell) + 1)
            .take(self.ring_size.ilog2() as usize)
    }
}

pub type CdksKeyOwned<T> = CdksKey<Vec<T>, AutomorphismMap>;
pub type CdksKeyCompact = CdksKey<Compact, AutomorphismMap>;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CdksKey<S, A> {
    param: CdksParam,
    aks: Vec<RlweAutoKey<S, A>>,
}

impl<S, A> CdksKey<S, A> {
    fn new(param: CdksParam, aks: Vec<RlweAutoKey<S, A>>) -> Self {
        Self { param, aks }
    }
}

impl<S: AsSlice, A: Borrow<AutomorphismMap>> CdksKey<S, A> {
    pub fn param(&self) -> &CdksParam {
        &self.param
    }

    pub fn aks(&self) -> impl Iterator<Item = RlweAutoKeyView<S::Elem>> {
        self.aks.iter().map(RlweAutoKey::as_view)
    }

    pub fn ak(&self, ell: usize) -> RlweAutoKeyView<S::Elem> {
        self.aks[ell - 1].as_view()
    }

    pub fn compact(&self, ring: &impl RingOps<Elem = S::Elem>) -> CdksKeyCompact {
        CdksKey::new(
            self.param,
            self.aks.iter().map(|ak| ak.compact(ring)).collect(),
        )
    }
}

impl<S: AsMutSlice, A: Borrow<AutomorphismMap>> CdksKey<S, A> {
    pub(crate) fn aks_mut(&mut self) -> impl Iterator<Item = RlweAutoKeyMutView<S::Elem>> {
        self.aks.iter_mut().map(RlweAutoKey::as_mut_view)
    }
}

impl<T: Clone + Default> CdksKeyOwned<T> {
    pub fn allocate(param: CdksParam) -> Self {
        let aks = param
            .aks()
            .map(|k| RlweAutoKey::allocate(param.ring_size, param.auto_decomposition_param, k))
            .collect();
        Self::new(param, aks)
    }

    pub fn allocate_eval(param: CdksParam, eval_size: usize) -> Self {
        let aks = param
            .aks()
            .map(|k| {
                RlweAutoKey::allocate_eval(
                    param.ring_size,
                    eval_size,
                    param.auto_decomposition_param,
                    k,
                )
            })
            .collect();
        Self::new(param, aks)
    }
}

impl CdksKeyCompact {
    pub fn uncompact<M: ModulusOps>(&self, ring: &M) -> CdksKeyOwned<M::Elem> {
        CdksKey::new(
            self.param,
            self.aks.iter().map(|ak| ak.uncompact(ring)).collect(),
        )
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
pub struct CdksCrs<S: HierarchicalSeedableRng> {
    seed: S::Seed,
    #[cfg_attr(feature = "serde", serde(skip))]
    _marker: PhantomData<S>,
}

impl<S: HierarchicalSeedableRng> CdksCrs<S> {
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

impl<S: HierarchicalSeedableRng> CdksCrs<S> {
    pub(crate) fn ak_rng<R: RngCore + SeedableRng>(&self, rng: &mut R) -> LweRng<R, S> {
        let private = R::from_rng(rng).unwrap();
        let seedable = self.hierarchical_rng(&[0]);
        LweRng::new(private, seedable)
    }

    pub(crate) fn unseed_ak_rng(&self) -> LweRng<(), S> {
        let seedable = self.hierarchical_rng(&[0]);
        LweRng::new((), seedable)
    }

    fn hierarchical_rng(&self, path: &[usize]) -> S {
        S::from_hierarchical_seed(self.seed, path)
    }
}

impl<S: HierarchicalSeedableRng> Clone for CdksCrs<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: HierarchicalSeedableRng> Copy for CdksCrs<S> {}

impl<S: HierarchicalSeedableRng> fmt::Debug for CdksCrs<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CdksCrs").field("seed", &self.seed).finish()
    }
}

impl<S: HierarchicalSeedableRng> PartialEq for CdksCrs<S> {
    fn eq(&self, other: &Self) -> bool {
        self.seed.eq(&other.seed)
    }
}

pub type CdksKeyShareOwned<T> = CdksKeyShare<Vec<T>>;
pub type CdksKeyShareCompact = CdksKeyShare<Compact>;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CdksKeyShare<S> {
    param: CdksParam,
    aks: Vec<SeededRlweAutoKey<S>>,
}

impl<S> CdksKeyShare<S> {
    fn new(param: CdksParam, aks: Vec<SeededRlweAutoKey<S>>) -> Self {
        Self { param, aks }
    }
}

impl<S: AsSlice> CdksKeyShare<S> {
    pub fn param(&self) -> &CdksParam {
        &self.param
    }

    pub fn aks(&self) -> impl Iterator<Item = SeededRlweAutoKeyView<S::Elem>> {
        self.aks.iter().map(SeededRlweAutoKey::as_view)
    }

    pub fn compact(&self, ring: &impl RingOps<Elem = S::Elem>) -> CdksKeyShareCompact {
        CdksKeyShare::new(
            self.param,
            self.aks.iter().map(|ak| ak.compact(ring)).collect(),
        )
    }
}

impl<S: AsMutSlice> CdksKeyShare<S> {
    pub(crate) fn aks_mut(&mut self) -> impl Iterator<Item = SeededRlweAutoKeyMutView<S::Elem>> {
        self.aks.iter_mut().map(SeededRlweAutoKey::as_mut_view)
    }
}

impl<T: Clone + Default> CdksKeyShareOwned<T> {
    pub fn allocate(param: CdksParam) -> Self {
        let aks = param
            .aks()
            .map(|k| {
                SeededRlweAutoKey::allocate(param.ring_size, param.auto_decomposition_param, k)
            })
            .collect();
        Self::new(param, aks)
    }
}

impl CdksKeyShareCompact {
    pub fn uncompact<M: ModulusOps>(&self, ring: &M) -> CdksKeyShareOwned<M::Elem> {
        CdksKeyShare::new(
            self.param,
            self.aks.iter().map(|ak| ak.uncompact(ring)).collect(),
        )
    }
}
