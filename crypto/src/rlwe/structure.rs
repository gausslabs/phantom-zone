use crate::{distribution::SecretKeyDistribution, lwe::LweSecretKey};
use core::iter::repeat_with;
use num_traits::{FromPrimitive, Signed};
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::DistributionSized,
    misc::as_slice::{AsMutSlice, AsSlice},
    poly::automorphism::{AutomorphismMap, AutomorphismMapView},
};
use rand::RngCore;

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlweSecretKey<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
}

impl<S: AsSlice> RlweSecretKey<S> {
    pub fn new(data: S, ring_size: usize) -> Self {
        Self { data, ring_size }
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }
}

impl<T: Default> RlweSecretKey<Vec<T>> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(repeat_with(T::default).take(ring_size).collect(), ring_size)
    }

    pub fn sample(
        ring_size: usize,
        sk_distribution: SecretKeyDistribution,
        rng: impl RngCore,
    ) -> Self
    where
        T: Signed + FromPrimitive,
    {
        Self::new(sk_distribution.sample_vec(ring_size, rng), ring_size)
    }
}

impl<S: AsSlice> From<RlweSecretKey<S>> for LweSecretKey<S> {
    fn from(sk: RlweSecretKey<S>) -> Self {
        LweSecretKey::new(sk.data)
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlwePlaintext<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
}

impl<S: AsSlice> RlwePlaintext<S> {
    pub fn new(data: S, ring_size: usize) -> Self {
        Self { data, ring_size }
    }
}

impl<T: Default> RlwePlaintext<Vec<T>> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(repeat_with(T::default).take(ring_size).collect(), ring_size)
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlweCiphertext<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
}

impl<S: AsSlice> RlweCiphertext<S> {
    pub fn new(data: S, ring_size: usize) -> Self {
        Self { data, ring_size }
    }

    pub fn ct_size(&self) -> usize {
        self.data.len()
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }

    pub fn a(&self) -> &[S::Elem] {
        self.a_b().0
    }

    pub fn b(&self) -> &[S::Elem] {
        self.a_b().1
    }

    pub fn a_b(&self) -> (&[S::Elem], &[S::Elem]) {
        self.data.split_at_mid()
    }
}

impl<S: AsMutSlice> RlweCiphertext<S> {
    pub fn a_mut(&mut self) -> &mut [S::Elem] {
        self.a_b_mut().0
    }

    pub fn b_mut(&mut self) -> &mut [S::Elem] {
        self.a_b_mut().1
    }

    pub fn a_b_mut(&mut self) -> (&mut [S::Elem], &mut [S::Elem]) {
        self.data.split_at_mid_mut()
    }
}

impl<T: Default> RlweCiphertext<Vec<T>> {
    pub fn allocate(ring_size: usize) -> Self {
        let ct_size = 2 * ring_size;
        Self::new(repeat_with(T::default).take(ct_size).collect(), ring_size)
    }

    pub fn allocate_prep(ring_size: usize, prep_size: usize) -> Self {
        let ct_size = 2 * prep_size;
        Self::new(repeat_with(T::default).take(ct_size).collect(), ring_size)
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlweCiphertextList<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
    ct_size: usize,
}

impl<S: AsSlice> RlweCiphertextList<S> {
    pub fn new(data: S, ring_size: usize, ct_size: usize) -> Self {
        debug_assert_eq!(data.len() % ct_size, 0);
        Self {
            data,
            ring_size,
            ct_size,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.ct_size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn ct_size(&self) -> usize {
        self.ct_size
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }

    pub fn iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_ref()
            .chunks(ct_size)
            .map(move |ct| RlweCiphertext::new(ct, ring_size))
    }

    pub fn chunks(
        &self,
        chunk_size: usize,
    ) -> impl Iterator<Item = RlweCiphertextListView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_ref()
            .chunks(chunk_size * ct_size)
            .map(move |ct| RlweCiphertextList::new(ct, ring_size, ct_size))
    }
}

impl<S: AsMutSlice> RlweCiphertextList<S> {
    pub fn iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_mut()
            .chunks_mut(ct_size)
            .map(move |ct| RlweCiphertext::new(ct, ring_size))
    }

    pub fn chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl Iterator<Item = RlweCiphertextListMutView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_mut()
            .chunks_mut(chunk_size * ct_size)
            .map(move |ct| RlweCiphertextList::new(ct, ring_size, ct_size))
    }
}

impl<T: Default> RlweCiphertextList<Vec<T>> {
    pub fn allocate(ring_size: usize, n: usize) -> Self {
        let ct_size = 2 * ring_size;
        Self::new(
            repeat_with(T::default).take(n * ct_size).collect(),
            ring_size,
            ct_size,
        )
    }

    pub fn allocate_prep(ring_size: usize, prep_size: usize, n: usize) -> Self {
        let ct_size = 2 * prep_size;
        Self::new(
            repeat_with(T::default).take(n * ct_size).collect(),
            ring_size,
            ct_size,
        )
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlwePublicKey<S>(#[as_slice(nested)] RlweCiphertext<S>);

impl<S: AsSlice> RlwePublicKey<S> {
    pub fn new(ct: RlweCiphertext<S>) -> Self {
        Self(ct)
    }

    pub fn ring_size(&self) -> usize {
        self.0.ring_size()
    }

    pub fn a(&self) -> &[S::Elem] {
        self.0.a()
    }

    pub fn b(&self) -> &[S::Elem] {
        self.0.b()
    }

    pub fn a_b(&self) -> (&[S::Elem], &[S::Elem]) {
        self.0.a_b()
    }

    pub fn as_ct(&self) -> RlweCiphertextView<S::Elem> {
        self.0.as_view()
    }
}

impl<S: AsMutSlice> RlwePublicKey<S> {
    pub fn as_ct_mut(&mut self) -> RlweCiphertextMutView<S::Elem> {
        self.0.as_mut_view()
    }
}

impl<T: Default> RlwePublicKey<Vec<T>> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(RlweCiphertext::allocate(ring_size))
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlweKeySwitchKey<S> {
    #[as_slice(nested)]
    cts: RlweCiphertextList<S>,
    decomposition_param: DecompositionParam,
}

impl<S: AsSlice> RlweKeySwitchKey<S> {
    pub fn new(cts: RlweCiphertextList<S>, decomposition_param: DecompositionParam) -> Self {
        debug_assert_eq!(cts.len(), decomposition_param.level);
        Self {
            cts,
            decomposition_param,
        }
    }

    pub fn ring_size(&self) -> usize {
        self.cts.ring_size
    }

    pub fn ct_size(&self) -> usize {
        self.cts.ct_size
    }

    pub fn decomposition_param(&self) -> DecompositionParam {
        self.decomposition_param
    }

    pub fn ct_iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        self.cts.iter()
    }
}

impl<S: AsMutSlice> RlweKeySwitchKey<S> {
    pub fn ct_iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        self.cts.iter_mut()
    }
}

impl<T: Default> RlweKeySwitchKey<Vec<T>> {
    pub fn allocate(ring_size: usize, decomposition_param: DecompositionParam) -> Self {
        Self::new(
            RlweCiphertextList::allocate(ring_size, decomposition_param.level),
            decomposition_param,
        )
    }

    pub fn allocate_prep(
        ring_size: usize,
        prep_size: usize,
        decomposition_param: DecompositionParam,
    ) -> Self {
        Self::new(
            RlweCiphertextList::allocate_prep(ring_size, prep_size, decomposition_param.level),
            decomposition_param,
        )
    }
}

#[derive(Clone, Debug, AsSliceWrapper)]
pub struct RlweAutoKey<S1, S2: AsSlice<Elem = usize>> {
    #[as_slice(nested)]
    ks_key: RlweKeySwitchKey<S1>,
    #[as_slice(nested)]
    map: AutomorphismMap<S2>,
}

impl<S1, S2: AsSlice<Elem = usize>> RlweAutoKey<S1, S2> {
    pub fn map(&self) -> AutomorphismMapView {
        self.map.as_view()
    }

    pub fn k(&self) -> usize {
        self.map.k()
    }
}

impl<S1: AsSlice, S2: AsSlice<Elem = usize>> RlweAutoKey<S1, S2> {
    pub fn new(ks_key: RlweKeySwitchKey<S1>, map: AutomorphismMap<S2>) -> Self {
        debug_assert_eq!(ks_key.ring_size(), map.ring_size());
        Self { ks_key, map }
    }

    pub fn as_ks_key(&self) -> RlweKeySwitchKeyView<S1::Elem> {
        self.ks_key.as_view()
    }
}

impl<S1: AsMutSlice, S2: AsSlice<Elem = usize>> RlweAutoKey<S1, S2> {
    pub fn split_into_map_and_ks_key_mut(
        &mut self,
    ) -> (AutomorphismMapView, RlweKeySwitchKeyMutView<S1::Elem>) {
        (self.map.as_view(), self.ks_key.as_mut_view())
    }
}

impl<T: Default> RlweAutoKey<Vec<T>, Vec<usize>> {
    pub fn allocate(ring_size: usize, decomposition_param: DecompositionParam, k: i64) -> Self {
        Self::new(
            RlweKeySwitchKey::allocate(ring_size, decomposition_param),
            AutomorphismMap::new(ring_size, k),
        )
    }

    pub fn allocate_prep(
        ring_size: usize,
        prep_size: usize,
        decomposition_param: DecompositionParam,
        k: i64,
    ) -> Self {
        Self::new(
            RlweKeySwitchKey::allocate_prep(ring_size, prep_size, decomposition_param),
            AutomorphismMap::new(ring_size, k),
        )
    }
}
