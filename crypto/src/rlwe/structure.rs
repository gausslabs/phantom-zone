use crate::lwe::LweSecretKey;
use core::iter::repeat_with;
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    misc::{
        as_slice::{AsMutSlice, AsSlice},
        automorphism::{AutomorphismMap, AutomorphismMapView},
    },
};

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlweSecretKey<S>(S);

impl<S: AsSlice> RlweSecretKey<S> {
    pub fn new(data: S) -> Self {
        Self(data)
    }

    pub fn ring_size(&self) -> usize {
        self.as_ref().len()
    }
}

impl<T: Default> RlweSecretKey<Vec<T>> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(repeat_with(T::default).take(ring_size).collect())
    }
}

impl<S: AsSlice> From<RlweSecretKey<S>> for LweSecretKey<S> {
    fn from(sk: RlweSecretKey<S>) -> Self {
        LweSecretKey::new(sk.0)
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlwePlaintext<S>(S);

impl<S: AsSlice> RlwePlaintext<S> {
    pub fn new(data: S) -> Self {
        Self(data)
    }
}

impl<T: Default> RlwePlaintext<Vec<T>> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(repeat_with(T::default).take(ring_size).collect())
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlweCiphertext<S>(S);

impl<S: AsSlice> RlweCiphertext<S> {
    pub fn new(data: S) -> Self {
        Self(data)
    }

    pub fn ring_size(&self) -> usize {
        self.len() / 2
    }

    pub fn a(&self) -> &[S::Elem] {
        self.a_b().0
    }

    pub fn b(&self) -> &[S::Elem] {
        self.a_b().1
    }

    pub fn a_b(&self) -> (&[S::Elem], &[S::Elem]) {
        let ring_size = self.ring_size();
        self.as_ref().split_at(ring_size)
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
        let ring_size = self.ring_size();
        self.as_mut().split_at_mut(ring_size)
    }
}

impl<T: Default> RlweCiphertext<Vec<T>> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(repeat_with(T::default).take(2 * ring_size).collect())
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlwePublicKey<S>(S);

impl<S: AsSlice> RlwePublicKey<S> {
    pub fn new(data: S) -> Self {
        Self(data)
    }

    pub fn ring_size(&self) -> usize {
        self.len() / 2
    }

    pub fn a(&self) -> &[S::Elem] {
        self.a_b().0
    }

    pub fn b(&self) -> &[S::Elem] {
        self.a_b().1
    }

    pub fn a_b(&self) -> (&[S::Elem], &[S::Elem]) {
        let ring_size = self.ring_size();
        self.as_ref().split_at(ring_size)
    }

    pub fn as_ct(&self) -> RlweCiphertextView<S::Elem> {
        RlweCiphertext::new(self.as_ref())
    }
}

impl<S: AsMutSlice> RlwePublicKey<S> {
    pub fn a_mut(&mut self) -> &mut [S::Elem] {
        self.a_b_mut().0
    }

    pub fn b_mut(&mut self) -> &mut [S::Elem] {
        self.a_b_mut().1
    }

    pub fn a_b_mut(&mut self) -> (&mut [S::Elem], &mut [S::Elem]) {
        let ring_size = self.ring_size();
        self.as_mut().split_at_mut(ring_size)
    }

    pub fn as_ct_mut(&mut self) -> RlweCiphertextMutView<S::Elem> {
        RlweCiphertext::new(self.as_mut())
    }
}

impl<T: Default> RlwePublicKey<Vec<T>> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(repeat_with(T::default).take(2 * ring_size).collect())
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlweKeySwitchKey<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
    decomposition_param: DecompositionParam,
}

impl<S: AsSlice> RlweKeySwitchKey<S> {
    pub fn new(data: S, ring_size: usize, decomposition_param: DecompositionParam) -> Self {
        debug_assert_eq!(data.len(), (2 * ring_size * decomposition_param.level));
        Self {
            data,
            ring_size,
            decomposition_param,
        }
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }

    pub fn decomposition_param(&self) -> DecompositionParam {
        self.decomposition_param
    }

    pub fn ct_iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        let ct_len = 2 * self.ring_size;
        self.as_ref().chunks(ct_len).map(RlweCiphertext)
    }
}

impl<S: AsMutSlice> RlweKeySwitchKey<S> {
    pub fn ct_iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        let ct_len = 2 * self.ring_size;
        self.as_mut().chunks_mut(ct_len).map(RlweCiphertext)
    }
}

impl<T: Default> RlweKeySwitchKey<Vec<T>> {
    pub fn allocate(ring_size: usize, decomposition_param: DecompositionParam) -> Self {
        let len = 2 * ring_size * decomposition_param.level;
        Self {
            data: repeat_with(T::default).take(len).collect(),
            ring_size,
            decomposition_param,
        }
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
    pub fn new(ks_key: RlweKeySwitchKey<S1>, map: AutomorphismMap<S2>) -> Self {
        Self { ks_key, map }
    }

    pub fn map(&self) -> AutomorphismMapView {
        self.map.as_view()
    }
}

impl<S1: AsSlice, S2: AsSlice<Elem = usize>> RlweAutoKey<S1, S2> {
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
}
