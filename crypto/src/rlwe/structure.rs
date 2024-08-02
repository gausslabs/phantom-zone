use crate::{
    lwe::LweSecretKey,
    misc::{AsMutSlice, AsSlice},
};
use core::iter::repeat_with;
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::decomposer::DecompositionParam;

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlweSecretKey<S>(S);

impl<S: AsSlice> RlweSecretKey<S> {
    pub fn ring_size(&self) -> usize {
        self.as_ref().len()
    }
}

impl<T: Default> RlweSecretKey<Vec<T>> {
    pub fn allocate(ring_size: usize) -> Self {
        Self(repeat_with(T::default).take(ring_size).collect())
    }
}

impl<S: AsSlice> From<RlweSecretKey<S>> for LweSecretKey<S> {
    fn from(sk: RlweSecretKey<S>) -> Self {
        LweSecretKey(sk.0)
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlwePlaintext<S>(S);

impl<T: Default> RlwePlaintext<Vec<T>> {
    pub fn allocate(ring_size: usize) -> Self {
        Self(repeat_with(T::default).take(ring_size).collect())
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlweCiphertext<S>(S);

impl<S: AsSlice> RlweCiphertext<S> {
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
        Self(repeat_with(T::default).take(2 * ring_size).collect())
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
