use crate::rlwe::{RlweCiphertext, RlweCiphertextMutView, RlweCiphertextView};
use core::iter::repeat_with;
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    misc::as_slice::{AsMutSlice, AsSlice},
};

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RgswCiphertext<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
    ct_size: usize,
    decomposition_log_base: usize,
    decomposition_level_a: usize,
    decomposition_level_b: usize,
}

impl<S: AsSlice> RgswCiphertext<S> {
    pub fn new(
        data: S,
        ring_size: usize,
        ct_size: usize,
        decomposition_log_base: usize,
        decomposition_level_a: usize,
        decomposition_level_b: usize,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            ct_size * (decomposition_level_a + decomposition_level_b)
        );
        Self {
            data,
            ring_size,
            ct_size,
            decomposition_log_base,
            decomposition_level_a,
            decomposition_level_b,
        }
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }

    pub fn ct_size(&self) -> usize {
        self.ct_size
    }

    pub fn decomposition_log_base(&self) -> usize {
        self.decomposition_log_base
    }

    pub fn decomposition_level_a(&self) -> usize {
        self.decomposition_level_a
    }

    pub fn decomposition_level_b(&self) -> usize {
        self.decomposition_level_b
    }

    pub fn decomposition_param_a(&self) -> DecompositionParam {
        DecompositionParam {
            level: self.decomposition_level_a,
            log_base: self.decomposition_log_base,
        }
    }

    pub fn decomposition_param_b(&self) -> DecompositionParam {
        DecompositionParam {
            level: self.decomposition_level_b,
            log_base: self.decomposition_log_base,
        }
    }

    pub fn ct_iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        self.as_ref()
            .chunks(self.ct_size())
            .map(|ct| RlweCiphertext::new(ct, self.ring_size()))
    }

    pub fn a_ct_iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        self.ct_iter().take(self.decomposition_level_a)
    }

    pub fn b_ct_iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        self.ct_iter().skip(self.decomposition_level_a)
    }
}

impl<S: AsMutSlice> RgswCiphertext<S> {
    pub fn ct_iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        let ct_size = self.ct_size();
        let ring_size = self.ring_size();
        self.as_mut()
            .chunks_mut(ct_size)
            .map(move |ct| RlweCiphertext::new(ct, ring_size))
    }

    pub fn a_ct_iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        let decomposition_level_a = self.decomposition_level_a;
        self.ct_iter_mut().take(decomposition_level_a)
    }

    pub fn b_ct_iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        let decomposition_level_a = self.decomposition_level_a;
        self.ct_iter_mut().skip(decomposition_level_a)
    }
}

impl<T: Default> RgswCiphertext<Vec<T>> {
    pub fn allocate(
        ring_size: usize,
        decomposition_log_base: usize,
        decomposition_level_a: usize,
        decomposition_level_b: usize,
    ) -> Self {
        let ct_size = 2 * ring_size;
        let len = ct_size * (decomposition_level_a + decomposition_level_b);
        Self::new(
            repeat_with(T::default).take(len).collect(),
            ring_size,
            ct_size,
            decomposition_log_base,
            decomposition_level_a,
            decomposition_level_b,
        )
    }

    pub fn allocate_prep(
        ring_size: usize,
        prep_size: usize,
        decomposition_log_base: usize,
        decomposition_level_a: usize,
        decomposition_level_b: usize,
    ) -> Self {
        let ct_size = 2 * prep_size;
        let len = ct_size * (decomposition_level_a + decomposition_level_b);
        Self::new(
            repeat_with(T::default).take(len).collect(),
            ring_size,
            ct_size,
            decomposition_log_base,
            decomposition_level_a,
            decomposition_level_b,
        )
    }
}
