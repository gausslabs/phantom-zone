use crate::rlwe::{RlweCiphertextList, RlweCiphertextMutView, RlweCiphertextView};
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    misc::{
        as_slice::{AsMutSlice, AsSlice},
        scratch::Scratch,
    },
};

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RgswCiphertext<S> {
    #[as_slice(nested)]
    cts: RlweCiphertextList<S>,
    decomposition_log_base: usize,
    decomposition_level_a: usize,
    decomposition_level_b: usize,
}

impl<S: AsSlice> RgswCiphertext<S> {
    pub fn new(
        cts: RlweCiphertextList<S>,
        decomposition_log_base: usize,
        decomposition_level_a: usize,
        decomposition_level_b: usize,
    ) -> Self {
        debug_assert_eq!(cts.len(), decomposition_level_a + decomposition_level_b);
        Self {
            cts,
            decomposition_log_base,
            decomposition_level_a,
            decomposition_level_b,
        }
    }

    pub fn ring_size(&self) -> usize {
        self.cts.ring_size()
    }

    pub fn ct_size(&self) -> usize {
        self.cts.ct_size()
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
        self.cts.iter()
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
        self.cts.iter_mut()
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
        Self::new(
            RlweCiphertextList::allocate(ring_size, decomposition_level_a + decomposition_level_b),
            decomposition_log_base,
            decomposition_level_a,
            decomposition_level_b,
        )
    }

    pub fn allocate_eval(
        ring_size: usize,
        eval_size: usize,
        decomposition_log_base: usize,
        decomposition_level_a: usize,
        decomposition_level_b: usize,
    ) -> Self {
        Self::new(
            RlweCiphertextList::allocate_eval(
                ring_size,
                eval_size,
                decomposition_level_a + decomposition_level_b,
            ),
            decomposition_log_base,
            decomposition_level_a,
            decomposition_level_b,
        )
    }
}

impl<'a, T> RgswCiphertext<&'a mut [T]> {
    pub fn scratch(
        ring_size: usize,
        eval_size: usize,
        decomposition_log_base: usize,
        decomposition_level_a: usize,
        decomposition_level_b: usize,
        scratch: &mut Scratch<'a>,
    ) -> Self {
        Self::new(
            RlweCiphertextList::scratch(
                ring_size,
                eval_size,
                decomposition_level_a + decomposition_level_b,
                scratch,
            ),
            decomposition_log_base,
            decomposition_level_a,
            decomposition_level_b,
        )
    }
}
