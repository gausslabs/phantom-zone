use crate::core::rlwe::{RlweCiphertextList, RlweCiphertextMutView, RlweCiphertextView};
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    util::{
        as_slice::{AsMutSlice, AsSlice},
        scratch::Scratch,
    },
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RgswDecompositionParam {
    pub log_base: usize,
    pub level_a: usize,
    pub level_b: usize,
}

impl RgswDecompositionParam {
    pub fn a(&self) -> DecompositionParam {
        DecompositionParam {
            level: self.level_a,
            log_base: self.log_base,
        }
    }

    pub fn b(&self) -> DecompositionParam {
        DecompositionParam {
            level: self.level_b,
            log_base: self.log_base,
        }
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RgswCiphertext<S> {
    #[as_slice(nested)]
    cts: RlweCiphertextList<S>,
    decomposition_param: RgswDecompositionParam,
}

impl<S: AsSlice> RgswCiphertext<S> {
    pub fn new(cts: RlweCiphertextList<S>, decomposition_param: RgswDecompositionParam) -> Self {
        debug_assert_eq!(
            cts.len(),
            decomposition_param.level_a + decomposition_param.level_b
        );
        Self {
            cts,
            decomposition_param,
        }
    }

    pub fn ring_size(&self) -> usize {
        self.cts.ring_size()
    }

    pub fn ct_size(&self) -> usize {
        self.cts.ct_size()
    }

    pub fn decomposition_param(&self) -> RgswDecompositionParam {
        self.decomposition_param
    }

    pub fn decomposition_param_a(&self) -> DecompositionParam {
        self.decomposition_param.a()
    }

    pub fn decomposition_param_b(&self) -> DecompositionParam {
        self.decomposition_param.b()
    }

    pub fn ct_iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        self.cts.iter()
    }

    pub fn a_ct_iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        self.ct_iter().take(self.decomposition_param.level_a)
    }

    pub fn b_ct_iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        self.ct_iter().skip(self.decomposition_param.level_a)
    }
}

impl<S: AsMutSlice> RgswCiphertext<S> {
    pub fn ct_iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        self.cts.iter_mut()
    }

    pub fn a_ct_iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        let decomposition_level_a = self.decomposition_param.level_a;
        self.ct_iter_mut().take(decomposition_level_a)
    }

    pub fn b_ct_iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        let decomposition_level_a = self.decomposition_param.level_a;
        self.ct_iter_mut().skip(decomposition_level_a)
    }
}

impl<T: Default> RgswCiphertext<Vec<T>> {
    pub fn allocate(ring_size: usize, decomposition_param: RgswDecompositionParam) -> Self {
        Self::new(
            RlweCiphertextList::allocate(
                ring_size,
                decomposition_param.level_a + decomposition_param.level_b,
            ),
            decomposition_param,
        )
    }

    pub fn allocate_eval(
        ring_size: usize,
        eval_size: usize,
        decomposition_param: RgswDecompositionParam,
    ) -> Self {
        Self::new(
            RlweCiphertextList::allocate_eval(
                ring_size,
                eval_size,
                decomposition_param.level_a + decomposition_param.level_b,
            ),
            decomposition_param,
        )
    }
}

impl<'a, T> RgswCiphertext<&'a mut [T]> {
    pub fn scratch(
        ring_size: usize,
        eval_size: usize,
        decomposition_param: RgswDecompositionParam,
        scratch: &mut Scratch<'a>,
    ) -> Self {
        Self::new(
            RlweCiphertextList::scratch(
                ring_size,
                eval_size,
                decomposition_param.level_a + decomposition_param.level_b,
                scratch,
            ),
            decomposition_param,
        )
    }
}
