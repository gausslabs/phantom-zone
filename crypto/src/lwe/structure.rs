use core::iter::repeat_with;
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    misc::as_slice::{AsMutSlice, AsSlice},
};

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct LweSecretKey<S>(S);

impl<S: AsSlice> LweSecretKey<S> {
    pub fn new(data: S) -> Self {
        Self(data)
    }

    pub fn dimension(&self) -> usize {
        self.as_ref().len()
    }
}

impl<T: Default> LweSecretKey<Vec<T>> {
    pub fn allocate(dimension: usize) -> Self {
        Self::new(repeat_with(T::default).take(dimension).collect())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LwePlaintext<T>(pub T);

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct LweCiphertext<S>(S);

impl<S: AsSlice> LweCiphertext<S> {
    pub fn new(data: S) -> Self {
        Self(data)
    }

    pub fn dimension(&self) -> usize {
        self.as_ref().len() - 1
    }

    pub fn a(&self) -> &[S::Elem] {
        self.a_b().0
    }

    pub fn b(&self) -> &S::Elem {
        self.a_b().1
    }

    pub fn a_b(&self) -> (&[S::Elem], &S::Elem) {
        let (b, a) = self.as_ref().split_last().unwrap();
        (a, b)
    }
}

impl<S: AsMutSlice> LweCiphertext<S> {
    pub fn a_mut(&mut self) -> &mut [S::Elem] {
        self.a_b_mut().0
    }

    pub fn b_mut(&mut self) -> &mut S::Elem {
        self.a_b_mut().1
    }

    pub fn a_b_mut(&mut self) -> (&mut [S::Elem], &mut S::Elem) {
        let (b, a) = self.as_mut().split_last_mut().unwrap();
        (a, b)
    }
}

impl<T: Default> LweCiphertext<Vec<T>> {
    pub fn allocate(dimension: usize) -> Self {
        Self::new(repeat_with(T::default).take(dimension + 1).collect())
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct LweKeySwitchKey<S> {
    #[as_slice]
    data: S,
    to_dimension: usize,
    decomposition_param: DecompositionParam,
}

impl<S: AsSlice> LweKeySwitchKey<S> {
    pub fn new(data: S, to_dimension: usize, decomposition_param: DecompositionParam) -> Self {
        debug_assert_eq!(
            data.len() % ((to_dimension + 1) * decomposition_param.level),
            0
        );
        Self {
            data,
            to_dimension,
            decomposition_param,
        }
    }

    pub fn to_dimension(&self) -> usize {
        self.to_dimension
    }

    pub fn from_dimension(&self) -> usize {
        self.len() / ((self.to_dimension + 1) * self.decomposition_param.level)
    }

    pub fn decomposition_param(&self) -> DecompositionParam {
        self.decomposition_param
    }

    pub fn ct_iter(&self) -> impl Iterator<Item = LweCiphertextView<S::Elem>> {
        let ct_len = self.to_dimension + 1;
        self.as_ref().chunks(ct_len).map(LweCiphertext)
    }
}

impl<S: AsMutSlice> LweKeySwitchKey<S> {
    pub fn ct_iter_mut(&mut self) -> impl Iterator<Item = LweCiphertextMutView<S::Elem>> {
        let ct_len = self.to_dimension + 1;
        self.as_mut().chunks_mut(ct_len).map(LweCiphertext)
    }
}

impl<T: Default> LweKeySwitchKey<Vec<T>> {
    pub fn allocate(
        from_dimension: usize,
        to_dimension: usize,
        decomposition_param: DecompositionParam,
    ) -> Self {
        let len = from_dimension * (to_dimension + 1) * decomposition_param.level;
        Self::new(
            repeat_with(T::default).take(len).collect(),
            to_dimension,
            decomposition_param,
        )
    }
}
