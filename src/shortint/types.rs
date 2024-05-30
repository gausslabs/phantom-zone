#[derive(Clone)]
pub(super) struct FheUint8<C> {
    pub(super) data: Vec<C>,
}

impl<C> FheUint8<C> {
    pub(super) fn data(&self) -> &[C] {
        &self.data
    }

    pub(super) fn data_mut(&mut self) -> &mut [C] {
        &mut self.data
    }
}
