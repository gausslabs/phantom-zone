use itertools::{izip, Itertools};
use num::UnsignedInteger;
use num_traits::{abs, Zero};
use rand::CryptoRng;
use random::{RandomGaussianDist, RandomUniformDist};
use utils::TryConvertFrom;

mod backend;
mod decomposer;
mod lwe;
mod ntt;
mod num;
mod random;
mod rgsw;
mod utils;
pub trait Matrix: AsRef<[Self::R]> {
    type MatElement;
    type R: Row<Element = Self::MatElement>;

    fn dimension(&self) -> (usize, usize);

    fn get_row(&self, row_idx: usize) -> impl Iterator<Item = &Self::MatElement> {
        self.as_ref()[row_idx].as_ref().iter().map(move |r| r)
    }

    fn get_row_slice(&self, row_idx: usize) -> &[Self::MatElement] {
        self.as_ref()[row_idx].as_ref()
    }

    fn iter_rows(&self) -> impl Iterator<Item = &Self::R> {
        self.as_ref().iter().map(move |r| r)
    }

    fn get(&self, row_idx: usize, column_idx: usize) -> &Self::MatElement {
        &self.as_ref()[row_idx].as_ref()[column_idx]
    }
}

pub trait MatrixMut: Matrix + AsMut<[<Self as Matrix>::R]>
where
    <Self as Matrix>::R: RowMut,
{
    fn get_row_mut(&mut self, row_index: usize) -> &mut [Self::MatElement] {
        self.as_mut()[row_index].as_mut()
    }

    fn iter_rows_mut(&mut self) -> impl Iterator<Item = &mut Self::R> {
        self.as_mut().iter_mut().map(move |r| r)
    }

    fn set(&mut self, row_idx: usize, column_idx: usize, val: <Self as Matrix>::MatElement) {
        self.as_mut()[row_idx].as_mut()[column_idx] = val;
    }

    fn split_at_row(
        &mut self,
        idx: usize,
    ) -> (&mut [<Self as Matrix>::R], &mut [<Self as Matrix>::R]) {
        self.as_mut().split_at_mut(idx)
    }
}

pub trait MatrixEntity: Matrix // where
// <Self as Matrix>::MatElement: Zero,
{
    fn zeros(row: usize, col: usize) -> Self;
}

pub trait Row: AsRef<[Self::Element]> {
    type Element;
}

pub trait RowMut: Row + AsMut<[<Self as Row>::Element]> {}

trait Secret {
    type Element;
    fn values(&self) -> &[Self::Element];
}

impl<T> Matrix for Vec<Vec<T>> {
    type MatElement = T;
    type R = Vec<T>;

    fn dimension(&self) -> (usize, usize) {
        (self.len(), self[0].len())
    }
}

impl<T> MatrixMut for Vec<Vec<T>> {}

impl<T: Zero + Clone> MatrixEntity for Vec<Vec<T>> {
    fn zeros(row: usize, col: usize) -> Self {
        vec![vec![T::zero(); col]; row]
    }
}

impl<T> Row for Vec<T> {
    type Element = T;
}

impl<T> RowMut for Vec<T> {}
