use std::{iter::Once, sync::OnceLock};

use itertools::{izip, Itertools};
use num::UnsignedInteger;
use num_traits::{abs, Zero};
use rand::CryptoRng;
use utils::TryConvertFrom1;

mod backend;
mod bool;
mod decomposer;
mod lwe;
mod multi_party;
mod noise;
mod ntt;
mod num;
mod pbs;
mod random;
mod rgsw;
mod shortint;
mod utils;

pub use backend::{
    ArithmeticLazyOps, ArithmeticOps, ModInit, ModularOpsU64, ShoupMatrixFMA, VectorOps,
};
pub use decomposer::{Decomposer, DecomposerIter, DefaultDecomposer};
pub use ntt::{Ntt, NttBackendU64, NttInit};

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

    fn split_at_row(&self, idx: usize) -> (&[<Self as Matrix>::R], &[<Self as Matrix>::R]) {
        self.as_ref().split_at(idx)
    }

    /// Does the matrix fit sub-matrix of dimension row x col
    fn fits(&self, row: usize, col: usize) -> bool;
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

    fn split_at_row_mut(
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

pub trait RowEntity: Row {
    fn zeros(col: usize) -> Self;
}

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

    fn fits(&self, row: usize, col: usize) -> bool {
        self.len() >= row && self[0].len() >= col
    }
}

impl<T> Matrix for &[Vec<T>] {
    type MatElement = T;
    type R = Vec<T>;

    fn dimension(&self) -> (usize, usize) {
        (self.len(), self[0].len())
    }

    fn fits(&self, row: usize, col: usize) -> bool {
        self.len() >= row && self[0].len() >= col
    }
}

impl<T> Matrix for &mut [Vec<T>] {
    type MatElement = T;
    type R = Vec<T>;

    fn dimension(&self) -> (usize, usize) {
        (self.len(), self[0].len())
    }

    fn fits(&self, row: usize, col: usize) -> bool {
        self.len() >= row && self[0].len() >= col
    }
}

impl<T> MatrixMut for Vec<Vec<T>> {}
impl<T> MatrixMut for &mut [Vec<T>] {}

impl<T: Zero + Clone> MatrixEntity for Vec<Vec<T>> {
    fn zeros(row: usize, col: usize) -> Self {
        vec![vec![T::zero(); col]; row]
    }
}

impl<T> Row for Vec<T> {
    type Element = T;
}

impl<T> Row for [T] {
    type Element = T;
}

impl<T> RowMut for Vec<T> {}

impl<T: Zero + Clone> RowEntity for Vec<T> {
    fn zeros(col: usize) -> Self {
        vec![T::zero(); col]
    }
}

trait Encryptor<M: ?Sized, C> {
    fn encrypt(&self, m: &M) -> C;
}

trait Decryptor<M, C> {
    fn decrypt(&self, c: &C) -> M;
}

trait MultiPartyDecryptor<M, C> {
    type DecryptionShare;

    fn gen_decryption_share(&self, c: &C) -> Self::DecryptionShare;
    fn aggregate_decryption_shares(&self, c: &C, shares: &[Self::DecryptionShare]) -> M;
}
