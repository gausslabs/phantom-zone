use crate::{
    decomposer::Decomposer,
    distribution::{DistributionSized, Sampler},
    izip_eq,
    util::serde::Serde,
};
use core::{borrow::Borrow, fmt::Debug, hash::Hash, iter::successors};
use itertools::izip;
use rand_distr::Distribution;

pub(crate) mod power_of_two;
pub(crate) mod prime;

pub use power_of_two::{Native, NonNativePowerOfTwo};
pub use prime::Prime;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Modulus {
    PowerOfTwo(usize),
    Prime(u64),
}

impl Modulus {
    pub fn bits(&self) -> usize {
        match *self {
            Self::PowerOfTwo(bits) => bits,
            Self::Prime(prime) => prime
                .checked_next_power_of_two()
                .map(u64::ilog2)
                .unwrap_or(u64::BITS) as _,
        }
    }

    pub fn as_f64(&self) -> f64 {
        match *self {
            Self::PowerOfTwo(bits) => 2f64.powi(bits as _),
            Self::Prime(prime) => prime as f64,
        }
    }
}

pub trait ElemOps: Clone + Debug + Send + Sync {
    type Elem: 'static + Copy + Debug + Default + Eq + Ord + Hash + Send + Sync + Serde;
}

pub trait ElemFrom<T>: ElemOps {
    fn elem_from(&self, v: T) -> Self::Elem;
}

pub trait ElemTo<T>: ElemOps {
    fn elem_to(&self, v: Self::Elem) -> T;
}

pub trait ModulusOps:
    ElemOps
    + ElemFrom<u64>
    + ElemFrom<i64>
    + ElemFrom<f64>
    + ElemFrom<u32>
    + ElemFrom<i32>
    + ElemTo<u64>
    + ElemTo<i64>
    + ElemTo<f64>
    + Sampler
{
    type ElemPrep: 'static + Copy + Debug + Default + Send + Sync;
    type Decomposer: Decomposer<Self::Elem>;

    fn new(modulus: Modulus) -> Self;

    fn modulus(&self) -> Modulus;

    fn uniform_distribution(&self)
        -> impl Distribution<Self::Elem> + DistributionSized<Self::Elem>;

    fn zero(&self) -> Self::Elem;

    fn one(&self) -> Self::Elem;

    fn neg_one(&self) -> Self::Elem;

    fn neg(&self, a: &Self::Elem) -> Self::Elem;

    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    fn add_assign(&self, a: &mut Self::Elem, b: &Self::Elem) {
        *a = self.add(a, b);
    }

    fn sub_assign(&self, a: &mut Self::Elem, b: &Self::Elem) {
        *a = self.sub(a, b);
    }

    fn mul_assign(&self, a: &mut Self::Elem, b: &Self::Elem) {
        *a = self.mul(a, b);
    }

    fn powers(&self, a: &Self::Elem) -> impl Iterator<Item = Self::Elem> {
        successors(Some(self.one()), move |v| self.mul(v, a).into())
    }

    fn inv(&self, _: &Self::Elem) -> Option<Self::Elem> {
        None
    }

    fn add_elem_from<T: Copy>(&self, a: &Self::Elem, b: &T) -> Self::Elem
    where
        Self: ElemFrom<T>,
    {
        self.add(a, &self.elem_from(*b))
    }

    fn sub_elem_from<T: Copy>(&self, a: &Self::Elem, b: &T) -> Self::Elem
    where
        Self: ElemFrom<T>,
    {
        self.sub(a, &self.elem_from(*b))
    }

    fn mul_elem_from<T: Copy>(&self, a: &Self::Elem, b: &T) -> Self::Elem
    where
        Self: ElemFrom<T>,
    {
        self.mul(a, &self.elem_from(*b))
    }

    fn prepare(&self, a: &Self::Elem) -> Self::ElemPrep;

    fn mul_prep(&self, a: &Self::Elem, b: &Self::ElemPrep) -> Self::Elem;

    fn to_u64(&self, a: Self::Elem) -> u64 {
        self.elem_to(a)
    }

    fn to_i64(&self, a: Self::Elem) -> i64 {
        self.elem_to(a)
    }

    fn mod_switch<M: ModulusOps>(&self, a: &Self::Elem, mod_to: &M) -> M::Elem {
        let delta = mod_to.modulus().as_f64() / self.modulus().as_f64();
        mod_to.elem_from((self.to_u64(*a) as f64 * delta).round() as u64)
    }

    fn slice_op_assign<T>(&self, b: &mut [Self::Elem], a: &[T], f: impl Fn(&mut Self::Elem, &T)) {
        izip_eq!(b, a).for_each(|(b, a)| f(b, a))
    }

    fn slice_op<T>(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: &[T],
        f: impl Fn(&mut Self::Elem, &Self::Elem, &T),
    ) {
        izip_eq!(c, a, b).for_each(|(c, a, b)| f(c, a, b))
    }

    fn slice_op_assign_iter<T>(
        &self,
        b: &mut [Self::Elem],
        a: impl IntoIterator<Item: Borrow<T>>,
        f: impl Fn(&mut Self::Elem, &T),
    ) {
        izip!(b, a).for_each(|(b, a)| f(b, a.borrow()))
    }

    fn slice_op_iter<T>(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: impl IntoIterator<Item: Borrow<T>>,
        f: impl Fn(&mut Self::Elem, &Self::Elem, &T),
    ) {
        izip!(c, a, b).for_each(|(c, a, b)| f(c, a, b.borrow()))
    }

    fn slice_prepare(&self, b: &mut [Self::ElemPrep], a: &[Self::Elem]) {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.prepare(a))
    }

    fn slice_elem_from<T: Copy>(&self, b: &mut [Self::Elem], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.slice_op_assign(b, a, |b, a| *b = self.elem_from(*a))
    }

    fn slice_elem_from_iter<T: Copy>(
        &self,
        b: &mut [Self::Elem],
        a: impl IntoIterator<Item: Borrow<T>>,
    ) where
        Self: ElemFrom<T>,
    {
        self.slice_op_assign_iter(b, a, |b, a| *b = self.elem_from(*a))
    }

    fn slice_neg_assign(&self, a: &mut [Self::Elem]) {
        a.iter_mut().for_each(|a| *a = self.neg(a))
    }

    fn slice_double_assign(&self, a: &mut [Self::Elem]) {
        a.iter_mut().for_each(|a| *a = self.add(a, a))
    }

    fn slice_add_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        self.slice_op_assign(b, a, |b, a| *b = self.add(b, a))
    }

    fn slice_sub_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        self.slice_op_assign(b, a, |b, a| *b = self.sub(b, a))
    }

    fn slice_mul_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        self.slice_op_assign(b, a, |b, a| *b = self.mul(b, a))
    }

    fn slice_mul_assign_prep(&self, b: &mut [Self::Elem], a: &[Self::ElemPrep]) {
        self.slice_op_assign(b, a, |b, a| *b = self.mul_prep(b, a))
    }

    fn slice_scalar_mul_assign(&self, b: &mut [Self::Elem], a: &Self::Elem) {
        b.iter_mut().for_each(|b| *b = self.mul(b, a))
    }

    fn slice_scalar_mul_assign_prep(&self, b: &mut [Self::Elem], a: &Self::ElemPrep) {
        b.iter_mut().for_each(|b| *b = self.mul_prep(b, a))
    }

    fn slice_add_assign_elem_from<T: Copy>(&self, b: &mut [Self::Elem], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.slice_op_assign(b, a, |b, a| *b = self.add_elem_from(b, a))
    }

    fn slice_sub_assign_elem_from<T: Copy>(&self, b: &mut [Self::Elem], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.slice_op_assign(b, a, |b, a| *b = self.sub_elem_from(b, a))
    }

    fn slice_mul_assign_elem_from<T: Copy>(&self, b: &mut [Self::Elem], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.slice_op_assign(b, a, |b, a| *b = self.mul_elem_from(b, a))
    }

    fn slice_add_assign_iter(
        &self,
        b: &mut [Self::Elem],
        a: impl IntoIterator<Item: Borrow<Self::Elem>>,
    ) {
        self.slice_op_assign_iter(b, a, |b, a| *b = self.add(b, a.borrow()))
    }

    fn slice_neg(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        self.slice_op_assign(b, a, |b, a| *b = self.neg(a))
    }

    fn slice_add(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        self.slice_op(c, a, b, |c, a, b| *c = self.add(a, b))
    }

    fn slice_sub(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        self.slice_op(c, a, b, |c, a, b| *c = self.sub(a, b))
    }

    fn slice_mul(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        self.slice_op(c, a, b, |c, a, b| *c = self.mul(a, b))
    }

    fn slice_mul_prep(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::ElemPrep]) {
        self.slice_op(c, a, b, |c, a, b| *c = self.mul_prep(a, b))
    }

    fn slice_scalar_mul(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Elem) {
        self.slice_op_assign(c, a, |c, a| *c = self.mul(a, b))
    }

    fn slice_add_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.slice_op(c, a, b, |c, a, b| *c = self.add_elem_from(a, b))
    }

    fn slice_sub_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.slice_op(c, a, b, |c, a, b| *c = self.sub_elem_from(a, b))
    }

    fn slice_mul_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.slice_op(c, a, b, |c, a, b| *c = self.mul_elem_from(a, b))
    }

    fn slice_dot(&self, a: &[Self::Elem], b: &[Self::Elem]) -> Self::Elem {
        izip_eq!(a, b)
            .map(|(a, b)| self.mul(a, b))
            .reduce(|a, b| self.add(&a, &b))
            .unwrap_or_else(|| self.zero())
    }

    fn slice_dot_elem_from<T: Copy>(&self, a: &[Self::Elem], b: &[T]) -> Self::Elem
    where
        Self: ElemFrom<T>,
    {
        izip_eq!(a, b)
            .map(|(a, b)| self.mul_elem_from(a, b))
            .reduce(|a, b| self.add(&a, &b))
            .unwrap_or_else(|| self.zero())
    }

    fn slice_fma(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        self.slice_op(c, a, b, |c, a, b| *c = self.add(c, &self.mul(a, b)))
    }

    fn slice_scalar_fma(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Elem) {
        self.slice_op_assign(c, a, |c, a| *c = self.add(c, &self.mul(a, b)))
    }

    fn slice_scalar_fma_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[T], b: &Self::Elem)
    where
        Self: ElemFrom<T>,
    {
        self.slice_op_assign(c, a, |c, a| *c = self.add(c, &self.mul_elem_from(b, a)))
    }

    fn slice_mod_switch<M: ModulusOps>(&self, b: &mut [M::Elem], a: &[Self::Elem], mod_to: &M) {
        let delta = mod_to.modulus().as_f64() / self.modulus().as_f64();
        let mod_swtich = |a| mod_to.elem_from((self.to_u64(a) as f64 * delta).round() as u64);
        izip_eq!(b, a).for_each(|(b, a)| *b = mod_swtich(*a))
    }

    fn slice_mod_switch_odd<M: ModulusOps>(&self, b: &mut [M::Elem], a: &[Self::Elem], mod_to: &M) {
        let delta = mod_to.modulus().as_f64() / self.modulus().as_f64();
        let mod_switch_odd = |a| {
            let a = self.to_u64(a) as f64 * delta;
            let t = a.floor() as u64;
            if t == 0 {
                mod_to.elem_from(a.round() as u64)
            } else {
                mod_to.elem_from(t | 1)
            }
        };
        izip_eq!(b, a).for_each(|(b, a)| *b = mod_switch_odd(*a))
    }
}
