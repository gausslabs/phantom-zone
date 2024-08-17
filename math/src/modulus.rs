use crate::distribution::DistributionSized;
use core::{fmt::Debug, hash::Hash, iter::successors};
use rand_distr::Distribution;

pub(crate) mod power_of_two;
pub(crate) mod prime;

pub use power_of_two::{Native, NonNativePowerOfTwo};
pub use prime::Prime;

#[derive(Clone, Copy, Debug)]
pub enum Modulus {
    Native(Native),
    NonNativePowerOfTwo(NonNativePowerOfTwo),
    Prime(Prime),
}

impl Modulus {
    pub fn native() -> Self {
        Native::native().into()
    }

    pub fn bits(&self) -> usize {
        match self {
            Self::Native(inner) => inner.bits(),
            Self::NonNativePowerOfTwo(inner) => inner.bits(),
            Self::Prime(inner) => inner.bits(),
        }
    }

    pub fn max(&self) -> u64 {
        match self {
            Self::Native(inner) => inner.max(),
            Self::NonNativePowerOfTwo(inner) => inner.max(),
            Self::Prime(inner) => inner.max(),
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Native(inner) => inner.as_f64(),
            Self::NonNativePowerOfTwo(inner) => inner.as_f64(),
            Self::Prime(inner) => inner.as_f64(),
        }
    }

    pub fn to_i64(&self, v: u64) -> i64 {
        match self {
            Self::Native(inner) => inner.to_i64(v),
            Self::NonNativePowerOfTwo(inner) => inner.to_i64(v),
            Self::Prime(inner) => inner.to_i64(v),
        }
    }
}

pub trait ModulusOps: Clone + Debug + Send + Sync {
    type Elem: 'static + Copy + Debug + Default + Send + Sync + Eq + Ord + Hash;
    type Prep: 'static + Copy + Debug + Default + Send + Sync;

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
        let a = self.prepare(a);
        successors(Some(self.one()), move |v| self.mul_prep(v, &a).into())
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

    fn prepare(&self, a: &Self::Elem) -> Self::Prep;

    fn mul_prep(&self, a: &Self::Elem, b: &Self::Prep) -> Self::Elem;

    fn to_u64(&self, a: Self::Elem) -> u64
    where
        Self: ElemTo<u64>,
    {
        self.elem_to(a)
    }

    fn to_i64(&self, a: Self::Elem) -> i64
    where
        Self: ElemTo<i64>,
    {
        self.elem_to(a)
    }

    fn mod_switch<M>(&self, a: &Self::Elem, mod_to: &M) -> M::Elem
    where
        Self: ElemTo<u64>,
        M: ElemFrom<u64>,
    {
        let delta = mod_to.modulus().as_f64() / self.modulus().as_f64();
        mod_to.elem_from((self.to_u64(*a) as f64 * delta).round() as _)
    }
}

pub trait ElemFrom<T>: ModulusOps {
    fn elem_from(&self, v: T) -> Self::Elem;
}

pub trait ElemTo<T>: ModulusOps {
    fn elem_to(&self, v: Self::Elem) -> T;
}
