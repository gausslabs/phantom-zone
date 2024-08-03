use crate::{decomposer::Decomposer, distribution::Sampler, izip_eq, modulus::Modulus};
use core::{borrow::Borrow, fmt::Debug};
use itertools::izip;

pub(crate) mod ffnt;
pub mod power_of_two;
pub mod prime;

pub trait ArithmeticOps {
    type Elem: Copy + Debug + Default + 'static;
    type Prep: Copy + Debug + Default + 'static;

    fn zero(&self) -> Self::Elem;

    fn one(&self) -> Self::Elem;

    fn neg_one(&self) -> Self::Elem;

    fn neg(&self, a: &Self::Elem) -> Self::Elem;

    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

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
}

pub trait ElemFrom<T>: ArithmeticOps {
    fn elem_from(&self, v: T) -> Self::Elem;
}

pub trait ElemTo<T>: ArithmeticOps {
    fn elem_to(&self, v: Self::Elem) -> T;
}

pub trait SliceOps: ArithmeticOps {
    fn slice_elem_from<T: Copy>(&self, b: &mut [Self::Elem], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.elem_from(*a))
    }

    fn slice_elem_from_iter<T: Copy>(
        &self,
        b: &mut [Self::Elem],
        a: impl IntoIterator<Item: Borrow<T>>,
    ) where
        Self: ElemFrom<T>,
    {
        izip!(b, a).for_each(|(b, a)| *b = self.elem_from(*a.borrow()))
    }

    fn slice_neg_assign(&self, a: &mut [Self::Elem]) {
        a.iter_mut().for_each(|a| *a = self.neg(a))
    }

    fn slice_add_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.add(b, a))
    }

    fn slice_sub_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.sub(b, a))
    }

    fn slice_mul_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.mul(b, a))
    }

    fn slice_scalar_mul_assign(&self, b: &mut [Self::Elem], a: &Self::Elem) {
        b.iter_mut().for_each(|b| *b = self.mul(b, a))
    }

    fn slice_scalar_mul_assign_prep(&self, b: &mut [Self::Elem], a: &Self::Prep) {
        b.iter_mut().for_each(|b| *b = self.mul_prep(b, a))
    }

    fn slice_add_assign_elem_from<T: Copy>(&self, b: &mut [Self::Elem], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.add_elem_from(b, a))
    }

    fn slice_sub_assign_elem_from<T: Copy>(&self, b: &mut [Self::Elem], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.sub_elem_from(b, a))
    }

    fn slice_mul_assign_elem_from<T: Copy>(&self, b: &mut [Self::Elem], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.mul_elem_from(b, a))
    }

    fn slice_add_assign_iter(
        &self,
        b: &mut [Self::Elem],
        a: impl IntoIterator<Item: Borrow<Self::Elem>>,
    ) {
        izip!(b, a).for_each(|(b, a)| *b = self.add(b, a.borrow()))
    }

    fn slice_neg(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        izip_eq!(b, a).for_each(|(b, a)| *b = self.neg(a))
    }

    fn slice_add(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        izip_eq!(c, a, b).for_each(|(c, a, b): (&mut _, _, _)| *c = self.add(a, b))
    }

    fn slice_sub(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        izip_eq!(c, a, b).for_each(|(c, a, b): (&mut _, _, _)| *c = self.sub(a, b))
    }

    fn slice_mul(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        izip_eq!(c, a, b).for_each(|(c, a, b): (&mut _, _, _)| *c = self.mul(a, b))
    }

    fn slice_scalar_mul(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Elem) {
        izip_eq!(c, a).for_each(|(c, a)| *c = self.mul(a, b))
    }

    fn slice_scalar_mul_prep(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Prep) {
        izip_eq!(c, a).for_each(|(c, a)| *c = self.mul_prep(a, b))
    }

    fn slice_add_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        izip_eq!(c, a, b).for_each(|(c, a, b)| *c = self.add_elem_from(a, b))
    }

    fn slice_sub_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        izip_eq!(c, a, b).for_each(|(c, a, b)| *c = self.sub_elem_from(a, b))
    }

    fn slice_mul_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        izip_eq!(c, a, b).for_each(|(c, a, b)| *c = self.mul_elem_from(a, b))
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
        izip_eq!(c, a, b).for_each(|(c, a, b)| *c = self.add(c, &self.mul(a, b)))
    }

    fn slice_scalar_fma(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Elem) {
        izip_eq!(c, a).for_each(|(c, a)| *c = self.add(c, &self.mul(a, b)))
    }

    fn slice_fma_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        izip_eq!(c, a, b).for_each(|(c, a, b)| *c = self.add(c, &self.mul_elem_from(a, b)))
    }

    fn matrix_fma_prep<'a>(
        &self,
        c: &mut [Self::Elem],
        a: impl IntoIterator<Item = &'a [Self::Elem]>,
        b: impl IntoIterator<Item = &'a [Self::Prep]>,
    ) {
        izip_eq!(a, b).for_each(|(a, b)| {
            izip_eq!(&mut *c, a, b).for_each(|(c, a, b)| *c = self.add(c, &self.mul_prep(a, b)))
        });
    }
}

pub trait RingOps:
    SliceOps
    + Sampler
    + ElemFrom<u64>
    + ElemFrom<i64>
    + ElemFrom<u32>
    + ElemFrom<i32>
    + ElemTo<u64>
    + ElemTo<i64>
{
    type Eval: Copy + Debug + Default + 'static;
    type Decomposer: Decomposer<Self::Elem>;

    fn new(modulus: Modulus, ring_size: usize) -> Self;

    fn modulus(&self) -> Modulus;

    fn ring_size(&self) -> usize;

    fn eval_size(&self) -> usize;

    fn scratch_size(&self) -> usize {
        2 * self.eval_size()
    }

    fn allocate_scratch(&self) -> Vec<Self::Eval> {
        vec![Default::default(); self.scratch_size()]
    }

    fn eval(&self) -> &impl SliceOps<Elem = Self::Eval>;

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem]);

    fn forward_elem_from<T: Copy>(&self, b: &mut [Self::Eval], a: &[T])
    where
        Self: ElemFrom<T>;

    fn forward_normalized(&self, b: &mut [Self::Eval], a: &[Self::Elem]);

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn add_backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn poly_mul(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: &[Self::Elem],
        scratch: &mut [Self::Eval],
    ) {
        let (a_eval, b_eval) = scratch.split_at_mut(self.eval_size());
        self.forward(a_eval, a);
        self.forward(b_eval, b);
        self.eval().slice_mul_assign(a_eval, &*b_eval);
        self.backward_normalized(c, a_eval)
    }

    fn poly_mul_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem], scratch: &mut [Self::Eval]) {
        let (a_eval, b_eval) = scratch.split_at_mut(self.eval_size());
        self.forward(a_eval, a);
        self.forward(b_eval, b);
        self.eval().slice_mul_assign(a_eval, &*b_eval);
        self.backward_normalized(b, a_eval)
    }

    fn poly_mul_elem_from<T: Copy>(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: &[T],
        scratch: &mut [Self::Eval],
    ) where
        Self: ElemFrom<T>,
    {
        let (a_eval, b_eval) = scratch.split_at_mut(self.eval_size());
        self.forward(a_eval, a);
        self.forward_elem_from(b_eval, b);
        self.eval().slice_mul_assign(a_eval, &*b_eval);
        self.backward_normalized(c, a_eval)
    }

    fn poly_mul_assign_elem_from<T: Copy>(
        &self,
        b: &mut [Self::Elem],
        a: &[T],
        scratch: &mut [Self::Eval],
    ) where
        Self: ElemFrom<T>,
    {
        let (a_eval, b_eval) = scratch.split_at_mut(self.eval_size());
        self.forward_elem_from(a_eval, a);
        self.forward(b_eval, b);
        self.eval().slice_mul_assign(a_eval, &*b_eval);
        self.backward_normalized(b, a_eval)
    }

    fn poly_fma(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: &[Self::Elem],
        scratch: &mut [Self::Eval],
    ) {
        let (a_eval, b_eval) = scratch.split_at_mut(self.eval_size());
        self.forward(a_eval, a);
        self.forward(b_eval, b);
        self.eval().slice_mul_assign(a_eval, &*b_eval);
        self.add_backward_normalized(c, a_eval)
    }

    fn poly_fma_elem_from<T: Copy>(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: &[T],
        scratch: &mut [Self::Eval],
    ) where
        Self: ElemFrom<T>,
    {
        let (a_eval, b_eval) = scratch.split_at_mut(self.eval_size());
        self.forward(a_eval, a);
        self.forward_elem_from(b_eval, b);
        self.eval().slice_mul_assign(a_eval, &*b_eval);
        self.add_backward_normalized(c, a_eval)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        izip_eq,
        ring::{ArithmeticOps, RingOps},
    };
    use itertools::izip;

    fn nega_cyclic_schoolbook_mul<T: ArithmeticOps>(
        arith: &T,
        a: &[T::Elem],
        b: &[T::Elem],
    ) -> Vec<T::Elem> {
        let n = a.len();
        let mut c = vec![Default::default(); n];
        izip!(0.., a.iter()).for_each(|(i, a)| {
            izip!(0.., b.iter()).for_each(|(j, b)| {
                if i + j < n {
                    c[i + j] = arith.add(&c[i + j], &arith.mul(a, b));
                } else {
                    c[i + j - n] = arith.sub(&c[i + j - n], &arith.mul(a, b));
                }
            })
        });
        c
    }

    pub(crate) fn test_round_trip<R: RingOps>(
        ring: &R,
        a: &[R::Elem],
        assert_fn: impl Fn(&R::Elem, &R::Elem),
    ) {
        let mut b = vec![Default::default(); ring.eval_size()];
        let mut c = vec![Default::default(); ring.ring_size()];

        ring.forward(&mut b, a);
        ring.backward_normalized(&mut c, &mut b);
        izip_eq!(a, &c).for_each(|(a, c)| assert_fn(a, c));

        b.fill_with(Default::default);
        c.fill_with(Default::default);
        ring.forward_normalized(&mut b, a);
        ring.backward(&mut c, &mut b);
        izip_eq!(a, &c).for_each(|(a, c)| assert_fn(a, c));

        b.fill_with(Default::default);
        c.fill_with(Default::default);
        ring.forward_normalized(&mut b, a);
        ring.add_backward(&mut c, &mut b);
        izip_eq!(a, &c).for_each(|(a, c)| assert_fn(a, c));
    }

    pub(crate) fn test_poly_mul<R: RingOps>(
        ring: &R,
        a: &[R::Elem],
        b: &[R::Elem],
        assert_fn: impl Fn(&R::Elem, &R::Elem),
    ) {
        let mut c = vec![Default::default(); ring.ring_size()];
        ring.poly_mul(&mut c, a, b, &mut ring.allocate_scratch());
        izip_eq!(&c, nega_cyclic_schoolbook_mul(ring, a, b)).for_each(|(a, b)| assert_fn(a, &b));
    }
}
