use crate::{
    decomposer::{Decomposer, DecompositionParam},
    distribution::Sampler,
    modulus::Modulus,
};
use core::fmt::Debug;
use itertools::izip;

pub(crate) mod ffnt;
pub mod power_of_two;
pub mod prime;

pub trait ArithmeticOps {
    type Elem: Clone + Copy + Debug + Default + 'static;
    type Prep: Clone + Copy + Debug + Default + 'static;

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
    fn slice_neg_assign(&self, a: &mut [Self::Elem]) {
        a.iter_mut().for_each(|a| *a = self.neg(a))
    }

    fn slice_add_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        debug_assert_eq!(a.len(), b.len());
        izip!(b, a).for_each(|(b, a)| *b = self.add(b, a))
    }

    fn slice_sub_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        debug_assert_eq!(a.len(), b.len());
        izip!(b, a).for_each(|(b, a)| *b = self.sub(b, a))
    }

    fn slice_mul_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        debug_assert_eq!(a.len(), b.len());
        izip!(b, a).for_each(|(b, a)| *b = self.mul(b, a))
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
        debug_assert_eq!(a.len(), b.len());
        izip!(b, a).for_each(|(b, a)| *b = self.add_elem_from(b, a))
    }

    fn slice_sub_assign_elem_from<T: Copy>(&self, b: &mut [Self::Elem], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        debug_assert_eq!(a.len(), b.len());
        izip!(b, a).for_each(|(b, a)| *b = self.sub_elem_from(b, a))
    }

    fn slice_mul_assign_elem_from<T: Copy>(&self, b: &mut [Self::Elem], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        debug_assert_eq!(a.len(), b.len());
        izip!(b, a).for_each(|(b, a)| *b = self.mul_elem_from(b, a))
    }

    fn slice_neg(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        debug_assert_eq!(a.len(), b.len());
        izip!(b, a).for_each(|(b, a)| *b = self.neg(a))
    }

    fn slice_add(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.add(a, b))
    }

    fn slice_sub(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.sub(a, b))
    }

    fn slice_mul(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.mul(a, b))
    }

    fn slice_scalar_mul(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Elem) {
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a).for_each(|(c, a)| *c = self.mul(a, b))
    }

    fn slice_scalar_mul_prep(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Prep) {
        izip!(c, a).for_each(|(c, a)| *c = self.mul_prep(a, b))
    }

    fn slice_add_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.add_elem_from(a, b))
    }

    fn slice_sub_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.sub_elem_from(a, b))
    }

    fn slice_mul_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.mul_elem_from(a, b))
    }

    fn slice_dot(&self, a: &[Self::Elem], b: &[Self::Elem]) -> Self::Elem {
        debug_assert_eq!(a.len(), b.len());
        izip!(a, b)
            .map(|(a, b)| self.mul(a, b))
            .reduce(|a, b| self.add(&a, &b))
            .unwrap_or_else(|| self.zero())
    }

    fn slice_dot_elem_from<T: Copy>(&self, a: &[Self::Elem], b: &[T]) -> Self::Elem
    where
        Self: ElemFrom<T>,
    {
        debug_assert_eq!(a.len(), b.len());
        izip!(a, b)
            .map(|(a, b)| self.mul_elem_from(a, b))
            .reduce(|a, b| self.add(&a, &b))
            .unwrap_or_else(|| self.zero())
    }

    fn slice_fma(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Elem]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.add(c, &self.mul(a, b)))
    }

    fn slice_scalar_fma(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Elem) {
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a).for_each(|(c, a)| *c = self.add(c, &self.mul(a, b)))
    }

    fn slice_fma_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.add(c, &self.mul_elem_from(a, b)))
    }

    fn matrix_fma_prep<'a>(
        &self,
        c: &mut [Self::Elem],
        a: impl IntoIterator<Item = &'a [Self::Elem]>,
        b: impl IntoIterator<Item = &'a [Self::Prep]>,
    ) {
        let (mut a, mut b) = (a.into_iter(), b.into_iter());
        izip!(a.by_ref(), b.by_ref()).for_each(|(a, b)| {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), c.len());
            izip!(&mut *c, a, b).for_each(|(c, a, b)| *c = self.add(c, &self.mul_prep(a, b)))
        });
        debug_assert!(a.next().is_none());
        debug_assert!(b.next().is_none());
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
    type Eval: Clone + Copy + Debug + Default + 'static;

    fn new(modulus: Modulus, ring_size: usize) -> Self;

    fn eval_ops(&self) -> &impl SliceOps<Elem = Self::Eval>;

    fn ring_size(&self) -> usize;

    fn eval_size(&self) -> usize;

    fn scratch_size(&self) -> usize {
        2 * self.eval_size()
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem]);

    fn forward_normalized(&self, b: &mut [Self::Eval], a: &[Self::Elem]);

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn poly_mul(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: &[Self::Elem],
        scratch: &mut [Self::Eval],
    ) {
        debug_assert_eq!(a.len(), self.ring_size());
        debug_assert_eq!(b.len(), self.ring_size());
        debug_assert_eq!(c.len(), self.ring_size());
        debug_assert_eq!(scratch.len(), 2 * self.eval_size());
        let (a_eval, b_eval) = scratch.split_at_mut(self.eval_size());
        self.forward(a_eval, a);
        self.forward(b_eval, b);
        self.eval_ops().slice_mul_assign(a_eval, b_eval);
        self.backward_normalized(c, a_eval)
    }

    fn decomposer(&self, decomposition_param: DecompositionParam) -> impl Decomposer<Self::Elem>;
}

#[cfg(test)]
mod test {
    use crate::ring::{ArithmeticOps, RingOps};
    use itertools::izip;

    fn nega_cyclic_schoolbook_mul<T: ArithmeticOps>(
        op: &T,
        a: &[T::Elem],
        b: &[T::Elem],
    ) -> Vec<T::Elem> {
        let n = a.len();
        let mut c = vec![Default::default(); n];
        izip!(0.., a.iter()).for_each(|(i, a)| {
            izip!(0.., b.iter()).for_each(|(j, b)| {
                if i + j < n {
                    c[i + j] = op.add(&c[i + j], &op.mul(a, b));
                } else {
                    c[i + j - n] = op.sub(&c[i + j - n], &op.mul(a, b));
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
        izip!(a, &c).for_each(|(a, c)| assert_fn(a, c));

        b.fill_with(Default::default);
        c.fill_with(Default::default);
        ring.forward_normalized(&mut b, a);
        ring.backward(&mut c, &mut b);
        izip!(a, &c).for_each(|(a, c)| assert_fn(a, c));

        b.fill_with(Default::default);
        c.fill_with(Default::default);
        ring.forward_normalized(&mut b, a);
        ring.add_backward(&mut c, &mut b);
        izip!(a, &c).for_each(|(a, c)| assert_fn(a, c));
    }

    pub(crate) fn test_poly_mul<R: RingOps>(
        ring: &R,
        a: &[R::Elem],
        b: &[R::Elem],
        assert_fn: impl Fn(&R::Elem, &R::Elem),
    ) {
        let mut c = vec![Default::default(); ring.ring_size()];
        let mut scratch = vec![Default::default(); ring.scratch_size()];
        ring.poly_mul(&mut c, a, b, &mut scratch);
        izip!(&c, nega_cyclic_schoolbook_mul(ring, a, b)).for_each(|(a, b)| assert_fn(a, &b));
    }
}
