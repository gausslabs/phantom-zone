use crate::{
    decomposer::Decomposer,
    distribution::Sampler,
    izip_eq,
    misc::scratch::{Scratch, ScratchOwned},
    modulus::Modulus,
};
use core::{borrow::Borrow, fmt::Debug, mem::size_of};
use itertools::izip;

pub(crate) mod power_of_two;
pub(crate) mod prime;

pub use power_of_two::{
    noisy::{NoisyNativeRing, NoisyNonNativePowerOfTwoRing},
    precise::{NativeRing, NonNativePowerOfTwoRing},
};
pub use prime::{noisy::NoisyPrimeRing, precise::PrimeRing};

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

    fn slice_prepare(&self, b: &mut [Self::Prep], a: &[Self::Elem]) {
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

    fn slice_add_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        self.slice_op_assign(b, a, |b, a| *b = self.add(b, a))
    }

    fn slice_sub_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        self.slice_op_assign(b, a, |b, a| *b = self.sub(b, a))
    }

    fn slice_mul_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem]) {
        self.slice_op_assign(b, a, |b, a| *b = self.mul(b, a))
    }

    fn slice_mul_assign_prep(&self, b: &mut [Self::Elem], a: &[Self::Prep]) {
        self.slice_op_assign(b, a, |b, a| *b = self.mul_prep(b, a))
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

    fn slice_mul_prep(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Prep]) {
        self.slice_op(c, a, b, |c, a, b| *c = self.mul_prep(a, b))
    }

    fn slice_scalar_mul(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Elem) {
        self.slice_op_assign(c, a, |c, a| *c = self.mul(a, b))
    }

    fn slice_scalar_mul_prep(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Prep) {
        self.slice_op_assign(c, a, |c, a| *c = self.mul_prep(a, b))
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

    fn slice_fma_prep(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[Self::Prep]) {
        self.slice_op(c, a, b, |c, a, b| *c = self.add(c, &self.mul_prep(a, b)))
    }

    fn slice_scalar_fma(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &Self::Elem) {
        self.slice_op_assign(c, a, |c, a| *c = self.add(c, &self.mul(a, b)))
    }

    fn slice_fma_elem_from<T: Copy>(&self, c: &mut [Self::Elem], a: &[Self::Elem], b: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.slice_op(c, a, b, |c, a, b| {
            *c = self.add(c, &self.mul_elem_from(a, b))
        })
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
    type EvalPrep: Copy + Debug + Default + 'static;
    type Decomposer: Decomposer<Self::Elem>;

    fn new(modulus: Modulus, ring_size: usize) -> Self;

    fn modulus(&self) -> Modulus;

    fn ring_size(&self) -> usize;

    fn eval_size(&self) -> usize;

    fn allocate_poly(&self) -> Vec<Self::Elem> {
        vec![Default::default(); self.ring_size()]
    }

    fn allocate_eval(&self) -> Vec<Self::Eval> {
        vec![Default::default(); self.eval_size()]
    }

    fn allocate_scratch(&self, poly: usize, eval: usize) -> ScratchOwned {
        let poly_bytes = size_of::<Self::Elem>() * self.ring_size() * poly;
        let eval_bytes = size_of::<Self::Eval>() * self.eval_size() * eval;
        ScratchOwned::allocate(poly_bytes + eval_bytes)
    }

    fn take_poly<'a>(&self, scratch: &mut Scratch<'a>) -> &'a mut [Self::Elem] {
        scratch.take_slice(self.ring_size())
    }

    fn take_polys<'a, const N: usize>(
        &self,
        scratch: &mut Scratch<'a>,
    ) -> [&'a mut [Self::Elem]; N] {
        scratch.take_slice_array(self.ring_size())
    }

    fn take_eval<'a>(&self, scratch: &mut Scratch<'a>) -> &'a mut [Self::Eval] {
        scratch.take_slice(self.eval_size())
    }

    fn take_evals<'a, const N: usize>(
        &self,
        scratch: &mut Scratch<'a>,
    ) -> [&'a mut [Self::Eval]; N] {
        scratch.take_slice_array(self.eval_size())
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem]);

    fn forward_elem_from<T: Copy>(&self, b: &mut [Self::Eval], a: &[T])
    where
        Self: ElemFrom<T>;

    fn forward_normalized(&self, b: &mut [Self::Eval], a: &[Self::Elem]);

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn add_backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]);

    fn eval_prepare(&self, b: &mut [Self::EvalPrep], a: &[Self::Eval]);

    fn eval_mul(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]);

    fn eval_mul_assign(&self, b: &mut [Self::Eval], a: &[Self::Eval]);

    fn eval_fma(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]);

    fn eval_mul_prep(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::EvalPrep]);

    fn eval_mul_assign_prep(&self, b: &mut [Self::Eval], a: &[Self::EvalPrep]);

    fn eval_fma_prep(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::EvalPrep]);

    fn poly_mul(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: &[Self::Elem],
        mut scratch: Scratch,
    ) {
        let a_eval = self.take_eval(&mut scratch);
        let b_eval = self.take_eval(&mut scratch);
        self.forward(a_eval, a);
        self.forward(b_eval, b);
        self.eval_mul_assign(a_eval, b_eval);
        self.backward_normalized(c, a_eval);
    }

    fn poly_mul_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem], mut scratch: Scratch) {
        let a_eval = self.take_eval(&mut scratch);
        let b_eval = self.take_eval(&mut scratch);
        self.forward(a_eval, a);
        self.forward(b_eval, b);
        self.eval_mul_assign(a_eval, b_eval);
        self.backward_normalized(b, a_eval);
    }

    fn poly_mul_elem_from<T: Copy>(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: &[T],
        mut scratch: Scratch,
    ) where
        Self: ElemFrom<T>,
    {
        let a_eval = self.take_eval(&mut scratch);
        let b_eval = self.take_eval(&mut scratch);
        self.forward(a_eval, a);
        self.forward_elem_from(b_eval, b);
        self.eval_mul_assign(a_eval, b_eval);
        self.backward_normalized(c, a_eval)
    }

    fn poly_mul_assign_elem_from<T: Copy>(
        &self,
        b: &mut [Self::Elem],
        a: &[T],
        mut scratch: Scratch,
    ) where
        Self: ElemFrom<T>,
    {
        let a_eval = self.take_eval(&mut scratch);
        let b_eval = self.take_eval(&mut scratch);
        self.forward_elem_from(a_eval, a);
        self.forward(b_eval, b);
        self.eval_mul_assign(a_eval, b_eval);
        self.backward_normalized(b, a_eval);
    }

    fn poly_fma(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: &[Self::Elem],
        mut scratch: Scratch,
    ) {
        let a_eval = self.take_eval(&mut scratch);
        let b_eval = self.take_eval(&mut scratch);
        self.forward(a_eval, a);
        self.forward(b_eval, b);
        self.eval_mul_assign(a_eval, b_eval);
        self.add_backward_normalized(c, a_eval)
    }

    fn poly_fma_elem_from<T: Copy>(
        &self,
        c: &mut [Self::Elem],
        a: &[Self::Elem],
        b: &[T],
        mut scratch: Scratch,
    ) where
        Self: ElemFrom<T>,
    {
        let a_eval = self.take_eval(&mut scratch);
        let b_eval = self.take_eval(&mut scratch);
        self.forward(a_eval, a);
        self.forward_elem_from(b_eval, b);
        self.eval_mul_assign(a_eval, b_eval);
        self.add_backward_normalized(c, a_eval)
    }
}

#[cfg(test)]
mod test {
    use crate::{izip_eq, poly::test::nega_cyclic_schoolbook_mul, ring::RingOps};

    pub(crate) fn test_round_trip<R: RingOps>(
        ring: &R,
        a: &[R::Elem],
        assert_fn: impl Fn(&R::Elem, &R::Elem),
    ) {
        let b = &mut ring.allocate_eval();
        let c = &mut ring.allocate_poly();

        ring.forward(b, a);
        ring.backward_normalized(c, b);
        izip_eq!(a, &*c).for_each(|(a, c)| assert_fn(a, c));

        ring.forward_normalized(b, a);
        ring.backward(c, b);
        izip_eq!(a, &*c).for_each(|(a, c)| assert_fn(a, c));

        c.fill_with(Default::default);
        ring.forward_normalized(b, a);
        ring.add_backward(c, b);
        izip_eq!(a, &*c).for_each(|(a, c)| assert_fn(a, c));
    }

    pub(crate) fn test_poly_mul<R: RingOps>(
        ring: &R,
        a: &[R::Elem],
        b: &[R::Elem],
        assert_fn: impl Fn(&R::Elem, &R::Elem),
    ) {
        let mut scratch = ring.allocate_scratch(2, 3);
        let scratch = &mut scratch.borrow_mut();

        let c = ring.take_poly(scratch);
        nega_cyclic_schoolbook_mul(ring, c, a, b);

        let d = ring.take_poly(scratch);
        ring.poly_mul(d, a, b, scratch.reborrow());
        izip_eq!(&*c, &*d).for_each(|(a, b)| assert_fn(a, b));

        let [a_eval, b_eval] = ring.take_evals(scratch);
        ring.forward(a_eval, a);
        ring.forward(b_eval, b);

        let d_eval = ring.take_eval(scratch);
        ring.eval_fma(d_eval, a_eval, b_eval);
        ring.backward_normalized(d, d_eval);
        izip_eq!(&*c, &*d).for_each(|(a, b)| assert_fn(a, b));

        ring.eval_mul(d_eval, a_eval, b_eval);
        ring.backward_normalized(d, d_eval);
        izip_eq!(&*c, &*d).for_each(|(a, b)| assert_fn(a, b));

        ring.eval_mul_assign(a_eval, b_eval);
        ring.backward_normalized(d, a_eval);
        izip_eq!(&*c, &*d).for_each(|(a, b)| assert_fn(a, b));
    }
}
