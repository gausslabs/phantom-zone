use crate::{
    izip_eq,
    modulus::{ElemFrom, Modulus, ModulusOps},
    poly::automorphism::AutomorphismMapView,
    util::scratch::{Scratch, ScratchOwned},
};
use core::{fmt::Debug, mem::size_of};

pub(crate) mod power_of_two;
pub(crate) mod prime;

pub use power_of_two::{
    noisy::{NoisyNativeRing, NoisyNonNativePowerOfTwoRing},
    precise::{NativeRing, NonNativePowerOfTwoRing},
};
pub use prime::{noisy::NoisyPrimeRing, precise::PrimeRing};

pub trait RingOps: ModulusOps {
    type Eval: 'static + Copy + Debug + Default + Send + Sync;
    type EvalPrep: 'static + Copy + Debug + Default + Send + Sync;

    fn new(modulus: Modulus, ring_size: usize) -> Self;

    fn ring_size(&self) -> usize;

    fn eval_size(&self) -> usize;

    fn eval_scratch_size(&self) -> usize;

    fn allocate_poly(&self) -> Vec<Self::Elem> {
        vec![Default::default(); self.ring_size()]
    }

    fn allocate_eval(&self) -> Vec<Self::Eval> {
        vec![Default::default(); self.eval_size()]
    }

    fn scratch_bytes(&self, poly: usize, eval: usize, eval_prep: usize) -> usize {
        let mut bytes;
        bytes = size_of::<Self::Elem>() * self.ring_size() * poly;
        bytes = bytes.next_multiple_of(size_of::<Self::Eval>());
        bytes += size_of::<Self::Eval>() * self.eval_size() * eval;
        bytes += size_of::<Self::Eval>() * self.eval_scratch_size();
        bytes = bytes.next_multiple_of(size_of::<Self::EvalPrep>());
        bytes += size_of::<Self::EvalPrep>() * self.eval_size() * eval_prep;
        bytes
    }

    fn allocate_scratch(&self, poly: usize, eval: usize, eval_prep: usize) -> ScratchOwned {
        ScratchOwned::allocate(self.scratch_bytes(poly, eval, eval_prep))
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

    fn take_eval_scratch<'a>(&self, scratch: &mut Scratch<'a>) -> &'a mut [Self::Eval] {
        scratch.take_slice(self.eval_scratch_size())
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem], eval_scratch: &mut [Self::Eval]);

    fn forward_elem_from<T: Copy>(
        &self,
        b: &mut [Self::Eval],
        a: &[T],
        eval_scratch: &mut [Self::Eval],
    ) where
        Self: ElemFrom<T>;

    fn forward_normalized(
        &self,
        b: &mut [Self::Eval],
        a: &[Self::Elem],
        eval_scratch: &mut [Self::Eval],
    );

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval], eval_scratch: &mut [Self::Eval]);

    fn backward_normalized(
        &self,
        b: &mut [Self::Elem],
        a: &mut [Self::Eval],
        eval_scratch: &mut [Self::Eval],
    );

    fn add_backward(
        &self,
        b: &mut [Self::Elem],
        a: &mut [Self::Eval],
        eval_scratch: &mut [Self::Eval],
    );

    fn add_backward_normalized(
        &self,
        b: &mut [Self::Elem],
        a: &mut [Self::Eval],
        eval_scratch: &mut [Self::Eval],
    );

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
        let eval_scratch = self.take_eval_scratch(&mut scratch);
        self.forward(a_eval, a, eval_scratch);
        self.forward(b_eval, b, eval_scratch);
        self.eval_mul_assign(a_eval, b_eval);
        self.backward_normalized(c, a_eval, eval_scratch);
    }

    fn poly_mul_assign(&self, b: &mut [Self::Elem], a: &[Self::Elem], mut scratch: Scratch) {
        let a_eval = self.take_eval(&mut scratch);
        let b_eval = self.take_eval(&mut scratch);
        let eval_scratch = self.take_eval_scratch(&mut scratch);
        self.forward(a_eval, a, eval_scratch);
        self.forward(b_eval, b, eval_scratch);
        self.eval_mul_assign(a_eval, b_eval);
        self.backward_normalized(b, a_eval, eval_scratch);
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
        let eval_scratch = self.take_eval_scratch(&mut scratch);
        self.forward(a_eval, a, eval_scratch);
        self.forward_elem_from(b_eval, b, eval_scratch);
        self.eval_mul_assign(a_eval, b_eval);
        self.backward_normalized(c, a_eval, eval_scratch)
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
        let eval_scratch = self.take_eval_scratch(&mut scratch);
        self.forward_elem_from(a_eval, a, eval_scratch);
        self.forward(b_eval, b, eval_scratch);
        self.eval_mul_assign(a_eval, b_eval);
        self.backward_normalized(b, a_eval, eval_scratch);
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
        let eval_scratch = self.take_eval_scratch(&mut scratch);
        self.forward(a_eval, a, eval_scratch);
        self.forward(b_eval, b, eval_scratch);
        self.eval_mul_assign(a_eval, b_eval);
        self.add_backward_normalized(c, a_eval, eval_scratch)
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
        let eval_scratch = self.take_eval_scratch(&mut scratch);
        self.forward(a_eval, a, eval_scratch);
        self.forward_elem_from(b_eval, b, eval_scratch);
        self.eval_mul_assign(a_eval, b_eval);
        self.add_backward_normalized(c, a_eval, eval_scratch)
    }

    fn poly_set_monomial(&self, a: &mut [Self::Elem], exp: i64) {
        a.fill(self.zero());
        let exp = exp.rem_euclid(2 * self.ring_size() as i64) as usize;
        if exp < self.ring_size() {
            a[exp] = self.one();
        } else {
            a[exp - self.ring_size()] = self.neg_one();
        }
    }

    fn poly_mul_monomial(&self, a: &mut [Self::Elem], exp: i64) {
        let exp = exp.rem_euclid(2 * self.ring_size() as i64) as usize;
        a.rotate_right(exp & (self.ring_size() - 1));
        if exp < self.ring_size() {
            self.slice_neg_assign(&mut a[..exp]);
        } else {
            self.slice_neg_assign(&mut a[exp - self.ring_size()..]);
        }
    }

    fn poly_add_auto(&self, a: &mut [Self::Elem], b: &[Self::Elem], auto_map: AutomorphismMapView) {
        izip_eq!(a, auto_map.iter()).for_each(|(a, (sign, idx))| {
            if sign {
                *a = self.sub(a, &b[idx]);
            } else {
                *a = self.add(a, &b[idx]);
            }
        })
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
        let mut scratch = ring.allocate_scratch(1, 1, 0);
        let mut scratch = scratch.borrow_mut();

        let b = ring.take_eval(&mut scratch);
        let c = ring.take_poly(&mut scratch);
        let eval_scratch = ring.take_eval_scratch(&mut scratch);

        ring.forward(b, a, eval_scratch);
        ring.backward_normalized(c, b, eval_scratch);
        izip_eq!(a, &*c).for_each(|(a, c)| assert_fn(a, c));

        ring.forward_normalized(b, a, eval_scratch);
        ring.backward(c, b, eval_scratch);
        izip_eq!(a, &*c).for_each(|(a, c)| assert_fn(a, c));

        c.fill_with(Default::default);
        ring.forward_normalized(b, a, eval_scratch);
        ring.add_backward(c, b, eval_scratch);
        izip_eq!(a, &*c).for_each(|(a, c)| assert_fn(a, c));
    }

    pub(crate) fn test_poly_mul<R: RingOps>(
        ring: &R,
        a: &[R::Elem],
        b: &[R::Elem],
        assert_fn: impl Fn(&R::Elem, &R::Elem),
    ) {
        let mut scratch = ring.allocate_scratch(2, 3, 0);
        let mut scratch = scratch.borrow_mut();

        let c = ring.take_poly(&mut scratch);
        nega_cyclic_schoolbook_mul(ring, c, a, b);

        let d = ring.take_poly(&mut scratch);
        ring.poly_mul(d, a, b, scratch.reborrow());
        izip_eq!(&*c, &*d).for_each(|(a, b)| assert_fn(a, b));

        let [a_eval, b_eval, d_eval] = ring.take_evals(&mut scratch);
        let eval_scratch = ring.take_eval_scratch(&mut scratch);
        ring.forward(a_eval, a, eval_scratch);
        ring.forward(b_eval, b, eval_scratch);

        d_eval.fill_with(Default::default);
        ring.eval_fma(d_eval, a_eval, b_eval);
        ring.backward_normalized(d, d_eval, eval_scratch);
        izip_eq!(&*c, &*d).for_each(|(a, b)| assert_fn(a, b));

        ring.eval_mul(d_eval, a_eval, b_eval);
        ring.backward_normalized(d, d_eval, eval_scratch);
        izip_eq!(&*c, &*d).for_each(|(a, b)| assert_fn(a, b));

        ring.eval_mul_assign(a_eval, b_eval);
        ring.backward_normalized(d, a_eval, eval_scratch);
        izip_eq!(&*c, &*d).for_each(|(a, b)| assert_fn(a, b));
    }
}
