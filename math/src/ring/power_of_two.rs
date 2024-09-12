use crate::{
    distribution::{DistributionSized, Sampler},
    modulus::{
        power_of_two::{f64_mod_u64, PowerOfTwo},
        ElemFrom, ElemOps, ElemTo, Modulus, ModulusOps,
    },
    poly::ffnt::Ffnt,
    ring::RingOps,
};
use core::fmt::Debug;
use num_complex::Complex64;
use rand::distributions::Distribution;
use unroll::unroll_for_loops;

pub mod noisy;
pub mod precise;

#[derive(Clone, Debug)]
pub struct PowerOfTwoRing<const NATIVE: bool, const LIMB_BITS: usize> {
    q: PowerOfTwo<NATIVE>,
    fft: Ffnt,
}

impl<const NATIVE: bool, const LIMB_BITS: usize> PowerOfTwoRing<NATIVE, LIMB_BITS> {
    fn new(q: PowerOfTwo<NATIVE>, fft: Ffnt) -> Self {
        Self { q, fft }
    }
}

impl<const NATIVE: bool, const LIMB_BITS: usize> ElemOps for PowerOfTwoRing<NATIVE, LIMB_BITS> {
    type Elem = <PowerOfTwo<NATIVE> as ElemOps>::Elem;
}

impl<const NATIVE: bool, const LIMB_BITS: usize> ModulusOps for PowerOfTwoRing<NATIVE, LIMB_BITS> {
    type ElemPrep = <PowerOfTwo<NATIVE> as ModulusOps>::ElemPrep;
    type Decomposer = <PowerOfTwo<NATIVE> as ModulusOps>::Decomposer;

    fn new(modulus: Modulus) -> Self {
        Self::new(modulus.try_into().unwrap(), Ffnt::default())
    }

    #[inline(always)]
    fn modulus(&self) -> Modulus {
        self.q.modulus()
    }

    #[inline(always)]
    fn uniform_distribution(
        &self,
    ) -> impl Distribution<Self::Elem> + DistributionSized<Self::Elem> {
        self.q.uniform_distribution()
    }

    #[inline(always)]
    fn zero(&self) -> Self::Elem {
        self.q.zero()
    }

    #[inline(always)]
    fn one(&self) -> Self::Elem {
        self.q.one()
    }

    #[inline(always)]
    fn neg_one(&self) -> Self::Elem {
        self.q.neg_one()
    }

    #[inline(always)]
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        self.q.neg(a)
    }

    #[inline(always)]
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.q.add(a, b)
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.q.sub(a, b)
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.q.mul(a, b)
    }

    #[inline(always)]
    fn powers(&self, a: &Self::Elem) -> impl Iterator<Item = Self::Elem> {
        self.q.powers(a)
    }

    #[inline(always)]
    fn inv(&self, a: &Self::Elem) -> Option<Self::Elem> {
        self.q.inv(a)
    }

    #[inline(always)]
    fn prepare(&self, a: &Self::Elem) -> Self::ElemPrep {
        self.q.prepare(a)
    }

    #[inline(always)]
    fn mul_prep(&self, a: &Self::Elem, b: &Self::ElemPrep) -> Self::Elem {
        self.q.mul_prep(a, b)
    }
}

impl<const NATIVE: bool, const LIMB_BITS: usize> ElemFrom<u64>
    for PowerOfTwoRing<NATIVE, LIMB_BITS>
{
    #[inline(always)]
    fn elem_from(&self, v: u64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<const NATIVE: bool, const LIMB_BITS: usize> ElemFrom<i64>
    for PowerOfTwoRing<NATIVE, LIMB_BITS>
{
    #[inline(always)]
    fn elem_from(&self, v: i64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<const NATIVE: bool, const LIMB_BITS: usize> ElemFrom<u32>
    for PowerOfTwoRing<NATIVE, LIMB_BITS>
{
    #[inline(always)]
    fn elem_from(&self, v: u32) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<const NATIVE: bool, const LIMB_BITS: usize> ElemFrom<i32>
    for PowerOfTwoRing<NATIVE, LIMB_BITS>
{
    #[inline(always)]
    fn elem_from(&self, v: i32) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<const NATIVE: bool, const LIMB_BITS: usize> ElemFrom<f64>
    for PowerOfTwoRing<NATIVE, LIMB_BITS>
{
    #[inline(always)]
    fn elem_from(&self, v: f64) -> Self::Elem {
        self.q.elem_from(v)
    }
}

impl<const NATIVE: bool, const LIMB_BITS: usize> ElemTo<u64> for PowerOfTwoRing<NATIVE, LIMB_BITS> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> u64 {
        self.q.elem_to(v)
    }
}

impl<const NATIVE: bool, const LIMB_BITS: usize> ElemTo<i64> for PowerOfTwoRing<NATIVE, LIMB_BITS> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> i64 {
        self.q.elem_to(v)
    }
}

impl<const NATIVE: bool, const LIMB_BITS: usize> ElemTo<f64> for PowerOfTwoRing<NATIVE, LIMB_BITS> {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> f64 {
        self.q.elem_to(v)
    }
}

impl<const NATIVE: bool, const LIMB_BITS: usize> Sampler for PowerOfTwoRing<NATIVE, LIMB_BITS> {}

impl<const NATIVE: bool, const LIMB_BITS: usize> PowerOfTwoRing<NATIVE, LIMB_BITS> {
    const SINGLE_LIMB: bool = LIMB_BITS == 64;

    const LIMBS: usize = {
        assert!(LIMB_BITS >= 8 && LIMB_BITS <= 64);
        (u64::BITS as usize).div_ceil(LIMB_BITS)
    };

    const LIMB_MASK: u64 = ((1u128 << LIMB_BITS) - 1) as u64;

    const LEFTOVER_MASK: u64 = !Self::LIMB_MASK;

    #[inline(always)]
    fn limb<'a, const I: usize>(&self, a: &'a [Complex64]) -> &'a [Complex64] {
        &a[I * self.fft.fft_size()..(I + 1) * self.fft.fft_size()]
    }

    #[inline(always)]
    fn limb_mut<'a, const I: usize>(&self, a: &'a mut [Complex64]) -> &'a mut [Complex64] {
        &mut a[I * self.fft.fft_size()..(I + 1) * self.fft.fft_size()]
    }

    #[inline(always)]
    fn split_limbs_at_mut<'a, const I: usize>(
        &self,
        a: &'a mut [Complex64],
    ) -> (&'a mut [Complex64], &'a mut [Complex64]) {
        a.split_at_mut(I * self.fft.fft_size())
    }

    #[inline(always)]
    fn forward_limb<T: Copy, const I: usize>(
        &self,
        b: &mut [Complex64],
        a: &[T],
        scratch: &mut [Complex64],
        leftover: &mut u64,
    ) where
        Self: ElemFrom<T, Elem = u64>,
    {
        if I != 0 && *leftover == 0 {
            set_limb_zero(self.limb_mut::<I>(b));
            return;
        }
        *leftover = 0;
        let to_f64 = |a: &_| {
            let a = self.elem_from(*a);
            let sign = a >> (self.q.bits() - 1);
            let mask = sign.wrapping_neg();
            let a = (a ^ mask).wrapping_add(sign);
            *leftover |= a & self.q.mask() & Self::LEFTOVER_MASK;
            let limb = (a >> (I * LIMB_BITS)) & Self::LIMB_MASK;
            (limb ^ mask).wrapping_add(sign) as i64 as _
        };
        self.fft.forward(self.limb_mut::<I>(b), a, to_f64, scratch);
    }

    #[inline(always)]
    fn add_backward_limb<const I: usize>(
        &self,
        b: &mut [u64],
        a: &mut [Complex64],
        scratch: &mut [Complex64],
    ) {
        if is_limb_zero(a) {
            return;
        }
        let add_from_f64 = |b: &mut u64, a| {
            *b = if I == Self::LIMBS - 1 {
                self.q
                    .reduce(b.wrapping_add(from_small_f64(a) << (I * LIMB_BITS)))
            } else {
                b.wrapping_add(from_small_f64(a) << (I * LIMB_BITS))
            }
        };
        self.fft
            .add_backward(b, self.limb_mut::<I>(a), add_from_f64, scratch);
    }

    #[inline(always)]
    fn eval_mul_assign_limb<const I: usize, const J: usize>(
        &self,
        b: &mut [Complex64],
        a: &[Complex64],
    ) {
        let (b, a) = (self.limb_mut::<I>(b), self.limb::<J>(a));
        if is_limb_zero(b) || is_limb_zero(a) {
            b.fill_with(Default::default);
        } else {
            self.fft.eval_mul_assign(b, a);
        }
    }

    #[inline(always)]
    fn eval_mul_limb<const I: usize, const J: usize, const K: usize>(
        &self,
        c: &mut [Complex64],
        a: &[Complex64],
        b: &[Complex64],
    ) {
        let (c, a, b) = (self.limb_mut::<I>(c), self.limb::<J>(a), self.limb::<K>(b));
        if is_limb_zero(b) || is_limb_zero(a) {
            c.fill_with(Default::default);
        } else {
            self.fft.eval_mul(c, b, a);
        }
    }

    #[inline(always)]
    fn eval_fma_limb<const I: usize, const J: usize, const K: usize>(
        &self,
        c: &mut [Complex64],
        a: &[Complex64],
        b: &[Complex64],
    ) {
        let (c, a, b) = (self.limb_mut::<I>(c), self.limb::<J>(a), self.limb::<K>(b));
        if !is_limb_zero(b) && !is_limb_zero(a) {
            self.fft.eval_fma(c, b, a);
        }
    }
}

#[allow(clippy::absurd_extreme_comparisons)]
impl<const NATIVE: bool, const LIMB_BITS: usize> RingOps for PowerOfTwoRing<NATIVE, LIMB_BITS>
where
    Self: ModulusOps<Elem = u64>,
{
    type Eval = Complex64;
    type EvalPrep = Complex64;

    fn new(modulus: Modulus, ring_size: usize) -> Self {
        Self::new(modulus.try_into().unwrap(), Ffnt::new(ring_size))
    }

    fn ring_size(&self) -> usize {
        self.fft.ring_size()
    }

    fn eval_size(&self) -> usize {
        Self::LIMBS * self.fft.fft_size()
    }

    fn eval_scratch_size(&self) -> usize {
        self.fft.fft_scratch_size()
    }

    #[unroll_for_loops]
    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem], scratch: &mut [Self::Eval]) {
        if Self::SINGLE_LIMB {
            self.fft.forward(b, a, |a| self.q.center(*a) as _, scratch);
            return;
        }

        let mut leftover = 0;
        for I in 0..8 {
            if I < Self::LIMBS {
                self.forward_limb::<_, I>(b, a, scratch, &mut leftover);
            }
        }
    }

    #[unroll_for_loops]
    fn forward_elem_from<T: Copy>(&self, b: &mut [Self::Eval], a: &[T], scratch: &mut [Self::Eval])
    where
        Self: ElemFrom<T>,
    {
        if Self::SINGLE_LIMB {
            let to_f64 = |a: &_| self.q.center(self.elem_from(*a)) as _;
            self.fft.forward(b, a, to_f64, scratch);
            return;
        }

        let mut leftover = 0;
        for I in 0..8 {
            if I < Self::LIMBS {
                self.forward_limb::<_, I>(b, a, scratch, &mut leftover);
            }
        }
    }

    fn forward_normalized(
        &self,
        b: &mut [Self::Eval],
        a: &[Self::Elem],
        scratch: &mut [Self::Eval],
    ) {
        if Self::SINGLE_LIMB {
            let to_f64 = |a: &_| self.q.center(*a) as _;
            self.fft.forward_normalized(b, a, to_f64, scratch);
            return;
        }

        self.forward(b, a, scratch);
        self.fft.normalize(b);
    }

    #[unroll_for_loops]
    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval], scratch: &mut [Self::Eval]) {
        if Self::SINGLE_LIMB {
            self.fft.backward(b, a, |a| self.elem_from(a), scratch);
            return;
        }

        self.fft
            .backward(b, self.limb_mut::<0>(a), from_small_f64, scratch);
        for I in 1..8 {
            if I < Self::LIMBS {
                self.add_backward_limb::<I>(b, a, scratch);
            }
        }
    }

    fn backward_normalized(
        &self,
        b: &mut [Self::Elem],
        a: &mut [Self::Eval],
        scratch: &mut [Self::Eval],
    ) {
        if Self::SINGLE_LIMB {
            self.fft
                .backward_normalized(b, a, |a| self.elem_from(a), scratch);
            return;
        }

        self.fft.normalize(a);
        self.backward(b, a, scratch);
    }

    #[unroll_for_loops]
    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval], scratch: &mut [Self::Eval]) {
        if Self::SINGLE_LIMB {
            let add_from_f64 = |b: &mut u64, a| *b = self.q.reduce(b.wrapping_add(f64_mod_u64(a)));
            self.fft.add_backward(b, a, add_from_f64, scratch);
            return;
        }

        for I in 0..8 {
            if I < Self::LIMBS {
                self.add_backward_limb::<I>(b, a, scratch);
            }
        }
    }

    fn add_backward_normalized(
        &self,
        b: &mut [Self::Elem],
        a: &mut [Self::Eval],
        scratch: &mut [Self::Eval],
    ) {
        self.fft.normalize(a);
        self.add_backward(b, a, scratch);
    }

    fn eval_prepare(&self, b: &mut [Self::EvalPrep], a: &[Self::Eval]) {
        b.copy_from_slice(a)
    }

    #[unroll_for_loops]
    fn eval_mul(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        if Self::SINGLE_LIMB {
            self.fft.eval_mul(c, a, b);
            return;
        }

        for H in 0..8 {
            const I: usize = 7 - H;
            if I < Self::LIMBS {
                self.eval_mul_limb::<I, 0, I>(c, a, b);
                for J in 1..8 {
                    if J <= I {
                        const K: usize = I.saturating_sub(J);
                        self.eval_fma_limb::<I, J, K>(c, a, b);
                    }
                }
            }
        }
    }

    #[unroll_for_loops]
    fn eval_mul_assign(&self, b: &mut [Self::Eval], a: &[Self::Eval]) {
        if Self::SINGLE_LIMB {
            self.fft.eval_mul_assign(b, a);
            return;
        }

        for H in 0..8 {
            const I: usize = 7 - H;
            if I < Self::LIMBS {
                let (b, c) = self.split_limbs_at_mut::<I>(b);
                self.eval_mul_assign_limb::<0, 0>(c, a);
                for J in 1..8 {
                    if J <= I {
                        const K: usize = I.saturating_sub(J);
                        self.eval_fma_limb::<0, J, K>(c, a, b);
                    }
                }
            }
        }
    }

    #[unroll_for_loops]
    fn eval_fma(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        if Self::SINGLE_LIMB {
            self.fft.eval_fma(c, a, b);
            return;
        }

        for H in 0..8 {
            const I: usize = 7 - H;
            if I < Self::LIMBS {
                for J in 0..8 {
                    if J <= I {
                        const K: usize = I.saturating_sub(J);
                        self.eval_fma_limb::<I, J, K>(c, a, b);
                    }
                }
            }
        }
    }

    fn eval_mul_prep(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::EvalPrep]) {
        self.eval_mul(c, a, b)
    }

    fn eval_mul_assign_prep(&self, b: &mut [Self::Eval], a: &[Self::EvalPrep]) {
        self.eval_mul_assign(b, a)
    }

    fn eval_fma_prep(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::EvalPrep]) {
        self.eval_fma(c, a, b)
    }
}

#[inline(always)]
fn from_small_f64(a: f64) -> u64 {
    a.round() as i64 as u64
}

#[inline(always)]
fn set_limb_zero(a: &mut [Complex64]) {
    a[0].re = f64::NAN
}

#[inline(always)]
fn is_limb_zero(a: &[Complex64]) -> bool {
    a[0].re.is_nan()
}
