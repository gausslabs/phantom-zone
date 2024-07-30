use crate::ring::{ArithmeticOps, SliceOps};
use core::{
    f64::consts::PI,
    fmt::{self, Debug, Formatter},
};
use itertools::{izip, Itertools};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Ffnt {
    ring_size: usize,
    fft_size: usize,
    fft_size_inv: f64,
    twiddle: Vec<Complex64>,
    twiddle_inv: Vec<Complex64>,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
}

impl Ffnt {
    pub(crate) fn new(ring_size: usize) -> Self {
        assert!(ring_size.is_power_of_two());

        let twiddle = (0..ring_size / 2)
            .map(|i| Complex64::cis((i as f64 * PI) / ring_size as f64))
            .collect_vec();
        let twiddle_inv = twiddle.iter().map(Complex64::conj).collect_vec();

        let fft_size = (ring_size / 2).max(1);
        let fft = FftPlanner::new().plan_fft_forward(fft_size);
        let ifft = FftPlanner::new().plan_fft_inverse(fft_size);

        Self {
            ring_size,
            fft_size,
            fft_size_inv: 1f64 / fft_size as f64,
            twiddle,
            twiddle_inv,
            fft,
            ifft,
        }
    }

    pub(crate) fn ring_size(&self) -> usize {
        self.ring_size
    }

    pub(crate) fn fft_size(&self) -> usize {
        self.fft_size
    }

    pub(crate) fn forward<T>(&self, b: &mut [Complex64], a: &[T], to_f64: impl Fn(&T) -> f64) {
        self.fold_twist(b, a, to_f64);
        self.fft.process(b);
    }

    pub(crate) fn forward_normalized<T>(
        &self,
        b: &mut [Complex64],
        a: &[T],
        to_f64: impl Fn(&T) -> f64,
    ) {
        self.forward(b, a, to_f64);
        self.normalize(b);
    }

    pub(crate) fn backward<T>(
        &self,
        b: &mut [T],
        a: &mut [Complex64],
        from_f64: impl Fn(f64) -> T,
    ) {
        self.ifft.process(a);
        self.unfold_untwist(b, a, from_f64);
    }

    pub(crate) fn backward_normalized<T>(
        &self,
        b: &mut [T],
        a: &mut [Complex64],
        from_f64: impl Fn(f64) -> T,
    ) {
        self.normalize(a);
        self.backward(b, a, from_f64);
    }

    pub(crate) fn add_backward<T>(
        &self,
        b: &mut [T],
        a: &mut [Complex64],
        add_from_f64: impl Fn(&mut T, f64),
    ) {
        self.ifft.process(a);
        self.add_unfold_untwist(b, a, add_from_f64);
    }

    fn normalize(&self, a: &mut [Complex64]) {
        a.iter_mut().for_each(|a| *a *= self.fft_size_inv);
    }

    fn fold_twist<T>(&self, b: &mut [Complex64], a: &[T], to_f64: impl Fn(&T) -> f64) {
        if a.len() == 1 {
            b[0] = to_f64(&a[0]).into();
        } else {
            let (lo, hi) = a.split_at(a.len() / 2);
            izip!(&mut *b, lo, hi, &self.twiddle)
                .for_each(|(b, lo, hi, t)| *b = Complex64::new(to_f64(lo), to_f64(hi)) * t);
        }
    }

    fn unfold_untwist<T>(&self, b: &mut [T], a: &[Complex64], from_f64: impl Fn(f64) -> T) {
        if b.len() == 1 {
            b[0] = from_f64(a[0].re);
        } else {
            let (lo, hi) = b.split_at_mut(b.len() / 2);
            izip!(lo, hi, a, &self.twiddle_inv).for_each(|(lo, hi, a, t)| {
                let a = *a * t;
                *lo = from_f64(a.re);
                *hi = from_f64(a.im);
            });
        }
    }

    fn add_unfold_untwist<T>(
        &self,
        b: &mut [T],
        a: &[Complex64],
        add_from_f64: impl Fn(&mut T, f64),
    ) {
        if b.len() == 1 {
            add_from_f64(&mut b[0], a[0].re);
        } else {
            let (lo, hi) = b.split_at_mut(b.len() / 2);
            izip!(lo, hi, a, &self.twiddle_inv).for_each(|(lo, hi, a, t)| {
                let a = *a * t;
                add_from_f64(lo, a.re);
                add_from_f64(hi, a.im);
            });
        }
    }
}

impl ArithmeticOps for Ffnt {
    type Elem = Complex64;
    type Prep = Complex64;

    #[inline(always)]
    fn zero(&self) -> Self::Elem {
        0f64.into()
    }

    #[inline(always)]
    fn one(&self) -> Self::Elem {
        1f64.into()
    }

    #[inline(always)]
    fn neg_one(&self) -> Self::Elem {
        (-1f64).into()
    }

    #[inline(always)]
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        -a
    }

    #[inline(always)]
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a + b
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a - b
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a * b
    }

    #[inline(always)]
    fn prepare(&self, a: &Self::Elem) -> Self::Prep {
        *a
    }

    #[inline(always)]
    fn mul_prep(&self, a: &Self::Elem, b: &Self::Prep) -> Self::Elem {
        a * b
    }
}

impl SliceOps for Ffnt {}

impl Debug for Ffnt {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ffnt")
            .field("ring_size", &self.ring_size)
            .field("fft_size", &self.fft_size)
            .field("fft_size_inv", &self.fft_size_inv)
            .field("twiddle", &self.twiddle)
            .field("twiddle_inv", &self.twiddle_inv)
            .finish()
    }
}
