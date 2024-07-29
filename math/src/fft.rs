use core::ops::{Add, Mul, MulAssign, Sub};
use itertools::izip;
use num_complex::Complex64;

pub fn bit_reverse<T, V: AsMut<[T]>>(mut values: V) -> V {
    let n = values.as_mut().len();
    if n > 2 {
        assert!(n.is_power_of_two());
        let log_len = n.ilog2();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - log_len);
            if i < j {
                values.as_mut().swap(i, j)
            }
        }
    }
    values
}

pub fn fft64_in_place(a: &mut [Complex64], twiddle_bo: &[Complex64]) {
    fft_in_place(a, twiddle_bo)
}

pub fn ifft64_in_place(a: &mut [Complex64], twiddle_inv_bo: &[Complex64]) {
    let n_inv = 1f64 / a.len() as f64;
    ifft_in_place(a, twiddle_inv_bo, &n_inv)
}

// Given normal order input and bit-reversed order twiddle factors,
// compute bit-reversed order output in place.
fn fft_in_place<T: Butterfly>(a: &mut [T], twiddle_bo: &[T]) {
    assert!(a.len().is_power_of_two());
    for layer in (0..a.len().ilog2()).rev() {
        let size = 1 << layer;
        izip!(a.chunks_mut(2 * size), twiddle_bo).for_each(|(values, twiddle)| {
            let (a, b) = values.split_at_mut(size);
            izip!(a, b).for_each(|(a, b): (_, &mut _)| Butterfly::dit(a, b, twiddle));
        });
    }
}

// Given bit-reversed order input and bit-reversed order twiddle factors,
// compute normal order output in place.
fn ifft_in_place<T, S>(a: &mut [T], twiddle_inv_bo: &[T], n_inv: &S)
where
    T: Butterfly + for<'t> MulAssign<&'t S>,
{
    assert!(a.len().is_power_of_two());
    for layer in 0..a.len().ilog2() {
        let size = 1 << layer;
        izip!(a.chunks_mut(2 * size), twiddle_inv_bo).for_each(|(values, twiddle)| {
            let (a, b) = values.split_at_mut(size);
            izip!(a, b).for_each(|(a, b): (_, &mut _)| Butterfly::dif(a, b, twiddle));
        });
    }
    a.iter_mut().for_each(|a| *a *= n_inv);
}

pub trait Butterfly {
    fn dit(a: &mut Self, b: &mut Self, t: &Self);

    fn dif(a: &mut Self, b: &mut Self, t: &Self);

    fn twiddle_free(a: &mut Self, b: &mut Self);
}

impl<T> Butterfly for T
where
    for<'t> &'t T: Mul<&'t T, Output = T> + Add<&'t T, Output = T> + Sub<&'t T, Output = T>,
{
    #[inline(always)]
    fn dit(a: &mut Self, b: &mut Self, t: &Self) {
        let tb = t * b;
        let c = &*a + &tb;
        let d = &*a - &tb;
        *a = c;
        *b = d;
    }

    #[inline(always)]
    fn dif(a: &mut Self, b: &mut Self, t: &Self) {
        let c = &*a + b;
        let d = &(&*a - b) * t;
        *a = c;
        *b = d;
    }

    #[inline(always)]
    fn twiddle_free(a: &mut Self, b: &mut Self) {
        let c = &*a + b;
        let d = &*a - b;
        *a = c;
        *b = d;
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::ring::ArithmeticOps;
    use itertools::izip;

    macro_rules! assert_precision {
        ($a:expr, $b:expr, $precision:expr $(,)?) => {
            itertools::izip!(&$a, &$b).for_each(|(a, b)| {
                let abs_diff = ((a.wrapping_sub(*b)) as i64).abs();
                assert!(
                    abs_diff < (1 << $precision),
                    "log(|{a} - {b}|) = {} >= {}",
                    abs_diff.ilog2(),
                    $precision
                );
            })
        };
    }

    pub(crate) use assert_precision;

    pub(crate) fn nega_cyclic_schoolbook_mul<T: ArithmeticOps>(
        op: &T,
        a: &[T::Element],
        b: &[T::Element],
    ) -> Vec<T::Element> {
        let n = a.len();
        let mut c = vec![T::Element::default(); n];
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
}
