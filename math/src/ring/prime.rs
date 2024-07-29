use crate::{
    fft::bit_reverse,
    prime::{mod_inv, mod_powers, two_adic_generator},
    ring::{ArithmeticOps, RingOps, SliceOps},
};
use itertools::{izip, Itertools};
use rand::{
    distributions::{Distribution, Uniform},
    RngCore,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Shoup(u64, u64);

impl Shoup {
    #[inline]
    fn new(v: u64, q: u64) -> Self {
        let quotient = (((v as u128) << 64) / q as u128) as _;
        Self(v, quotient)
    }

    #[inline]
    fn value(&self) -> u64 {
        self.0
    }

    #[inline]
    fn quotient(&self) -> u64 {
        self.1
    }
}

#[derive(Clone, Debug)]
pub struct PrimeRing {
    q: u64,
    q_half: u64,
    q_twice: u64,
    q_quart: u64,
    log_q: usize,
    barrett_mu: u128,
    barrett_alpha: usize,
    ring_size: usize,
    twiddle_bo: Vec<Shoup>,
    twiddle_bo_inv: Vec<Shoup>,
    n_inv: Shoup,
}

impl PrimeRing {
    pub fn new(q: u64, ring_size: usize) -> Self {
        assert!(ring_size.is_power_of_two());

        // largest unsigned value modulus fits is modulus-1
        let log_q = q.next_power_of_two().ilog2() as usize;

        // barrett calculation
        let barrett_mu = (1u128 << (log_q * 2 + 3)) / (q as u128);
        let barrett_alpha = log_q + 3;

        let g = two_adic_generator(q, ring_size.ilog2() as usize + 1);
        let twiddle_bo = bit_reverse(
            mod_powers(g, q)
                .take(ring_size)
                .map(|v| Shoup::new(v, q))
                .collect_vec(),
        );
        let twiddle_bo_inv = bit_reverse(
            mod_powers(mod_inv(g, q), q)
                .take(ring_size)
                .map(|v| Shoup::new(v, q))
                .collect_vec(),
        );
        let n_inv = Shoup::new(mod_inv(ring_size as _, q), q);

        PrimeRing {
            q,
            q_half: q >> 1,
            q_twice: q << 1,
            q_quart: q << 2,
            log_q,
            barrett_mu,
            barrett_alpha,
            ring_size,
            twiddle_bo,
            twiddle_bo_inv,
            n_inv,
        }
    }

    fn dit_lazy(&self, a: &mut u64, b: &mut u64, t: &Shoup) {
        debug_assert!(*a < self.q_quart);
        debug_assert!(*b < self.q_quart);
        if *a >= self.q_twice {
            *a -= self.q_twice;
        }
        let bt = self.mul_prepared(b, t);
        let c = *a + bt;
        let d = *a + self.q_twice - bt;
        *a = c;
        *b = d;
    }

    fn dit(&self, a: &mut u64, b: &mut u64, t: &Shoup) {
        debug_assert!(*a < self.q_quart);
        debug_assert!(*b < self.q_quart);
        if *a >= self.q_twice {
            *a -= self.q_twice;
        }
        let bt = self.mul_prepared(b, t);
        let c = a.wrapping_add(bt);
        let d = a.wrapping_sub(bt);
        *a = (c).min(c.wrapping_sub(self.q_twice));
        *b = (d).min(d.wrapping_add(self.q_twice));
    }

    fn dif(&self, a: &mut u64, b: &mut u64, t: &Shoup) {
        debug_assert!(*a < self.q_twice);
        debug_assert!(*b < self.q_twice);
        let mut c = *a + *b;
        if c >= self.q_twice {
            c -= self.q_twice
        }
        let d = self.mul_prepared(&(*a + self.q_twice - *b), t);
        *a = c;
        *b = d;
    }
}

impl ArithmeticOps for PrimeRing {
    type Element = u64;
    type Prepared = Shoup;

    #[inline]
    fn zero(&self) -> Self::Element {
        0
    }
    #[inline]
    fn one(&self) -> Self::Element {
        1
    }
    #[inline]
    fn neg_one(&self) -> Self::Element {
        self.q - 1
    }

    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        self.q - a
    }
    #[inline]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        debug_assert!(*a < self.q);
        debug_assert!(*b < self.q);
        let c = a + b;
        if c >= self.q {
            c - self.q
        } else {
            c
        }
    }
    #[inline]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        debug_assert!(*a < self.q);
        debug_assert!(*b < self.q);
        if a >= b {
            a - b
        } else {
            self.q + a - b
        }
    }
    #[inline]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        debug_assert!(*a < self.q_twice);
        debug_assert!(*b < self.q_twice);

        let ab = *a as u128 * *b as u128;
        // ab / (2^{n + \beta})
        // note: \beta is assumed to -2
        let tmp = ab >> (self.log_q - 2);
        // k = ((ab / (2^{n + \beta})) * \mu) / 2^{\alpha - (-2)}
        let k = (tmp * self.barrett_mu) >> (self.barrett_alpha + 2);
        // ab - k*p
        let tmp = k * (self.q as u128);

        let c = (ab - tmp) as u64;
        if c >= self.q {
            c - self.q
        } else {
            c
        }
    }

    #[inline]
    fn prepare(&self, a: &Self::Element) -> Self::Prepared {
        Shoup::new(*a, self.q)
    }
    #[inline]
    fn mul_prepared(&self, a: &Self::Element, b: &Self::Prepared) -> Self::Element {
        let t = ((b.quotient() as u128 * *a as u128) >> 64) as _;
        (a.wrapping_mul(b.value())).wrapping_sub(self.q.wrapping_mul(t))
    }

    #[inline]
    fn from_f64(&self, v: f64) -> Self::Element {
        self.from_i64(v.round() as _)
    }
    #[inline]
    fn from_i64(&self, v: i64) -> Self::Element {
        v.rem_euclid(self.q as _) as _
    }
    #[inline]
    fn to_i64(&self, v: &Self::Element) -> i64 {
        if v < &self.q_half {
            *v as _
        } else {
            -((self.q - v) as i64)
        }
    }

    fn sample_uniform(&self, mut rng: impl RngCore) -> Self::Element {
        Uniform::new(0, self.q).sample(&mut rng)
    }
}

impl SliceOps for PrimeRing {
    fn matrix_fma_prepared<'a>(
        &self,
        c: &mut [Self::Element],
        a: impl IntoIterator<Item = &'a [Self::Element]>,
        b: impl IntoIterator<Item = &'a [Self::Prepared]>,
    ) {
        let (mut a, mut b) = (a.into_iter(), b.into_iter());
        izip!(a.by_ref(), b.by_ref()).for_each(|(a, b)| {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), c.len());
            izip!(&mut *c, a, b).for_each(|(c, a, b)| {
                let t = c.wrapping_add(self.mul_prepared(a, b));
                *c = if t >= self.q_twice {
                    t - self.q_twice
                } else {
                    t
                }
            })
        });
        debug_assert!(a.next().is_none());
        debug_assert!(b.next().is_none());
    }
}

impl RingOps for PrimeRing {
    type Evaluation = u64;
    type Twiddle = Shoup;

    fn ring_size(&self) -> usize {
        self.ring_size
    }

    fn evaluation_size(&self) -> usize {
        self.ring_size
    }

    fn forward(&self, mut a: Vec<Self::Element>) -> Vec<Self::Evaluation> {
        debug_assert_eq!(a.len(), self.ring_size());
        let log_n = a.len().ilog2();
        for layer in 0..log_n {
            let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
            izip!(a.chunks_exact_mut(2 * size), &self.twiddle_bo[m..]).for_each(|(a, t)| {
                let (a, b) = a.split_at_mut(size);
                if layer == log_n - 1 {
                    izip!(a, b).for_each(|(a, b)| self.dit(a, b, t));
                } else {
                    izip!(a, b).for_each(|(a, b)| self.dit_lazy(a, b, t));
                }
            });
        }
        a
    }

    fn backward(&self, mut a: Vec<Self::Evaluation>) -> Vec<Self::Element> {
        debug_assert_eq!(a.len(), self.evaluation_size());
        let log_n = a.len().ilog2();
        for layer in (0..log_n).rev() {
            let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
            izip!(a.chunks_exact_mut(2 * size), &self.twiddle_bo_inv[m..]).for_each(|(a, t)| {
                let (a, b) = a.split_at_mut(size);
                if layer == 0 {
                    izip!(a, b).for_each(|(a, b)| {
                        self.dif(a, b, t);
                        *a = self.mul_prepared(a, &self.n_inv);
                        *b = self.mul_prepared(b, &self.n_inv);
                    });
                } else {
                    izip!(a, b).for_each(|(a, b)| self.dif(a, b, t));
                }
            });
        }
        a
    }

    fn evaluation_mul_assign(&self, a: &mut [Self::Evaluation], b: &[Self::Evaluation]) {
        self.slice_mul_assign(a, b)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        fft::test::nega_cyclic_schoolbook_mul,
        prime::two_adic_primes,
        ring::{prime::PrimeRing, ArithmeticOps, RingOps},
    };
    use rand::thread_rng;

    #[test]
    fn ring_ops() {
        let mut rng = thread_rng();
        for log_ring_size in 0..12usize {
            let ring_size = 1 << log_ring_size;
            for q in two_adic_primes(50, log_ring_size + 1).take(10) {
                let ring = PrimeRing::new(q, ring_size);
                let a = ring.sample_uniform_vec(ring_size, &mut rng);
                let b = ring.sample_uniform_vec(ring_size, &mut rng);
                assert_eq!(a, ring.backward(ring.forward(a.clone())));
                assert_eq!(
                    ring.ring_mul(a.clone(), b.clone()),
                    nega_cyclic_schoolbook_mul(&ring, &a, &b)
                );
            }
        }
    }
}
