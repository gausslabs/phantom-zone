use crate::{
    modulus::{prime::Shoup, ModulusOps, Prime},
    util::{as_slice::AsMutSlice, bit_reverse},
};
use core::fmt::{self, Debug};
use itertools::izip;

/// Negacyclic NTT
#[derive(Clone)]
pub struct Ntt {
    q: Prime,
    q_twice: u64,
    q_quart: u64,
    ring_size: usize,
    log_ring_size: usize,
    twiddle_bo: Vec<Shoup>,
    twiddle_inv_bo: Vec<Shoup>,
    n_inv: Shoup,
}

impl Ntt {
    pub fn new(q: Prime, ring_size: usize) -> Self {
        assert!(ring_size.is_power_of_two());
        let log_ring_size = ring_size.ilog2() as usize;
        let g = (log_ring_size > 0)
            .then(|| q.two_adic_generator(log_ring_size + 1))
            .unwrap_or(1);
        let [twiddle_bo, twiddle_inv_bo] = [g, q.inv(&g).unwrap()]
            .map(|b| Vec::from_iter(q.powers(&b).take(ring_size).map(|v| Shoup::new(v, *q))))
            .map(bit_reverse);
        let n_inv = Shoup::new(q.inv(&(ring_size as u64)).unwrap(), *q);
        Self {
            q,
            q_twice: *q << 1,
            q_quart: *q << 2,
            ring_size,
            log_ring_size,
            twiddle_bo,
            twiddle_inv_bo,
            n_inv,
        }
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }

    pub fn eval_size(&self) -> usize {
        self.ring_size
    }

    pub fn forward<const NORMALIZE: bool>(&self, a: &mut [u64]) {
        debug_assert_eq!(a.len(), self.ring_size);
        for layer in 0..self.log_ring_size {
            let (m, size) = (1 << layer, 1 << (self.log_ring_size - layer - 1));
            if layer == self.log_ring_size - 1 {
                izip!(a.chunks_exact_mut(2), &self.twiddle_bo[m..]).for_each(|(a, t)| {
                    let (a, b) = a.split_at_mid_mut();
                    self.dit::<false>(&mut a[0], &mut b[0], t);
                    if NORMALIZE {
                        self.normalize::<true>(&mut a[0]);
                        self.normalize::<true>(&mut b[0]);
                    }
                });
            } else {
                izip!(a.chunks_exact_mut(2 * size), &self.twiddle_bo[m..]).for_each(|(a, t)| {
                    let (a, b) = a.split_at_mid_mut();
                    izip!(a, b).for_each(|(a, b)| self.dit::<true>(a, b, t));
                });
            }
        }
    }

    pub fn backward<const NORMALIZE: bool>(&self, a: &mut [u64]) {
        debug_assert_eq!(a.len(), self.ring_size);
        for layer in (0..self.log_ring_size).rev() {
            let (m, size) = (1 << layer, 1 << (self.log_ring_size - layer - 1));
            if layer == 0 {
                izip!(a.chunks_exact_mut(2 * size), &self.twiddle_inv_bo[m..]).for_each(
                    |(a, t)| {
                        let (a, b) = a.split_at_mid_mut();
                        izip!(a, b).for_each(|(a, b)| {
                            if NORMALIZE {
                                self.dif::<true>(a, b, t);
                                self.normalize::<false>(a);
                                self.normalize::<false>(b);
                            } else {
                                self.dif::<false>(a, b, t);
                            }
                        });
                    },
                );
            } else {
                izip!(a.chunks_exact_mut(2 * size), &self.twiddle_inv_bo[m..]).for_each(
                    |(a, t)| {
                        let (a, b) = a.split_at_mid_mut();
                        izip!(a, b).for_each(|(a, b)| self.dif::<true>(a, b, t));
                    },
                );
            }
        }
    }

    #[inline(always)]
    fn reduce_twice_assign(&self, a: &mut u64) {
        if *a >= self.q_twice {
            *a -= self.q_twice
        }
    }

    #[inline(always)]
    fn dit<const LAZY: bool>(&self, a: &mut u64, b: &mut u64, t: &Shoup) {
        debug_assert!(*a < self.q_quart);
        debug_assert!(*b < self.q_quart);
        self.reduce_twice_assign(a);
        let bt = self.q.mul_prep(b, t);
        let c = a.wrapping_add(bt);
        let d = a.wrapping_sub(bt);
        if LAZY {
            *a = c;
            *b = d.wrapping_add(self.q_twice);
        } else {
            *a = c.min(c.wrapping_sub(self.q_twice));
            *b = d.min(d.wrapping_add(self.q_twice));
        }
    }

    #[inline(always)]
    fn dif<const LAZY: bool>(&self, a: &mut u64, b: &mut u64, t: &Shoup) {
        debug_assert!(*a < self.q_twice);
        debug_assert!(*b < self.q_twice);
        let mut c = *a + *b;
        self.reduce_twice_assign(&mut c);
        let d = self.q.mul_prep(&(*a + self.q_twice - *b), t);
        if LAZY {
            *a = c;
            *b = d;
        } else {
            *a = c.min(c.wrapping_sub(*self.q));
            *b = d.min(d.wrapping_sub(*self.q));
        }
    }

    #[inline(always)]
    fn normalize<const LAZY: bool>(&self, a: &mut u64) {
        *a = self.q.mul_prep(a, &self.n_inv);
        if !LAZY {
            *a = (*a).min(a.wrapping_sub(*self.q));
        }
    }
}

impl Default for Ntt {
    fn default() -> Self {
        Self::new(Prime::new(2), 1)
    }
}

impl Debug for Ntt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ntt")
            .field("q", &self.q)
            .field("q_twice", &self.q_twice)
            .field("q_quart", &self.q_quart)
            .field("ring_size", &self.ring_size)
            .field("log_ring_size", &self.log_ring_size)
            .field(
                "twiddle_bo",
                &format_args!(
                    "bit_reverse(powers({:?}))",
                    &self.twiddle_bo[self.ring_size / 2],
                ),
            )
            .field(
                "twiddle_inv_bo",
                &format_args!(
                    "bit_reverse(powers({:?}))",
                    &self.twiddle_inv_bo[self.ring_size / 2],
                ),
            )
            .field("n_inv", &self.n_inv)
            .finish()
    }
}
