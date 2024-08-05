use crate::{
    misc::{as_slice::AsMutSlice, bit_reverse},
    modulus::{inv_mod, pow_mod, powers_mod, Prime},
    ring::prime::Shoup,
};
use itertools::izip;

/// Negacyclic NTT
pub struct Ntt {
    q: u64,
    q_twice: u64,
    q_quart: u64,
    ring_size: usize,
    twiddle_bo: Vec<Shoup>,
    twiddle_bo_inv: Vec<Shoup>,
    n_inv: Shoup,
}

impl Ntt {
    pub fn new(Prime(q): Prime, ring_size: usize) -> Self {
        assert!(ring_size.is_power_of_two());

        let g = two_adic_generator(q, ring_size.ilog2() as usize + 1);
        let [twiddle_bo, twiddle_bo_inv] = [g, inv_mod(g, q).unwrap()]
            .map(|b| powers_mod(b, q).take(ring_size).map(|v| Shoup::new(v, q)))
            .map(FromIterator::from_iter)
            .map(bit_reverse);
        let n_inv = Shoup::new(inv_mod(ring_size as _, q).unwrap(), q);

        Self {
            q,
            q_twice: q << 1,
            q_quart: q << 2,
            ring_size,
            twiddle_bo,
            twiddle_bo_inv,
            n_inv,
        }
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }

    pub fn eval_size(&self) -> usize {
        self.ring_size
    }

    pub fn n_inv(&self) -> &Shoup {
        &self.n_inv
    }

    pub fn forward(&self, a: &mut [u64]) {
        debug_assert_eq!(a.len(), self.ring_size);
        let log_n = a.len().ilog2();
        for layer in 0..log_n {
            let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
            izip!(a.chunks_exact_mut(2 * size), &self.twiddle_bo[m..]).for_each(|(a, t)| {
                let (a, b) = a.split_at_mid_mut();
                if layer == log_n - 1 {
                    izip!(a, b).for_each(|(a, b)| self.dit(a, b, t));
                } else {
                    izip!(a, b).for_each(|(a, b)| self.dit_lazy(a, b, t));
                }
            });
        }
    }

    pub fn backward(&self, a: &mut [u64]) {
        debug_assert_eq!(a.len(), self.ring_size);
        let log_n = a.len().ilog2();
        for layer in (0..log_n).rev() {
            let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
            izip!(a.chunks_exact_mut(2 * size), &self.twiddle_bo_inv[m..]).for_each(|(a, t)| {
                let (a, b) = a.split_at_mid_mut();
                izip!(a, b).for_each(|(a, b)| self.dif(a, b, t));
            });
        }
    }

    pub fn normalize(&self, a: &mut [u64]) {
        a.iter_mut().for_each(|a| *a = self.n_inv.mul(*a, self.q));
    }

    #[inline(always)]
    fn reduce_twice_assign(&self, a: &mut u64) {
        if *a >= self.q_twice {
            *a -= self.q_twice
        }
    }

    #[inline(always)]
    fn dit_lazy(&self, a: &mut u64, b: &mut u64, t: &Shoup) {
        debug_assert!(*a < self.q_quart);
        debug_assert!(*b < self.q_quart);
        self.reduce_twice_assign(a);
        let bt = t.mul(*b, self.q);
        let c = *a + bt;
        let d = *a + self.q_twice - bt;
        *a = c;
        *b = d;
    }

    #[inline(always)]
    fn dit(&self, a: &mut u64, b: &mut u64, t: &Shoup) {
        debug_assert!(*a < self.q_quart);
        debug_assert!(*b < self.q_quart);
        self.reduce_twice_assign(a);
        let bt = t.mul(*b, self.q);
        let c = a.wrapping_add(bt);
        let d = a.wrapping_sub(bt);
        *a = (c).min(c.wrapping_sub(self.q_twice));
        *b = (d).min(d.wrapping_add(self.q_twice));
    }

    #[inline(always)]
    fn dif(&self, a: &mut u64, b: &mut u64, t: &Shoup) {
        debug_assert!(*a < self.q_twice);
        debug_assert!(*b < self.q_twice);
        let mut c = *a + *b;
        self.reduce_twice_assign(&mut c);
        let d = t.mul(*a + self.q_twice - *b, self.q);
        *a = c;
        *b = d;
    }
}

pub fn two_adic_generator(q: u64, two_adicity: usize) -> u64 {
    assert_eq!((q - 1) % (1 << two_adicity) as u64, 0);
    pow_mod(multiplicative_generator(q), (q - 1) >> two_adicity, q)
}

pub fn multiplicative_generator(q: u64) -> u64 {
    let order = q - 1;
    (1..order)
        .find(|g| pow_mod(*g, order >> 1, q) == order)
        .unwrap()
}
