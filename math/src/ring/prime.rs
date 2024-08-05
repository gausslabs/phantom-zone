use crate::{
    decomposer::PrimeDecomposer,
    distribution::{DistributionSized, Sampler},
    izip_eq,
    misc::bit_reverse,
    modulus::{inv_mod, pow_mod, powers_mod, Modulus, Prime},
    ring::{ArithmeticOps, ElemFrom, ElemTo, RingOps, SliceOps},
};
use itertools::izip;
use rand::distributions::{Distribution, Uniform};

#[derive(Clone, Copy, Debug, Default)]
pub struct Shoup(u64, u64);

impl Shoup {
    #[inline(always)]
    fn new(v: u64, q: u64) -> Self {
        let quotient = (((v as u128) << 64) / q as u128) as _;
        Self(v, quotient)
    }

    #[inline(always)]
    fn value(&self) -> u64 {
        self.0
    }

    #[inline(always)]
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
    fn new(Prime(q): Prime, ring_size: usize) -> Self {
        assert!(ring_size.is_power_of_two());

        let log_q = q.next_power_of_two().ilog2() as usize;

        let barrett_mu = (1u128 << (log_q * 2 + 3)) / (q as u128);
        let barrett_alpha = log_q + 3;

        let g = two_adic_generator(q, ring_size.ilog2() as usize + 1);
        let [twiddle_bo, twiddle_bo_inv] = [g, inv_mod(g, q).unwrap()]
            .map(|b| powers_mod(b, q).take(ring_size).map(|v| Shoup::new(v, q)))
            .map(FromIterator::from_iter)
            .map(bit_reverse);
        let n_inv = Shoup::new(inv_mod(ring_size as _, q).unwrap(), q);

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

    #[inline(always)]
    pub fn to_i64(&self, v: u64) -> i64 {
        if v < self.q_half {
            v as _
        } else {
            -((self.q - v) as i64)
        }
    }

    #[inline(always)]
    fn reduce_assign(&self, a: &mut u64) {
        if *a >= self.q {
            *a -= self.q
        }
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
        let bt = self.mul_prep(b, t);
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
        let bt = self.mul_prep(b, t);
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
        let d = self.mul_prep(&(*a + self.q_twice - *b), t);
        *a = c;
        *b = d;
    }

    fn ntt(&self, a: &mut [u64]) {
        debug_assert_eq!(a.len(), self.ring_size);
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
    }

    fn intt(&self, a: &mut [u64]) {
        debug_assert_eq!(a.len(), self.ring_size);
        let log_n = a.len().ilog2();
        for layer in (0..log_n).rev() {
            let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
            izip!(a.chunks_exact_mut(2 * size), &self.twiddle_bo_inv[m..]).for_each(|(a, t)| {
                let (a, b) = a.split_at_mut(size);
                izip!(a, b).for_each(|(a, b)| self.dif(a, b, t));
            });
        }
    }

    fn normalize(&self, a: &mut [u64]) {
        self.slice_scalar_mul_assign_prep(a, &self.n_inv);
    }
}

impl ArithmeticOps for PrimeRing {
    type Elem = u64;
    type Prep = Shoup;

    #[inline(always)]
    fn zero(&self) -> Self::Elem {
        0
    }

    #[inline(always)]
    fn one(&self) -> Self::Elem {
        1
    }

    #[inline(always)]
    fn neg_one(&self) -> Self::Elem {
        self.q - 1
    }

    #[inline(always)]
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        self.q - a
    }

    #[inline(always)]
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        debug_assert!(*a < self.q);
        debug_assert!(*b < self.q);
        let mut c = a + b;
        self.reduce_assign(&mut c);
        c
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        debug_assert!(*a < self.q);
        debug_assert!(*b < self.q);
        if a >= b {
            a - b
        } else {
            self.q + a - b
        }
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
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

        let mut c = (ab - tmp) as u64;
        self.reduce_assign(&mut c);
        c
    }

    #[inline(always)]
    fn prepare(&self, a: &Self::Elem) -> Self::Prep {
        Shoup::new(*a, self.q)
    }

    #[inline(always)]
    fn mul_prep(&self, a: &Self::Elem, b: &Self::Prep) -> Self::Elem {
        let t = ((b.quotient() as u128 * *a as u128) >> 64) as _;
        (a.wrapping_mul(b.value())).wrapping_sub(self.q.wrapping_mul(t))
    }
}

impl ElemFrom<u64> for PrimeRing {
    #[inline(always)]
    fn elem_from(&self, v: u64) -> Self::Elem {
        v % self.q
    }
}

impl ElemFrom<i64> for PrimeRing {
    #[inline(always)]
    fn elem_from(&self, v: i64) -> Self::Elem {
        v.rem_euclid(self.q as _) as _
    }
}

impl ElemFrom<u32> for PrimeRing {
    #[inline(always)]
    fn elem_from(&self, v: u32) -> Self::Elem {
        self.elem_from(v as u64)
    }
}

impl ElemFrom<i32> for PrimeRing {
    #[inline(always)]
    fn elem_from(&self, v: i32) -> Self::Elem {
        self.elem_from(v as i64)
    }
}

impl ElemTo<u64> for PrimeRing {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> u64 {
        v
    }
}

impl ElemTo<i64> for PrimeRing {
    #[inline(always)]
    fn elem_to(&self, v: Self::Elem) -> i64 {
        self.to_i64(v)
    }
}

impl SliceOps for PrimeRing {
    fn matrix_fma_prep<'a>(
        &self,
        c: &mut [Self::Elem],
        a: impl IntoIterator<Item = &'a [Self::Elem]>,
        b: impl IntoIterator<Item = &'a [Self::Prep]>,
    ) {
        izip_eq!(a, b).for_each(|(a, b)| {
            izip_eq!(&mut *c, a, b).for_each(|(c, a, b)| {
                *c = c.wrapping_add(self.mul_prep(a, b));
                self.reduce_twice_assign(c);
            })
        });
    }
}

impl Sampler for PrimeRing {
    fn uniform_distribution(
        &self,
    ) -> impl Distribution<Self::Elem> + DistributionSized<Self::Elem> {
        Uniform::new(0, self.q)
    }
}

impl RingOps for PrimeRing {
    type Eval = u64;
    type Decomposer = PrimeDecomposer;

    fn new(modulus: Modulus, ring_size: usize) -> Self {
        Self::new(modulus.try_into().unwrap(), ring_size)
    }

    fn modulus(&self) -> Modulus {
        Prime(self.q).into()
    }

    fn ring_size(&self) -> usize {
        self.ring_size
    }

    fn eval_size(&self) -> usize {
        self.ring_size
    }

    fn forward(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        b.copy_from_slice(a);
        self.ntt(b);
    }

    fn forward_elem_from<T: Copy>(&self, b: &mut [Self::Eval], a: &[T])
    where
        Self: ElemFrom<T>,
    {
        self.slice_elem_from(b, a);
        self.ntt(b);
    }

    fn forward_normalized(&self, b: &mut [Self::Eval], a: &[Self::Elem]) {
        self.forward(b, a);
        self.normalize(b);
    }

    fn backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        b.copy_from_slice(a);
        self.intt(b);
        b.iter_mut().for_each(|b| self.reduce_assign(b));
    }

    fn backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        b.copy_from_slice(a);
        self.intt(b);
        self.normalize(b);
    }

    fn add_backward(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.intt(a);
        izip_eq!(b, a).for_each(|(b, a)| {
            self.reduce_assign(a);
            *b = self.add(a, b);
        })
    }

    fn add_backward_normalized(&self, b: &mut [Self::Elem], a: &mut [Self::Eval]) {
        self.intt(a);
        izip_eq!(b, a).for_each(|(b, a)| *b = self.add(&self.mul_prep(a, &self.n_inv), b))
    }

    fn eval_mul(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        self.slice_mul(c, a, b)
    }

    fn eval_mul_assign(&self, b: &mut [Self::Eval], a: &[Self::Eval]) {
        self.slice_mul_assign(b, a)
    }

    fn eval_fma(&self, c: &mut [Self::Eval], a: &[Self::Eval], b: &[Self::Eval]) {
        self.slice_fma(c, a, b)
    }
}

fn two_adic_generator(q: u64, two_adicity: usize) -> u64 {
    assert_eq!((q - 1) % (1 << two_adicity) as u64, 0);
    pow_mod(multiplicative_generator(q), (q - 1) >> two_adicity, q)
}

fn multiplicative_generator(q: u64) -> u64 {
    let order = q - 1;
    (1..order)
        .find(|g| pow_mod(*g, order >> 1, q) == order)
        .unwrap()
}

#[cfg(test)]
mod test {
    use crate::{
        distribution::Sampler,
        modulus::Prime,
        ring::{
            prime::PrimeRing,
            test::{test_poly_mul, test_round_trip},
            RingOps,
        },
    };
    use rand::thread_rng;

    #[test]
    fn round_trip() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for q in Prime::gen_iter(50, log_ring_size + 1).take(10) {
                let ring = PrimeRing::new(q, 1 << log_ring_size);
                let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                test_round_trip(&ring, &a, |a, b| assert_eq!(a, b));
            }
        }
    }

    #[test]
    fn poly_mul() {
        let mut rng = thread_rng();
        for log_ring_size in 0..10 {
            for q in Prime::gen_iter(50, log_ring_size + 1).take(10) {
                let ring = PrimeRing::new(q, 1 << log_ring_size);
                let a = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                let b = ring.sample_uniform_vec(ring.ring_size(), &mut rng);
                test_poly_mul(&ring, &a, &b, |a, b| assert_eq!(a, b));
            }
        }
    }
}
