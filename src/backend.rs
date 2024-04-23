use itertools::izip;

pub trait VectorOps {
    type Element;

    fn elwise_scalar_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &Self::Element);
    fn elwise_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &[Self::Element]);

    fn elwise_add_mut(&self, a: &mut [Self::Element], b: &[Self::Element]);
    fn elwise_mul_mut(&self, a: &mut [Self::Element], b: &[Self::Element]);
    fn elwise_neg_mut(&self, a: &mut [Self::Element]);
    /// inplace mutates `a`: a = a + b*c
    fn elwise_fma_mut(&self, a: &mut [Self::Element], b: &[Self::Element], c: &[Self::Element]);

    fn modulus(&self) -> Self::Element;
}

pub trait ArithmeticOps {
    type Element;

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    fn modulus(&self) -> Self::Element;
}

pub struct ModularOpsU64 {
    q: u64,
    logq: usize,
    barrett_mu: u128,
    barrett_alpha: usize,
}

impl ModularOpsU64 {
    pub fn new(q: u64) -> ModularOpsU64 {
        let logq = 64 - q.leading_zeros();

        // barrett calculation
        let mu = (1u128 << (logq * 2 + 3)) / (q as u128);
        let alpha = logq + 3;

        ModularOpsU64 {
            q,
            logq: logq as usize,
            barrett_alpha: alpha as usize,
            barrett_mu: mu,
        }
    }

    fn add_mod_fast(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q);
        debug_assert!(b < self.q);

        let mut o = a + b;
        if o >= self.q {
            o -= self.q;
        }
        o
    }

    fn sub_mod_fast(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q);
        debug_assert!(b < self.q);

        if a > b {
            a - b
        } else {
            (self.q + a) - b
        }
    }

    /// returns (a * b)  % q
    ///
    /// - both a and b must be in range [0, 2q)
    /// - output is in range [0 , q)
    fn mul_mod_fast(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < 2 * self.q);
        debug_assert!(b < 2 * self.q);

        let ab = a as u128 * b as u128;

        // ab / (2^{n + \beta})
        // note: \beta is assumed to -2
        let tmp = ab >> (self.logq - 2);

        // k = ((ab / (2^{n + \beta})) * \mu) / 2^{\alpha - (-2)}
        let k = (tmp * self.barrett_mu) >> (self.barrett_alpha + 2);

        // ab - k*p
        let tmp = k * (self.q as u128);

        let mut out = (ab - tmp) as u64;

        if out >= self.q {
            out -= self.q;
        }

        return out;
    }
}

impl ArithmeticOps for ModularOpsU64 {
    type Element = u64;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.add_mod_fast(*a, *b)
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul_mod_fast(*a, *b)
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.sub_mod_fast(*a, *b)
    }

    fn modulus(&self) -> Self::Element {
        self.q
    }
}

impl VectorOps for ModularOpsU64 {
    type Element = u64;

    fn elwise_add_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| {
            *ai = self.add_mod_fast(*ai, *bi);
        });
    }

    fn elwise_mul_mut(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| {
            *ai = self.mul_mod_fast(*ai, *bi);
        });
    }

    fn elwise_neg_mut(&self, a: &mut [Self::Element]) {
        a.iter_mut().for_each(|ai| *ai = self.q - *ai);
    }

    fn elwise_scalar_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &Self::Element) {
        izip!(out.iter_mut(), a.iter()).for_each(|(oi, ai)| {
            *oi = self.mul_mod_fast(*ai, *b);
        });
    }

    fn elwise_mul(&self, out: &mut [Self::Element], a: &[Self::Element], b: &[Self::Element]) {
        izip!(out.iter_mut(), a.iter(), b.iter()).for_each(|(oi, ai, bi)| {
            *oi = self.mul_mod_fast(*ai, *bi);
        });
    }

    fn elwise_fma_mut(&self, a: &mut [Self::Element], b: &[Self::Element], c: &[Self::Element]) {
        izip!(a.iter_mut(), b.iter(), c.iter()).for_each(|(ai, bi, ci)| {
            *ai = self.add_mod_fast(*ai, self.mul_mod_fast(*bi, *ci));
        });
    }

    fn modulus(&self) -> Self::Element {
        self.q
    }
}
