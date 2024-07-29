use core::{fmt::Debug, iter::repeat_with};
use itertools::izip;
use rand::{distributions::Distribution, RngCore};

pub mod po2;
pub mod prime;
pub mod word;

#[allow(clippy::wrong_self_convention)]
pub trait ArithmeticOps {
    type Element: Clone + Copy + Debug + Default + 'static;
    type Prepared: Clone + Copy + Debug + Default + 'static;

    fn zero(&self) -> Self::Element;
    fn one(&self) -> Self::Element;
    fn neg_one(&self) -> Self::Element;

    fn neg(&self, a: &Self::Element) -> Self::Element;
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    fn prepare(&self, a: &Self::Element) -> Self::Prepared;
    fn mul_prepared(&self, a: &Self::Element, b: &Self::Prepared) -> Self::Element;

    fn from_f64(&self, v: f64) -> Self::Element;
    fn from_i64(&self, v: i64) -> Self::Element;
    fn to_i64(&self, v: &Self::Element) -> i64;

    fn sample(&self, dist: impl Distribution<i64>, mut rng: impl RngCore) -> Self::Element {
        self.from_i64(dist.sample(&mut rng))
    }
    fn sample_iter(
        &self,
        dist: impl Distribution<i64>,
        rng: impl RngCore,
    ) -> impl Iterator<Item = Self::Element> {
        dist.sample_iter(rng).map(|v| self.from_i64(v))
    }
    fn sample_vec(
        &self,
        n: usize,
        dist: impl Distribution<i64>,
        mut rng: impl RngCore,
    ) -> Vec<Self::Element> {
        self.sample_iter(dist, &mut rng).take(n).collect()
    }
    fn sample_uniform(&self, rng: impl RngCore) -> Self::Element;
    fn sample_uniform_iter(&self, mut rng: impl RngCore) -> impl Iterator<Item = Self::Element> {
        repeat_with(move || self.sample_uniform(&mut rng))
    }
    fn sample_uniform_vec(&self, n: usize, rng: impl RngCore) -> Vec<Self::Element> {
        self.sample_uniform_iter(rng).take(n).collect()
    }
}

pub trait SliceOps: ArithmeticOps {
    fn slice_neg_assign(&self, a: &mut [Self::Element]) {
        a.iter_mut().for_each(|a| *a = self.neg(a))
    }
    fn slice_add_assign(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        debug_assert_eq!(a.len(), b.len());
        izip!(a, b).for_each(|(a, b)| *a = self.add(a, b))
    }
    fn slice_sub_assign(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        debug_assert_eq!(a.len(), b.len());
        izip!(a, b).for_each(|(a, b)| *a = self.sub(a, b))
    }
    fn slice_mul_assign(&self, a: &mut [Self::Element], b: &[Self::Element]) {
        debug_assert_eq!(a.len(), b.len());
        izip!(a, b).for_each(|(a, b)| *a = self.mul(a, b))
    }
    fn slice_mul_scalar_assign(&self, a: &mut [Self::Element], b: &Self::Element) {
        a.iter_mut().for_each(|a| *a = self.mul(a, b))
    }

    fn slice_neg(&self, b: &mut [Self::Element], a: &[Self::Element]) {
        debug_assert_eq!(a.len(), b.len());
        izip!(b, a).for_each(|(b, a)| *b = self.neg(a))
    }
    fn slice_add(&self, c: &mut [Self::Element], a: &[Self::Element], b: &[Self::Element]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.add(a, b))
    }
    fn slice_sub(&self, c: &mut [Self::Element], a: &[Self::Element], b: &[Self::Element]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.sub(a, b))
    }
    fn slice_mul(&self, c: &mut [Self::Element], a: &[Self::Element], b: &[Self::Element]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.mul(a, b))
    }
    fn slice_mul_scalar(&self, c: &mut [Self::Element], a: &[Self::Element], b: &Self::Element) {
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a).for_each(|(c, a)| *c = self.mul(a, b))
    }

    fn fma(&self, c: &mut [Self::Element], a: &[Self::Element], b: &[Self::Element]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a, b).for_each(|(c, a, b)| *c = self.add(c, &self.mul(a, b)))
    }
    fn fma_scalar(&self, c: &mut [Self::Element], a: &[Self::Element], b: &Self::Element) {
        debug_assert_eq!(a.len(), c.len());
        izip!(c, a).for_each(|(c, a)| *c = self.add(c, &self.mul(a, b)))
    }
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
            izip!(&mut *c, a, b).for_each(|(c, a, b)| *c = self.add(c, &self.mul_prepared(a, b)))
        });
        debug_assert!(a.next().is_none());
        debug_assert!(b.next().is_none());
    }
}

pub trait RingOps: SliceOps {
    type Evaluation: Clone + Copy + Debug + Default + 'static;
    type Twiddle: Clone + Copy + Debug + Default + 'static;

    fn ring_size(&self) -> usize;

    fn evaluation_size(&self) -> usize;

    fn forward(&self, a: Vec<Self::Element>) -> Vec<Self::Evaluation>;

    fn backward(&self, a: Vec<Self::Evaluation>) -> Vec<Self::Element>;

    fn evaluation_mul_assign(&self, a: &mut [Self::Evaluation], b: &[Self::Evaluation]);

    fn ring_mul(&self, a: Vec<Self::Element>, b: Vec<Self::Element>) -> Vec<Self::Element> {
        debug_assert_eq!(a.len(), b.len());
        let mut a = self.forward(a);
        let b = self.forward(b);
        self.evaluation_mul_assign(&mut a, &b);
        self.backward(a)
    }
}
