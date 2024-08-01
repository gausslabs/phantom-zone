use crate::misc::{AsMutSlice, AsSlice, SecretKeyDistribution};
use core::array::from_fn;
use phantom_zone_math::{
    decomposer::{Decomposer, DecompositionParam},
    distribution::{sample_gaussian_vec, sample_ternary_vec},
    izip_eq,
    modulus::Modulus,
    ring::{RingOps, SliceOps},
};
use rand::RngCore;

#[derive(Clone, Copy, Debug)]
pub struct RlweParam {
    pub message_modulus: u64,
    pub ciphertext_modulus: Modulus,
    pub ring_size: usize,
    pub sk_dist: SecretKeyDistribution,
    pub noise_std_dev: f64,
    pub ks_decomposition_param: Option<DecompositionParam>,
}

impl RlweParam {
    pub fn build<R: RingOps>(self) -> Rlwe<R> {
        let delta = self.ciphertext_modulus.to_f64() / self.message_modulus as f64;
        let ring = RingOps::new(self.ciphertext_modulus, self.ring_size);
        Rlwe {
            param: self,
            delta,
            ring,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rlwe<R> {
    param: RlweParam,
    delta: f64,
    ring: R,
}

impl<R> Rlwe<R> {
    pub fn param(&self) -> &RlweParam {
        &self.param
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }
}

impl<R: RingOps> Rlwe<R> {
    pub fn ks_decomposer(&self) -> impl Decomposer<R::Elem> + '_ {
        self.ring
            .decomposer(self.param.ks_decomposition_param.unwrap())
    }
}

#[derive(Clone, Debug)]
pub struct RlweSecretKey(Vec<i32>);

#[derive(Clone, Copy, Debug)]
pub struct RlwePlaintext<S>(S);

pub type RlwePlaintextOwned<T> = RlwePlaintext<Vec<T>>;

pub type RlwePlaintextView<'a, T> = RlwePlaintext<&'a [T]>;

impl<S: AsSlice> RlwePlaintext<S> {
    pub fn as_view(&self) -> RlwePlaintextView<S::Elem> {
        RlwePlaintext(self.0.as_ref())
    }
}

impl<S: AsSlice> AsRef<[S::Elem]> for RlwePlaintext<S> {
    fn as_ref(&self) -> &[S::Elem] {
        self.0.as_ref()
    }
}

impl<S: AsMutSlice> AsMut<[S::Elem]> for RlwePlaintext<S> {
    fn as_mut(&mut self) -> &mut [S::Elem] {
        self.0.as_mut()
    }
}

#[derive(Clone, Debug)]
pub struct RlweCiphertext<S>(S);

pub type RlweCiphertextOwned<T> = RlweCiphertext<Vec<T>>;

pub type RlweCiphertextView<'a, T> = RlweCiphertext<&'a [T]>;

pub type RlweCiphertextMutView<'a, T> = RlweCiphertext<&'a mut [T]>;

impl<S: AsSlice> RlweCiphertext<S> {
    pub fn ring_size(&self) -> usize {
        self.0.len() / 2
    }
    pub fn a(&self) -> &[S::Elem] {
        self.a_b().0
    }

    pub fn b(&self) -> &[S::Elem] {
        self.a_b().1
    }

    pub fn a_b(&self) -> (&[S::Elem], &[S::Elem]) {
        let (b, a) = self.0.as_ref().split_at(self.ring_size());
        (a, b)
    }

    pub fn as_view(&self) -> RlweCiphertextView<S::Elem> {
        RlweCiphertext(self.0.as_ref())
    }
}

impl<S: AsMutSlice> RlweCiphertext<S> {
    pub fn a_mut(&mut self) -> &mut [S::Elem] {
        self.a_b_mut().0
    }

    pub fn b_mut(&mut self) -> &mut [S::Elem] {
        self.a_b_mut().1
    }

    pub fn a_b_mut(&mut self) -> (&mut [S::Elem], &mut [S::Elem]) {
        let ring_size = self.ring_size();
        let (b, a) = self.0.as_mut().split_at_mut(ring_size);
        (a, b)
    }

    pub fn as_mut_view(&mut self) -> RlweCiphertextMutView<S::Elem> {
        RlweCiphertext(self.0.as_mut())
    }
}

impl<T> RlweCiphertext<Vec<T>> {
    pub fn into_b(mut self) -> Vec<T> {
        let ring_size = self.ring_size();
        self.0.truncate(ring_size);
        self.0
    }
}

impl<S: AsSlice> AsRef<[S::Elem]> for RlweCiphertext<S> {
    fn as_ref(&self) -> &[S::Elem] {
        self.0.as_ref()
    }
}

impl<S: AsMutSlice> AsMut<[S::Elem]> for RlweCiphertext<S> {
    fn as_mut(&mut self) -> &mut [S::Elem] {
        self.0.as_mut()
    }
}

#[derive(Clone, Debug)]
pub struct RlweKeySwitchKey<T>(Vec<T>);

impl<T> RlweKeySwitchKey<T> {
    pub fn ct_view_iter(&self, param: &RlweParam) -> impl Iterator<Item = RlweCiphertextView<T>> {
        self.0.chunks(2 * param.ring_size).map(RlweCiphertext)
    }

    pub fn ct_mut_view_iter(
        &mut self,
        param: &RlweParam,
    ) -> impl Iterator<Item = RlweCiphertextMutView<T>> {
        self.0.chunks_mut(2 * param.ring_size).map(RlweCiphertext)
    }
}

impl<R: RingOps> Rlwe<R> {
    pub fn sk_gen(&self, rng: impl RngCore) -> RlweSecretKey {
        let sk = match self.param.sk_dist {
            SecretKeyDistribution::Gaussian(std_dev) => {
                sample_gaussian_vec(std_dev, self.param.ring_size, rng)
            }
            SecretKeyDistribution::Ternary(hamming_weight) => {
                sample_ternary_vec(hamming_weight, self.param.ring_size, rng)
            }
        };
        RlweSecretKey(sk)
    }

    pub fn encode(&self, m: Vec<u64>) -> RlwePlaintextOwned<R::Elem> {
        let encode = |m| self.ring.elem_from((self.delta * m as f64).round() as u64);
        let pt = m.into_iter().map(encode).collect();
        RlwePlaintext(pt)
    }

    pub fn decode(&self, RlwePlaintext(pt): RlwePlaintextView<R::Elem>) -> Vec<u64> {
        let decode = |pt: &_| {
            let pt: u64 = self.ring.elem_to(*pt);
            ((pt as f64 / self.delta).round() as u64) % self.param.message_modulus
        };
        pt.iter().map(decode).collect()
    }

    pub fn sk_encrypt(
        &self,
        sk: &RlweSecretKey,
        pt: RlwePlaintextView<R::Elem>,
        rng: impl RngCore,
    ) -> RlweCiphertextOwned<R::Elem> {
        let mut ct = RlweCiphertext(vec![self.ring.zero(); 2 * self.param.ring_size]);
        let mut scratch = vec![R::Eval::default(); self.ring.scratch_size()];
        self.sk_encrypt_inner(ct.as_mut_view(), sk, pt, &mut scratch, rng);
        ct
    }

    fn sk_encrypt_inner(
        &self,
        mut ct: RlweCiphertextMutView<R::Elem>,
        RlweSecretKey(sk): &RlweSecretKey,
        RlwePlaintext(pt): RlwePlaintextView<R::Elem>,
        scratch: &mut [R::Eval],
        mut rng: impl RngCore,
    ) {
        let (param, ring) = (self.param(), self.ring());
        let (a, b) = ct.a_b_mut();
        ring.sample_uniform_into(a, &mut rng);
        ring.poly_mul_elem_from(b, a, sk, scratch);
        let e = ring.sample_gaussian_iter(param.noise_std_dev, &mut rng);
        ring.slice_add_assign_iter(b, e);
        ring.slice_add_assign(b, pt);
    }

    pub fn decrypt(
        &self,
        RlweSecretKey(sk): &RlweSecretKey,
        mut ct: RlweCiphertextOwned<R::Elem>,
    ) -> RlwePlaintextOwned<R::Elem> {
        let ring = self.ring();
        let (a, b) = ct.a_b_mut();
        let mut scratch = vec![R::Eval::default(); self.ring.scratch_size()];
        ring.poly_mul_assign_elem_from(a, sk, &mut scratch);
        ring.slice_sub_assign(b, &*a);
        RlwePlaintext(ct.into_b())
    }

    pub fn ksk_gen(
        &self,
        sk_to: &RlweSecretKey,
        RlweSecretKey(sk_from): &RlweSecretKey,
        mut rng: impl RngCore,
    ) -> RlweKeySwitchKey<R::Elem> {
        let (param, ring, decomposer) = (self.param(), self.ring(), self.ks_decomposer());
        let mut ksk = RlweKeySwitchKey(vec![
            ring.zero();
            (2 * param.ring_size) * decomposer.level()
        ]);
        let mut scratch = vec![R::Eval::default(); self.ring.scratch_size()];
        let mut pt = RlwePlaintext(vec![ring.zero(); param.ring_size]);
        izip_eq!(ksk.ct_mut_view_iter(param), decomposer.gadget_iter()).for_each(|(ct, beta_j)| {
            ring.slice_elem_from(pt.as_mut(), sk_from);
            ring.slice_scalar_mul_assign(pt.as_mut(), &ring.neg(&beta_j));
            self.sk_encrypt_inner(ct, sk_to, pt.as_view(), &mut scratch, &mut rng)
        });
        ksk
    }

    pub fn key_switch(
        &self,
        ksk: &RlweKeySwitchKey<R::Elem>,
        mut ct_from: RlweCiphertextOwned<R::Elem>,
    ) -> RlweCiphertextOwned<R::Elem> {
        let (param, ring, decomposer) = (self.param(), self.ring(), self.ks_decomposer());
        let [mut a_eval, mut b_eval] = from_fn(|_| vec![R::Eval::default(); ring.eval_size()]);
        let mut scratch: [_; 3] = from_fn(|_| vec![R::Eval::default(); ring.eval_size()]);
        decomposer.slice_decompose_zip_for_each(
            ct_from.a_mut(),
            ksk.ct_view_iter(param),
            &mut vec![self.ring.zero(); self.param.ring_size],
            |(a_i, ct)| {
                ring.forward(&mut scratch[0], a_i);
                ring.forward(&mut scratch[1], ct.a());
                ring.forward(&mut scratch[2], ct.b());
                ring.eval().slice_fma(&mut a_eval, &scratch[0], &scratch[1]);
                ring.eval().slice_fma(&mut b_eval, &scratch[0], &scratch[2]);
            },
        );
        let mut ct_to = RlweCiphertext(vec![self.ring.zero(); 2 * self.param.ring_size]);
        ring.backward_normalized(ct_to.a_mut(), &mut a_eval);
        ring.backward_normalized(ct_to.b_mut(), &mut b_eval);
        ring.slice_add_assign(ct_to.b_mut(), ct_from.b());
        ct_to
    }
}

#[cfg(test)]
mod test {
    use crate::{
        misc::SecretKeyDistribution,
        rlwe::{Rlwe, RlweParam},
    };
    use phantom_zone_math::{
        decomposer::DecompositionParam,
        distribution::sample_uniform_vec,
        modulus::{Modulus, PowerOfTwo, Prime},
        ring::{
            power_of_two::{NativeRing, NonNativePowerOfTwoRing},
            prime::PrimeRing,
            RingOps,
        },
    };
    use rand::thread_rng;

    fn test_param(ciphertext_modulus: impl Into<Modulus>) -> RlweParam {
        RlweParam {
            message_modulus: 1 << 6,
            ciphertext_modulus: ciphertext_modulus.into(),
            ring_size: 256,
            sk_dist: SecretKeyDistribution::Gaussian(3.2),
            noise_std_dev: 3.2,
            ks_decomposition_param: Some(DecompositionParam {
                log_base: 8,
                level: 6,
            }),
        }
    }

    #[test]
    fn encrypt_decrypt() {
        fn run(rlwe: Rlwe<impl RingOps>) {
            let mut rng = thread_rng();
            let param = rlwe.param();
            let sk = rlwe.sk_gen(&mut rng);
            for _ in 0..100 {
                let m = sample_uniform_vec(param.ring_size, 0..param.message_modulus, &mut rng);
                let pt = rlwe.encode(m.clone());
                let ct = rlwe.sk_encrypt(&sk, pt.as_view(), &mut rng);
                assert_eq!(m, rlwe.decode(pt.as_view()));
                assert_eq!(m, rlwe.decode(rlwe.decrypt(&sk, ct).as_view()));
            }
        }

        run(test_param(Modulus::native()).build::<NativeRing>());
        run(test_param(PowerOfTwo::new(50)).build::<NonNativePowerOfTwoRing>());
        run(test_param(Prime::gen(50, 9)).build::<PrimeRing>());
    }

    #[test]
    fn key_switch() {
        let mut rng = thread_rng();
        let rlwe = test_param(Modulus::native()).build::<NativeRing>();
        let param = rlwe.param();
        let sk_from = rlwe.sk_gen(&mut rng);
        let sk_to = rlwe.sk_gen(&mut rng);
        let ksk = rlwe.ksk_gen(&sk_to, &sk_from, &mut rng);
        for _ in 0..100 {
            let m = sample_uniform_vec(param.ring_size, 0..param.message_modulus, &mut rng);
            let ct0 = rlwe.sk_encrypt(&sk_from, rlwe.encode(m.clone()).as_view(), &mut rng);
            let ct1 = rlwe.key_switch(&ksk, ct0);
            assert_eq!(m, rlwe.decode(rlwe.decrypt(&sk_to, ct1).as_view()));
        }
    }
}
