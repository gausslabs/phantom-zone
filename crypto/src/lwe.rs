use crate::misc::{AsMutSlice, AsSlice, SecretKeyDistribution};
use itertools::{izip, Itertools};
use phantom_zone_math::{
    decomposer::{Decomposer, DecompositionParam},
    distribution::{sample_gaussian_vec, sample_ternary_vec},
    modulus::Modulus,
    ring::RingOps,
};
use rand::RngCore;

#[derive(Clone, Copy, Debug)]
pub struct LweParam {
    pub plaintext_modulus: u64,
    pub ciphertext_modulus: Modulus,
    pub dimension: usize,
    pub sk_dist: SecretKeyDistribution,
    pub noise_std_dev: f64,
    pub ks_decomposition_param: Option<DecompositionParam>,
}

impl LweParam {
    pub fn build<R: RingOps>(self) -> Lwe<R> {
        let delta = self.ciphertext_modulus.to_f64() / self.plaintext_modulus as f64;
        let ring = RingOps::new(self.ciphertext_modulus, 1);
        Lwe {
            param: self,
            delta,
            ring,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Lwe<R> {
    param: LweParam,
    delta: f64,
    ring: R,
}

impl<R> Lwe<R> {
    pub fn param(&self) -> &LweParam {
        &self.param
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }
}
impl<R: RingOps> Lwe<R> {
    pub fn ks_decomposer(&self) -> impl Decomposer<R::Elem> + '_ {
        self.ring
            .decomposer(self.param.ks_decomposition_param.unwrap())
    }
}

#[derive(Clone, Debug)]
pub struct LweSecretKey(Vec<i32>);

#[derive(Clone, Copy, Debug)]
pub struct LwePlaintext<T>(T);

#[derive(Clone, Debug)]
pub struct LweCiphertext<S>(S);

pub type LweCiphertextOwned<T> = LweCiphertext<Vec<T>>;

pub type LweCiphertextView<'a, T> = LweCiphertext<&'a [T]>;

pub type LweCiphertextMutView<'a, T> = LweCiphertext<&'a mut [T]>;

impl<S: AsSlice> LweCiphertext<S> {
    pub fn a(&self) -> &[S::Elem] {
        self.a_b().0
    }

    pub fn b(&self) -> &S::Elem {
        self.a_b().1
    }

    pub fn a_b(&self) -> (&[S::Elem], &S::Elem) {
        let (b, a) = self.0.as_ref().split_last().unwrap();
        (a, b)
    }

    pub fn as_view(&self) -> LweCiphertextView<S::Elem> {
        LweCiphertext(self.0.as_ref())
    }
}

impl<S: AsMutSlice> LweCiphertext<S> {
    pub fn a_mut(&mut self) -> &mut [S::Elem] {
        self.a_b_mut().0
    }

    pub fn b_mut(&mut self) -> &mut S::Elem {
        self.a_b_mut().1
    }

    pub fn a_b_mut(&mut self) -> (&mut [S::Elem], &mut S::Elem) {
        let (b, a) = self.0.as_mut().split_last_mut().unwrap();
        (a, b)
    }

    pub fn as_mut_view(&mut self) -> LweCiphertextMutView<S::Elem> {
        LweCiphertext(self.0.as_mut())
    }
}

impl<S: AsSlice> AsRef<[S::Elem]> for LweCiphertext<S> {
    fn as_ref(&self) -> &[S::Elem] {
        self.0.as_ref()
    }
}

impl<S: AsMutSlice> AsMut<[S::Elem]> for LweCiphertext<S> {
    fn as_mut(&mut self) -> &mut [S::Elem] {
        self.0.as_mut()
    }
}

#[derive(Clone, Debug)]
pub struct LweKeySwitchKey<T>(Vec<T>);

impl<T> LweKeySwitchKey<T> {
    pub fn ct_view_iter(&self, param: &LweParam) -> impl Iterator<Item = LweCiphertextView<T>> {
        self.0.chunks(param.dimension + 1).map(LweCiphertext)
    }

    pub fn ct_mut_view_iter(
        &mut self,
        param: &LweParam,
    ) -> impl Iterator<Item = LweCiphertextMutView<T>> {
        self.0.chunks_mut(param.dimension + 1).map(LweCiphertext)
    }
}

impl<R: RingOps> Lwe<R> {
    pub fn sk_gen(&self, rng: impl RngCore) -> LweSecretKey {
        let sk = match self.param.sk_dist {
            SecretKeyDistribution::Gaussian(std_dev) => {
                sample_gaussian_vec(std_dev, self.param.dimension, rng)
            }
            SecretKeyDistribution::Ternary(hamming_weight) => {
                sample_ternary_vec(hamming_weight, self.param.dimension, rng)
            }
        };
        LweSecretKey(sk)
    }

    pub fn encode(&self, m: u64) -> LwePlaintext<R::Elem> {
        let pt = self.ring.elem_from((self.delta * m as f64).round() as u64);
        LwePlaintext(pt)
    }

    pub fn decode(&self, LwePlaintext(pt): LwePlaintext<R::Elem>) -> u64 {
        let pt: u64 = self.ring.elem_to(pt);
        ((pt as f64 / self.delta).round() as u64) % self.param.plaintext_modulus
    }

    pub fn sk_encrypt(
        &self,
        sk: &LweSecretKey,
        LwePlaintext(pt): LwePlaintext<R::Elem>,
        rng: impl RngCore,
    ) -> LweCiphertextOwned<R::Elem> {
        let mut ct = LweCiphertext(vec![self.ring.zero(); self.param.dimension + 1]);
        self.sk_encrypt_inner(ct.as_mut_view(), sk, pt, rng);
        ct
    }

    fn sk_encrypt_inner(
        &self,
        mut ct: LweCiphertextMutView<R::Elem>,
        LweSecretKey(sk): &LweSecretKey,
        pt: R::Elem,
        mut rng: impl RngCore,
    ) {
        let (param, ring) = (self.param(), self.ring());
        ring.sample_uniform_into(ct.a_mut(), &mut rng);
        let a_sk = ring.slice_dot_elem_from(ct.a(), sk);
        let e = ring.sample_gaussian(param.noise_std_dev, &mut rng);
        *ct.b_mut() = ring.add(&ring.add(&a_sk, &e), &pt);
    }

    pub fn decrypt(
        &self,
        LweSecretKey(sk): &LweSecretKey,
        ct: LweCiphertextOwned<R::Elem>,
    ) -> LwePlaintext<R::Elem> {
        let ring = self.ring();
        let a_sk = ring.slice_dot_elem_from(ct.a(), sk);
        let pt = ring.sub(ct.b(), &a_sk);
        LwePlaintext(pt)
    }

    pub fn ksk_gen(
        &self,
        sk_to: &LweSecretKey,
        LweSecretKey(sk_from): &LweSecretKey,
        mut rng: impl RngCore,
    ) -> LweKeySwitchKey<R::Elem> {
        let (param, ring, decomposer) = (self.param(), self.ring(), self.ks_decomposer());
        let mut ksk = LweKeySwitchKey(vec![
            ring.zero();
            (param.dimension + 1)
                * sk_from.len()
                * decomposer.level()
        ]);
        izip!(
            &ksk.ct_mut_view_iter(param).chunks(decomposer.level()),
            sk_from,
        )
        .for_each(|(cts, sk_from_i)| {
            izip!(cts, decomposer.gadget_iter()).for_each(|(ct, b_j)| {
                self.sk_encrypt_inner(ct, sk_to, ring.mul_elem_from(&b_j, &-sk_from_i), &mut rng)
            })
        });
        ksk
    }

    pub fn key_switch(
        &self,
        ksk: &LweKeySwitchKey<R::Elem>,
        ct_from: LweCiphertextOwned<R::Elem>,
    ) -> LweCiphertextOwned<R::Elem> {
        let (param, ring, decomposer) = (self.param(), self.ring(), self.ks_decomposer());
        let mut ct_to = LweCiphertext(vec![self.ring.zero(); self.param.dimension + 1]);
        izip!(
            &ksk.ct_view_iter(param).chunks(decomposer.level()),
            ct_from.a()
        )
        .for_each(|(cts, a_i)| {
            izip!(cts, decomposer.decompose_iter(a_i))
                .for_each(|(ct, a_i_j)| ring.slice_scalar_fma(ct_to.as_mut(), ct.as_ref(), &a_i_j))
        });
        *ct_to.b_mut() = ring.add(ct_to.b(), ct_from.b());
        ct_to
    }
}

#[cfg(test)]
mod test {
    use crate::{
        lwe::{Lwe, LweParam},
        misc::SecretKeyDistribution,
    };
    use phantom_zone_math::{
        decomposer::DecompositionParam,
        modulus::{Modulus, PowerOfTwo, Prime},
        ring::{
            power_of_two::{NativeRing, NonNativePowerOfTwoRing},
            prime::PrimeRing,
            RingOps,
        },
    };
    use rand::{
        distributions::{Distribution, Uniform},
        thread_rng,
    };

    fn test_param(ciphertext_modulus: impl Into<Modulus>) -> LweParam {
        LweParam {
            plaintext_modulus: 1 << 6,
            ciphertext_modulus: ciphertext_modulus.into(),
            dimension: 256,
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
        fn run(lwe: Lwe<impl RingOps>) {
            let mut rng = thread_rng();
            let sk = lwe.sk_gen(&mut rng);
            for _ in 0..100 {
                let m = Uniform::new(0, lwe.param.plaintext_modulus).sample(&mut rng);
                let pt = lwe.encode(m);
                let ct = lwe.sk_encrypt(&sk, pt, &mut rng);
                assert_eq!(m, lwe.decode(pt));
                assert_eq!(m, lwe.decode(lwe.decrypt(&sk, ct)));
            }
        }

        run(test_param(Modulus::native()).build::<NativeRing>());
        run(test_param(PowerOfTwo::new(50)).build::<NonNativePowerOfTwoRing>());
        run(test_param(Prime::gen(50, 0)).build::<PrimeRing>());
    }

    #[test]
    fn key_switch() {
        let lwe = test_param(Modulus::native()).build::<NativeRing>();
        let mut rng = thread_rng();
        let sk_from = lwe.sk_gen(&mut rng);
        let sk_to = lwe.sk_gen(&mut rng);
        let ksk = lwe.ksk_gen(&sk_to, &sk_from, &mut rng);
        for _ in 0..100 {
            let m = Uniform::new(0, lwe.param.plaintext_modulus).sample(&mut rng);
            let ct0 = lwe.sk_encrypt(&sk_from, lwe.encode(m), &mut rng);
            let ct1 = lwe.key_switch(&ksk, ct0);
            assert_eq!(m, lwe.decode(lwe.decrypt(&sk_to, ct1)));
        }
    }
}
