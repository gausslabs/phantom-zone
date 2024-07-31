use crate::misc::SecretKeyDistribution;
use phantom_zone_math::{
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

pub struct LweSecretKey(Vec<i32>);

pub struct LwePlaintext<R: RingOps>(R::Elem);

pub struct LweCiphertext<R: RingOps>(Vec<R::Elem>);

impl<R: RingOps> LweCiphertext<R> {
    pub fn a(&self) -> &[R::Elem] {
        &self.0[1..]
    }

    pub fn b(&self) -> &R::Elem {
        &self.0[0]
    }

    pub fn a_mut(&mut self) -> &mut [R::Elem] {
        &mut self.0[1..]
    }

    pub fn b_mut(&mut self) -> &mut R::Elem {
        &mut self.0[0]
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

    pub fn encode(&self, m: u64) -> LwePlaintext<R> {
        let pt = self.ring.elem_from((self.delta * m as f64).round() as u64);
        LwePlaintext(pt)
    }

    pub fn decode(&self, LwePlaintext(pt): LwePlaintext<R>) -> u64 {
        let pt: u64 = self.ring.elem_to(pt);
        ((pt as f64 / self.delta).round() as u64) % self.param.plaintext_modulus
    }

    pub fn sk_encrypt(
        &self,
        LweSecretKey(sk): &LweSecretKey,
        LwePlaintext(pt): LwePlaintext<R>,
        mut rng: impl RngCore,
    ) -> LweCiphertext<R> {
        let (param, ring) = (self.param(), self.ring());
        let mut ct = LweCiphertext(vec![ring.zero(); param.dimension + 1]);
        ring.sample_uniform_into(ct.a_mut(), &mut rng);
        let a_sk = ring.slice_dot_elem_from(ct.a(), sk);
        let e = ring.sample_gaussian(param.noise_std_dev, &mut rng);
        *ct.b_mut() = ring.add(&ring.add(&a_sk, &e), &pt);
        ct
    }

    pub fn decrypt(
        &self,
        LweSecretKey(sk): &LweSecretKey,
        ct: LweCiphertext<R>,
    ) -> LwePlaintext<R> {
        let ring = self.ring();
        let a_sk = ring.slice_dot_elem_from(ct.a(), sk);
        let pt = ring.sub(ct.b(), &a_sk);
        LwePlaintext(pt)
    }

    pub fn ksk_gen(&self) {
        todo!()
    }

    pub fn key_switch(&self) {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        lwe::{Lwe, LweParam},
        misc::SecretKeyDistribution,
    };
    use phantom_zone_math::{
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
                let ct = lwe.sk_encrypt(&sk, lwe.encode(m), &mut rng);
                assert_eq!(m, lwe.decode(pt));
                assert_eq!(m, lwe.decode(lwe.decrypt(&sk, ct)));
            }
        }

        run(test_param(Modulus::native()).build::<NativeRing>());
        run(test_param(PowerOfTwo::new(50)).build::<NonNativePowerOfTwoRing>());
        run(test_param(Prime::gen(50, 0)).build::<PrimeRing>());
    }
}
