use crate::core::{
    lwe::LweCiphertextView,
    rlwe::{self, RlweAutoKeyOwned, RlweCiphertext, RlweCiphertextMutView, RlweCiphertextOwned},
};
use itertools::Itertools;
use phantom_zone_math::{ring::RingOps, util::scratch::Scratch};

pub fn pack_lwes<'a, 'b, R: RingOps>(
    ring: &R,
    aks: &[RlweAutoKeyOwned<R::EvalPrep>],
    cts: impl IntoIterator<Item: Into<LweCiphertextView<'b, R::Elem>>>,
) -> RlweCiphertextOwned<R::Elem> {
    let cts = cts.into_iter().map_into().collect_vec();
    let ell = cts.len().next_power_of_two().ilog2() as usize;
    let mut scratch = ring.allocate_scratch(2 + 2 * (ell + 2), 3, 0);
    let (ct, _) = recurse(ring, aks, &cts, 1, 0, scratch.borrow_mut());
    return ct
        .map(|ct| ct.cloned())
        .unwrap_or_else(|| RlweCiphertextOwned::allocate(ring.ring_size()));

    fn recurse<'a, R: RingOps>(
        ring: &R,
        aks: &[RlweAutoKeyOwned<R::EvalPrep>],
        cts: &[LweCiphertextView<R::Elem>],
        step: usize,
        skip: usize,
        mut scratch: Scratch<'a>,
    ) -> (Option<RlweCiphertextMutView<'a, R::Elem>>, Scratch<'a>) {
        let n = ring.ring_size();
        let k = cts.len().next_power_of_two();
        let ell = (k / step).ilog2() as usize;
        let inv = ring.inv(&ring.elem_from(k as u64));

        if ell == 0 {
            return (
                cts.get(skip).map(|ct_0| {
                    let mut ct = RlweCiphertext::scratch(n, n, &mut scratch);
                    if let Some(inv) = &inv {
                        ring.slice_scalar_mul(ct.a_mut(), ct_0.a(), inv);
                        ct.b_mut()[0] = ring.mul(ct_0.b(), inv);
                    } else {
                        ct.a_mut().copy_from_slice(ct_0.a());
                        ct.b_mut()[0] = *ct_0.b();
                    }
                    ct.b_mut()[1..].fill(ring.zero());
                    ct
                }),
                scratch,
            );
        }

        let (ct_even, mut scratch) = recurse(ring, aks, cts, 2 * step, skip, scratch);
        let (ct_odd, mut tmp) = recurse(ring, aks, cts, 2 * step, skip + step, scratch.reborrow());
        let mut ct = match (ct_even, ct_odd) {
            (Some(mut ct_even), Some(mut ct_odd)) => {
                let ct_odd_rot = {
                    ring.poly_mul_assign_monomial(ct_odd.a_mut(), (n >> ell) as _);
                    ring.poly_mul_assign_monomial(ct_odd.b_mut(), (n >> ell) as _);
                    ct_odd
                };
                let ct_auto = {
                    let mut ct_auto = RlweCiphertext::scratch(n, n, &mut tmp);
                    ring.slice_sub(ct_auto.as_mut(), ct_even.as_ref(), ct_odd_rot.as_ref());
                    rlwe::automorphism_prep_in_place(ring, &mut ct_auto, &aks[ell - 1], tmp);
                    ct_auto
                };
                ring.slice_add_assign(ct_even.as_mut(), ct_odd_rot.as_ref());
                ring.slice_add_assign(ct_even.as_mut(), ct_auto.as_ref());
                Some(ct_even)
            }
            (Some(mut ct_even), None) => {
                let ct_auto = {
                    let mut ct_auto = RlweCiphertext::scratch(n, n, &mut tmp);
                    ct_auto.as_mut().copy_from_slice(ct_even.as_ref());
                    rlwe::automorphism_prep_in_place(ring, &mut ct_auto, &aks[ell - 1], tmp);
                    ct_auto
                };
                ring.slice_add_assign(ct_even.as_mut(), ct_auto.as_ref());
                Some(ct_even)
            }
            (None, None) => None,
            _ => unreachable!(),
        };
        match &mut ct {
            Some(ct) if inv.is_none() => {
                let rounding_shr = |a: &_| ring.elem_from(ring.to_u64(*a).wrapping_add(1) >> 1);
                ct.as_mut().iter_mut().for_each(|a| *a = rounding_shr(a))
            }
            _ => {}
        }
        (ct, scratch)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        core::rlwe::{test::RlweParam, RlweSecretKey},
        scheme::packing,
        util::rng::StdLweRng,
    };
    use core::{iter::repeat_with, ops::Neg};
    use itertools::{izip, Itertools};
    use phantom_zone_math::{
        decomposer::DecompositionParam,
        distribution::{Gaussian, Ternary},
        modulus::{Modulus, Native, Prime},
        poly::automorphism::AutomorphismMap,
        ring::{NativeRing, PrimeRing, RingOps},
    };
    use rand::{Rng, SeedableRng};

    pub fn test_param(ciphertext_modulus: impl Into<Modulus>) -> RlweParam {
        let ring_size = 1024usize;
        RlweParam {
            message_modulus: 4,
            ciphertext_modulus: ciphertext_modulus.into(),
            ring_size,
            sk_distribution: Ternary.into(),
            u_distribution: Ternary.into(),
            noise_distribution: Gaussian(3.19).into(),
            ks_decomposition_param: DecompositionParam {
                log_base: 17,
                level: 1,
            },
        }
    }

    #[test]
    fn pack_lwes() {
        fn run<R: RingOps>(modulus: impl Into<Modulus>) {
            let mut rng = StdLweRng::from_entropy();
            let param = test_param(modulus);
            let rlwe = param.build::<R>();
            let lwe = param.to_lwe().build::<R>();

            let sk = lwe.sk_gen(&mut rng);
            let sk_rlwe = RlweSecretKey::new(
                AutomorphismMap::new(rlwe.ring_size(), 2 * rlwe.ring_size() - 1)
                    .apply(sk.as_ref(), |i| i.neg())
                    .collect(),
                rlwe.ring_size(),
            );
            let aks = (1..)
                .map(|ell| {
                    rlwe.prepare_auto_key(&rlwe.auto_key_gen(&sk_rlwe, (1 << ell) + 1, &mut rng))
                })
                .take(rlwe.ring_size().ilog2() as usize)
                .collect_vec();
            let ms = repeat_with(|| rng.gen_range(0..param.message_modulus))
                .take(rlwe.ring_size())
                .collect_vec();
            let cts = ms
                .iter()
                .map(|m| lwe.sk_encrypt(&sk, lwe.encode(*m), &mut rng))
                .collect_vec();
            for k in 0..=rlwe.ring_size() {
                let ell = k.next_power_of_two().ilog2() as usize;
                let ct = packing::pack_lwes(rlwe.ring(), &aks, &cts[..k]);
                let m = rlwe.decode(&rlwe.decrypt(&sk_rlwe, &ct));
                izip!(&ms[..k], m.iter().step_by(rlwe.ring_size() >> ell)).for_each(|(a, b)| {
                    assert!(
                        *a == *b
                            || (matches!(param.ciphertext_modulus, Modulus::PowerOfTwo(_))
                                && (*a as i64 - *b as i64).unsigned_abs()
                                    == param.message_modulus / 2)
                    );
                });
            }
        }

        run::<NativeRing>(Native::native());
        run::<PrimeRing>(Prime::gen(54, 11));
    }
}
