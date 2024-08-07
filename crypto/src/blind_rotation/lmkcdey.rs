use crate::{
    lwe::{self, LweCiphertext, LweCiphertextMutView, LweKeySwitchKeyView},
    rgsw::{self, RgswCiphertextOwned},
    rlwe::{self, RlweAutoKeyOwned, RlweCiphertext, RlweCiphertextMutView, RlwePlaintextView},
};
use core::cmp::Reverse;
use itertools::{izip, Itertools};
use phantom_zone_math::{
    misc::scratch::Scratch,
    modulus::{powers_mod, PowerOfTwo},
    ring::{ModulusOps, NonNativePowerOfTwoRing, RingOps},
};

// Figure 2 + Algorithm 7 in 2022/198.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap<'a, 'b, 'c, R1: RingOps, R2: RingOps>(
    ring: &R1,
    ring_ks: &R2,
    ct: impl Into<LweCiphertextMutView<'a, R1::Elem>>,
    ks_key: impl Into<LweKeySwitchKeyView<'b, R2::Elem>>,
    f_auto_neg_g: impl Into<RlwePlaintextView<'c, R1::Elem>>,
    brk: &[RgswCiphertextOwned<R1::EvalPrep>],
    ak: &[RlweAutoKeyOwned<R1::EvalPrep>],
    mut scratch: Scratch,
) {
    let (ct, ks_key, f_auto_neg_g) = (ct.into(), ks_key.into(), f_auto_neg_g.into());
    let n_twice_mod = {
        let modulus = PowerOfTwo::new((2 * ring.ring_size()).ilog2() as _);
        NonNativePowerOfTwoRing::new(modulus.into(), 1)
    };

    let mut ct_big_q_ks = LweCiphertext::scratch(ks_key.from_dimension(), &mut scratch);
    ring.slice_mod_switch(ct_big_q_ks.as_mut(), ct.as_ref(), ring_ks);

    let mut ct_ks = LweCiphertext::scratch(ks_key.to_dimension(), &mut scratch);
    lwe::key_switch(ring_ks, &mut ct_ks, ks_key, &ct_big_q_ks);

    let mut ct_n_twice = LweCiphertext::scratch(ks_key.to_dimension(), &mut scratch);
    ring_ks.slice_mod_switch_odd(ct_n_twice.as_mut(), ct_ks.as_ref(), &n_twice_mod);

    let mut acc = RlweCiphertext::scratch(ring.ring_size(), ring.ring_size(), &mut scratch);
    acc.a_mut().fill_with(Default::default);
    acc.b_mut().copy_from_slice(f_auto_neg_g.as_ref());
    let gb = ak[1].k() as i64 * n_twice_mod.to_u64(*ct_n_twice.b()) as i64;
    ring.poly_mul_monomial(acc.b_mut(), gb as _);

    blind_rotate_core(ring, &mut acc, ct_n_twice.a(), brk, ak, scratch);

    rlwe::sample_extract(ring, ct, &acc, 0);
}

// Algorithm 3 in 2022/198.
pub fn blind_rotate_core<'a, R: RingOps>(
    ring: &R,
    acc: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    a: &[u64],
    brk: &[RgswCiphertextOwned<R::EvalPrep>],
    ak: &[RlweAutoKeyOwned<R::EvalPrep>],
    mut scratch: Scratch,
) {
    let [i_m, i_p] = &mut i_m_i_p(ring, ak[1].k(), a, &mut scratch).map(|i| i.iter().peekable());
    let mut acc = acc.into();
    let mut v = 0;
    for l in (1..ring.ring_size() / 2).rev() {
        for (_, j) in i_m.take_while_ref(|(log, _)| *log == l) {
            rgsw::rlwe_by_rgsw_prep_in_place(ring, &mut acc, &brk[*j], scratch.reborrow());
        }
        v += 1;
        let has_adj = i_m.peek().filter(|(log, _)| (*log == l - 1)).is_some();
        if has_adj || v == ak.len() - 1 || l == 1 {
            rlwe::automorphism_prep_in_place(ring, &mut acc, &ak[v], scratch.reborrow());
            v = 0
        }
    }
    for (_, j) in i_m {
        rgsw::rlwe_by_rgsw_prep_in_place(ring, &mut acc, &brk[*j], scratch.reborrow());
    }
    rlwe::automorphism_prep_in_place(ring, &mut acc, &ak[0], scratch.reborrow());
    for l in (1..ring.ring_size() / 2).rev() {
        for (_, j) in i_p.take_while_ref(|(log, _)| *log == l) {
            rgsw::rlwe_by_rgsw_prep_in_place(ring, &mut acc, &brk[*j], scratch.reborrow());
        }
        v += 1;
        let has_adj = i_p.peek().filter(|(log, _)| (*log == l - 1)).is_some();
        if has_adj || v == ak.len() - 1 || l == 1 {
            rlwe::automorphism_prep_in_place(ring, &mut acc, &ak[v], scratch.reborrow());
            v = 0
        }
    }
    for (_, j) in i_p {
        rgsw::rlwe_by_rgsw_prep_in_place(ring, &mut acc, &brk[*j], scratch.reborrow());
    }
}

fn i_m_i_p<'a, R: RingOps>(
    ring: &R,
    g: usize,
    a: &[u64],
    scratch: &mut Scratch<'a>,
) -> [&'a [(usize, usize)]; 2] {
    let [i_m, i_p] = scratch.take_slice_array::<(usize, usize), 2>(a.len());
    let mut i_m_count = 0;
    let mut i_p_count = 0;
    let log_g_map = log_g_map(ring, g, scratch.reborrow());
    izip!(0.., a).for_each(|(j, a_j)| {
        let log = log_g_map[*a_j as usize];
        if *a_j != 0 {
            debug_assert_ne!(log, usize::MAX);
            if log & 1 == 1 {
                i_m[i_m_count] = (log >> 1, j);
                i_m_count += 1
            } else {
                i_p[i_p_count] = (log >> 1, j);
                i_p_count += 1
            }
        }
    });
    i_m[..i_m_count].sort_by_key(|(l, _)| Reverse(*l));
    i_p[..i_p_count].sort_by_key(|(l, _)| Reverse(*l));
    [&i_m[..i_m_count], &i_p[..i_p_count]]
}

fn log_g_map<'a, R: RingOps>(ring: &R, g: usize, mut scratch: Scratch<'a>) -> &'a mut [usize] {
    let n_twice = 2 * ring.ring_size();
    let log_g_map = scratch.take_slice(n_twice as _);
    #[cfg(debug_assertions)]
    log_g_map.fill(usize::MAX);
    izip!(powers_mod(g as _, n_twice as _), 0..ring.ring_size() / 2).for_each(|(v, i)| {
        log_g_map[v as usize] = i << 1;
        log_g_map[n_twice - v as usize] = i << 1 | 1;
    });
    log_g_map
}

#[cfg(test)]
mod test {
    use crate::{
        blind_rotation::lmkcdey::bootstrap,
        lwe::test::LweParam,
        rgsw::test::RgswParam,
        rlwe::{test::RlweParam, RlwePlaintext},
    };
    use core::{array::from_fn, iter::repeat, mem::size_of};
    use itertools::{chain, Itertools};
    use phantom_zone_math::{
        decomposer::DecompositionParam,
        distribution::Gaussian,
        misc::scratch::ScratchOwned,
        modulus::{powers_mod, Modulus, PowerOfTwo, Prime},
        ring::{
            NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
            NonNativePowerOfTwoRing, PrimeRing, RingOps,
        },
    };
    use rand::thread_rng;

    fn bootstrapping_param(
        modulus: impl Into<Modulus>,
    ) -> (RgswParam, LweParam, LweParam, u64, usize) {
        let rgsw_param = RgswParam {
            rlwe: RlweParam {
                message_modulus: 4,
                ciphertext_modulus: modulus.into(),
                ring_size: 512,
                sk_distribution: Gaussian::new(3.2).into(),
                noise_distribution: Gaussian::new(3.2).into(),
                ks_decomposition_param: DecompositionParam {
                    log_base: 24,
                    level: 1,
                },
            },
            decomposition_log_base: 17,
            decomposition_level_a: 1,
            decomposition_level_b: 1,
        };
        let lwe_param = LweParam {
            message_modulus: rgsw_param.message_modulus,
            ciphertext_modulus: rgsw_param.ciphertext_modulus,
            dimension: rgsw_param.ring_size,
            sk_distribution: rgsw_param.sk_distribution,
            noise_distribution: rgsw_param.noise_distribution,
            ks_decomposition_param: DecompositionParam {
                log_base: 0,
                level: 0,
            },
        };
        let lwe_ks_param = LweParam {
            message_modulus: rgsw_param.message_modulus,
            ciphertext_modulus: PowerOfTwo::new(16).into(),
            dimension: 200,
            sk_distribution: Gaussian::new(3.2).into(),
            noise_distribution: Gaussian::new(3.2).into(),
            ks_decomposition_param: DecompositionParam {
                log_base: 1,
                level: 13,
            },
        };
        let g = 5;
        let w = 10;
        (rgsw_param, lwe_param, lwe_ks_param, g, w)
    }

    #[test]
    fn nand() {
        fn run<R: RingOps>(big_q: impl Copy + Into<Modulus>) {
            let mut rng = thread_rng();
            let big_q = big_q.into();
            let (rgsw_param, lwe_param, lwe_ks_param, g, w) = bootstrapping_param(big_q);
            let rgsw = rgsw_param.build::<R>();
            let lwe = lwe_param.build::<R>();
            let lwe_ks = lwe_ks_param.build::<NonNativePowerOfTwoRing>();
            let rlwe = rgsw.rlwe();
            let ring = rlwe.ring();
            let mut scratch = {
                let elem = 5 * ring.ring_size()
                    + (ring.ring_size() + 1)
                    + 2 * (lwe_ks.dimension() + 1)
                    + (4 * lwe_ks.dimension() + 1);
                let eval = 2 * ring.eval_size();
                ScratchOwned::allocate(size_of::<R::Elem>() * elem + size_of::<R::Eval>() * eval)
            };
            let mut scratch = scratch.borrow_mut();

            let rlwe_sk = rlwe.sk_gen(&mut rng);
            let lwe_sk = rlwe_sk.clone().into();
            let lwe_ks_sk = lwe_ks.sk_gen(&mut rng);
            let ks_key = lwe_ks.ks_key_gen(&lwe_sk, &lwe_ks_sk, &mut rng);
            let brk = lwe_ks_sk
                .as_ref()
                .iter()
                .map(|lwe_ks_sk_i| {
                    let mut pt = RlwePlaintext::allocate(ring.ring_size());
                    ring.poly_set_monomial(pt.as_mut(), *lwe_ks_sk_i as _);
                    rgsw.prepare_rgsw(&rgsw.sk_encrypt(&rlwe_sk, &pt, &mut rng))
                })
                .collect_vec();
            let ak = {
                let n_twice = 2 * ring.ring_size() as u64;
                chain![[n_twice - g], powers_mod(g, n_twice).skip(1)]
                    .take(w)
                    .map(|k| rlwe.prepare_auto_key(&rlwe.auto_key_gen(&rlwe_sk, k as _, &mut rng)))
                    .collect_vec()
            };
            let big_q_by_8 = ring.elem_from(big_q.to_f64() / 8f64);
            let nand_lut_auto_neg_g = {
                let nand_lut = [true, true, true, false]
                    .into_iter()
                    .map(|v| if v { big_q_by_8 } else { ring.neg(&big_q_by_8) })
                    .flat_map(|v| repeat(v).take(ring.ring_size() / 4))
                    .collect_vec();
                let mut nand_lut_auto_neg_g = RlwePlaintext::allocate(ring.ring_size());
                ring.poly_add_auto(nand_lut_auto_neg_g.as_mut(), &nand_lut, ak[0].auto_map());
                nand_lut_auto_neg_g
            };

            for m in 0..1 << 2 {
                let [a, b] = from_fn(|i| (m >> i) & 1 == 1);
                let ct_a = lwe.sk_encrypt(&lwe_sk, lwe.encode(a as _), &mut rng);
                let ct_b = lwe.sk_encrypt(&lwe_sk, lwe.encode(b as _), &mut rng);
                let mut ct = lwe.add(&ct_a, &ct_b);
                bootstrap(
                    ring,
                    lwe_ks.ring(),
                    &mut ct,
                    &ks_key,
                    &nand_lut_auto_neg_g,
                    &brk,
                    &ak,
                    scratch.reborrow(),
                );
                *ct.b_mut() = ring.add(ct.b(), &big_q_by_8);
                assert_eq!(!(a & b) as u64, lwe.decode(lwe.decrypt(&lwe_sk, &ct)));
            }
        }

        run::<PrimeRing>(Prime::gen(50, 10));
        run::<NoisyNativeRing>(Modulus::native());
        run::<NoisyNonNativePowerOfTwoRing>(PowerOfTwo::new(50));
        run::<NativeRing>(Modulus::native());
        run::<NonNativePowerOfTwoRing>(PowerOfTwo::new(50));
        run::<NoisyPrimeRing>(Prime::gen(50, 10));
    }
}
