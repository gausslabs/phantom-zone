use crate::core::{
    lwe::{self, LweCiphertext, LweCiphertextMutView, LweCiphertextView, LweKeySwitchKeyView},
    rgsw::{self, RgswCiphertextOwned},
    rlwe::{self, RlweAutoKeyOwned, RlweCiphertext, RlweCiphertextMutView, RlwePlaintextView},
};
use core::{cmp::Reverse, iter::successors};
use itertools::{izip, Itertools};
use phantom_zone_math::{
    izip_eq, modulus::NonNativePowerOfTwo, ring::RingOps, util::scratch::Scratch,
};

pub fn bootstrap_scratch_bytes<R: RingOps>(ring: &R, lwe_dimension: usize) -> usize {
    ring.scratch_bytes(
        // 2 (acc) + 2 (automorphism/rlwe_by_rgsw) + ceil(5 * lwe_dimension / ring_size) (ct_ks_mod_switch + i_n_i_p).
        2 + 2 + ((1 + 4) * lwe_dimension).div_ceil(ring.ring_size()),
        // 3 (automorphism/rlwe_by_rgsw)
        3,
        0,
    )
}

/// Implementation of Figure 2 + Algorithm 7 in 2022/198.
///
/// Because we don't need `ak_{g^0}`, `ak_{-g}` is assumed to be stored in
/// `ak[0]` from argument.
pub fn bootstrap<'a, 'b, 'c, R1: RingOps, R2: RingOps>(
    ring: &R1,
    ring_ks: &R2,
    log_g_map: &LogGMap,
    ct: impl Into<LweCiphertextMutView<'a, R1::Elem>>,
    ks_key: impl Into<LweKeySwitchKeyView<'b, R2::Elem>>,
    brk: &[RgswCiphertextOwned<R1::EvalPrep>],
    ak: &[RlweAutoKeyOwned<R1::EvalPrep>],
    f_auto_neg_g: impl Into<RlwePlaintextView<'c, R1::Elem>>,
    mut scratch: Scratch,
) {
    debug_assert_eq!((2 * ring.ring_size()) % log_g_map.q(), 0);
    let (ct, ks_key, f_auto_neg_g) = (ct.into(), ks_key.into(), f_auto_neg_g.into());

    let mut ct_ks_mod_switch = LweCiphertext::scratch(ks_key.to_dimension(), &mut scratch);
    key_switch_mod_switch_odd(
        ring,
        ring_ks,
        log_g_map.q(),
        ct_ks_mod_switch.as_mut_view(),
        ct.as_view(),
        ks_key.as_view(),
        scratch.reborrow(),
    );

    let mut acc = RlweCiphertext::scratch(ring.ring_size(), ring.ring_size(), &mut scratch);
    acc.a_mut().fill(ring.zero());
    let embedding_factor = 2 * ring.ring_size() / log_g_map.q();
    if embedding_factor == 1 {
        acc.b_mut().copy_from_slice(f_auto_neg_g.as_ref());
    } else {
        let acc_b = acc.b_mut().iter_mut().step_by(embedding_factor);
        izip_eq!(acc_b, f_auto_neg_g.as_ref()).for_each(|(b, a)| *b = *a);
    }
    let gb = ak[1].k() * *ct_ks_mod_switch.b() as usize;
    ring.poly_mul_monomial(acc.b_mut(), (embedding_factor * gb) as _);

    blind_rotate_core(
        ring,
        log_g_map,
        &mut acc,
        brk,
        ak,
        ct_ks_mod_switch.a(),
        scratch,
    );

    rlwe::sample_extract(ring, ct, &acc, 0);
}

fn key_switch_mod_switch_odd<R1: RingOps, R2: RingOps>(
    ring: &R1,
    ring_ks: &R2,
    q: usize,
    mut ct_ks_mod_switch: LweCiphertextMutView<u64>,
    ct: LweCiphertextView<R1::Elem>,
    ks_key: LweKeySwitchKeyView<R2::Elem>,
    mut scratch: Scratch,
) {
    let mut ct_mod_switch = LweCiphertext::scratch(ks_key.from_dimension(), &mut scratch);
    ring.slice_mod_switch(ct_mod_switch.as_mut(), ct.as_ref(), ring_ks);

    let mut ct_ks = LweCiphertext::scratch(ks_key.to_dimension(), &mut scratch);
    lwe::key_switch(ring_ks, &mut ct_ks, ks_key, &ct_mod_switch);

    let mod_q = NonNativePowerOfTwo::new(q.ilog2() as _);
    ring_ks.slice_mod_switch_odd(ct_ks_mod_switch.as_mut(), ct_ks.as_ref(), &mod_q);
}

/// Implementation of Algorithm 3 in 2022/198.
///
/// Because we don't need `ak_{g^0}`, `ak_{-g}` is assumed to be stored in
/// `ak[0]` from argument.
pub fn blind_rotate_core<'a, R: RingOps>(
    ring: &R,
    log_g_map: &LogGMap,
    acc: impl Into<RlweCiphertextMutView<'a, R::Elem>>,
    brk: &[RgswCiphertextOwned<R::EvalPrep>],
    ak: &[RlweAutoKeyOwned<R::EvalPrep>],
    a: &[u64],
    mut scratch: Scratch,
) {
    let [i_n, i_p] = &mut i_n_i_p(log_g_map, a, &mut scratch).map(|i| i.iter().peekable());
    let mut acc = acc.into();
    let mut v = 0;
    for l in (1..log_g_map.q() / 4).rev() {
        for (_, j) in i_n.take_while_ref(|(log, _)| *log == l) {
            rgsw::rlwe_by_rgsw_prep_in_place(ring, &mut acc, &brk[*j], scratch.reborrow());
        }
        v += 1;
        let has_adj = i_n.peek().filter(|(log, _)| (*log == l - 1)).is_some();
        if has_adj || v == ak.len() - 1 || l == 1 {
            rlwe::automorphism_prep_in_place(ring, &mut acc, &ak[v], scratch.reborrow());
            v = 0
        }
    }
    for (_, j) in i_n {
        rgsw::rlwe_by_rgsw_prep_in_place(ring, &mut acc, &brk[*j], scratch.reborrow());
    }
    rlwe::automorphism_prep_in_place(ring, &mut acc, &ak[0], scratch.reborrow());
    for l in (1..log_g_map.q() / 4).rev() {
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

/// Returns negative and positive sets of indices `j` (of `a_j`) where
/// `a_j = -g^log` and `a_j = g^log`, and sets are sorted by `log` descendingly.
fn i_n_i_p<'a>(
    log_g_map: &LogGMap,
    a: &[u64],
    scratch: &mut Scratch<'a>,
) -> [&'a [(usize, usize)]; 2] {
    let [i_n, i_p] = scratch.take_slice_array::<(usize, usize), 2>(a.len());
    let mut i_n_count = 0;
    let mut i_p_count = 0;
    izip!(0.., a).for_each(|(j, a_j)| {
        if *a_j != 0 {
            let (sign, log) = log_g_map.index(*a_j as usize);
            if sign {
                i_n[i_n_count] = (log, j);
                i_n_count += 1
            } else {
                i_p[i_p_count] = (log, j);
                i_p_count += 1
            }
        }
    });
    i_n[..i_n_count].sort_by_key(|(log, _)| Reverse(*log));
    i_p[..i_p_count].sort_by_key(|(log, _)| Reverse(*log));
    [&i_n[..i_n_count], &i_p[..i_p_count]]
}

/// Map for both `v` to `log_g(v) mod q` and `-v` to `log_g(v) mod q`, where
/// `q` is power of two and `g` is odd.
///
/// The `map` contains `sign` bit and `log` encoded as `log << 1 | sign`.
/// Also because `g` is odd, `v` will only be odd, the `map` stores the output
/// of `v` in index `v >> 1` to make use of all space.
#[derive(Clone, Debug)]
pub struct LogGMap {
    g: usize,
    q: usize,
    map: Vec<usize>,
}

impl LogGMap {
    /// Returns `LogGMap`.
    ///
    /// # Panics
    ///
    /// Panics if `q` is not power of two or `g` is not odd.
    pub fn new(g: usize, q: usize) -> Self {
        debug_assert!(q.is_power_of_two());
        debug_assert_eq!(g & 1, 1);
        let mut map = vec![0; q / 2];
        izip!(power_g_mod_q(g, q), 0..q / 4).for_each(|(v, log)| {
            map[(v) >> 1] = log << 1;
            map[(q - v) >> 1] = log << 1 | 1;
        });
        Self { g, q, map }
    }

    pub fn g(&self) -> usize {
        self.g
    }

    pub fn q(&self) -> usize {
        self.q
    }

    #[inline(always)]
    pub fn index(&self, v: usize) -> (bool, usize) {
        debug_assert_eq!(v & 1, 1);
        let l = self.map[v >> 1];
        (l & 1 == 1, l >> 1)
    }
}

pub fn power_g_mod_q(g: usize, q: usize) -> impl Iterator<Item = usize> {
    debug_assert!(q.is_power_of_two());
    let mask = q - 1;
    successors(Some(1), move |v| ((v * g) & mask).into())
}

#[cfg(test)]
mod test {
    use crate::{
        core::{
            lwe::test::{Lwe, LweParam},
            rgsw::{
                test::{Rgsw, RgswParam},
                RgswDecompositionParam,
            },
            rlwe::{test::RlweParam, RlwePlaintext},
        },
        scheme::blind_rotation::lmkcdey::{
            bootstrap, bootstrap_scratch_bytes, power_g_mod_q, LogGMap,
        },
        util::rng::StdLweRng,
    };
    use core::{array::from_fn, iter::repeat};
    use itertools::{chain, Itertools};
    use phantom_zone_math::{
        decomposer::DecompositionParam,
        distribution::{Gaussian, Ternary},
        modulus::{Modulus, NonNativePowerOfTwo, Prime},
        poly::automorphism::AutomorphismMap,
        ring::{
            NativeRing, NoisyNativeRing, NoisyNonNativePowerOfTwoRing, NoisyPrimeRing,
            NonNativePowerOfTwoRing, PrimeRing, RingOps,
        },
        util::scratch::ScratchOwned,
    };
    use rand::SeedableRng;

    #[derive(Clone, Copy, Debug)]
    struct BootstrappingParam {
        rgsw: RgswParam,
        lwe_ks: LweParam,
        q: usize,
        g: usize,
        w: usize,
    }

    impl BootstrappingParam {
        fn build<R: RingOps>(self) -> (Rgsw<R>, Lwe<R>, Lwe<NonNativePowerOfTwoRing>) {
            (
                self.rgsw.build(),
                self.rgsw.rlwe.to_lwe().build(),
                self.lwe_ks.build(),
            )
        }
    }

    fn testing_param(big_q: impl Into<Modulus>, embedding_factor: usize) -> BootstrappingParam {
        let message_modulus = 4;
        let ring_size = 1024;
        BootstrappingParam {
            rgsw: RgswParam {
                rlwe: RlweParam {
                    message_modulus,
                    ciphertext_modulus: big_q.into(),
                    ring_size,
                    sk_distribution: Gaussian(3.2).into(),
                    noise_distribution: Gaussian(3.2).into(),
                    u_distribution: Ternary(ring_size / 2).into(),
                    ks_decomposition_param: DecompositionParam {
                        log_base: 24,
                        level: 1,
                    },
                },
                decomposition_param: RgswDecompositionParam {
                    log_base: 17,
                    level_a: 1,
                    level_b: 1,
                },
            },
            lwe_ks: LweParam {
                message_modulus,
                ciphertext_modulus: NonNativePowerOfTwo::new(16).into(),
                dimension: 100,
                sk_distribution: Gaussian(3.2).into(),
                noise_distribution: Gaussian(3.2).into(),
                ks_decomposition_param: DecompositionParam {
                    log_base: 1,
                    level: 13,
                },
            },
            q: 2 * ring_size / embedding_factor,
            g: 5,
            w: 10,
        }
    }

    #[test]
    fn nand() {
        fn run<R: RingOps>(big_q: impl Into<Modulus>, embedding_factor: usize) {
            let mut rng = StdLweRng::from_entropy();
            let param = testing_param(big_q, embedding_factor);
            let (rgsw, lwe, lwe_ks) = param.build::<R>();
            let rlwe = rgsw.rlwe();
            let ring = rlwe.ring();
            let ring_ks = lwe_ks.ring();
            let log_g_map = LogGMap::new(param.g, param.q);
            let mut scratch =
                ScratchOwned::allocate(bootstrap_scratch_bytes(ring, lwe_ks.dimension()));

            let sk = rlwe.sk_gen();
            let sk_ks = lwe_ks.sk_gen();
            let brk = sk_ks
                .as_ref()
                .iter()
                .map(|lwe_ks_sk_i| {
                    let exp = embedding_factor as i32 * lwe_ks_sk_i;
                    let mut pt = RlwePlaintext::allocate(ring.ring_size());
                    ring.poly_set_monomial(pt.as_mut(), exp as _);
                    rgsw.prepare_rgsw(&rgsw.sk_encrypt(&sk, &pt, &mut rng))
                })
                .collect_vec();
            let ak = chain![[param.q - param.g], power_g_mod_q(param.g, param.q).skip(1)]
                .take(param.w + 1)
                .map(|k| rlwe.prepare_auto_key(&rlwe.auto_key_gen(&sk, k as _, &mut rng)))
                .collect_vec();
            let sk = sk.into();
            let ks_key = lwe_ks.ks_key_gen(&sk, &sk_ks, &mut rng);

            let big_q_by_8 = ring.elem_from(ring.modulus().as_f64() / 8f64);
            let nand_lut_auto_neg_g = {
                let q_half = param.q / 2;
                let nand_lut = [true, true, true, false]
                    .into_iter()
                    .map(|v| if v { big_q_by_8 } else { ring.neg(&big_q_by_8) })
                    .flat_map(|v| repeat(v).take(q_half / 4))
                    .collect_vec();
                RlwePlaintext::new(
                    AutomorphismMap::new(q_half, -(param.g as i64))
                        .apply(&nand_lut, |v| ring.neg(v))
                        .collect_vec(),
                    q_half,
                )
            };

            for m in 0..1 << 2 {
                let [a, b] = from_fn(|i| (m >> i) & 1 == 1);
                let ct_a = lwe.sk_encrypt(&sk, lwe.encode(a as _), &mut rng);
                let ct_b = lwe.sk_encrypt(&sk, lwe.encode(b as _), &mut rng);
                let mut ct = lwe.add(&ct_a, &ct_b);
                bootstrap(
                    ring,
                    ring_ks,
                    &log_g_map,
                    &mut ct,
                    &ks_key,
                    &brk,
                    &ak,
                    &nand_lut_auto_neg_g,
                    scratch.borrow_mut(),
                );
                *ct.b_mut() = ring.add(ct.b(), &big_q_by_8);
                assert_eq!(!(a & b) as u64, lwe.decode(lwe.decrypt(&sk, &ct)));
            }
        }

        for embedding_factor in [1, 2] {
            run::<NoisyNativeRing>(Modulus::native(), embedding_factor);
            run::<NoisyNonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(50), embedding_factor);
            run::<NativeRing>(Modulus::native(), embedding_factor);
            run::<NonNativePowerOfTwoRing>(NonNativePowerOfTwo::new(50), embedding_factor);
            run::<NoisyPrimeRing>(Prime::gen(50, 12), embedding_factor);
            run::<PrimeRing>(Prime::gen(50, 12), embedding_factor);
        }
    }
}
