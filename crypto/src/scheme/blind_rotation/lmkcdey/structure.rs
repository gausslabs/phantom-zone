use crate::{
    core::{
        lwe::{
            LweKeySwitchKey, LweKeySwitchKeyMutView, LweKeySwitchKeyOwned, LweKeySwitchKeyView,
            SeededLweKeySwitchKey, SeededLweKeySwitchKeyMutView, SeededLweKeySwitchKeyOwned,
            SeededLweKeySwitchKeyView,
        },
        rgsw::{
            RgswCiphertext, RgswCiphertextMutView, RgswCiphertextOwned, RgswCiphertextView,
            RgswDecompositionParam,
        },
        rlwe::{
            RlweAutoKey, RlweAutoKeyMutView, RlweAutoKeyOwned, RlweAutoKeyView, SeededRlweAutoKey,
            SeededRlweAutoKeyMutView, SeededRlweAutoKeyOwned, SeededRlweAutoKeyView,
        },
    },
    util::distribution::{NoiseDistribution, SecretDistribution},
};
use core::{
    iter::{repeat, successors},
    ops::Deref,
};
use itertools::{chain, izip};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    modulus::{Modulus, ModulusOps},
    ring::RingOps,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmkcdeyParam {
    // Rlwe param
    pub modulus: Modulus,
    pub ring_size: usize,
    pub sk_distribution: SecretDistribution,
    pub noise_distribution: NoiseDistribution,
    pub auto_decomposition_param: DecompositionParam,
    pub rlwe_by_rgsw_decomposition_param: RgswDecompositionParam,
    // Lwe param
    pub lwe_modulus: Modulus,
    pub lwe_dimension: usize,
    pub lwe_sk_distribution: SecretDistribution,
    pub lwe_noise_distribution: NoiseDistribution,
    pub lwe_ks_decomposition_param: DecompositionParam,
    // Blind rotation param
    pub q: usize,
    pub g: usize,
    pub w: usize,
}

impl LmkcdeyParam {
    pub fn embedding_factor(&self) -> usize {
        2 * self.ring_size / self.q
    }

    pub fn scratch_bytes<R: RingOps, M: ModulusOps>(&self, ring: &R, _: &M) -> usize {
        let mut bytes;
        bytes = ring.scratch_bytes(
            // 2 (acc) + 2 (automorphism/rlwe_by_rgsw).
            2 + 2,
            // 3 (automorphism/rlwe_by_rgsw)
            3,
            0,
        );
        // ct_ks_mod_switch
        bytes = bytes.next_multiple_of(size_of::<M::Elem>());
        bytes += self.lwe_dimension * size_of::<M::Elem>();
        // i_n_i_p
        bytes = bytes.next_multiple_of(size_of::<(usize, usize)>());
        bytes += 4 * self.lwe_dimension * size_of::<(usize, usize)>();
        bytes
    }
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
        izip!(powers_mod_q(g, q), 0..q / 4).for_each(|(v, log)| {
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

fn powers_mod_q(g: usize, q: usize) -> impl Iterator<Item = usize> {
    debug_assert!(q.is_power_of_two());
    let mask = q - 1;
    successors(Some(1), move |v| ((v * g) & mask).into())
}

pub struct LmkcdeyKey<T1, T2> {
    param: LmkcdeyParam,
    ks_key: LweKeySwitchKeyOwned<T2>,
    brks: Vec<RgswCiphertextOwned<T1>>,
    aks: Vec<RlweAutoKeyOwned<T1>>,
    log_g_map: LogGMap,
}

impl<T1, T2> LmkcdeyKey<T1, T2> {
    fn new(
        param: LmkcdeyParam,
        ks_key: LweKeySwitchKeyOwned<T2>,
        brks: Vec<RgswCiphertextOwned<T1>>,
        aks: Vec<RlweAutoKeyOwned<T1>>,
    ) -> Self {
        let log_g_map = LogGMap::new(param.g, param.q);
        Self {
            param,
            ks_key,
            brks,
            aks,
            log_g_map,
        }
    }

    pub fn param(&self) -> &LmkcdeyParam {
        &self.param
    }

    pub fn q(&self) -> usize {
        self.param.q
    }

    pub fn g(&self) -> usize {
        self.param.g
    }

    pub fn w(&self) -> usize {
        self.param.w
    }

    pub fn embedding_factor(&self) -> usize {
        self.param.embedding_factor()
    }

    pub fn ks_key(&self) -> LweKeySwitchKeyView<T2> {
        self.ks_key.as_view()
    }

    pub fn brks(&self) -> impl Iterator<Item = RgswCiphertextView<T1>> {
        self.brks.iter().map(RgswCiphertext::as_view)
    }

    pub fn brk(&self, idx: usize) -> RgswCiphertextView<T1> {
        self.brks[idx].as_view()
    }

    pub fn aks(&self) -> impl Iterator<Item = RlweAutoKeyView<T1>> {
        self.aks.iter().map(RlweAutoKey::as_view)
    }

    pub fn ak(&self, idx: usize) -> RlweAutoKeyView<T1> {
        self.aks[idx].as_view()
    }

    pub fn ak_neg_g(&self) -> RlweAutoKeyView<T1> {
        self.ak(0)
    }

    pub fn log_g_map(&self) -> &LogGMap {
        &self.log_g_map
    }

    pub(crate) fn ks_key_mut(&mut self) -> LweKeySwitchKeyMutView<T2> {
        self.ks_key.as_mut_view()
    }

    pub(crate) fn brks_mut(&mut self) -> impl Iterator<Item = RgswCiphertextMutView<T1>> {
        self.brks.iter_mut().map(RgswCiphertext::as_mut_view)
    }

    pub(crate) fn aks_mut(&mut self) -> impl Iterator<Item = RlweAutoKeyMutView<T1>> {
        self.aks.iter_mut().map(RlweAutoKey::as_mut_view)
    }
}

impl<T1: Default + Clone, T2: Default> LmkcdeyKey<T1, T2> {
    pub fn allocate(param: LmkcdeyParam) -> Self {
        let ks_key = LweKeySwitchKey::allocate(
            param.ring_size,
            param.lwe_dimension,
            param.lwe_ks_decomposition_param,
        );
        let brks = repeat(RgswCiphertext::allocate(
            param.ring_size,
            param.rlwe_by_rgsw_decomposition_param,
        ))
        .take(param.lwe_dimension)
        .collect();
        let aks = chain![[param.q - param.g], powers_mod_q(param.g, param.q).skip(1)]
            .take(param.w + 1)
            .map(|k| RlweAutoKey::allocate(param.ring_size, param.auto_decomposition_param, k as _))
            .collect();
        Self::new(param, ks_key, brks, aks)
    }

    pub fn allocate_eval(param: LmkcdeyParam, eval_size: usize) -> Self {
        let ks_key = LweKeySwitchKey::allocate(
            param.ring_size,
            param.lwe_dimension,
            param.lwe_ks_decomposition_param,
        );
        let brks = repeat(RgswCiphertext::allocate_eval(
            param.ring_size,
            eval_size,
            param.rlwe_by_rgsw_decomposition_param,
        ))
        .take(param.lwe_dimension)
        .collect();
        let aks = chain![[param.q - param.g], powers_mod_q(param.g, param.q).skip(1)]
            .take(param.w + 1)
            .map(|k| {
                RlweAutoKey::allocate_eval(
                    param.ring_size,
                    eval_size,
                    param.auto_decomposition_param,
                    k as _,
                )
            })
            .collect();
        Self::new(param, ks_key, brks, aks)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LmkcdeyInteractiveParam {
    pub param: LmkcdeyParam,
    pub u_distribution: SecretDistribution,
    pub rgsw_by_rgsw_decomposition_param: RgswDecompositionParam,
}

impl Deref for LmkcdeyInteractiveParam {
    type Target = LmkcdeyParam;

    fn deref(&self) -> &Self::Target {
        &self.param
    }
}

pub struct LmkcdeyKeyShare<T1, T2> {
    param: LmkcdeyInteractiveParam,
    ks_key: SeededLweKeySwitchKeyOwned<T2>,
    brks: Vec<RgswCiphertextOwned<T1>>,
    aks: Vec<SeededRlweAutoKeyOwned<T1>>,
    share_idx: usize,
    total_shares: usize,
}

impl<T1, T2> LmkcdeyKeyShare<T1, T2> {
    pub fn param(&self) -> &LmkcdeyInteractiveParam {
        &self.param
    }

    pub fn share_idx(&self) -> usize {
        self.share_idx
    }

    pub fn total_shares(&self) -> usize {
        self.total_shares
    }

    pub fn ks_key(&self) -> SeededLweKeySwitchKeyView<T2> {
        self.ks_key.as_view()
    }

    pub fn brks(&self) -> impl Iterator<Item = RgswCiphertextView<T1>> {
        self.brks.iter().map(RgswCiphertext::as_view)
    }

    pub fn aks(&self) -> impl Iterator<Item = SeededRlweAutoKeyView<T1>> {
        self.aks.iter().map(SeededRlweAutoKey::as_view)
    }

    pub(crate) fn ks_key_mut(&mut self) -> SeededLweKeySwitchKeyMutView<T2> {
        self.ks_key.as_mut_view()
    }

    pub(crate) fn brks_mut(&mut self) -> impl Iterator<Item = RgswCiphertextMutView<T1>> {
        self.brks.iter_mut().map(RgswCiphertext::as_mut_view)
    }

    pub(crate) fn aks_mut(&mut self) -> impl Iterator<Item = SeededRlweAutoKeyMutView<T1>> {
        self.aks.iter_mut().map(SeededRlweAutoKey::as_mut_view)
    }
}

impl<T1: Default + Clone, T2: Default> LmkcdeyKeyShare<T1, T2> {
    fn new(
        param: LmkcdeyInteractiveParam,
        ks_key: SeededLweKeySwitchKeyOwned<T2>,
        brks: Vec<RgswCiphertextOwned<T1>>,
        aks: Vec<SeededRlweAutoKeyOwned<T1>>,
        share_idx: usize,
        total_shares: usize,
    ) -> Self {
        debug_assert!(total_shares > 0);
        debug_assert!(share_idx < total_shares);
        Self {
            param,
            ks_key,
            brks,
            aks,
            share_idx,
            total_shares,
        }
    }

    pub fn allocate(param: LmkcdeyInteractiveParam, share_idx: usize, total_shares: usize) -> Self {
        let ks_key = SeededLweKeySwitchKey::allocate(
            param.ring_size,
            param.lwe_dimension,
            param.lwe_ks_decomposition_param,
        );
        let brks = {
            let chunk_size = param.lwe_dimension.div_ceil(total_shares);
            let init_range = chunk_size * share_idx..chunk_size * (share_idx + 1);
            (0..param.lwe_dimension)
                .map(|idx| {
                    let rlwe_by_rgsw_decomposition_param = if init_range.contains(&idx) {
                        param.rlwe_by_rgsw_decomposition_param
                    } else {
                        param.rgsw_by_rgsw_decomposition_param
                    };
                    RgswCiphertext::allocate(param.ring_size, rlwe_by_rgsw_decomposition_param)
                })
                .collect()
        };
        let aks = chain![[param.q - param.g], powers_mod_q(param.g, param.q).skip(1)]
            .take(param.w + 1)
            .map(|k| {
                SeededRlweAutoKey::allocate(param.ring_size, param.auto_decomposition_param, k as _)
            })
            .collect();
        Self::new(param, ks_key, brks, aks, share_idx, total_shares)
    }
}
