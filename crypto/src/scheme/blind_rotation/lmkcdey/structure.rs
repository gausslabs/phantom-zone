use crate::{
    core::{
        lwe::{LweKeySwitchKey, LweKeySwitchKeyMutView, LweKeySwitchKeyView},
        rgsw::{RgswCiphertext, RgswCiphertextMutView, RgswCiphertextView, RgswDecompositionParam},
        rlwe::{RlweAutoKey, RlweAutoKeyMutView, RlweAutoKeyView},
    },
    util::distribution::{NoiseDistribution, SecretDistribution},
};
use core::{
    borrow::Borrow,
    fmt::Debug,
    iter::{repeat, successors},
};
use itertools::{chain, izip};
use phantom_zone_math::{
    decomposer::DecompositionParam,
    modulus::{Modulus, ModulusOps},
    poly::automorphism::AutomorphismMap,
    ring::RingOps,
    util::{
        as_slice::{AsMutSlice, AsSlice},
        compact::Compact,
    },
};

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    pub fn aks(&self) -> impl Iterator<Item = usize> {
        chain![
            [self.q - self.g],
            powers_mod_q(self.g, self.q).skip(1).take(self.w)
        ]
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
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(into = "SerdeLogGMap", from = "SerdeLogGMap")
)]
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

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct SerdeLogGMap {
    g: usize,
    q: usize,
}

#[cfg(feature = "serde")]
impl From<SerdeLogGMap> for LogGMap {
    fn from(value: SerdeLogGMap) -> Self {
        Self::new(value.g, value.q)
    }
}

#[cfg(feature = "serde")]
impl From<LogGMap> for SerdeLogGMap {
    fn from(value: LogGMap) -> Self {
        Self {
            g: value.g,
            q: value.q,
        }
    }
}

pub type LmkcdeyKeyOwned<T1, T2> = LmkcdeyKey<Vec<T1>, Vec<T2>, AutomorphismMap>;
pub type LmkcdeyKeyCompact = LmkcdeyKey<Compact, Compact, AutomorphismMap>;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LmkcdeyKey<S1, S2, A> {
    param: LmkcdeyParam,
    ks_key: LweKeySwitchKey<S2>,
    brks: Vec<RgswCiphertext<S1>>,
    aks: Vec<RlweAutoKey<S1, A>>,
    log_g_map: LogGMap,
}

impl<S1, S2, A> LmkcdeyKey<S1, S2, A> {
    fn new(
        param: LmkcdeyParam,
        ks_key: LweKeySwitchKey<S2>,
        brks: Vec<RgswCiphertext<S1>>,
        aks: Vec<RlweAutoKey<S1, A>>,
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
}

impl<S1: AsSlice, S2: AsSlice, A: Borrow<AutomorphismMap>> LmkcdeyKey<S1, S2, A> {
    pub fn ks_key(&self) -> LweKeySwitchKeyView<S2::Elem> {
        self.ks_key.as_view()
    }

    pub fn brks(&self) -> impl Iterator<Item = RgswCiphertextView<S1::Elem>> {
        self.brks.iter().map(RgswCiphertext::as_view)
    }

    pub fn brk(&self, idx: usize) -> RgswCiphertextView<S1::Elem> {
        self.brks[idx].as_view()
    }

    pub fn aks(&self) -> impl Iterator<Item = RlweAutoKeyView<S1::Elem>> {
        self.aks.iter().map(RlweAutoKey::as_view)
    }

    pub fn ak(&self, idx: usize) -> RlweAutoKeyView<S1::Elem> {
        self.aks[idx].as_view()
    }

    pub fn ak_neg_g(&self) -> RlweAutoKeyView<S1::Elem> {
        self.ak(0)
    }

    pub fn log_g_map(&self) -> &LogGMap {
        &self.log_g_map
    }

    pub fn compact(
        &self,
        ring: &impl ModulusOps<Elem = S1::Elem>,
        mod_ks: &impl ModulusOps<Elem = S2::Elem>,
    ) -> LmkcdeyKeyCompact {
        LmkcdeyKey::new(
            self.param,
            self.ks_key.compact(mod_ks),
            self.brks.iter().map(|brk| brk.compact(ring)).collect(),
            self.aks.iter().map(|ak| ak.compact(ring)).collect(),
        )
    }
}

impl<S1: AsMutSlice, S2: AsMutSlice, A: Borrow<AutomorphismMap>> LmkcdeyKey<S1, S2, A> {
    pub(crate) fn ks_key_mut(&mut self) -> LweKeySwitchKeyMutView<S2::Elem> {
        self.ks_key.as_mut_view()
    }

    pub(crate) fn brks_mut(&mut self) -> impl Iterator<Item = RgswCiphertextMutView<S1::Elem>> {
        self.brks.iter_mut().map(RgswCiphertext::as_mut_view)
    }

    pub(crate) fn aks_mut(&mut self) -> impl Iterator<Item = RlweAutoKeyMutView<S1::Elem>> {
        self.aks.iter_mut().map(RlweAutoKey::as_mut_view)
    }
}

impl<T1: Clone + Default, T2: Default> LmkcdeyKeyOwned<T1, T2> {
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
        let aks = param
            .aks()
            .map(|k| RlweAutoKey::allocate(param.ring_size, param.auto_decomposition_param, k))
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
        let aks = param
            .aks()
            .map(|k| {
                RlweAutoKey::allocate_eval(
                    param.ring_size,
                    eval_size,
                    param.auto_decomposition_param,
                    k,
                )
            })
            .collect();
        Self::new(param, ks_key, brks, aks)
    }
}

impl LmkcdeyKeyCompact {
    pub fn uncompact<M1, M2>(&self, ring: &M1, mod_ks: &M2) -> LmkcdeyKeyOwned<M1::Elem, M2::Elem>
    where
        M1: ModulusOps,
        M2: ModulusOps,
    {
        LmkcdeyKey::new(
            self.param,
            self.ks_key.uncompact(mod_ks),
            self.brks.iter().map(|brk| brk.uncompact(ring)).collect(),
            self.aks.iter().map(|ak| ak.uncompact(ring)).collect(),
        )
    }
}
