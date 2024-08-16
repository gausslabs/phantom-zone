use itertools::Itertools;
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    modulus::Modulus,
    util::as_slice::{self, AsMutSlice, AsSlice},
};
use std::iter::repeat_with;

use crate::{
    core::{
        lwe::{SeededLweKeySwitchKey, SeededLweKeySwitchKeyMutView},
        rgsw::RgswDecompositionParam,
        rlwe::{RlweAutoKey, SeededRlweAutoKey},
    },
    scheme::blind_rotation::lmkcdey::power_g_mod_q,
    util::distribution::NoiseDistribution,
};

// trait Parameters {
//     fn lwe_n(&self) -> usize;
//     fn rlwe_n(&self) -> usize;
//     fn lwe_q(&self) -> usize;
//     fn rlwe_q(&self) -> usize;
//     fn lwe_ksk_decomposer(&self) -> DecompositionParam;
//     fn rlwe_x_rgsw_decomposer(&self) -> RgswDecompositionParam;
//     fn auto_decomposer(&self) -> DecompositionParam;
//     fn w(&self) -> usize;
// }

pub struct InteractiveMpcParameters {
    lwe_n: usize,
    rlwe_n: usize,
    lwe_q: Modulus,
    rlwe_q: Modulus,
    br_q: usize,
    lwe_ksk_decomposer: DecompositionParam,
    rlwe_x_rgsw_decomposer: RgswDecompositionParam,
    auto_decomposer: DecompositionParam,
    rgsw_x_rgsw_decomposer: RgswDecompositionParam,
    noise_distribution: NoiseDistribution,
    w: usize,
    g: usize,
}

impl InteractiveMpcParameters {
    pub fn auto_decomposer(&self) -> DecompositionParam {
        self.auto_decomposer.clone()
    }
    pub fn lwe_n(&self) -> usize {
        self.lwe_n
    }
    pub fn rlwe_n(&self) -> usize {
        self.rlwe_n
    }
    pub fn rlwe_x_rgsw_decomposer(&self) -> RgswDecompositionParam {
        self.rlwe_x_rgsw_decomposer.clone()
    }
    pub fn lwe_ksk_decomposer(&self) -> DecompositionParam {
        self.lwe_ksk_decomposer.clone()
    }
    pub fn w(&self) -> usize {
        self.w
    }
    pub fn lwe_q(&self) -> Modulus {
        self.lwe_q
    }
    pub fn rlwe_q(&self) -> Modulus {
        self.rlwe_q
    }
    pub fn noise_distribution(&self) -> NoiseDistribution {
        self.noise_distribution
    }
    fn g(&self) -> usize {
        self.g
    }
    fn br_q(&self) -> usize {
        self.br_q
    }
}

impl InteractiveMpcParameters {
    fn rgsw_x_rgsw_decomposer(&self) -> RgswDecompositionParam {
        self.rgsw_x_rgsw_decomposer.clone()
    }
}

/// Divides LWE dimension into two chunks. The first contains indices for which `self` generates RGSW ciphertexts for RLWE x RGSW and second contains indices for which `self`  generates RGSW for RGSW x RGSW
fn split_lwe_dimension_for_rgsws(
    lwe_n: usize,
    user_id: usize,
    total_users: usize,
) -> (Vec<usize>, Vec<usize>) {
    assert!(user_id < total_users);
    let chunk_size = lwe_n / total_users;
    let self_indices =
        ((user_id * chunk_size)..std::cmp::min((user_id + 1) * chunk_size, lwe_n)).collect_vec();
    let not_self_indices = (0..(user_id * chunk_size))
        .chain(((user_id + 1) * chunk_size)..lwe_n)
        .collect_vec();
    (self_indices, not_self_indices)
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RlwePublicKeyShare<S> {
    #[as_slice]
    data: S,
}

impl<S: AsSlice> RlwePublicKeyShare<S> {
    pub fn new(data: S) -> Self {
        Self { data }
    }
    pub fn ring_size(&self) -> usize {
        self.data.len()
    }
}

impl<T: Default> RlwePublicKeyShare<Vec<T>> {
    pub fn allocate(n: usize) -> Self {
        Self::new(repeat_with(T::default).take(n).collect())
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct RgswCiphertextList<S> {
    #[as_slice]
    data: S,
}

impl<S: AsSlice> RgswCiphertextList<S> {
    fn new(data: S) -> Self {
        Self { data }
    }
}

impl<T: Default> RgswCiphertextList<Vec<T>> {
    fn allocate(
        count: usize,
        ring_size: usize,
        decomposition_param: RgswDecompositionParam,
    ) -> Self {
        Self::new(
            repeat_with(T::default)
                .take(
                    count
                        * ((2 * ring_size * decomposition_param.level_a)
                            + (2 * ring_size * decomposition_param.level_b)),
                )
                .collect(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct InteractiveServerKeyShare<S1, S2: AsSlice<Elem = usize>> {
    seeded_lwe_ksk: SeededLweKeySwitchKey<S1>,
    seeded_auto_keys: Vec<SeededRlweAutoKey<S1, S2>>,
    // #[as_slice]
    // seeded_self_rgsw_cts: S,
    // #[as_slice]
    // seeded_not_self_rgsw_cts: S,
}

impl<S1: AsSlice, S2: AsSlice<Elem = usize>> InteractiveServerKeyShare<S1, S2> {
    pub fn new(
        lwe_ksk: SeededLweKeySwitchKey<S1>,
        auto_keys: Vec<SeededRlweAutoKey<S1, S2>>,
        // self_rgsw_cts: S,
        // not_self_rgsw_cts: S,
    ) -> Self {
        Self {
            seeded_lwe_ksk: lwe_ksk,
            seeded_auto_keys: auto_keys,
            // seeded_self_rgsw_cts: self_rgsw_cts,
            // seeded_not_self_rgsw_cts: not_self_rgsw_cts,
        }
    }
}

impl<T: Default> InteractiveServerKeyShare<Vec<T>, Vec<usize>> {
    fn allocate(parameters: InteractiveMpcParameters, user_id: usize, total_users: usize) -> Self {
        let (self_count, not_self_count) =
            split_lwe_dimension_for_rgsws(parameters.lwe_n(), user_id, total_users);

        Self::new(
            SeededLweKeySwitchKey::allocate(
                parameters.rlwe_n(),
                parameters.lwe_n(),
                parameters.lwe_ksk_decomposer(),
            ),
            // Auto keys for k = [-g, g^1, ..., g^w]
            (([(parameters.br_q() - parameters.g()) as i64].into_iter()).chain(
                power_g_mod_q(parameters.g(), parameters.br_q())
                    .take(parameters.w() + 1)
                    .skip(1)
                    .map(|k| k as i64),
            ))
            .map(|k| {
                SeededRlweAutoKey::allocate(parameters.rlwe_n(), parameters.auto_decomposer(), k)
            })
            .collect(),
            // repeat_with(T::default)
            //     .take(
            //         (parameters.w() + 1) * parameters.auto_decomposer().level * parameters.rlwe_n(),
            //     )
            //     .collect(),
            // repeat_with(T::default)
            //     .take(
            //         self_count.len()
            //             * ((2 * parameters.rlwe_n() * parameters.rlwe_x_rgsw_decomposer().level_a)
            //                 + (2 * parameters.rlwe_n()
            //                     * parameters.rlwe_x_rgsw_decomposer().level_b)),
            //     )
            //     .collect(),
            // repeat_with(T::default)
            //     .take(
            //         not_self_count.len()
            //             * ((2 * parameters.rlwe_n() * parameters.rgsw_x_rgsw_decomposer().level_a)
            //                 + (2 * parameters.rlwe_n()
            //                     * parameters.rgsw_x_rgsw_decomposer().level_b)),
            //     )
            //     .collect(),
        )
    }
}

// impl<S: AsMutSlice> InteractiveServerKeyShare<S> {
//     pub fn as_mut_lwe_ksk(&mut self) -> SeededLweKeySwitchKeyMutView<S::Elem> {
//         self.seeded_lwe_ksk.as_mut_view()
//     }

// pub fn as_mut_auto_keys(&mut self) -> S::MutView {
//     self.seeded_auto_keys.as_mut_view()
// }

// pub fn as_mut_self_rgsw_cts(&mut self) -> RgswCiphertextListMutView<S::Elem> {
//     self.seeded_self_rgsw_cts.as_mut_view()
// }

// pub fn as_mut_not_self_rgsw_cts(&mut self) -> RgswCiphertextListMutView<S::Elem> {
//     self.seeded_not_self_rgsw_cts.as_mut_view()
// }
// }
