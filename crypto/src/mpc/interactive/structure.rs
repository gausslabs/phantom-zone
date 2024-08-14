use itertools::Itertools;
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    util::as_slice::{AsMutSlice, AsSlice},
};
use std::iter::repeat_with;

use crate::core::{lwe::LweKeySwitchKey, rgsw::RgswDecompositionParam};

trait Parameters {
    fn lwe_n(&self) -> usize;
    fn rlwe_n(&self) -> usize;
    fn lwe_ksk_decomposer(&self) -> DecompositionParam;
    fn rlwe_x_rgsw_decomposer(&self) -> RgswDecompositionParam;
    fn auto_decomposer(&self) -> DecompositionParam;
    fn w(&self) -> usize;
}

struct InteractiveMpcParameters {
    lwe_n: usize,
    rlwe_n: usize,
    lwe_ksk_decomposer: DecompositionParam,
    rlwe_x_rgsw_decomposer: RgswDecompositionParam,
    auto_decomposer: DecompositionParam,
    rgsw_x_rgsw_decomposer: RgswDecompositionParam,
    w: usize,
}

impl Parameters for InteractiveMpcParameters {
    fn auto_decomposer(&self) -> DecompositionParam {
        self.auto_decomposer.clone()
    }
    fn lwe_n(&self) -> usize {
        self.lwe_n
    }
    fn rlwe_n(&self) -> usize {
        self.rlwe_n
    }
    fn rlwe_x_rgsw_decomposer(&self) -> RgswDecompositionParam {
        self.rlwe_x_rgsw_decomposer.clone()
    }
    fn lwe_ksk_decomposer(&self) -> DecompositionParam {
        self.lwe_ksk_decomposer.clone()
    }
    fn w(&self) -> usize {
        self.w
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

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct SeededRlweKeyCiphertext<S> {
    data: S,
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct SeededLweKeySwitchKey<S> {
    data: S,
}

impl<S: AsSlice> SeededLweKeySwitchKey<S> {
    fn new(data: S) -> Self {
        Self { data }
    }
}

#[derive(Clone, Copy, Debug, AsSliceWrapper)]
pub struct InteractiveServerKeyShare<S0, S1, S2, S3> {
    #[as_slice]
    lwe_ksk: S0,
    #[as_slice]
    auto_keys: S1,
    #[as_slice]
    self_rgsw_cts: S2,
    #[as_slice]
    not_self_rgsw_cts: S3,
}

// impl<S: AsSlice> InteractiveServerKeyShare<S> {
//     pub fn new(
//         lwe_ksk: LweKeySwitchKey<S>,
//         auto_keys: S,
//         self_rgsw_cts: RgswCiphertextList<S>,
//         not_self_rgsw_cts: RgswCiphertextList<S>,
//     ) -> Self {
//         Self {
//             lwe_ksk,
//             auto_keys,
//             self_rgsw_cts,
//             not_self_rgsw_cts,
//         }
//     }
// }

// impl<T: Default> InteractiveServerKeyShare<Vec<T>> {
//     fn allocate(parameters: InteractiveMpcParameters, user_id: usize, total_users: usize) -> Self {
//         let (self_count, not_self_count) =
//             split_lwe_dimension_for_rgsws(parameters.lwe_n(), user_id, total_users);
//         Self::new(
//             repeat_with(T::default)
//                 .take(parameters.rlwe_n() * parameters.lwe_ksk_decomposer().level)
//                 .collect(),
//             repeat_with(T::default)
//                 .take(
//                     (parameters.w() + 1) * parameters.auto_decomposer().level * parameters.rlwe_n(),
//                 )
//                 .collect(),
//             RgswCiphertextList::allocate(
//                 self_count.len(),
//                 parameters.rlwe_n(),
//                 parameters.rlwe_x_rgsw_decomposer(),
//             ),
//             RgswCiphertextList::allocate(
//                 not_self_count.len(),
//                 parameters.rlwe_n(),
//                 parameters.rgsw_x_rgsw_decomposer(),
//             ),
//         )
//     }
// }
