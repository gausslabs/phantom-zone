mod method;
mod structure;

#[cfg(test)]
pub(crate) mod test;

pub use method::{
    prepare_rgsw, rgsw_by_rgsw_in_place, rlwe_by_rgsw_in_place, rlwe_by_rgsw_prep_in_place,
    sk_encrypt,
};
pub use structure::{
    RgswCiphertext, RgswCiphertextMutView, RgswCiphertextOwned, RgswCiphertextView,
    RgswDecompositionParam,
};
