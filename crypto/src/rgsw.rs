mod method;
mod structure;

#[cfg(test)]
mod test;

pub use method::{rlwe_by_rgsw_in_place, sk_encrypt};
pub use structure::{
    RgswCiphertext, RgswCiphertextMutView, RgswCiphertextOwned, RgswCiphertextView,
};
