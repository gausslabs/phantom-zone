mod method;
mod structure;

#[cfg(test)]
pub(crate) mod test;

pub use method::{
    auto_key_gen, automorphism, automorphism_prep, decrypt, key_switch, key_switch_prep,
    ks_key_gen, pk_encrypt, pk_gen, prepare_auto_key, prepare_ks_key, sample_extract, sk_encrypt,
    sk_encrypt_with_pt_in_b,
};
pub use structure::{
    RlweAutoKey, RlweAutoKeyMutView, RlweAutoKeyOwned, RlweAutoKeyView, RlweCiphertext,
    RlweCiphertextMutView, RlweCiphertextOwned, RlweCiphertextView, RlweKeySwitchKey,
    RlweKeySwitchKeyMutView, RlweKeySwitchKeyOwned, RlweKeySwitchKeyView, RlwePlaintext,
    RlwePlaintextMutView, RlwePlaintextOwned, RlwePlaintextView, RlwePublicKey,
    RlwePublicKeyMutView, RlwePublicKeyOwned, RlwePublicKeyView, RlweSecretKey,
    RlweSecretKeyMutView, RlweSecretKeyOwned, RlweSecretKeyView,
};
