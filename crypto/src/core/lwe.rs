mod method;
mod structure;

#[cfg(test)]
pub(crate) mod test;

pub use method::{
    aggregate_decryption_shares, decrypt, decrypt_share, key_switch, ks_key_gen, sk_encrypt,
};
pub use structure::{
    LweCiphertext, LweCiphertextList, LweCiphertextListMutView, LweCiphertextListOwned,
    LweCiphertextListView, LweCiphertextMutView, LweCiphertextOwned, LweCiphertextView,
    LweDecryptionShare, LweKeySwitchKey, LweKeySwitchKeyMutView, LweKeySwitchKeyOwned,
    LweKeySwitchKeyView, LwePlaintext, LweSecretKey, LweSecretKeyMutView, LweSecretKeyOwned,
    LweSecretKeyView,
};

pub use method::{seeded_ks_key_gen, unseed_ks_key};
pub use structure::{
    SeededLweCiphertext, SeededLweCiphertextList, SeededLweCiphertextListMutView,
    SeededLweCiphertextListOwned, SeededLweCiphertextListView, SeededLweKeySwitchKey,
    SeededLweKeySwitchKeyMutView, SeededLweKeySwitchKeyOwned, SeededLweKeySwitchKeyView,
};
