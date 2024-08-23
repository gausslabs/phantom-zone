mod method;
mod structure;

#[cfg(test)]
pub(crate) mod test;

pub(crate) use method::{decomposed_fma, decomposed_fma_prep, pk_encrypt_zero, sk_encrypt_zero};

pub use method::{
    auto_key_gen, automorphism_in_place, automorphism_prep_in_place, decrypt, key_switch_in_place,
    key_switch_prep_in_place, ks_key_gen, pk_encrypt, pk_gen, prepare_auto_key, prepare_ks_key,
    sample_extract, sk_encrypt,
};
pub use structure::{
    RlweAutoKey, RlweAutoKeyMutView, RlweAutoKeyOwned, RlweAutoKeyView, RlweCiphertext,
    RlweCiphertextList, RlweCiphertextListMutView, RlweCiphertextListOwned, RlweCiphertextListView,
    RlweCiphertextMutView, RlweCiphertextOwned, RlweCiphertextView, RlweKeySwitchKey,
    RlweKeySwitchKeyMutView, RlweKeySwitchKeyOwned, RlweKeySwitchKeyView, RlwePlaintext,
    RlwePlaintextMutView, RlwePlaintextOwned, RlwePlaintextView, RlwePublicKey,
    RlwePublicKeyMutView, RlwePublicKeyOwned, RlwePublicKeyView, RlweSecretKey,
    RlweSecretKeyMutView, RlweSecretKeyOwned, RlweSecretKeyView,
};

pub use method::{
    seeded_auto_key_gen, seeded_pk_gen, unseed_auto_key, unseed_ct, unseed_ks_key, unseed_pk,
};
pub use structure::{
    SeededRlweAutoKey, SeededRlweAutoKeyMutView, SeededRlweAutoKeyOwned, SeededRlweAutoKeyView,
    SeededRlweCiphertext, SeededRlweCiphertextList, SeededRlweCiphertextListMutView,
    SeededRlweCiphertextListOwned, SeededRlweCiphertextListView, SeededRlweCiphertextMutView,
    SeededRlweCiphertextOwned, SeededRlweCiphertextView, SeededRlweKeySwitchKey,
    SeededRlweKeySwitchKeyMutView, SeededRlweKeySwitchKeyOwned, SeededRlweKeySwitchKeyView,
    SeededRlwePublicKey, SeededRlwePublicKeyMutView, SeededRlwePublicKeyOwned,
    SeededRlwePublicKeyView,
};
