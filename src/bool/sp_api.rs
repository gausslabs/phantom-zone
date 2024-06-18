mod impl_enc_dec {
    use crate::{Decryptor, Encryptor};

    use super::super::keys::SinglePartyClientKey;

    impl<K> Encryptor<bool, Vec<u64>> for K
    where
        K: SinglePartyClientKey,
    {
        fn encrypt(&self, m: &bool) -> Vec<u64> {
            todo!()
            // BoolEvaluator::with_local(|e| e.sk_encrypt(*m, self))
        }
    }

    impl<K> Decryptor<bool, Vec<u64>> for K
    where
        K: SinglePartyClientKey,
    {
        fn decrypt(&self, c: &Vec<u64>) -> bool {
            todo!()
            // BoolEvaluator::with_local(|e| e.sk_decrypt(c, self))
        }
    }
}
