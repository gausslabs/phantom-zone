use core::{array::from_fn, iter::repeat_with};
use itertools::{izip, Itertools};
use phantom_zone_evaluator::boolean::{
    dev::MockBoolEvaluator,
    fhew::{
        param::{I_4P, I_4P_60},
        prelude::*,
    },
};
use pz::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

#[allow(clippy::type_complexity)]
pub mod pz {
    use core::{fmt::Debug, ops::Deref};
    use itertools::Itertools;
    use phantom_zone_evaluator::boolean::fhew::prelude::*;
    use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
    use serde::{Deserialize, Serialize};

    pub trait Ops: Debug {
        type Ring: RingOps;
        type EvaluationRing: RingOps<Elem = Elem<Self::Ring>>;
        type KeySwitchMod: ModulusOps;
        type PackingRing: RingOps;

        fn new(param: Param) -> Self;

        fn param(&self) -> &FhewBoolParam;

        fn ring_packing_param(&self) -> &RingPackingParam;

        fn ring(&self) -> &Self::Ring;

        fn mod_ks(&self) -> &Self::KeySwitchMod;

        fn ring_rp(&self) -> &Self::PackingRing;

        fn pk_encrypt(
            &self,
            pk: &RlwePublicKeyOwned<Elem<Self::Ring>>,
            m: bool,
        ) -> FhewBoolCiphertextOwned<Elem<Self::Ring>> {
            FhewBoolCiphertextOwned::pk_encrypt(
                self.param(),
                self.ring(),
                pk,
                m,
                &mut StdLweRng::from_entropy(),
            )
        }

        fn batched_pk_encrypt(
            &self,
            pk: &RlwePublicKeyOwned<Elem<Self::Ring>>,
            ms: impl IntoIterator<Item = bool>,
        ) -> FhewBoolBatchedCiphertextOwned<Elem<Self::Ring>> {
            FhewBoolBatchedCiphertextOwned::pk_encrypt(
                self.param(),
                self.ring(),
                pk,
                ms,
                &mut StdLweRng::from_entropy(),
            )
        }

        fn pack<'a>(
            &self,
            rp_key: &RingPackingKeyOwned<<Self::PackingRing as RingOps>::EvalPrep>,
            cts: impl IntoIterator<Item = &'a FhewBoolCiphertextOwned<Elem<Self::Ring>>>,
        ) -> FhewBoolPackedCiphertextOwned<Elem<Self::PackingRing>>;

        fn aggregate_rp_decryption_shares<'a>(
            &self,
            ct: &FhewBoolPackedCiphertextOwned<Elem<Self::PackingRing>>,
            dec_shares: impl IntoIterator<
                Item = &'a RlweDecryptionShareListOwned<Elem<Self::PackingRing>>,
            >,
        ) -> Vec<bool> {
            ct.aggregate_decryption_shares(self.ring_rp(), dec_shares)
        }

        fn aggregate_decryption_shares<'a>(
            &self,
            ct: &FhewBoolCiphertextOwned<Elem<Self::Ring>>,
            dec_shares: impl IntoIterator<Item = &'a LweDecryptionShare<Elem<Self::Ring>>>,
        ) -> bool {
            ct.aggregate_decryption_shares(self.ring(), dec_shares)
        }

        fn serialize_pk_share(
            &self,
            pk_share: &SeededRlwePublicKeyOwned<Elem<Self::Ring>>,
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(&pk_share.compact(self.ring()))
        }

        fn deserialize_pk_share(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<SeededRlwePublicKeyOwned<Elem<Self::Ring>>> {
            let pk_share_compact: SeededRlwePublicKey<Compact> = bincode::deserialize(bytes)?;
            Ok(pk_share_compact.uncompact(self.ring()))
        }

        fn serialize_rp_key_share(
            &self,
            rp_key_share: &RingPackingKeyShareOwned<Elem<Self::PackingRing>>,
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(&rp_key_share.compact(self.ring_rp()))
        }

        fn deserialize_rp_key_share(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<RingPackingKeyShareOwned<Elem<Self::PackingRing>>> {
            let rp_key_share_compact: RingPackingKeyShareCompact = bincode::deserialize(bytes)?;
            Ok(rp_key_share_compact.uncompact(self.ring_rp()))
        }

        fn serialize_bs_key_share(
            &self,
            bs_key_share: &FhewBoolMpiKeyShareOwned<Elem<Self::Ring>, Elem<Self::KeySwitchMod>>,
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(&bs_key_share.compact(self.ring(), self.mod_ks()))
        }

        fn deserialize_bs_key_share(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<FhewBoolMpiKeyShareOwned<Elem<Self::Ring>, Elem<Self::KeySwitchMod>>>
        {
            let bs_key_share_compact: FhewBoolMpiKeyShareCompact = bincode::deserialize(bytes)?;
            Ok(bs_key_share_compact.uncompact(self.ring(), self.mod_ks()))
        }

        fn serialize_pk(
            &self,
            pk: &RlwePublicKeyOwned<Elem<Self::Ring>>,
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(&pk.compact(self.ring()))
        }

        fn deserialize_pk(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<RlwePublicKeyOwned<Elem<Self::Ring>>> {
            let pk_compact: RlwePublicKey<Compact> = bincode::deserialize(bytes)?;
            Ok(pk_compact.uncompact(self.ring()))
        }

        fn serialize_rp_key(
            &self,
            rp_key: &RingPackingKeyOwned<Elem<Self::PackingRing>>,
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(&rp_key.compact(self.ring_rp()))
        }

        fn deserialize_rp_key(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<RingPackingKeyOwned<Elem<Self::PackingRing>>> {
            let rp_key_compact: RingPackingKeyCompact = bincode::deserialize(bytes)?;
            Ok(rp_key_compact.uncompact(self.ring_rp()))
        }

        fn serialize_bs_key(
            &self,
            bs_key: &FhewBoolKeyOwned<Elem<Self::EvaluationRing>, Elem<Self::KeySwitchMod>>,
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(&bs_key.compact(self.ring(), self.mod_ks()))
        }

        fn deserialize_bs_key(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<FhewBoolKeyOwned<Elem<Self::EvaluationRing>, Elem<Self::KeySwitchMod>>>
        {
            let bs_key_compact: FhewBoolKeyCompact = bincode::deserialize(bytes)?;
            Ok(bs_key_compact.uncompact(self.ring(), self.mod_ks()))
        }

        fn serialize_ct(
            &self,
            ct: &FhewBoolCiphertextOwned<Elem<Self::Ring>>,
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(&ct.compact(self.ring()))
        }

        fn deserialize_ct(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<FhewBoolCiphertextOwned<Elem<Self::Ring>>> {
            let ct_compact: FhewBoolCiphertext<Compact> = bincode::deserialize(bytes)?;
            Ok(ct_compact.uncompact(self.ring()))
        }

        fn serialize_batched_ct(
            &self,
            ct: &FhewBoolBatchedCiphertextOwned<Elem<Self::Ring>>,
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(&ct.compact(self.ring()))
        }

        fn deserialize_batched_ct(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<FhewBoolBatchedCiphertextOwned<Elem<Self::Ring>>> {
            let ct_compact: FhewBoolBatchedCiphertext<Compact> = bincode::deserialize(bytes)?;
            Ok(ct_compact.uncompact(self.ring()))
        }

        fn serialize_rp_ct(
            &self,
            ct: &FhewBoolPackedCiphertextOwned<Elem<Self::PackingRing>>,
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(&ct.compact(self.ring_rp()))
        }

        fn deserialize_rp_ct(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<FhewBoolPackedCiphertextOwned<Elem<Self::PackingRing>>> {
            let ct_compact: FhewBoolPackedCiphertext<Compact> = bincode::deserialize(bytes)?;
            Ok(ct_compact.uncompact(self.ring_rp()))
        }

        fn serialize_dec_shares(
            &self,
            dec_shares: &[LweDecryptionShare<Elem<Self::Ring>>],
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(dec_shares)
        }

        fn deserialize_dec_shares(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<Vec<LweDecryptionShare<Elem<Self::Ring>>>> {
            bincode::deserialize(bytes)
        }

        fn serialize_rp_dec_share(
            &self,
            dec_share: &RlweDecryptionShareListOwned<Elem<Self::PackingRing>>,
        ) -> bincode::Result<Vec<u8>> {
            bincode::serialize(&dec_share.compact(self.ring_rp()))
        }

        fn deserialize_rp_dec_share(
            &self,
            bytes: &[u8],
        ) -> bincode::Result<RlweDecryptionShareListOwned<Elem<Self::PackingRing>>> {
            let dec_share_compact: RlweDecryptionShareList<Compact> = bincode::deserialize(bytes)?;
            Ok(dec_share_compact.uncompact(self.ring_rp()))
        }
    }

    #[derive(Debug)]
    pub struct NativeOps {
        param: FhewBoolParam,
        ring_packing_param: RingPackingParam,
        ring: NativeRing,
        mod_ks: NonNativePowerOfTwo,
        ring_rp: PrimeRing,
    }

    impl Ops for NativeOps {
        type Ring = NativeRing;
        type EvaluationRing = NoisyNativeRing;
        type KeySwitchMod = NonNativePowerOfTwo;
        type PackingRing = PrimeRing;

        fn new(param: Param) -> Self {
            let ring_packing_param = RingPackingParam {
                modulus: param.ring_packing_modulus.unwrap(),
                ring_size: param.ring_size,
                sk_distribution: param.sk_distribution,
                noise_distribution: param.noise_distribution,
                auto_decomposition_param: param.ring_packing_auto_decomposition_param,
            };
            Self {
                param: **param,
                ring_packing_param,
                ring: RingOps::new(param.modulus, param.ring_size),
                mod_ks: ModulusOps::new(param.lwe_modulus),
                ring_rp: RingOps::new(ring_packing_param.modulus, param.ring_size),
            }
        }

        fn param(&self) -> &FhewBoolParam {
            &self.param
        }

        fn ring_packing_param(&self) -> &RingPackingParam {
            &self.ring_packing_param
        }

        fn ring(&self) -> &Self::Ring {
            &self.ring
        }

        fn mod_ks(&self) -> &Self::KeySwitchMod {
            &self.mod_ks
        }

        fn ring_rp(&self) -> &Self::PackingRing {
            &self.ring_rp
        }

        fn pack<'a>(
            &self,
            rp_key: &RingPackingKeyOwned<<Self::PackingRing as RingOps>::EvalPrep>,
            cts: impl IntoIterator<Item = &'a FhewBoolCiphertextOwned<Elem<Self::Ring>>>,
        ) -> FhewBoolPackedCiphertextOwned<Elem<Self::PackingRing>> {
            FhewBoolPackedCiphertext::pack_ms(self.ring(), self.ring_rp(), rp_key, cts)
        }
    }

    #[derive(Debug)]
    pub struct PrimeOps {
        param: FhewBoolParam,
        ring_packing_param: RingPackingParam,
        ring: PrimeRing,
        mod_ks: NonNativePowerOfTwo,
    }

    impl Ops for PrimeOps {
        type Ring = PrimeRing;
        type EvaluationRing = NoisyPrimeRing;
        type KeySwitchMod = NonNativePowerOfTwo;
        type PackingRing = PrimeRing;

        fn new(param: Param) -> Self {
            let ring_packing_param = RingPackingParam {
                modulus: param.modulus,
                ring_size: param.ring_size,
                sk_distribution: param.sk_distribution,
                noise_distribution: param.noise_distribution,
                auto_decomposition_param: param.ring_packing_auto_decomposition_param,
            };
            Self {
                param: **param,
                ring_packing_param,
                ring: RingOps::new(param.modulus, param.ring_size),
                mod_ks: ModulusOps::new(param.lwe_modulus),
            }
        }

        fn param(&self) -> &FhewBoolParam {
            &self.param
        }

        fn ring_packing_param(&self) -> &RingPackingParam {
            &self.ring_packing_param
        }

        fn ring(&self) -> &Self::Ring {
            &self.ring
        }

        fn mod_ks(&self) -> &Self::KeySwitchMod {
            &self.mod_ks
        }

        fn ring_rp(&self) -> &Self::PackingRing {
            &self.ring
        }

        fn pack<'a>(
            &self,
            rp_key: &RingPackingKeyOwned<<Self::PackingRing as RingOps>::EvalPrep>,
            cts: impl IntoIterator<Item = &'a FhewBoolCiphertextOwned<Elem<Self::Ring>>>,
        ) -> FhewBoolPackedCiphertextOwned<Elem<Self::PackingRing>> {
            FhewBoolPackedCiphertext::pack(self.ring_rp(), rp_key, cts)
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct Param {
        pub param: FhewBoolMpiParam,
        pub ring_packing_modulus: Option<Modulus>,
        pub ring_packing_auto_decomposition_param: DecompositionParam,
    }

    impl Deref for Param {
        type Target = FhewBoolMpiParam;

        fn deref(&self) -> &Self::Target {
            &self.param
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct Crs(<StdRng as SeedableRng>::Seed);

    impl Crs {
        pub fn new(seed: <StdRng as SeedableRng>::Seed) -> Self {
            Self(seed)
        }

        pub fn from_entropy() -> Self {
            Self::new(thread_rng().gen())
        }

        fn fhew(&self) -> FhewBoolMpiCrs<StdRng> {
            FhewBoolMpiCrs::new(StdRng::from_hierarchical_seed(self.0, &[0]).gen())
        }

        fn ring_packing(&self) -> RingPackingCrs<StdRng> {
            RingPackingCrs::new(StdRng::from_hierarchical_seed(self.0, &[1]).gen())
        }
    }

    #[derive(Debug)]
    pub struct Client<O: Ops> {
        param: Param,
        crs: Crs,
        ops: O,
        share_idx: usize,
        seed: <StdRng as SeedableRng>::Seed,
        pk: Option<RlwePublicKeyOwned<Elem<O::Ring>>>,
    }

    impl<O: Ops> Deref for Client<O> {
        type Target = O;

        fn deref(&self) -> &Self::Target {
            &self.ops
        }
    }

    impl<O: Ops> Client<O> {
        pub fn new(
            param: Param,
            crs: Crs,
            share_idx: usize,
            seed: <StdRng as SeedableRng>::Seed,
            pk_bytes: Option<&[u8]>,
        ) -> bincode::Result<Self> {
            let mut client = Self {
                param,
                crs,
                ops: O::new(param),
                share_idx,
                seed,
                pk: None,
            };
            if let Some(pk_bytes) = pk_bytes {
                client.with_pk(client.deserialize_pk(pk_bytes)?);
            }
            Ok(client)
        }

        pub fn seed(&self) -> <StdRng as SeedableRng>::Seed {
            self.seed
        }

        pub fn sk(&self) -> RlweSecretKeyOwned<i64> {
            RlweSecretKey::sample(
                self.param.ring_size,
                self.param.sk_distribution,
                &mut StdRng::from_hierarchical_seed(self.seed, &[0, 0]),
            )
        }

        pub fn sk_ks(&self) -> LweSecretKeyOwned<i64> {
            LweSecretKey::sample(
                self.param.lwe_dimension,
                self.param.lwe_sk_distribution,
                &mut StdRng::from_hierarchical_seed(self.seed, &[0, 1]),
            )
        }

        pub fn pk(&self) -> &RlwePublicKeyOwned<Elem<O::Ring>> {
            self.pk.as_ref().unwrap()
        }

        pub fn pk_share_gen(&self) -> SeededRlwePublicKeyOwned<Elem<O::Ring>> {
            let mut pk = SeededRlwePublicKey::allocate(self.param.ring_size);
            pk_share_gen(
                self.ring(),
                &mut pk,
                &self.param,
                &self.crs.fhew(),
                &self.sk(),
                &mut StdRng::from_hierarchical_seed(self.seed, &[1, 0]),
            );
            pk
        }

        pub fn rp_key_share_gen(&self) -> RingPackingKeyShareOwned<Elem<O::PackingRing>> {
            let mut rp_key = RingPackingKeyShareOwned::allocate(*self.ring_packing_param());
            rp_key_share_gen(
                self.ring_rp(),
                &mut rp_key,
                &self.crs.ring_packing(),
                &self.sk(),
                &mut StdRng::from_hierarchical_seed(self.seed, &[1, 1]),
            );
            rp_key
        }

        pub fn with_pk(&mut self, pk: RlwePublicKeyOwned<Elem<O::Ring>>) {
            self.pk = Some(pk);
        }

        pub fn bs_key_share_gen(
            &self,
        ) -> FhewBoolMpiKeyShareOwned<Elem<O::Ring>, Elem<O::KeySwitchMod>> {
            let mut bs_key_share = FhewBoolMpiKeyShareOwned::allocate(*self.param, self.share_idx);
            bs_key_share_gen(
                self.ring(),
                self.mod_ks(),
                &mut bs_key_share,
                &self.crs.fhew(),
                &self.sk(),
                self.pk(),
                &self.sk_ks(),
                &mut StdRng::from_hierarchical_seed(self.seed, &[1, 1]),
            );
            bs_key_share
        }

        pub fn batched_pk_encrypt(
            &self,
            ms: impl IntoIterator<Item = bool>,
        ) -> FhewBoolBatchedCiphertextOwned<Elem<O::Ring>> {
            self.ops.batched_pk_encrypt(self.pk(), ms)
        }

        pub fn decrypt_share(
            &self,
            ct: &FhewBoolCiphertextOwned<Elem<O::Ring>>,
        ) -> LweDecryptionShare<Elem<O::Ring>> {
            ct.decrypt_share(
                &self.param,
                self.ring(),
                self.sk().as_view(),
                &mut StdLweRng::from_entropy(),
            )
        }

        pub fn rp_decrypt_share(
            &self,
            ct: &FhewBoolPackedCiphertextOwned<Elem<O::PackingRing>>,
        ) -> RlweDecryptionShareListOwned<Elem<O::PackingRing>> {
            ct.decrypt_share(
                &self.param,
                self.ring_rp(),
                self.sk().as_view(),
                &mut StdLweRng::from_entropy(),
            )
        }

        pub fn serialize_pk(&self) -> bincode::Result<Vec<u8>> {
            self.ops.serialize_pk(self.pk())
        }
    }

    #[derive(Debug)]
    pub struct Server<O: Ops> {
        param: Param,
        crs: Crs,
        ops: O,
        pk: Option<RlwePublicKeyOwned<Elem<O::Ring>>>,
        rp_key: Option<RingPackingKeyOwned<Elem<O::PackingRing>>>,
        rp_key_prep: Option<RingPackingKeyOwned<<O::PackingRing as RingOps>::EvalPrep>>,
        bs_key: Option<FhewBoolKeyOwned<Elem<O::Ring>, Elem<O::KeySwitchMod>>>,
        evaluator: Option<FhewBoolEvaluator<O::EvaluationRing, O::KeySwitchMod>>,
    }

    impl<O: Ops> Deref for Server<O> {
        type Target = O;

        fn deref(&self) -> &Self::Target {
            &self.ops
        }
    }

    impl<O: Ops> Server<O> {
        pub fn new(
            param: Param,
            crs: Crs,
            pk_bytes: Option<&[u8]>,
            rp_key_bytes: Option<&[u8]>,
            bs_key_bytes: Option<&[u8]>,
        ) -> bincode::Result<Self> {
            let mut server = Self {
                param,
                crs,
                ops: O::new(param),
                pk: None,
                rp_key: None,
                rp_key_prep: None,
                bs_key: None,
                evaluator: None,
            };
            if let Some(pk_bytes) = pk_bytes {
                server.with_pk(server.deserialize_pk(pk_bytes)?);
            }
            if let Some(rp_key_bytes) = rp_key_bytes {
                server.with_rp_key(server.deserialize_rp_key(rp_key_bytes)?);
            }
            if let Some(bs_key_bytes) = bs_key_bytes {
                server.with_bs_key(server.deserialize_bs_key(bs_key_bytes)?);
            }
            Ok(server)
        }

        pub fn param(&self) -> &FhewBoolMpiParam {
            &self.param
        }

        pub fn crs(&self) -> &Crs {
            &self.crs
        }

        pub fn pk(&self) -> &RlwePublicKeyOwned<Elem<O::Ring>> {
            self.pk.as_ref().unwrap()
        }

        pub fn rp_key(&self) -> &RingPackingKeyOwned<Elem<O::PackingRing>> {
            self.rp_key.as_ref().unwrap()
        }

        fn rp_key_prep(&self) -> &RingPackingKeyOwned<<O::PackingRing as RingOps>::EvalPrep> {
            self.rp_key_prep.as_ref().unwrap()
        }

        pub fn evaluator(&self) -> &FhewBoolEvaluator<O::EvaluationRing, O::KeySwitchMod> {
            self.evaluator.as_ref().unwrap()
        }

        pub fn bs_key(&self) -> &FhewBoolKeyOwned<Elem<O::EvaluationRing>, Elem<O::KeySwitchMod>> {
            self.bs_key.as_ref().unwrap()
        }

        pub fn with_pk(&mut self, pk: RlwePublicKeyOwned<Elem<O::Ring>>) {
            self.pk = Some(pk)
        }

        pub fn with_rp_key(&mut self, rp_key: RingPackingKeyOwned<Elem<O::PackingRing>>) {
            let mut rp_key_prep = RingPackingKeyOwned::allocate_eval(
                *self.ring_packing_param(),
                self.ring_rp().eval_size(),
            );
            prepare_rp_key(self.ring_rp(), &mut rp_key_prep, &rp_key);
            self.rp_key = Some(rp_key);
            self.rp_key_prep = Some(rp_key_prep);
        }

        pub fn with_bs_key(
            &mut self,
            bs_key: FhewBoolKeyOwned<Elem<O::EvaluationRing>, Elem<O::KeySwitchMod>>,
        ) {
            let bs_key_prep = {
                let ring: O::EvaluationRing =
                    RingOps::new(self.param.modulus, self.param.ring_size);
                let mut bs_key_prep =
                    FhewBoolKeyOwned::allocate_eval(**self.param, ring.eval_size());
                prepare_bs_key(&ring, &mut bs_key_prep, &bs_key);
                bs_key_prep
            };
            self.bs_key = Some(bs_key);
            self.evaluator = Some(FhewBoolEvaluator::new(bs_key_prep));
        }

        pub fn aggregate_pk_shares(
            &mut self,
            pk_shares: &[SeededRlwePublicKeyOwned<Elem<O::Ring>>],
        ) {
            let crs = self.crs.fhew();
            let mut pk = RlwePublicKey::allocate(self.param.ring_size);
            aggregate_pk_shares(self.ring(), &mut pk, &crs, pk_shares);
            self.with_pk(pk);
        }

        pub fn aggregate_rp_key_shares(
            &mut self,
            rp_key_shares: &[RingPackingKeyShareOwned<Elem<O::PackingRing>>],
        ) {
            let crs = self.crs.ring_packing();
            let mut rp_key = RingPackingKeyOwned::allocate(*self.ring_packing_param());
            aggregate_rp_key_shares(self.ring_rp(), &mut rp_key, &crs, rp_key_shares);
            self.with_rp_key(rp_key);
        }

        pub fn aggregate_bs_key_shares(
            &mut self,
            bs_key_shares: &[FhewBoolMpiKeyShareOwned<Elem<O::Ring>, Elem<O::KeySwitchMod>>],
        ) {
            let crs = self.crs.fhew();
            let bs_key = {
                let mut bs_key = FhewBoolKeyOwned::allocate(**self.param);
                aggregate_bs_key_shares(
                    self.ring(),
                    self.mod_ks(),
                    &mut bs_key,
                    &crs,
                    bs_key_shares,
                );
                bs_key
            };
            self.with_bs_key(bs_key);
        }

        pub fn wrap_batched_ct(
            &self,
            ct: &FhewBoolBatchedCiphertextOwned<Elem<O::Ring>>,
        ) -> Vec<FheBool<&FhewBoolEvaluator<O::EvaluationRing, O::KeySwitchMod>>> {
            ct.extract_all(self.ring())
                .into_iter()
                .map(|ct| FheBool::new(self.evaluator(), ct))
                .collect_vec()
        }

        pub fn serialize_pk(&self) -> bincode::Result<Vec<u8>> {
            self.ops.serialize_pk(self.pk())
        }

        pub fn serialize_rp_key(&self) -> bincode::Result<Vec<u8>> {
            self.ops.serialize_rp_key(self.rp_key())
        }

        pub fn serialize_bs_key(&self) -> bincode::Result<Vec<u8>> {
            self.ops.serialize_bs_key(self.bs_key())
        }

        pub fn pack<'a>(
            &self,
            cts: impl IntoIterator<Item = &'a FhewBoolCiphertextOwned<Elem<O::Ring>>>,
        ) -> FhewBoolPackedCiphertextOwned<Elem<O::PackingRing>> {
            self.ops.pack(self.rp_key_prep(), cts)
        }
    }
}

macro_rules! timed {
    ($description:literal, $code:expr) => {{
        print!(
            "  {} {} ",
            $description,
            "·".repeat(70 - $description.len())
        );
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let start = std::time::Instant::now();
        let out = $code;
        println!(
            "{:>6.3?}s",
            start.elapsed().as_nanos() as f64 / 1000000000.0
        );
        out
    }};
}

fn e2e<O: Ops>(param: Param) {
    println!(
        "Running e2e example on {}-bits {} modulus",
        param.modulus.bits(),
        match param.modulus {
            Modulus::PowerOfTwo(_) => "native",
            Modulus::Prime(_) => "prime",
        }
    );

    let crs = Crs::from_entropy();
    let mut server = Server::<O>::new(param, crs, None, None, None).unwrap();
    let mut clients = (0..param.total_shares)
        .map(|share_idx| {
            let seed = StdRng::from_entropy().gen();
            Client::<O>::new(param, crs, share_idx, seed, None).unwrap()
        })
        .collect_vec();

    // Key generation round 1.

    let (pk_shares, rp_key_shares) =
        timed!("client: generate public and ring packing key shares", {
            let pk_shares = clients
                .iter()
                .map(|client| client.serialize_pk_share(&client.pk_share_gen()).unwrap())
                .collect_vec();
            let rp_key_shares = clients
                .iter()
                .map(|client| {
                    client
                        .serialize_rp_key_share(&client.rp_key_share_gen())
                        .unwrap()
                })
                .collect_vec();
            (pk_shares, rp_key_shares)
        });

    let pk = timed!("server: aggregate public and ring packing key shares", {
        server.aggregate_pk_shares(
            &pk_shares
                .into_iter()
                .map(|bytes| server.deserialize_pk_share(&bytes).unwrap())
                .collect_vec(),
        );
        server.aggregate_rp_key_shares(
            &rp_key_shares
                .into_iter()
                .map(|bytes| server.deserialize_rp_key_share(&bytes).unwrap())
                .collect_vec(),
        );
        server.serialize_pk().unwrap()
    });

    // Key generation round 2.

    let bs_key_shares = timed!(
        "client: generate bootstrapping key shares",
        clients
            .iter_mut()
            .map(|client| {
                client.with_pk(client.deserialize_pk(&pk).unwrap());
                client
                    .serialize_bs_key_share(&client.bs_key_share_gen())
                    .unwrap()
            })
            .collect_vec()
    );

    timed!(
        "server: aggregate bootstrapping key shares",
        server.aggregate_bs_key_shares(
            &bs_key_shares
                .into_iter()
                .map(|bytes| server.deserialize_bs_key_share(&bytes).unwrap())
                .collect_vec(),
        )
    );

    // FHE evaluation.

    fn function<E: BoolEvaluator>(
        a: &[FheBool<E>],
        b: &[FheBool<E>],
        c: &[FheBool<E>],
        d: &[FheBool<E>],
    ) -> Vec<FheBool<E>> {
        a.par_iter()
            .zip(b)
            .zip(c)
            .zip(d)
            .map(|(((a, b), c), d)| a ^ b ^ c ^ d)
            .collect()
    }

    let ms: [Vec<bool>; 4] = {
        let mut rng = StdRng::from_entropy();
        let n = 2048;
        from_fn(|_| repeat_with(|| rng.gen()).take(n).collect())
    };
    let out = {
        let [a, b, c, d] = &ms
            .clone()
            .map(|m| m.into_iter().map(|m| m.into()).collect_vec());
        function::<MockBoolEvaluator>(a, b, c, d)
            .into_iter()
            .map(FheBool::into_ct)
            .collect_vec()
    };

    let run = |server: &Server<O>, clients: &[Client<O>]| {
        let cts = timed!("client: batched encrypt inputs", {
            from_fn(|i| {
                let ct = clients[i].batched_pk_encrypt(ms[i].clone());
                clients[i].serialize_batched_ct(&ct).unwrap()
            })
        });

        let ct_out = timed!("server: perform FHE evaluation on inputs parallelly", {
            let [a, b, c, d] = &cts.map(|bytes| {
                let ct = server.deserialize_batched_ct(&bytes).unwrap();
                server.wrap_batched_ct(&ct)
            });
            function(a, b, c, d)
                .into_iter()
                .map(FheBool::into_ct)
                .collect_vec()
        });

        // With ring packing.

        let rp_ct_out = timed!(
            "server: perform ring packing on output",
            server.serialize_rp_ct(&server.pack(&ct_out)).unwrap()
        );

        let rp_ct_out_dec_shares = timed!(
            "client: generate ring packing decryption shares",
            clients
                .iter()
                .map(|client| {
                    let dec_share =
                        client.rp_decrypt_share(&client.deserialize_rp_ct(&rp_ct_out).unwrap());
                    client.serialize_rp_dec_share(&dec_share).unwrap()
                })
                .collect_vec()
        );

        timed!("anyone: aggregate ring packing decryption shares", {
            assert_eq!(
                out.to_vec(),
                clients[0].aggregate_rp_decryption_shares(
                    &clients[0].deserialize_rp_ct(&rp_ct_out).unwrap(),
                    &rp_ct_out_dec_shares
                        .iter()
                        .map(|dec_share| clients[0].deserialize_rp_dec_share(dec_share).unwrap())
                        .collect_vec(),
                )
            )
        });

        // Without ring packing.

        let ct_out = ct_out
            .iter()
            .map(|ct| server.serialize_ct(ct).unwrap())
            .collect_vec();

        let ct_out_dec_shares = timed!(
            "client: generate decryption shares",
            clients
                .iter()
                .map(|client| {
                    let dec_shares = ct_out
                        .iter()
                        .map(|ct| client.decrypt_share(&client.deserialize_ct(ct).unwrap()))
                        .collect_vec();
                    client.serialize_dec_shares(&dec_shares).unwrap()
                })
                .collect_vec()
        );

        timed!("anyone: aggregate decryption shares", {
            assert_eq!(out.to_vec(), {
                let ct_out = ct_out
                    .iter()
                    .map(|ct| clients[0].deserialize_ct(ct).unwrap())
                    .collect_vec();
                let ct_out_dec_shares = ct_out_dec_shares
                    .iter()
                    .map(|bytes| clients[0].deserialize_dec_shares(bytes).unwrap())
                    .collect_vec();
                (0..out.len())
                    .map(|idx| {
                        clients[0].aggregate_decryption_shares(
                            &ct_out[idx],
                            ct_out_dec_shares.iter().map(|dec_shares| &dec_shares[idx]),
                        )
                    })
                    .collect_vec()
            })
        });
    };

    run(&server, &clients);

    // Store and restart.

    let server = timed!("server: store aggregated keys and restart", {
        let pk_bytes = server.serialize_pk().unwrap();
        let rp_key_bytes = server.serialize_rp_key().unwrap();
        let bs_key_bytes = server.serialize_bs_key().unwrap();
        Server::<O>::new(
            param,
            crs,
            Some(&pk_bytes),
            Some(&rp_key_bytes),
            Some(&bs_key_bytes),
        )
        .unwrap()
    });

    let clients = timed!("client: store aggregated pk and seed and restart", {
        let seeds = clients.iter().map(|client| client.seed()).collect_vec();
        let pk_bytes = clients[0].serialize_pk().unwrap();
        izip!(0.., seeds)
            .map(|(share_idx, seed)| {
                Client::<O>::new(param, crs, share_idx, seed, Some(&pk_bytes)).unwrap()
            })
            .collect_vec()
    });

    run(&server, &clients);
}

fn main() {
    e2e::<NativeOps>(Param {
        param: I_4P_60,
        ring_packing_modulus: Some(Modulus::Prime(2305843009213554689)),
        ring_packing_auto_decomposition_param: DecompositionParam {
            log_base: 20,
            level: 1,
        },
    });
    e2e::<PrimeOps>(Param {
        param: I_4P,
        ring_packing_modulus: None,
        ring_packing_auto_decomposition_param: DecompositionParam {
            log_base: 20,
            level: 1,
        },
    });
}
