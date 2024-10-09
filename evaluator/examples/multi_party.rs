use core::{array::from_fn, fmt::Debug, iter::repeat_with, ops::Deref};
use itertools::Itertools;
use phantom_zone_evaluator::boolean::{
    dev::MockBoolEvaluator,
    fhew::{param::I_4P, prelude::*},
};
use rand::{rngs::StdRng, thread_rng, Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub trait Ops: Debug {
    type Ring: RingOps;
    type EvaluationRing: RingOps<Elem = Elem<Self::Ring>>;
    type KeySwitchMod: ModulusOps;
    type PackingRing: RingOps;

    fn new(param: FhewBoolParam) -> Self;

    fn param(&self) -> &FhewBoolParam;

    fn ring_packing_param(&self) -> RingPackingParam {
        let param = self.param();
        RingPackingParam {
            modulus: self.ring_rp().modulus(),
            ring_size: param.ring_size,
            sk_distribution: param.sk_distribution,
            noise_distribution: param.noise_distribution,
            auto_decomposition_param: param.auto_decomposition_param,
        }
    }

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
        dec_shares: impl IntoIterator<Item = &'a RlweDecryptionShareListOwned<Elem<Self::PackingRing>>>,
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

    fn serialize_pk_share(&self, pk_share: &SeededRlwePublicKeyOwned<Elem<Self::Ring>>) -> Vec<u8> {
        bincode::serialize(&pk_share.compact(self.ring())).unwrap()
    }

    fn deserialize_pk_share(&self, bytes: &[u8]) -> SeededRlwePublicKeyOwned<Elem<Self::Ring>> {
        let pk_share_compact: SeededRlwePublicKey<Compact> = bincode::deserialize(bytes).unwrap();
        pk_share_compact.uncompact(self.ring())
    }

    fn serialize_rp_key_share(
        &self,
        rp_key_share: &RingPackingKeyShareOwned<Elem<Self::Ring>>,
    ) -> Vec<u8> {
        bincode::serialize(&rp_key_share.compact(self.ring())).unwrap()
    }

    fn deserialize_rp_key_share(&self, bytes: &[u8]) -> RingPackingKeyShareOwned<Elem<Self::Ring>> {
        let rp_key_share_compact: RingPackingKeyShareCompact = bincode::deserialize(bytes).unwrap();
        rp_key_share_compact.uncompact(self.ring())
    }

    fn serialize_pk(&self, pk: &RlwePublicKeyOwned<Elem<Self::Ring>>) -> Vec<u8> {
        bincode::serialize(&pk.compact(self.ring())).unwrap()
    }

    fn deserialize_pk(&self, bytes: &[u8]) -> RlwePublicKeyOwned<Elem<Self::Ring>> {
        let pk_compact: RlwePublicKey<Compact> = bincode::deserialize(bytes).unwrap();
        pk_compact.uncompact(self.ring())
    }

    fn serialize_bs_key_share(
        &self,
        bs_key_share: &FhewBoolMpiKeyShareOwned<Elem<Self::Ring>, Elem<Self::KeySwitchMod>>,
    ) -> Vec<u8> {
        bincode::serialize(&bs_key_share.compact(self.ring(), self.mod_ks())).unwrap()
    }

    fn deserialize_bs_key_share(
        &self,
        bytes: &[u8],
    ) -> FhewBoolMpiKeyShareOwned<Elem<Self::Ring>, Elem<Self::KeySwitchMod>> {
        let bs_key_share_compact: FhewBoolMpiKeyShareCompact = bincode::deserialize(bytes).unwrap();
        bs_key_share_compact.uncompact(self.ring(), self.mod_ks())
    }

    fn serialize_ct(&self, ct: &FhewBoolCiphertextOwned<Elem<Self::Ring>>) -> Vec<u8> {
        bincode::serialize(&ct.compact(self.ring())).unwrap()
    }

    fn deserialize_ct(&self, bytes: &[u8]) -> FhewBoolCiphertextOwned<Elem<Self::Ring>> {
        let ct: FhewBoolCiphertext<Compact> = bincode::deserialize(bytes).unwrap();
        ct.uncompact(self.ring())
    }

    fn serialize_batched_ct(
        &self,
        ct: &FhewBoolBatchedCiphertextOwned<Elem<Self::Ring>>,
    ) -> Vec<u8> {
        bincode::serialize(&ct.compact(self.ring())).unwrap()
    }

    fn deserialize_batched_ct(
        &self,
        bytes: &[u8],
    ) -> FhewBoolBatchedCiphertextOwned<Elem<Self::Ring>> {
        let ct: FhewBoolBatchedCiphertext<Compact> = bincode::deserialize(bytes).unwrap();
        ct.uncompact(self.ring())
    }

    fn serialize_rp_ct(
        &self,
        ct: &FhewBoolPackedCiphertextOwned<Elem<Self::PackingRing>>,
    ) -> Vec<u8> {
        bincode::serialize(&ct.compact(self.ring_rp())).unwrap()
    }

    fn deserialize_rp_ct(
        &self,
        bytes: &[u8],
    ) -> FhewBoolPackedCiphertextOwned<Elem<Self::PackingRing>> {
        let ct: FhewBoolPackedCiphertext<Compact> = bincode::deserialize(bytes).unwrap();
        ct.uncompact(self.ring_rp())
    }

    fn serialize_dec_shares(&self, dec_shares: &[LweDecryptionShare<Elem<Self::Ring>>]) -> Vec<u8> {
        bincode::serialize(dec_shares).unwrap()
    }

    fn deserialize_dec_shares(&self, bytes: &[u8]) -> Vec<LweDecryptionShare<Elem<Self::Ring>>> {
        bincode::deserialize(bytes).unwrap()
    }

    fn serialize_rp_dec_share(
        &self,
        dec_share: &RlweDecryptionShareListOwned<Elem<Self::PackingRing>>,
    ) -> Vec<u8> {
        bincode::serialize(&dec_share.compact(self.ring_rp())).unwrap()
    }

    fn deserialize_rp_dec_share(
        &self,
        bytes: &[u8],
    ) -> RlweDecryptionShareListOwned<Elem<Self::PackingRing>> {
        let dec_share: RlweDecryptionShareList<Compact> = bincode::deserialize(bytes).unwrap();
        dec_share.uncompact(self.ring_rp())
    }
}

#[derive(Debug)]
pub struct NativeOps {
    param: FhewBoolParam,
    ring: NativeRing,
    mod_ks: NonNativePowerOfTwo,
    ring_rp: PrimeRing,
}

impl Ops for NativeOps {
    type Ring = NativeRing;
    type EvaluationRing = NoisyNativeRing;
    type KeySwitchMod = NonNativePowerOfTwo;
    type PackingRing = PrimeRing;

    fn new(param: FhewBoolParam) -> Self {
        Self {
            param,
            ring: RingOps::new(param.modulus, param.ring_size),
            mod_ks: ModulusOps::new(param.lwe_modulus),
            ring_rp: RingOps::new(Modulus::Prime(2305843009213554689), param.ring_size),
        }
    }

    fn param(&self) -> &FhewBoolParam {
        &self.param
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
    ring: PrimeRing,
    mod_ks: NonNativePowerOfTwo,
}

impl Ops for PrimeOps {
    type Ring = PrimeRing;
    type EvaluationRing = NoisyPrimeRing;
    type KeySwitchMod = NonNativePowerOfTwo;
    type PackingRing = PrimeRing;

    fn new(param: FhewBoolParam) -> Self {
        Self {
            param,
            ring: RingOps::new(param.modulus, param.ring_size),
            mod_ks: ModulusOps::new(param.lwe_modulus),
        }
    }

    fn param(&self) -> &FhewBoolParam {
        &self.param
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
    param: FhewBoolMpiParam,
    crs: Crs,
    share_idx: usize,
    sk_seed: <StdRng as SeedableRng>::Seed,
    pk: RlwePublicKeyOwned<Elem<O::Ring>>,
    ops: O,
}

impl<O: Ops> Deref for Client<O> {
    type Target = O;

    fn deref(&self) -> &Self::Target {
        &self.ops
    }
}

impl<O: Ops> Client<O> {
    pub fn new(param: FhewBoolMpiParam, crs: Crs, share_idx: usize) -> Self {
        let mut sk_seed = <StdRng as SeedableRng>::Seed::default();
        StdRng::from_entropy().fill_bytes(sk_seed.as_mut());
        Self {
            param,
            crs,
            share_idx,
            sk_seed,
            pk: RlwePublicKey::allocate(param.ring_size),
            ops: O::new(*param),
        }
    }

    pub fn sk(&self) -> RlweSecretKeyOwned<i64> {
        RlweSecretKey::sample(
            self.param.ring_size,
            self.param.sk_distribution,
            &mut StdRng::from_hierarchical_seed(self.sk_seed, &[0]),
        )
    }

    pub fn sk_ks(&self) -> LweSecretKeyOwned<i64> {
        LweSecretKey::sample(
            self.param.lwe_dimension,
            self.param.lwe_sk_distribution,
            &mut StdRng::from_hierarchical_seed(self.sk_seed, &[1]),
        )
    }

    pub fn pk_share_gen(&self) -> SeededRlwePublicKeyOwned<Elem<O::Ring>> {
        let mut pk = SeededRlwePublicKey::allocate(self.param.ring_size);
        pk_share_gen(
            self.ring(),
            &mut pk,
            &self.param,
            &self.crs.fhew(),
            &self.sk(),
            &mut StdRng::from_entropy(),
        );
        pk
    }

    pub fn rp_key_share_gen(&self) -> RingPackingKeyShareOwned<Elem<O::PackingRing>> {
        let mut rp_key = RingPackingKeyShareOwned::allocate(self.ring_packing_param());
        rp_key_share_gen(
            self.ring_rp(),
            &mut rp_key,
            &self.crs.ring_packing(),
            &self.sk(),
            &mut StdRng::from_entropy(),
        );
        rp_key
    }

    pub fn receive_pk(&mut self, pk: &RlwePublicKeyOwned<Elem<O::Ring>>) {
        self.pk = pk.cloned();
    }

    pub fn bs_key_share_gen(
        &self,
    ) -> FhewBoolMpiKeyShareOwned<Elem<O::Ring>, Elem<O::KeySwitchMod>> {
        let mut bs_key_share = FhewBoolMpiKeyShareOwned::allocate(self.param, self.share_idx);
        bs_key_share_gen(
            self.ring(),
            self.mod_ks(),
            &mut bs_key_share,
            &self.crs.fhew(),
            &self.sk(),
            &self.pk,
            &self.sk_ks(),
            &mut StdRng::from_entropy(),
        );
        bs_key_share
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
}

#[derive(Debug)]
pub struct Server<O: Ops> {
    param: FhewBoolMpiParam,
    crs: Crs,
    pk: RlwePublicKeyOwned<Elem<O::Ring>>,
    rp_key: RingPackingKeyOwned<<O::PackingRing as RingOps>::EvalPrep>,
    evaluator: FhewBoolEvaluator<O::EvaluationRing, O::KeySwitchMod>,
    ops: O,
}

impl<O: Ops> Deref for Server<O> {
    type Target = O;

    fn deref(&self) -> &Self::Target {
        &self.ops
    }
}

impl<O: Ops> Server<O> {
    pub fn new(param: FhewBoolMpiParam, crs: Crs) -> Self {
        let ops = O::new(*param);
        Self {
            param,
            crs,
            pk: RlwePublicKey::allocate(param.ring_size),
            rp_key: RingPackingKeyOwned::allocate_eval(
                ops.ring_packing_param(),
                ops.ring_rp().eval_size(),
            ),
            evaluator: FhewBoolEvaluator::new(FhewBoolKeyOwned::allocate(*param)),
            ops,
        }
    }

    pub fn aggregate_pk_shares(&mut self, pk_shares: &[SeededRlwePublicKeyOwned<Elem<O::Ring>>]) {
        let crs = self.crs.fhew();
        aggregate_pk_shares(self.evaluator.ring(), &mut self.pk, &crs, pk_shares);
    }

    pub fn aggregate_rp_key_shares(
        &mut self,
        rp_key_shares: &[RingPackingKeyShareOwned<Elem<O::PackingRing>>],
    ) {
        let crs = self.crs.ring_packing();
        let mut rp_key = RingPackingKeyOwned::allocate(self.ring_packing_param());
        aggregate_rp_key_shares(self.ops.ring_rp(), &mut rp_key, &crs, rp_key_shares);
        prepare_rp_key(self.ops.ring_rp(), &mut self.rp_key, &rp_key);
    }

    #[allow(clippy::type_complexity)]
    pub fn aggregate_bs_key_shares(
        &mut self,
        bs_key_shares: &[FhewBoolMpiKeyShareOwned<Elem<O::Ring>, Elem<O::KeySwitchMod>>],
    ) {
        let crs = self.crs.fhew();
        let bs_key = {
            let mut bs_key = FhewBoolKeyOwned::allocate(*self.param);
            aggregate_bs_key_shares(self.ring(), self.mod_ks(), &mut bs_key, &crs, bs_key_shares);
            bs_key
        };
        let bs_key_prep = {
            let ring: O::EvaluationRing = RingOps::new(self.param.modulus, self.param.ring_size);
            let mut bs_key_prep = FhewBoolKeyOwned::allocate_eval(*self.param, ring.eval_size());
            prepare_bs_key(&ring, &mut bs_key_prep, &bs_key);
            bs_key_prep
        };
        self.evaluator = FhewBoolEvaluator::new(bs_key_prep);
    }
}

fn function<'a, E: BoolEvaluator>(
    a: &[FheBool<'a, E>],
    b: &[FheBool<'a, E>],
    c: &[FheBool<'a, E>],
    d: &[FheBool<'a, E>],
) -> Vec<FheBool<'a, E>> {
    a.par_iter()
        .zip(b)
        .zip(c)
        .zip(d)
        .map(|(((a, b), c), d)| a ^ b ^ c ^ d)
        .collect()
}

macro_rules! describe {
    ($description:literal, $code:expr) => {{
        print!("{} {} ", $description, "·".repeat(70 - $description.len()));
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let start = std::time::Instant::now();
        let out = $code;
        println!("{:.3?}", start.elapsed());
        out
    }};
}

fn main() {
    let mut server = Server::<PrimeOps>::new(I_4P, Crs::from_entropy());
    let mut clients = (0..server.param.total_shares)
        .map(|share_idx| Client::<PrimeOps>::new(server.param, server.crs, share_idx))
        .collect_vec();

    // Key generation round 1

    let (pk_shares, rp_key_shares) =
        describe!("client: generate public and ring packing key shares", {
            let pk_shares = clients
                .iter()
                .map(|client| client.serialize_pk_share(&client.pk_share_gen()))
                .collect_vec();
            let rp_key_shares = clients
                .iter()
                .map(|client| client.serialize_rp_key_share(&client.rp_key_share_gen()))
                .collect_vec();
            (pk_shares, rp_key_shares)
        });

    let pk = describe!("server: aggregate public and ring packing key shares", {
        server.aggregate_pk_shares(
            &pk_shares
                .into_iter()
                .map(|bytes| server.deserialize_pk_share(&bytes))
                .collect_vec(),
        );
        server.aggregate_rp_key_shares(
            &rp_key_shares
                .into_iter()
                .map(|bytes| server.deserialize_rp_key_share(&bytes))
                .collect_vec(),
        );
        server.serialize_pk(&server.pk)
    });

    // Key generation round 2

    let bs_key_shares = describe!(
        "client: generate bootstrapping key shares",
        clients
            .iter_mut()
            .map(|client| {
                client.receive_pk(&client.deserialize_pk(&pk));
                client.serialize_bs_key_share(&client.bs_key_share_gen())
            })
            .collect_vec()
    );

    describe!(
        "server: aggregate bootstrapping key shares",
        server.aggregate_bs_key_shares(
            &bs_key_shares
                .into_iter()
                .map(|bytes| server.deserialize_bs_key_share(&bytes))
                .collect_vec(),
        )
    );

    // FHE evaluation

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

    let cts = describe!("client: batched encrypt inputs", {
        from_fn(|i| {
            let ct = clients[i].batched_pk_encrypt(&clients[i].pk, ms[i].clone());
            clients[i].serialize_batched_ct(&ct)
        })
    });

    let ct_out = describe!("server: perform FHE evaluation on inputs parallelly", {
        let [a, b, c, d] = &cts.map(|bytes| {
            let ct = server.deserialize_batched_ct(&bytes);
            ct.extract_all(server.ring())
                .into_iter()
                .map(|ct| FheBool::new(&server.evaluator, ct))
                .collect_vec()
        });
        function(a, b, c, d)
            .into_iter()
            .map(FheBool::into_ct)
            .collect_vec()
    });

    // With ring packing

    let rp_ct_out = describe!(
        "server: perform ring packing on output",
        server.serialize_rp_ct(&server.pack(&server.rp_key, &ct_out))
    );

    let rp_ct_out_dec_shares = describe!(
        "client: generate ring packing decryption shares",
        clients
            .iter()
            .map(|client| {
                let dec_share = client.rp_decrypt_share(&client.deserialize_rp_ct(&rp_ct_out));
                client.serialize_rp_dec_share(&dec_share)
            })
            .collect_vec()
    );

    describe!("anyone: aggregate ring packing decryption shares", {
        assert_eq!(
            out.to_vec(),
            clients[0].aggregate_rp_decryption_shares(
                &clients[0].deserialize_rp_ct(&rp_ct_out),
                &rp_ct_out_dec_shares
                    .iter()
                    .map(|dec_share| clients[0].deserialize_rp_dec_share(dec_share))
                    .collect_vec(),
            )
        )
    });

    // Without ring packing

    let ct_out = ct_out
        .iter()
        .map(|ct| server.serialize_ct(ct))
        .collect_vec();

    let ct_out_dec_shares = describe!(
        "client: generate decryption shares",
        clients
            .iter()
            .map(|client| {
                let dec_shares = ct_out
                    .iter()
                    .map(|ct| client.decrypt_share(&client.deserialize_ct(ct)))
                    .collect_vec();
                client.serialize_dec_shares(&dec_shares)
            })
            .collect_vec()
    );

    describe!("anyone: aggregate decryption shares", {
        assert_eq!(out.to_vec(), {
            let ct_out = ct_out
                .iter()
                .map(|ct| clients[0].deserialize_ct(ct))
                .collect_vec();
            let ct_out_dec_shares = ct_out_dec_shares
                .iter()
                .map(|bytes| clients[0].deserialize_dec_shares(bytes))
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
}
