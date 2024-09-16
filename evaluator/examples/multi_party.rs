use core::{array::from_fn, cell::OnceCell, num::Wrapping};
use itertools::Itertools;
use num_traits::NumOps;
use phantom_zone_evaluator::boolean::fhew::{param::I_4P, prelude::*};
use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
struct Client<R: RingOps, M: ModulusOps> {
    param: FhewBoolMpiParam,
    crs: FhewBoolMpiCrs<StdRng>,
    share_idx: usize,
    sk_seed: <StdRng as SeedableRng>::Seed,
    pk: RlwePublicKeyOwned<R::Elem>,
    #[serde(skip)]
    ring: OnceCell<R>,
    #[serde(skip)]
    mod_ks: OnceCell<M>,
}

impl<R: RingOps, M: ModulusOps> Client<R, M> {
    fn new(param: FhewBoolMpiParam, crs: FhewBoolMpiCrs<StdRng>, share_idx: usize) -> Self {
        let mut sk_seed = <StdRng as SeedableRng>::Seed::default();
        StdRng::from_entropy().fill_bytes(sk_seed.as_mut());
        Self {
            param,
            crs,
            share_idx,
            sk_seed,
            pk: RlwePublicKey::allocate(param.ring_size),
            ring: Default::default(),
            mod_ks: Default::default(),
        }
    }

    fn ring(&self) -> &R {
        self.ring
            .get_or_init(|| RingOps::new(self.param.modulus, self.param.ring_size))
    }

    fn mod_ks(&self) -> &M {
        self.mod_ks.get_or_init(|| M::new(self.param.lwe_modulus))
    }

    fn sk(&self) -> RlweSecretKeyOwned<i64> {
        RlweSecretKey::sample(
            self.param.ring_size,
            self.param.sk_distribution,
            &mut StdRng::from_hierarchical_seed(self.sk_seed, &[0]),
        )
    }

    fn sk_ks(&self) -> LweSecretKeyOwned<i64> {
        LweSecretKey::sample(
            self.param.lwe_dimension,
            self.param.lwe_sk_distribution,
            &mut StdRng::from_hierarchical_seed(self.sk_seed, &[1]),
        )
    }

    fn pk_share_gen(&self) -> SeededRlwePublicKeyOwned<R::Elem> {
        let mut pk = SeededRlwePublicKey::allocate(self.param.ring_size);
        pk_share_gen(
            self.ring(),
            &mut pk,
            &self.param,
            &self.crs,
            &self.sk(),
            &mut StdRng::from_entropy(),
        );
        pk
    }

    fn receive_pk(&mut self, pk: &RlwePublicKeyOwned<R::Elem>) {
        self.pk = pk.cloned();
    }

    fn bs_key_share_gen(&self) -> FhewBoolMpiKeyShareOwned<R::Elem, M::Elem> {
        let mut bs_key_share = FhewBoolMpiKeyShare::allocate(self.param, self.share_idx);
        bs_key_share_gen(
            self.ring(),
            self.mod_ks(),
            &mut bs_key_share,
            &self.crs,
            &self.sk(),
            &self.pk,
            &self.sk_ks(),
            &mut StdRng::from_entropy(),
        );
        bs_key_share
    }

    fn decrypt_share(
        &self,
        ct: [FhewBoolCiphertextOwned<R::Elem>; 8],
    ) -> [LweDecryptionShare<R::Elem>; 8] {
        ct.map(|ct| {
            ct.decrypt_share(
                self.ring(),
                self.sk().as_view(),
                self.param.noise_distribution,
                &mut StdLweRng::from_entropy(),
            )
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
struct Server<R: RingOps, M: ModulusOps> {
    param: FhewBoolMpiParam,
    crs: FhewBoolMpiCrs<StdRng>,
    pk: RlwePublicKeyOwned<R::Elem>,
    #[serde(rename = "bs_key")]
    evaluator: FhewBoolEvaluator<R, M>,
}

impl<R: RingOps, M: ModulusOps> Server<R, M> {
    fn new(param: FhewBoolMpiParam) -> Self {
        Self {
            param,
            crs: FhewBoolMpiCrs::sample(StdRng::from_entropy()),
            pk: RlwePublicKey::allocate(param.ring_size),
            evaluator: FhewBoolEvaluator::new(FhewBoolKey::allocate(*param)),
        }
    }

    fn ring(&self) -> &R {
        self.evaluator.ring()
    }

    fn mod_ks(&self) -> &M {
        self.evaluator.mod_ks()
    }

    fn aggregate_pk_shares(&mut self, pk_shares: &[SeededRlwePublicKeyOwned<R::Elem>]) {
        aggregate_pk_shares(self.evaluator.ring(), &mut self.pk, &self.crs, pk_shares);
    }

    fn aggregate_bs_key_shares<R2: RingOps<Elem = R::Elem>>(
        &mut self,
        bs_key_shares: &[FhewBoolMpiKeyShareOwned<R::Elem, M::Elem>],
    ) {
        let bs_key = {
            let ring = <R2 as RingOps>::new(self.param.modulus, self.param.ring_size);
            let mut bs_key = FhewBoolKey::allocate(*self.param);
            aggregate_bs_key_shares(&ring, self.mod_ks(), &mut bs_key, &self.crs, bs_key_shares);
            bs_key
        };
        let bs_key_prep = {
            let mut bs_key_prep = FhewBoolKey::allocate_eval(*self.param, self.ring().eval_size());
            prepare_bs_key(self.ring(), &mut bs_key_prep, &bs_key);
            bs_key_prep
        };
        self.evaluator = FhewBoolEvaluator::new(bs_key_prep);
    }

    fn pk_encrypt(&self, m: u8) -> [FhewBoolCiphertextOwned<R::Elem>; 8] {
        pk_encrypt(&self.param, self.ring(), &self.pk, m)
    }
}

fn pk_encrypt<R: RingOps>(
    param: &FhewBoolParam,
    ring: &R,
    pk: &RlwePublicKeyOwned<R::Elem>,
    m: u8,
) -> [FhewBoolCiphertextOwned<R::Elem>; 8] {
    FhewBoolCiphertext::batched_pk_encrypt(
        param,
        ring,
        pk,
        (0..8).map(|idx| (m >> idx) & 1 == 1),
        &mut StdLweRng::from_entropy(),
    )
    .try_into()
    .unwrap()
}

fn aggregate_decryption_shares<R: RingOps>(
    ring: &R,
    ct: [FhewBoolCiphertextOwned<R::Elem>; 8],
    dec_shares: &[[LweDecryptionShare<R::Elem>; 8]],
) -> u8 {
    (0..8)
        .map(|idx| {
            let dec_shares = dec_shares.iter().map(|dec_shares| &dec_shares[idx]);
            ct[idx].aggregate_decryption_shares(ring, dec_shares)
        })
        .rev()
        .fold(0, |m, b| (m << 1) | b as u8)
}

fn serialize_pk_share<R: RingOps>(
    ring: &R,
    pk_share: &SeededRlwePublicKeyOwned<R::Elem>,
) -> Vec<u8> {
    bincode::serialize(&pk_share.compact(ring)).unwrap()
}

fn deserialize_pk_share<R: RingOps>(ring: &R, bytes: &[u8]) -> SeededRlwePublicKeyOwned<R::Elem> {
    let pk_share_compact: SeededRlwePublicKey<Compact> = bincode::deserialize(bytes).unwrap();
    pk_share_compact.uncompact(ring)
}

fn serialize_pk<R: RingOps>(ring: &R, pk: &RlwePublicKeyOwned<R::Elem>) -> Vec<u8> {
    bincode::serialize(&pk.compact(ring)).unwrap()
}

fn deserialize_pk<R: RingOps>(ring: &R, bytes: &[u8]) -> RlwePublicKeyOwned<R::Elem> {
    let pk_compact: RlwePublicKey<Compact> = bincode::deserialize(bytes).unwrap();
    pk_compact.uncompact(ring)
}

fn serialize_bs_key_share<R: RingOps, M: ModulusOps>(
    ring: &R,
    mod_ks: &M,
    bs_key_share: &FhewBoolMpiKeyShareOwned<R::Elem, M::Elem>,
) -> Vec<u8> {
    bincode::serialize(&bs_key_share.compact(ring, mod_ks)).unwrap()
}

fn deserialize_bs_key_share<R: RingOps, M: ModulusOps>(
    ring: &R,
    mod_ks: &M,
    bytes: &[u8],
) -> FhewBoolMpiKeyShareOwned<R::Elem, M::Elem> {
    let bs_key_share_compact: FhewBoolMpiKeyShareCompact = bincode::deserialize(bytes).unwrap();
    bs_key_share_compact.uncompact(ring, mod_ks)
}

fn serialize_cts<R: RingOps>(ring: &R, cts: [FhewBoolCiphertextOwned<R::Elem>; 8]) -> Vec<u8> {
    bincode::serialize(&cts.map(|ct| ct.compact(ring))).unwrap()
}

fn deserialize_cts<R: RingOps>(ring: &R, bytes: &[u8]) -> [FhewBoolCiphertextOwned<R::Elem>; 8] {
    let cts: [FhewBoolCiphertext<Compact>; 8] = bincode::deserialize(bytes).unwrap();
    cts.map(|ct| ct.uncompact(ring))
}

fn function<T>(a: &T, b: &T, c: &T, d: &T, e: &T) -> T
where
    T: for<'t> NumOps<&'t T, T>,
    for<'t> &'t T: NumOps<&'t T, T>,
{
    (((a + b) - c) * d) % e
}

fn main() {
    let mut server = Server::<NoisyPrimeRing, NonNativePowerOfTwo>::new(I_4P);
    let mut clients = (0..server.param.total_shares)
        .map(|share_idx| {
            Client::<PrimeRing, NonNativePowerOfTwo>::new(server.param, server.crs, share_idx)
        })
        .collect_vec();

    // Round 1

    // Clients generate public key shares
    let pk_shares = clients
        .iter()
        .map(|client| serialize_pk_share(client.ring(), &client.pk_share_gen()))
        .collect_vec();

    // Server aggregates public key shares
    server.aggregate_pk_shares(
        &pk_shares
            .into_iter()
            .map(|bytes| deserialize_pk_share(server.ring(), &bytes))
            .collect_vec(),
    );
    let pk = serialize_pk(server.ring(), &server.pk);

    // Round 2

    // Clients generate bootstrapping key shares
    let bs_key_shares = clients
        .iter_mut()
        .map(|client| {
            client.receive_pk(&deserialize_pk(client.ring(), &pk));
            serialize_bs_key_share(client.ring(), client.mod_ks(), &client.bs_key_share_gen())
        })
        .collect_vec();

    // Server aggregates bootstrapping key shares
    server.aggregate_bs_key_shares::<PrimeRing>(
        &bs_key_shares
            .into_iter()
            .map(|bytes| deserialize_bs_key_share(server.ring(), server.mod_ks(), &bytes))
            .collect_vec(),
    );

    // Server performs FHE evaluation
    let m = from_fn(|_| StdRng::from_entropy().gen());
    let g = {
        let [a, b, c, d, e] = &m.map(Wrapping);
        function(a, b, c, d, e).0
    };
    let ct_g = {
        let [a, b, c, d, e] = &m.map(|m| FheU8::from_cts(&server.evaluator, server.pk_encrypt(m)));
        serialize_cts(server.ring(), function(a, b, c, d, e).into_cts())
    };

    // Clients generate decryption share of evaluation output
    let ct_g_dec_shares = clients
        .iter()
        .map(|client| client.decrypt_share(deserialize_cts(client.ring(), &ct_g)))
        .collect_vec();

    // Aggregate decryption shares
    assert_eq!(g, {
        let ct_g = deserialize_cts(clients[0].ring(), &ct_g);
        aggregate_decryption_shares(clients[0].ring(), ct_g, &ct_g_dec_shares)
    });
}
