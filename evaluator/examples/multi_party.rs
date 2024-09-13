use core::{array::from_fn, num::Wrapping};
use itertools::Itertools;
use num_traits::NumOps;
use phantom_zone_evaluator::boolean::evaluator::fhew::prelude::*;
use rand::{rngs::StdRng, RngCore, SeedableRng};

const PARAM: FhewBoolMpiParam = FhewBoolMpiParam {
    param: FhewBoolParam {
        message_bits: 2,
        modulus: Modulus::PowerOfTwo(64),
        ring_size: 2048,
        sk_distribution: SecretDistribution::Ternary(Ternary),
        noise_distribution: NoiseDistribution::Gaussian(Gaussian(3.19)),
        u_distribution: SecretDistribution::Ternary(Ternary),
        auto_decomposition_param: DecompositionParam {
            log_base: 24,
            level: 1,
        },
        rlwe_by_rgsw_decomposition_param: RgswDecompositionParam {
            log_base: 17,
            level_a: 1,
            level_b: 1,
        },
        lwe_modulus: Modulus::PowerOfTwo(16),
        lwe_dimension: 620,
        lwe_sk_distribution: SecretDistribution::Ternary(Ternary),
        lwe_noise_distribution: NoiseDistribution::Gaussian(Gaussian(3.19)),
        lwe_ks_decomposition_param: DecompositionParam {
            log_base: 1,
            level: 13,
        },
        q: 2048,
        g: 5,
        w: 10,
    },
    rgsw_by_rgsw_decomposition_param: RgswDecompositionParam {
        log_base: 6,
        level_a: 7,
        level_b: 6,
    },
    total_shares: 4,
};

struct Client<R, M> {
    param: FhewBoolMpiParam,
    crs: FhewBoolMpiCrs<StdRng>,
    share_idx: usize,
    ring: R,
    mod_ks: M,
    sk: LweSecretKeyOwned<i32>,
}

impl<R: RingOps, M: ModulusOps> Client<R, M> {
    fn new(param: FhewBoolMpiParam, crs: FhewBoolMpiCrs<StdRng>, share_idx: usize) -> Self {
        Self {
            param,
            crs,
            share_idx,
            ring: <R as RingOps>::new(param.modulus, param.ring_size),
            mod_ks: M::new(param.lwe_modulus),
            sk: LweSecretKey::sample(
                param.ring_size,
                param.sk_distribution,
                StdRng::from_entropy(),
            ),
        }
    }

    fn pk_share_gen(&self) -> SeededRlwePublicKeyOwned<R::Elem> {
        let mut scratch = self.ring.allocate_scratch(2, 2, 0);
        let mut pk = SeededRlwePublicKey::allocate(self.param.ring_size);
        pk_share_gen(
            &self.ring,
            &mut pk,
            &self.param,
            &self.crs,
            self.sk.as_view(),
            scratch.borrow_mut(),
            &mut StdRng::from_entropy(),
        );
        pk
    }

    fn bs_key_share_gen(
        &self,
        pk: &RlwePublicKeyOwned<R::Elem>,
    ) -> FhewBoolMpiKeyShareOwned<R::Elem, M::Elem> {
        let sk_ks = LweSecretKey::sample(
            self.param.lwe_dimension,
            self.param.lwe_sk_distribution,
            StdRng::from_entropy(),
        );
        let mut scratch = self.ring.allocate_scratch(2, 3, 0);
        let mut bs_key_share = FhewBoolMpiKeyShare::allocate(self.param, self.share_idx);
        bs_key_share_gen(
            &self.ring,
            &self.mod_ks,
            &mut bs_key_share,
            &self.crs,
            self.sk.as_view(),
            pk,
            &sk_ks,
            scratch.borrow_mut(),
            &mut StdRng::from_entropy(),
        );
        bs_key_share
    }

    fn decrypt_share<R2: RingOps<Elem = R::Elem>>(
        &self,
        ct: &FheU8<FhewBoolEvaluator<R2, M>>,
    ) -> [LweDecryptionShare<R::Elem>; 8] {
        ct.cts().map(|ct| {
            ct.decrypt_share(
                &self.ring,
                &self.sk,
                self.param.noise_distribution,
                &mut StdLweRng::from_entropy(),
            )
        })
    }
}

struct Server<R: RingOps, M: ModulusOps> {
    param: FhewBoolMpiParam,
    crs: FhewBoolMpiCrs<StdRng>,
    pk: RlwePublicKeyOwned<R::Elem>,
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

    fn aggregate_pk_shares(&mut self, pk_shares: &[SeededRlwePublicKeyOwned<R::Elem>]) {
        aggregate_pk_shares(self.evaluator.ring(), &mut self.pk, &self.crs, pk_shares);
    }

    fn aggregate_bs_key_shares<R2: RingOps<Elem = R::Elem>>(
        &mut self,
        bs_key_shares: &[FhewBoolMpiKeyShareOwned<R::Elem, M::Elem>],
    ) {
        let bs_key = {
            let ring = <R2 as RingOps>::new(self.param.modulus, self.param.ring_size);
            let mut scratch = ring.allocate_scratch(
                2,
                3,
                2 * (self.param.rgsw_by_rgsw_decomposition_param.level_a
                    + self.param.rgsw_by_rgsw_decomposition_param.level_b),
            );
            let mut bs_key = FhewBoolKey::allocate(*self.param);
            aggregate_bs_key_shares(
                &ring,
                self.evaluator.mod_ks(),
                &mut bs_key,
                &self.crs,
                bs_key_shares,
                scratch.borrow_mut(),
            );
            bs_key
        };
        let bs_key_prep = {
            let mut scratch = self.evaluator.ring().allocate_scratch(0, 3, 0);
            let mut bs_key_prep =
                FhewBoolKey::allocate_eval(*self.param, self.evaluator.ring().eval_size());
            prepare_bs_key(
                self.evaluator.ring(),
                &mut bs_key_prep,
                &bs_key,
                scratch.borrow_mut(),
            );
            bs_key_prep
        };
        self.evaluator = FhewBoolEvaluator::new(bs_key_prep);
    }

    fn encrypt(
        &self,
        m: u8,
        rng: &mut LweRng<impl RngCore, impl RngCore>,
    ) -> FheU8<FhewBoolEvaluator<R, M>> {
        let cts = FhewBoolCiphertext::batched_pk_encrypt(
            self.evaluator.param(),
            self.evaluator.ring(),
            &self.pk,
            (0..8).map(|idx| (m >> idx) & 1 == 1),
            rng,
        );
        FheU8::from_cts(&self.evaluator, cts.try_into().unwrap())
    }

    fn aggregate_decryption_shares(
        &self,
        ct: &FheU8<FhewBoolEvaluator<R, M>>,
        dec_shares: &[[LweDecryptionShare<R::Elem>; 8]],
    ) -> u8 {
        (0..8)
            .map(|idx| {
                let dec_shares = dec_shares.iter().map(|dec_shares| &dec_shares[idx]);
                ct.cts()[idx].aggregate_decryption_shares(self.evaluator.ring(), dec_shares)
            })
            .rev()
            .fold(0, |m, b| (m << 1) | b as u8)
    }
}

fn function<T>(a: &T, b: &T, c: &T, d: &T, e: &T) -> T
where
    T: for<'t> NumOps<&'t T, T>,
    for<'t> &'t T: NumOps<&'t T, T>,
{
    (((a + b) - c) * d) % e
}

fn main() {
    let mut rng = StdLweRng::from_entropy();
    let mut server = Server::<NoisyNativeRing, NonNativePowerOfTwo>::new(PARAM);
    let clients = (0..PARAM.total_shares)
        .map(|share_idx| {
            Client::<NativeRing, NonNativePowerOfTwo>::new(PARAM, server.crs, share_idx)
        })
        .collect_vec();

    let pk_shares = clients
        .iter()
        .map(|client| client.pk_share_gen())
        .collect_vec();
    server.aggregate_pk_shares(&pk_shares);

    let bs_key_shares = clients
        .iter()
        .map(|client| client.bs_key_share_gen(&server.pk))
        .collect_vec();
    server.aggregate_bs_key_shares::<NativeRing>(&bs_key_shares);

    let m = from_fn(|_| rng.next_u64() as u8);
    let g = {
        let [a, b, c, d, e] = &m.map(Wrapping);
        function(a, b, c, d, e).0
    };
    let ct_g = {
        let [a, b, c, d, e] = &m.map(|m| server.encrypt(m, &mut rng));
        function(a, b, c, d, e)
    };

    let dec_shares = clients
        .iter()
        .map(|client| client.decrypt_share(&ct_g))
        .collect_vec();
    assert_eq!(g, server.aggregate_decryption_shares(&ct_g, &dec_shares));
}
