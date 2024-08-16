use phantom_zone_math::{
    decomposer::DecompositionParam,
    modulus::ElemFrom,
    ring::RingOps,
    util::scratch::{self, Scratch},
};
use rand::{RngCore, SeedableRng};
use structure::{InteractiveMpcParameters, RlwePublicKeyShare, RlwePublicKeyShareMutView};

use crate::{
    core::{
        lwe::{
            ks_key_gen, seeded_ks_key_gen, LweKeySwitchKey, LweSecretKey, LweSecretKeyView,
            SeededLweKeySwitchKey,
        },
        rgsw::{self, RgswDecompositionParam},
        rlwe::{sk_encrypt_zero, RlweSecretKeyView},
    },
    util::{distribution::NoiseDistribution, rng::LweRng},
};

mod structure;

fn collective_rlwe_public_key_share<'a, R, T>(
    ring: &R,
    pk_share: impl Into<RlwePublicKeyShareMutView<'a, R::Elem>>,
    sk: impl Into<RlweSecretKeyView<'a, T>>,
    mut scratch: Scratch,
    noise_distribution: NoiseDistribution,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'a + Copy,
{
    let (mut pk_share, sk) = (pk_share.into(), sk.into());

    let t0 = ring.take_poly(&mut scratch);
    ring.sample_uniform_into(t0, rng.a());
    ring.poly_mul_assign_elem_from(t0, sk.as_ref(), scratch.reborrow());

    ring.sample_into::<f64>(pk_share.as_mut(), noise_distribution, rng.noise());
    ring.slice_add_assign(pk_share.as_mut(), t0);
}

struct InteractiveCrs<S> {
    seed: S,
}

impl<S: Copy> InteractiveCrs<S> {
    fn lwe_ksk_rng<R: SeedableRng<Seed = S>>(&self) -> LweRng<R, R> {
        // TODO: make sure 1st rng is derive from self.seed (punctured appropriate no. of times)
        // and 2nd rng is derived from secret that user stores. The latter assures that MPC transcript remains same across multiple runs, given same CRS
        LweRng::new(R::from_entropy(), R::from_entropy())
    }
}

fn crs_server_key_share<'a, R, T, S>(
    parameters: InteractiveMpcParameters,
    lwe_sk: impl Into<LweSecretKeyView<'a, T>>,
    rlwe_sk: impl Into<RlweSecretKeyView<'a, T>>,
    user_id: usize,
    total_users: usize,
    crs: InteractiveCrs<S>,
    scratch: &mut Scratch,
    rng: &mut LweRng<impl RngCore, impl RngCore>,
) where
    R: RingOps + ElemFrom<T>,
    T: 'a + Copy,
    S: Copy,
{
    let ring_rlwe = R::new(parameters.rlwe_q(), parameters.rlwe_n());
    let ring_lwe = R::new(parameters.lwe_q(), parameters.lwe_n());

    // Seeded LWE key switching key
    let mut seeded_ks_key = SeededLweKeySwitchKey::allocate(
        parameters.lwe_n(),
        parameters.rlwe_n(),
        parameters.lwe_ksk_decomposer(),
    );
    seeded_ks_key_gen(
        &ring_lwe,
        seeded_ks_key.as_mut_view(),
        LweSecretKey::from(rlwe_sk.into()).as_view(),
        lwe_sk,
        parameters.noise_distribution(),
        scratch,
        rng,
    );

    // Seeded auto keys
    
}
