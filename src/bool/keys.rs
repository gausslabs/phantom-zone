use std::{collections::HashMap, marker::PhantomData};

use serde::{Deserialize, Serialize};

use crate::{
    backend::{ModInit, VectorOps},
    pbs::WithShoupRepr,
    random::{NewWithSeed, RandomFillUniformInModulus},
    utils::ToShoup,
    Matrix, MatrixEntity, MatrixMut, RowEntity, RowMut,
};

use super::parameters::{BoolParameters, CiphertextModulus};

pub(crate) trait SinglePartyClientKey {
    type Element;
    fn sk_rlwe(&self) -> Vec<Self::Element>;
    fn sk_lwe(&self) -> Vec<Self::Element>;
}

pub(crate) trait InteractiveMultiPartyClientKey {
    type Element;
    fn sk_rlwe(&self) -> Vec<Self::Element>;
    fn sk_lwe(&self) -> Vec<Self::Element>;
}

pub(crate) trait NonInteractiveMultiPartyClientKey {
    type Element;
    fn sk_rlwe(&self) -> Vec<Self::Element>;
    fn sk_u_rlwe(&self) -> Vec<Self::Element>;
    fn sk_lwe(&self) -> Vec<Self::Element>;
}

/// Client key
///
/// Key is used for all parameter varians - Single party, interactive
/// multi-party, and non-interactive multi-party. The only stored the main seed
/// and seeds of the Rlwe/Lwe secrets are derived at puncturing the seed desired
/// number of times.
///
/// ### Punctures required:
///
///     Puncture 1 -> Seed of RLWE secret used as main RLWE secret for
///                   single-party, interactive/non-interactive multi-party
///
///     Puncture 2 -> Seed of LWE secret used main LWE secret for single-party,
///                   interactive/non-interactive multi-party
///
///     Puncture 3 -> Seed of RLWE secret used as `u` in
///                   non-interactive multi-party.
#[derive(Clone, Serialize, Deserialize)]
pub struct ClientKey<S, E> {
    seed: S,
    parameters: BoolParameters<E>,
}

mod impl_ck {
    use crate::{
        parameters::SecretKeyDistribution,
        random::{DefaultSecureRng, RandomFillGaussian},
        utils::{fill_random_ternary_secret_with_hamming_weight, puncture_p_rng},
    };

    use super::*;

    impl<E> ClientKey<[u8; 32], E> {
        pub(in super::super) fn new(parameters: BoolParameters<E>) -> ClientKey<[u8; 32], E> {
            let mut rng = DefaultSecureRng::new();
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);
            Self { seed, parameters }
        }
    }

    impl<E> SinglePartyClientKey for ClientKey<[u8; 32], E> {
        type Element = i32;
        fn sk_lwe(&self) -> Vec<Self::Element> {
            let mut p_rng = DefaultSecureRng::new_seeded(self.seed);
            let lwe_seed = puncture_p_rng::<[u8; 32], DefaultSecureRng>(&mut p_rng, 2);

            let mut lwe_prng = DefaultSecureRng::new_seeded(lwe_seed);

            let mut out = vec![0i32; self.parameters.lwe_n().0];

            match self.parameters.lwe_secret_key_dist() {
                &SecretKeyDistribution::ErrorDistribution => {
                    RandomFillGaussian::random_fill(&mut lwe_prng, &mut out);
                }
                &SecretKeyDistribution::TernaryDistribution => {
                    fill_random_ternary_secret_with_hamming_weight(
                        &mut out,
                        self.parameters.lwe_n().0 >> 1,
                        &mut lwe_prng,
                    );
                }
            }
            out
        }
        fn sk_rlwe(&self) -> Vec<Self::Element> {
            assert!(
                self.parameters.rlwe_secret_key_dist()
                    == &SecretKeyDistribution::TernaryDistribution
            );

            let mut p_rng = DefaultSecureRng::new_seeded(self.seed);
            let rlwe_seed = puncture_p_rng::<[u8; 32], DefaultSecureRng>(&mut p_rng, 1);

            let mut rlwe_prng = DefaultSecureRng::new_seeded(rlwe_seed);
            let mut out = vec![0i32; self.parameters.rlwe_n().0];
            fill_random_ternary_secret_with_hamming_weight(
                &mut out,
                self.parameters.rlwe_n().0 >> 1,
                &mut rlwe_prng,
            );
            out
        }
    }

    #[cfg(feature = "interactive_mp")]
    impl<E> InteractiveMultiPartyClientKey for ClientKey<[u8; 32], E> {
        type Element = i32;
        fn sk_lwe(&self) -> Vec<Self::Element> {
            <Self as SinglePartyClientKey>::sk_lwe(&self)
        }
        fn sk_rlwe(&self) -> Vec<Self::Element> {
            <Self as SinglePartyClientKey>::sk_rlwe(&self)
        }
    }

    #[cfg(feature = "non_interactive_mp")]
    impl<E> NonInteractiveMultiPartyClientKey for ClientKey<[u8; 32], E> {
        type Element = i32;
        fn sk_lwe(&self) -> Vec<Self::Element> {
            <Self as SinglePartyClientKey>::sk_lwe(&self)
        }
        fn sk_rlwe(&self) -> Vec<Self::Element> {
            <Self as SinglePartyClientKey>::sk_rlwe(&self)
        }
        fn sk_u_rlwe(&self) -> Vec<Self::Element> {
            assert!(
                self.parameters.rlwe_secret_key_dist()
                    == &SecretKeyDistribution::TernaryDistribution
            );

            let mut p_rng = DefaultSecureRng::new_seeded(self.seed);
            let rlwe_seed = puncture_p_rng::<[u8; 32], DefaultSecureRng>(&mut p_rng, 3);

            let mut rlwe_prng = DefaultSecureRng::new_seeded(rlwe_seed);
            let mut out = vec![0i32; self.parameters.rlwe_n().0];
            fill_random_ternary_secret_with_hamming_weight(
                &mut out,
                self.parameters.rlwe_n().0 >> 1,
                &mut rlwe_prng,
            );
            out
        }
    }
}

/// Public key
pub struct PublicKey<M, Rng, ModOp> {
    key: M,
    _phantom: PhantomData<(Rng, ModOp)>,
}

pub(super) mod impl_pk {
    use super::*;

    impl<M, R, Mo> PublicKey<M, R, Mo> {
        pub(in super::super) fn key(&self) -> &M {
            &self.key
        }
    }

    impl<
            M: MatrixMut + MatrixEntity,
            Rng: NewWithSeed
                + RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>>,
            ModOp,
        > From<SeededPublicKey<M::R, Rng::Seed, BoolParameters<M::MatElement>, ModOp>>
        for PublicKey<M, Rng, ModOp>
    where
        <M as Matrix>::R: RowMut,
        M::MatElement: Copy,
    {
        fn from(
            value: SeededPublicKey<M::R, Rng::Seed, BoolParameters<M::MatElement>, ModOp>,
        ) -> Self {
            let mut prng = Rng::new_with_seed(value.seed);

            let mut key = M::zeros(2, value.parameters.rlwe_n().0);
            // sample A
            RandomFillUniformInModulus::random_fill(
                &mut prng,
                value.parameters.rlwe_q(),
                key.get_row_mut(0),
            );
            // Copy over B
            key.get_row_mut(1).copy_from_slice(value.part_b.as_ref());

            PublicKey {
                key,
                _phantom: PhantomData,
            }
        }
    }

    impl<
            M: MatrixMut + MatrixEntity,
            Rng: NewWithSeed
                + RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>>,
            ModOp: VectorOps<Element = M::MatElement> + ModInit<M = CiphertextModulus<M::MatElement>>,
        >
        From<
            &[CommonReferenceSeededCollectivePublicKeyShare<
                M::R,
                Rng::Seed,
                BoolParameters<M::MatElement>,
            >],
        > for PublicKey<M, Rng, ModOp>
    where
        <M as Matrix>::R: RowMut,
        Rng::Seed: Copy + PartialEq,
        M::MatElement: PartialEq + Copy,
    {
        fn from(
            value: &[CommonReferenceSeededCollectivePublicKeyShare<
                M::R,
                Rng::Seed,
                BoolParameters<M::MatElement>,
            >],
        ) -> Self {
            assert!(value.len() > 0);

            let parameters = &value[0].parameters;
            let mut key = M::zeros(2, parameters.rlwe_n().0);

            // sample A
            let seed = value[0].cr_seed;
            let mut main_rng = Rng::new_with_seed(seed);
            RandomFillUniformInModulus::random_fill(
                &mut main_rng,
                parameters.rlwe_q(),
                key.get_row_mut(0),
            );

            // Sum all Bs
            let rlweq_modop = ModOp::new(parameters.rlwe_q().clone());
            value.iter().for_each(|share_i| {
                assert!(share_i.cr_seed == seed);
                assert!(&share_i.parameters == parameters);

                rlweq_modop.elwise_add_mut(key.get_row_mut(1), share_i.share.as_ref());
            });

            PublicKey {
                key,
                _phantom: PhantomData,
            }
        }
    }
}

/// Seeded public key
struct SeededPublicKey<Ro, S, P, ModOp> {
    part_b: Ro,
    seed: S,
    parameters: P,
    _phantom: PhantomData<ModOp>,
}

mod impl_seeded_pk {
    use super::*;

    impl<R, S, ModOp>
        From<&[CommonReferenceSeededCollectivePublicKeyShare<R, S, BoolParameters<R::Element>>]>
        for SeededPublicKey<R, S, BoolParameters<R::Element>, ModOp>
    where
        ModOp: VectorOps<Element = R::Element> + ModInit<M = CiphertextModulus<R::Element>>,
        S: PartialEq + Clone,
        R: RowMut + RowEntity + Clone,
        R::Element: Clone + PartialEq,
    {
        fn from(
            value: &[CommonReferenceSeededCollectivePublicKeyShare<
                R,
                S,
                BoolParameters<R::Element>,
            >],
        ) -> Self {
            assert!(value.len() > 0);

            let parameters = &value[0].parameters;
            let cr_seed = value[0].cr_seed.clone();

            // Sum all Bs
            let rlweq_modop = ModOp::new(parameters.rlwe_q().clone());
            let mut part_b = value[0].share.clone();
            value.iter().skip(1).for_each(|share_i| {
                assert!(&share_i.cr_seed == &cr_seed);
                assert!(&share_i.parameters == parameters);

                rlweq_modop.elwise_add_mut(part_b.as_mut(), share_i.share.as_ref());
            });

            Self {
                part_b,
                seed: cr_seed,
                parameters: parameters.clone(),
                _phantom: PhantomData,
            }
        }
    }
}

/// CRS seeded collective public key share
#[derive(Clone, Serialize, Deserialize)]
pub struct CommonReferenceSeededCollectivePublicKeyShare<Ro, S, P> {
    /// Public key share polynomial
    share: Ro,
    /// Common reference seed
    cr_seed: S,
    /// Parameters
    parameters: P,
}
impl<Ro, S, P> CommonReferenceSeededCollectivePublicKeyShare<Ro, S, P> {
    pub(super) fn new(share: Ro, cr_seed: S, parameters: P) -> Self {
        CommonReferenceSeededCollectivePublicKeyShare {
            share,
            cr_seed,
            parameters,
        }
    }
}

/// Common reference seed seeded interactive multi-party server key share
#[derive(Clone, Serialize, Deserialize)]
pub struct CommonReferenceSeededInteractiveMultiPartyServerKeyShare<M: Matrix, P, S> {
    /// Public key encrypted RGSW(m = X^{s[i]}) ciphertexts for LWE secret
    /// indices for which `Self` is the leader. Note that when `Self` is
    /// leader RGSW ciphertext is encrypted using RLWE x RGSW decomposer
    self_leader_rgsws: Vec<M>,
    /// Public key encrypted RGSW(m = X^{s[i]}) ciphertext for LWE secret
    /// indices for which `Self` is `not` the leader. Note that when `Self`
    /// is not the leader RGSW ciphertext is encrypted using RGSW1
    /// decomposer for RGSW0 x RGSW1
    not_self_leader_rgsws: Vec<M>,
    /// Auto key shares for auto elements [-g, g, g^2, .., g^{w}] where `w`
    /// is the window size parameter. Share corresponding to auto element -g
    /// is stored at key `0` and share corresponding to auto element g^{k} is
    /// stored at key `k`.
    auto_keys: HashMap<usize, M>,
    /// LWE key switching key share to key switching ciphertext LWE_{q, s}(m) to
    /// LWE_{q, z}(m) where q is LWE ciphertext modulus, `s` is the ideal RLWE
    /// secret with dimension N, and `z` is the ideal LWE secret of dimension n.
    lwe_ksk: M::R,
    /// Common reference seed
    cr_seed: S,
    parameters: P,
    /// User id assigned by the server.
    ///
    /// User id must be unique and  a number in range [0, total_users)
    user_id: usize,
}

impl<M: Matrix, P, S> CommonReferenceSeededInteractiveMultiPartyServerKeyShare<M, P, S> {
    pub(super) fn new(
        self_leader_rgsws: Vec<M>,
        not_self_leader_rgsws: Vec<M>,
        auto_keys: HashMap<usize, M>,
        lwe_ksk: M::R,
        cr_seed: S,
        parameters: P,
        user_id: usize,
    ) -> Self {
        CommonReferenceSeededInteractiveMultiPartyServerKeyShare {
            self_leader_rgsws,
            not_self_leader_rgsws,
            auto_keys,
            lwe_ksk,
            cr_seed,
            parameters,
            user_id,
        }
    }

    pub(super) fn cr_seed(&self) -> &S {
        &self.cr_seed
    }

    pub(super) fn parameters(&self) -> &P {
        &self.parameters
    }

    pub(super) fn auto_keys(&self) -> &HashMap<usize, M> {
        &self.auto_keys
    }

    pub(crate) fn self_leader_rgsws(&self) -> &[M] {
        &self.self_leader_rgsws
    }

    pub(super) fn not_self_leader_rgsws(&self) -> &[M] {
        &self.not_self_leader_rgsws
    }

    pub(super) fn lwe_ksk(&self) -> &M::R {
        &self.lwe_ksk
    }

    pub(super) fn user_id(&self) -> usize {
        self.user_id
    }
}

/// Common reference seeded interactive multi-party server key
pub struct SeededInteractiveMultiPartyServerKey<M: Matrix, S, P> {
    /// RGSW ciphertexts RGSW(X^{s[i]}) encrypted under ideal RLWE secret key
    /// where `s` is ideal LWE secret key for each LWE secret dimension.
    rgsw_cts: Vec<M>,
    /// Seeded auto keys under ideal RLWE secret for RLWE automorphisms with
    /// auto elements [-g, g, g^2,..., g^{w}]. Auto key corresponidng to
    /// auto element -g is stored at key `0` and key corresponding to auto
    /// element g^{k} is stored at key `k`
    auto_keys: HashMap<usize, M>,
    /// Seeded LWE key switching key under ideal LWE secret to switch LWE_{q,
    /// s}(m) to LWE_{q, z}(m) where s is ideal RLWE secret and z is ideal LWE
    /// secret.
    lwe_ksk: M::R,
    /// Common reference seed
    cr_seed: S,
    parameters: P,
}

impl<M: Matrix, S, P> SeededInteractiveMultiPartyServerKey<M, S, P> {
    pub(super) fn new(
        rgsw_cts: Vec<M>,
        auto_keys: HashMap<usize, M>,
        lwe_ksk: M::R,
        cr_seed: S,
        parameters: P,
    ) -> Self {
        SeededInteractiveMultiPartyServerKey {
            rgsw_cts,
            auto_keys,
            lwe_ksk,
            cr_seed,
            parameters,
        }
    }

    pub(super) fn rgsw_cts(&self) -> &[M] {
        &self.rgsw_cts
    }
}

/// Seeded single party server key
pub struct SeededSinglePartyServerKey<M: Matrix, P, S> {
    /// Rgsw cts of LWE secret elements
    pub(crate) rgsw_cts: Vec<M>,
    /// Auto keys. Key corresponding to g^{k} is at index `k`. Key corresponding
    /// to -g is at 0
    pub(crate) auto_keys: HashMap<usize, M>,
    /// LWE ksk to key switching LWE ciphertext from RLWE secret to LWE secret
    pub(crate) lwe_ksk: M::R,
    /// Parameters
    pub(crate) parameters: P,
    /// Main seed
    pub(crate) seed: S,
}
impl<M: Matrix, S> SeededSinglePartyServerKey<M, BoolParameters<M::MatElement>, S> {
    pub(super) fn from_raw(
        auto_keys: HashMap<usize, M>,
        rgsw_cts: Vec<M>,
        lwe_ksk: M::R,
        parameters: BoolParameters<M::MatElement>,
        seed: S,
    ) -> Self {
        // sanity checks
        auto_keys.iter().for_each(|v| {
            assert!(
                v.1.dimension()
                    == (
                        parameters.auto_decomposition_count().0,
                        parameters.rlwe_n().0
                    )
            )
        });

        let (part_a_d, part_b_d) = parameters.rlwe_rgsw_decomposition_count();
        rgsw_cts.iter().for_each(|v| {
            assert!(v.dimension() == (part_a_d.0 * 2 + part_b_d.0, parameters.rlwe_n().0))
        });
        assert!(
            lwe_ksk.as_ref().len()
                == (parameters.lwe_decomposition_count().0 * parameters.rlwe_n().0)
        );

        SeededSinglePartyServerKey {
            rgsw_cts,
            auto_keys,
            lwe_ksk,
            parameters,
            seed,
        }
    }
}

/// Server key in evaluation domain
pub(crate) struct ServerKeyEvaluationDomain<M, P, R, N> {
    /// RGSW ciphertext RGSW(X^{s[i]}) for each LWE index in evaluation domain
    rgsw_cts: Vec<M>,
    /// Auto keys for all auto elements [-g, g, g^2,..., g^w] in evaluation
    /// domain
    galois_keys: HashMap<usize, M>,
    /// LWE key switching key to key switch LWE_{q, s}(m) to LWE_{q, z}(m)
    lwe_ksk: M,
    parameters: P,
    _phanton: PhantomData<(R, N)>,
}

pub(super) mod impl_server_key_eval_domain {
    use itertools::{izip, Itertools};

    use crate::{
        bool::evaluator::InteractiveMultiPartyCrs,
        ntt::{Ntt, NttInit},
        pbs::PbsKey,
        random::RandomFill,
    };

    use super::*;

    impl<M, Mod, R, N> ServerKeyEvaluationDomain<M, Mod, R, N> {
        pub(in super::super) fn rgsw_cts(&self) -> &[M] {
            &self.rgsw_cts
        }
    }

    impl<
            M: MatrixMut + MatrixEntity,
            R: RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>>
                + NewWithSeed,
            N: NttInit<CiphertextModulus<M::MatElement>> + Ntt<Element = M::MatElement>,
        > From<&SeededSinglePartyServerKey<M, BoolParameters<M::MatElement>, R::Seed>>
        for ServerKeyEvaluationDomain<M, BoolParameters<M::MatElement>, R, N>
    where
        <M as Matrix>::R: RowMut,
        M::MatElement: Copy,
        R::Seed: Clone,
    {
        fn from(
            value: &SeededSinglePartyServerKey<M, BoolParameters<M::MatElement>, R::Seed>,
        ) -> Self {
            let mut main_prng = R::new_with_seed(value.seed.clone());
            let parameters = &value.parameters;
            let g = parameters.g() as isize;
            let ring_size = value.parameters.rlwe_n().0;
            let lwe_n = value.parameters.lwe_n().0;
            let rlwe_q = value.parameters.rlwe_q();
            let lwq_q = value.parameters.lwe_q();

            let nttop = N::new(rlwe_q, ring_size);

            // galois keys
            let mut auto_keys = HashMap::new();
            let auto_decomp_count = parameters.auto_decomposition_count().0;
            let auto_element_dlogs = parameters.auto_element_dlogs();
            for i in auto_element_dlogs.into_iter() {
                let seeded_auto_key = value.auto_keys.get(&i).unwrap();
                assert!(seeded_auto_key.dimension() == (auto_decomp_count, ring_size));

                let mut data = M::zeros(auto_decomp_count * 2, ring_size);

                // sample RLWE'_A(-s(X^k))
                data.iter_rows_mut().take(auto_decomp_count).for_each(|ri| {
                    RandomFillUniformInModulus::random_fill(&mut main_prng, &rlwe_q, ri.as_mut())
                });

                // copy over RLWE'B_(-s(X^k))
                izip!(
                    data.iter_rows_mut().skip(auto_decomp_count),
                    seeded_auto_key.iter_rows()
                )
                .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

                // Send to Evaluation domain
                data.iter_rows_mut()
                    .for_each(|ri| nttop.forward(ri.as_mut()));

                auto_keys.insert(i, data);
            }

            // RGSW ciphertexts
            let (rlrg_a_decomp, rlrg_b_decomp) = parameters.rlwe_rgsw_decomposition_count();
            let rgsw_cts = value
                .rgsw_cts
                .iter()
                .map(|seeded_rgsw_si| {
                    assert!(
                        seeded_rgsw_si.dimension()
                            == (rlrg_a_decomp.0 * 2 + rlrg_b_decomp.0, ring_size)
                    );

                    let mut data = M::zeros(rlrg_a_decomp.0 * 2 + rlrg_b_decomp.0 * 2, ring_size);

                    // copy over RLWE'(-sm)
                    izip!(
                        data.iter_rows_mut().take(rlrg_a_decomp.0 * 2),
                        seeded_rgsw_si.iter_rows().take(rlrg_a_decomp.0 * 2)
                    )
                    .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

                    // sample RLWE'_A(m)
                    data.iter_rows_mut()
                        .skip(rlrg_a_decomp.0 * 2)
                        .take(rlrg_b_decomp.0)
                        .for_each(|ri| {
                            RandomFillUniformInModulus::random_fill(
                                &mut main_prng,
                                &rlwe_q,
                                ri.as_mut(),
                            )
                        });

                    // copy over RLWE'_B(m)
                    izip!(
                        data.iter_rows_mut()
                            .skip(rlrg_a_decomp.0 * 2 + rlrg_b_decomp.0),
                        seeded_rgsw_si.iter_rows().skip(rlrg_a_decomp.0 * 2)
                    )
                    .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

                    // send polynomials to evaluation domain
                    data.iter_rows_mut()
                        .for_each(|ri| nttop.forward(ri.as_mut()));

                    data
                })
                .collect_vec();

            // LWE ksk
            let lwe_ksk = {
                let d = parameters.lwe_decomposition_count().0;
                assert!(value.lwe_ksk.as_ref().len() == d * ring_size);

                let mut data = M::zeros(d * ring_size, lwe_n + 1);
                izip!(data.iter_rows_mut(), value.lwe_ksk.as_ref().iter()).for_each(
                    |(lwe_i, bi)| {
                        RandomFillUniformInModulus::random_fill(
                            &mut main_prng,
                            &lwq_q,
                            &mut lwe_i.as_mut()[1..],
                        );
                        lwe_i.as_mut()[0] = *bi;
                    },
                );

                data
            };

            ServerKeyEvaluationDomain {
                rgsw_cts,
                galois_keys: auto_keys,
                lwe_ksk,
                parameters: parameters.clone(),
                _phanton: PhantomData,
            }
        }
    }

    impl<
            M: MatrixMut + MatrixEntity,
            Rng: NewWithSeed,
            N: NttInit<CiphertextModulus<M::MatElement>> + Ntt<Element = M::MatElement>,
        >
        From<
            &SeededInteractiveMultiPartyServerKey<
                M,
                InteractiveMultiPartyCrs<Rng::Seed>,
                BoolParameters<M::MatElement>,
            >,
        > for ServerKeyEvaluationDomain<M, BoolParameters<M::MatElement>, Rng, N>
    where
        <M as Matrix>::R: RowMut,
        Rng::Seed: Copy + Default,
        Rng: RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>>
            + RandomFill<Rng::Seed>,
        M::MatElement: Copy,
    {
        fn from(
            value: &SeededInteractiveMultiPartyServerKey<
                M,
                InteractiveMultiPartyCrs<Rng::Seed>,
                BoolParameters<M::MatElement>,
            >,
        ) -> Self {
            let g = value.parameters.g() as isize;
            let rlwe_n = value.parameters.rlwe_n().0;
            let lwe_n = value.parameters.lwe_n().0;
            let rlwe_q = value.parameters.rlwe_q();
            let lwe_q = value.parameters.lwe_q();

            let rlwe_nttop = N::new(rlwe_q, rlwe_n);

            // auto keys
            let mut auto_keys = HashMap::new();
            {
                let mut auto_prng = Rng::new_with_seed(value.cr_seed.auto_keys_cts_seed::<Rng>());
                let auto_d_count = value.parameters.auto_decomposition_count().0;
                let auto_element_dlogs = value.parameters.auto_element_dlogs();
                for i in auto_element_dlogs.into_iter() {
                    let mut key = M::zeros(auto_d_count * 2, rlwe_n);

                    // sample a
                    key.iter_rows_mut().take(auto_d_count).for_each(|ri| {
                        RandomFillUniformInModulus::random_fill(
                            &mut auto_prng,
                            &rlwe_q,
                            ri.as_mut(),
                        )
                    });

                    let key_part_b = value.auto_keys.get(&i).unwrap();
                    assert!(key_part_b.dimension() == (auto_d_count, rlwe_n));
                    izip!(
                        key.iter_rows_mut().skip(auto_d_count),
                        key_part_b.iter_rows()
                    )
                    .for_each(|(to_ri, from_ri)| {
                        to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                    });

                    // send to evaluation domain
                    key.iter_rows_mut()
                        .for_each(|ri| rlwe_nttop.forward(ri.as_mut()));

                    auto_keys.insert(i, key);
                }
            }

            // rgsw cts
            let (rlrg_d_a, rlrg_d_b) = value.parameters.rlwe_rgsw_decomposition_count();
            let rgsw_ct_out = rlrg_d_a.0 * 2 + rlrg_d_b.0 * 2;
            let rgsw_cts = value
                .rgsw_cts
                .iter()
                .map(|ct_i_in| {
                    assert!(ct_i_in.dimension() == (rgsw_ct_out, rlwe_n));
                    let mut eval_ct_i_out = M::zeros(rgsw_ct_out, rlwe_n);

                    izip!(eval_ct_i_out.iter_rows_mut(), ct_i_in.iter_rows()).for_each(
                        |(to_ri, from_ri)| {
                            to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                            rlwe_nttop.forward(to_ri.as_mut());
                        },
                    );

                    eval_ct_i_out
                })
                .collect_vec();

            // lwe ksk
            let mut lwe_ksk_prng = Rng::new_with_seed(value.cr_seed.lwe_ksk_cts_seed_seed::<Rng>());
            let d_lwe = value.parameters.lwe_decomposition_count().0;
            let mut lwe_ksk = M::zeros(rlwe_n * d_lwe, lwe_n + 1);
            izip!(lwe_ksk.iter_rows_mut(), value.lwe_ksk.as_ref().iter()).for_each(
                |(lwe_i, bi)| {
                    RandomFillUniformInModulus::random_fill(
                        &mut lwe_ksk_prng,
                        &lwe_q,
                        &mut lwe_i.as_mut()[1..],
                    );
                    lwe_i.as_mut()[0] = *bi;
                },
            );

            ServerKeyEvaluationDomain {
                rgsw_cts,
                galois_keys: auto_keys,
                lwe_ksk,
                parameters: value.parameters.clone(),
                _phanton: PhantomData,
            }
        }
    }

    impl<M: Matrix, P, R, N> PbsKey for ServerKeyEvaluationDomain<M, P, R, N> {
        type AutoKey = M;
        type LweKskKey = M;
        type RgswCt = M;

        fn galois_key_for_auto(&self, k: usize) -> &Self::AutoKey {
            self.galois_keys.get(&k).unwrap()
        }
        fn rgsw_ct_lwe_si(&self, si: usize) -> &Self::RgswCt {
            &self.rgsw_cts[si]
        }

        fn lwe_ksk(&self) -> &Self::LweKskKey {
            &self.lwe_ksk
        }
    }

    #[cfg(test)]
    impl<M, P, R, N> super::super::print_noise::CollectRuntimeServerKeyStats
        for ServerKeyEvaluationDomain<M, P, R, N>
    {
        type M = M;
        fn galois_key_for_auto(&self, k: usize) -> &Self::M {
            self.galois_keys.get(&k).unwrap()
        }
        fn lwe_ksk(&self) -> &Self::M {
            &self.lwe_ksk
        }
        fn rgsw_cts_lwe_si(&self, s_index: usize) -> &Self::M {
            &self.rgsw_cts[s_index]
        }
    }
}

/// Non-interactive multi-party server key in evaluation domain.
///
/// The key is derived from Seeded non-interactive mmulti-party server key
/// `SeededNonInteractiveMultiPartyServerKey`.
pub(crate) struct NonInteractiveServerKeyEvaluationDomain<M, P, R, N> {
    /// RGSW ciphertexts RGSW(X^{s[i]}) under ideal RLWE secret key in
    /// evaluation domain
    rgsw_cts: Vec<M>,
    /// Auto keys for all auto elements [-g, g, g^2, g^w] in evaluation
    /// domain
    auto_keys: HashMap<usize, M>,
    /// LWE key switching key to key switch LWE_{q, s}(m) to LWE_{q, z}(m)
    lwe_ksk: M,
    /// Key switching key from user j's secret u_j to ideal RLWE secret key `s`
    /// in evaluation domain. User j's key switching key is at j'th index.
    ui_to_s_ksks: Vec<M>,
    parameters: P,
    _phanton: PhantomData<(R, N)>,
}

pub(super) mod impl_non_interactive_server_key_eval_domain {
    use itertools::{izip, Itertools};

    use crate::{bool::evaluator::NonInteractiveMultiPartyCrs, random::RandomFill, Ntt, NttInit};

    use super::*;

    impl<M, P, R, N> NonInteractiveServerKeyEvaluationDomain<M, P, R, N> {
        pub(in super::super) fn rgsw_cts(&self) -> &[M] {
            &self.rgsw_cts
        }
    }

    impl<M, Rng, N>
        From<
            &SeededNonInteractiveMultiPartyServerKey<
                M,
                NonInteractiveMultiPartyCrs<Rng::Seed>,
                BoolParameters<M::MatElement>,
            >,
        > for NonInteractiveServerKeyEvaluationDomain<M, BoolParameters<M::MatElement>, Rng, N>
    where
        M: MatrixMut + MatrixEntity + Clone,
        Rng: NewWithSeed
            + RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>>
            + RandomFill<<Rng as NewWithSeed>::Seed>,
        N: Ntt<Element = M::MatElement> + NttInit<CiphertextModulus<M::MatElement>>,
        M::R: RowMut,
        M::MatElement: Copy,
        Rng::Seed: Clone + Copy + Default,
    {
        fn from(
            value: &SeededNonInteractiveMultiPartyServerKey<
                M,
                NonInteractiveMultiPartyCrs<Rng::Seed>,
                BoolParameters<M::MatElement>,
            >,
        ) -> Self {
            let rlwe_nttop = N::new(value.parameters.rlwe_q(), value.parameters.rlwe_n().0);
            let ring_size = value.parameters.rlwe_n().0;

            // RGSW cts
            // copy over rgsw cts and send to evaluation domain
            let mut rgsw_cts = value.rgsw_cts.clone();
            rgsw_cts.iter_mut().for_each(|c| {
                c.iter_rows_mut()
                    .for_each(|ri| rlwe_nttop.forward(ri.as_mut()))
            });

            // Auto keys
            // populate pseudo random part of auto keys. Then send auto keys to
            // evaluation domain
            let mut auto_keys = HashMap::new();
            let auto_seed = value.cr_seed.auto_keys_cts_seed::<Rng>();
            let mut auto_prng = Rng::new_with_seed(auto_seed);
            let auto_element_dlogs = value.parameters.auto_element_dlogs();
            let d_auto = value.parameters.auto_decomposition_count().0;
            auto_element_dlogs.iter().for_each(|el| {
                let auto_part_b = value
                    .auto_keys
                    .get(el)
                    .expect(&format!("Auto key for element g^{el} not found"));

                assert!(auto_part_b.dimension() == (d_auto, ring_size));

                let mut auto_ct = M::zeros(d_auto * 2, ring_size);

                // sample part A
                auto_ct.iter_rows_mut().take(d_auto).for_each(|ri| {
                    RandomFillUniformInModulus::random_fill(
                        &mut auto_prng,
                        value.parameters.rlwe_q(),
                        ri.as_mut(),
                    )
                });

                // Copy over part B
                izip!(
                    auto_ct.iter_rows_mut().skip(d_auto),
                    auto_part_b.iter_rows()
                )
                .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

                // send to evaluation domain
                auto_ct
                    .iter_rows_mut()
                    .for_each(|r| rlwe_nttop.forward(r.as_mut()));

                auto_keys.insert(*el, auto_ct);
            });

            // LWE ksk
            // populate pseudo random part of lwe ciphertexts in ksk and copy over part b
            // elements
            let lwe_ksk_seed = value.cr_seed.lwe_ksk_cts_seed::<Rng>();
            let mut lwe_ksk_prng = Rng::new_with_seed(lwe_ksk_seed);
            let mut lwe_ksk = M::zeros(
                value.parameters.lwe_decomposition_count().0 * ring_size,
                value.parameters.lwe_n().0 + 1,
            );
            lwe_ksk.iter_rows_mut().for_each(|ri| {
                // first element is resereved for part b. Only sample a_is in the rest
                RandomFillUniformInModulus::random_fill(
                    &mut lwe_ksk_prng,
                    value.parameters.lwe_q(),
                    &mut ri.as_mut()[1..],
                )
            });
            // copy over part bs
            assert!(
                value.lwe_ksk.as_ref().len()
                    == value.parameters.lwe_decomposition_count().0 * ring_size
            );
            izip!(value.lwe_ksk.as_ref().iter(), lwe_ksk.iter_rows_mut()).for_each(
                |(b_el, lwe_ct)| {
                    lwe_ct.as_mut()[0] = *b_el;
                },
            );

            // u_i to s ksk
            let d_uitos = value
                .parameters
                .non_interactive_ui_to_s_key_switch_decomposition_count()
                .0;
            let ui_to_s_ksks = value
                .ui_to_s_ksks
                .iter()
                .enumerate()
                .map(|(user_id, incoming_ksk_partb)| {
                    let user_i_seed = value.cr_seed.ui_to_s_ks_seed_for_user_i::<Rng>(user_id);
                    let mut prng = Rng::new_with_seed(user_i_seed);

                    let mut ksk_ct = M::zeros(d_uitos * 2, ring_size);

                    ksk_ct.iter_rows_mut().take(d_uitos).for_each(|r| {
                        RandomFillUniformInModulus::random_fill(
                            &mut prng,
                            value.parameters.rlwe_q(),
                            r.as_mut(),
                        );
                    });

                    assert!(incoming_ksk_partb.dimension() == (d_uitos, ring_size));
                    izip!(
                        ksk_ct.iter_rows_mut().skip(d_uitos),
                        incoming_ksk_partb.iter_rows()
                    )
                    .for_each(|(to_ri, from_ri)| {
                        to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                    });

                    ksk_ct
                        .iter_rows_mut()
                        .for_each(|r| rlwe_nttop.forward(r.as_mut()));
                    ksk_ct
                })
                .collect_vec();

            NonInteractiveServerKeyEvaluationDomain {
                rgsw_cts,
                auto_keys,
                lwe_ksk,
                ui_to_s_ksks,
                parameters: value.parameters.clone(),
                _phanton: PhantomData,
            }
        }
    }

    #[cfg(test)]
    impl<M, P, R, N> super::super::print_noise::CollectRuntimeServerKeyStats
        for NonInteractiveServerKeyEvaluationDomain<M, P, R, N>
    {
        type M = M;
        fn galois_key_for_auto(&self, k: usize) -> &Self::M {
            self.auto_keys.get(&k).unwrap()
        }
        fn lwe_ksk(&self) -> &Self::M {
            &self.lwe_ksk
        }
        fn rgsw_cts_lwe_si(&self, s_index: usize) -> &Self::M {
            &self.rgsw_cts[s_index]
        }
    }
}

/// Seeded non-interactive multi-party server key.
///
/// Given common reference seeded non-interactive multi-party key shares of each
/// users with unique user-ids, seeded non-interactive can be generated using
/// `BoolEvaluator::aggregate_non_interactive_multi_party_key_share`
pub struct SeededNonInteractiveMultiPartyServerKey<M: Matrix, S, P> {
    /// Key switching key from user j's secret u_j to ideal RLWE secret key `s`.
    /// User j's key switching key is at j'th index.
    ui_to_s_ksks: Vec<M>,
    /// RGSW ciphertexts RGSW(X^{s[i]}) under ideal RLWE secret key
    rgsw_cts: Vec<M>,
    /// Auto keys for all auto elements [-g, g, g^2, g^w]
    auto_keys: HashMap<usize, M>,
    /// LWE key switching key to key switch LWE_{q, s}(m) to LWE_{q, z}(m)
    lwe_ksk: M::R,
    /// Common reference seed
    cr_seed: S,
    parameters: P,
}

impl<M: Matrix, S, P> SeededNonInteractiveMultiPartyServerKey<M, S, P> {
    pub(super) fn new(
        ui_to_s_ksks: Vec<M>,
        rgsw_cts: Vec<M>,
        auto_keys: HashMap<usize, M>,
        lwe_ksk: M::R,
        cr_seed: S,
        parameters: P,
    ) -> Self {
        Self {
            ui_to_s_ksks,

            rgsw_cts,
            auto_keys,
            lwe_ksk,
            cr_seed,
            parameters,
        }
    }
}

/// This key is equivalent to NonInteractiveServerKeyEvaluationDomain with the
/// addition that each polynomial in evaluation domain has a corresponding shoup
/// representation suitable for shoup multiplication.
pub(crate) struct ShoupNonInteractiveServerKeyEvaluationDomain<M> {
    rgsw_cts: Vec<NormalAndShoup<M>>,
    auto_keys: HashMap<usize, NormalAndShoup<M>>,
    lwe_ksk: M,
    ui_to_s_ksks: Vec<NormalAndShoup<M>>,
}

mod impl_shoup_non_interactive_server_key_eval_domain {
    use itertools::Itertools;
    use num_traits::{FromPrimitive, PrimInt, ToPrimitive};

    use super::*;
    use crate::{backend::Modulus, decomposer::NumInfo, pbs::PbsKey};

    impl<M> ShoupNonInteractiveServerKeyEvaluationDomain<M> {
        pub(in super::super) fn ui_to_s_ksk(&self, user_id: usize) -> &NormalAndShoup<M> {
            &self.ui_to_s_ksks[user_id]
        }
    }

    impl<M: Matrix + ToShoup<Modulus = M::MatElement>, R, N>
        From<NonInteractiveServerKeyEvaluationDomain<M, BoolParameters<M::MatElement>, R, N>>
        for ShoupNonInteractiveServerKeyEvaluationDomain<M>
    where
        M::MatElement: FromPrimitive + ToPrimitive + PrimInt + NumInfo,
    {
        fn from(
            value: NonInteractiveServerKeyEvaluationDomain<M, BoolParameters<M::MatElement>, R, N>,
        ) -> Self {
            let rlwe_q = value.parameters.rlwe_q().q().unwrap();

            let rgsw_dim = (
                value.parameters.rlwe_rgsw_decomposition_count().0 .0 * 2
                    + value.parameters.rlwe_rgsw_decomposition_count().1 .0 * 2,
                value.parameters.rlwe_n().0,
            );
            let rgsw_cts = value
                .rgsw_cts
                .into_iter()
                .map(|m| {
                    assert!(m.dimension() == rgsw_dim);
                    NormalAndShoup::new_with_modulus(m, rlwe_q)
                })
                .collect_vec();

            let auto_dim = (
                value.parameters.auto_decomposition_count().0 * 2,
                value.parameters.rlwe_n().0,
            );
            let mut auto_keys = HashMap::new();
            value.auto_keys.into_iter().for_each(|(k, v)| {
                assert!(v.dimension() == auto_dim);
                auto_keys.insert(k, NormalAndShoup::new_with_modulus(v, rlwe_q));
            });

            let ui_ks_dim = (
                value
                    .parameters
                    .non_interactive_ui_to_s_key_switch_decomposition_count()
                    .0
                    * 2,
                value.parameters.rlwe_n().0,
            );
            let ui_to_s_ksks = value
                .ui_to_s_ksks
                .into_iter()
                .map(|m| {
                    assert!(m.dimension() == ui_ks_dim);
                    NormalAndShoup::new_with_modulus(m, rlwe_q)
                })
                .collect_vec();

            assert!(
                value.lwe_ksk.dimension()
                    == (
                        value.parameters.rlwe_n().0 * value.parameters.lwe_decomposition_count().0,
                        value.parameters.lwe_n().0 + 1
                    )
            );

            Self {
                rgsw_cts,
                auto_keys,
                lwe_ksk: value.lwe_ksk,
                ui_to_s_ksks,
            }
        }
    }

    impl<M: Matrix> PbsKey for ShoupNonInteractiveServerKeyEvaluationDomain<M> {
        type AutoKey = NormalAndShoup<M>;
        type LweKskKey = M;
        type RgswCt = NormalAndShoup<M>;

        fn galois_key_for_auto(&self, k: usize) -> &Self::AutoKey {
            let d = self.auto_keys.get(&k).unwrap();
            d
        }
        fn rgsw_ct_lwe_si(&self, si: usize) -> &Self::RgswCt {
            &self.rgsw_cts[si]
        }

        fn lwe_ksk(&self) -> &Self::LweKskKey {
            &self.lwe_ksk
        }
    }
}

/// This is equivalent to ServerKeyEvaluationDomain with the addition that each
/// polynomial in evaluation domain has corresponding shoup representation
/// suitable for shoup multiplication.
pub(crate) struct ShoupServerKeyEvaluationDomain<M> {
    rgsw_cts: Vec<NormalAndShoup<M>>,
    galois_keys: HashMap<usize, NormalAndShoup<M>>,
    lwe_ksk: M,
}

mod shoup_server_key_eval_domain {
    use itertools::{izip, Itertools};
    use num_traits::{FromPrimitive, PrimInt};

    use crate::{backend::Modulus, decomposer::NumInfo, pbs::PbsKey};

    use super::*;

    impl<M: MatrixMut + MatrixEntity + ToShoup<Modulus = M::MatElement>, R, N>
        From<ServerKeyEvaluationDomain<M, BoolParameters<M::MatElement>, R, N>>
        for ShoupServerKeyEvaluationDomain<M>
    where
        <M as Matrix>::R: RowMut,
        M::MatElement: PrimInt + FromPrimitive + NumInfo,
    {
        fn from(value: ServerKeyEvaluationDomain<M, BoolParameters<M::MatElement>, R, N>) -> Self {
            let q = value.parameters.rlwe_q().q().unwrap();
            // Rgsw ciphertexts
            let rgsw_cts = value
                .rgsw_cts
                .into_iter()
                .map(|ct| NormalAndShoup::new_with_modulus(ct, q))
                .collect_vec();

            let mut auto_keys = HashMap::new();
            value.galois_keys.into_iter().for_each(|(index, key)| {
                auto_keys.insert(index, NormalAndShoup::new_with_modulus(key, q));
            });

            Self {
                rgsw_cts,
                galois_keys: auto_keys,
                lwe_ksk: value.lwe_ksk,
            }
        }
    }

    impl<M: Matrix> PbsKey for ShoupServerKeyEvaluationDomain<M> {
        type AutoKey = NormalAndShoup<M>;
        type LweKskKey = M;
        type RgswCt = NormalAndShoup<M>;

        fn galois_key_for_auto(&self, k: usize) -> &Self::AutoKey {
            self.galois_keys.get(&k).unwrap()
        }
        fn rgsw_ct_lwe_si(&self, si: usize) -> &Self::RgswCt {
            &self.rgsw_cts[si]
        }

        fn lwe_ksk(&self) -> &Self::LweKskKey {
            &self.lwe_ksk
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<M: Matrix, P, S> {
    /// Non-interactive RGSW ciphertexts for LWE secret indices for which user
    /// is the leader
    self_leader_ni_rgsw_cts: Vec<M>,
    /// Non-interactive RGSW ciphertexts for LWE secret indices for which user
    /// is not the leader
    not_self_leader_ni_rgsw_cts: Vec<M>,
    /// Zero encryptions for RGSW ciphertexts for all indices
    ni_rgsw_zero_encs: Vec<M>,

    /// Key switching key from u_j to s where u_j is user j's RLWE secret `u`
    /// and `s` is ideal RLWE secret. Note that in server key share the key
    /// switching key is encrypted under user j's RLWE secret `s_j`. It is
    /// then switched to ideal RLWE secret after adding zero encryptions
    /// generated using same `a_k`s from other users.
    ///
    /// That is the key share has the following key switching key:
    ///      (a_k*s_j + e + \beta u_j, a_k*s_j + e)
    ui_to_s_ksk: M,
    /// Zero encryptions to switch user l's key switching key u_l to s from
    /// user l's RLWE secret s_l to ideal RLWE secret `s`.
    ///
    /// If there are P total parties then zero encryption sets are generated for
    /// each party l \in [0, P) and l != j where j self's user_id.
    ///
    /// Zero encryption set for user `l` is stored at index l is l < j otherwise
    /// it is stored at index l - 1, where j is self's user_id
    ksk_zero_encs_for_others: Vec<M>,

    /// RLWE auto key shares for auto elements [-g, g, g^2, g^{w}] where `w`
    /// is the window size. Auto key share corresponding to auto element -g
    /// is stored at key 0 and key share corresponding to auto element g^{k} is
    /// stored at key `k`
    auto_keys_share: HashMap<usize, M>,
    /// LWE key switching key share to key switching LWE_{q, s}(m) to LWE_{q,
    /// z}(m)
    lwe_ksk_share: M::R,

    /// User's id.
    ///
    /// If there are P total parties, then user id must be inque and in range
    /// [0, P)
    user_id: usize,
    /// Total users participating in multi-party compute
    total_users: usize,
    /// LWE dimension
    lwe_n: usize,
    /// Common reference seed
    cr_seed: S,
    parameters: P,
}

mod impl_common_ref_non_interactive_multi_party_server_share {
    use crate::bool::evaluator::multi_party_user_id_lwe_segment;

    use super::*;

    impl<M: Matrix, P, S> CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<M, P, S> {
        pub(in super::super) fn new(
            self_leader_ni_rgsw_cts: Vec<M>,
            not_self_leader_ni_rgsw_cts: Vec<M>,
            ni_rgsw_zero_encs: Vec<M>,
            ui_to_s_ksk: M,
            ksk_zero_encs_for_others: Vec<M>,
            auto_keys_share: HashMap<usize, M>,
            lwe_ksk_share: M::R,
            user_index: usize,
            total_users: usize,
            lwe_n: usize,
            cr_seed: S,
            parameters: P,
        ) -> Self {
            Self {
                self_leader_ni_rgsw_cts,
                not_self_leader_ni_rgsw_cts,
                ni_rgsw_zero_encs,
                ui_to_s_ksk,
                ksk_zero_encs_for_others,
                auto_keys_share,
                lwe_ksk_share,
                user_id: user_index,
                total_users,
                lwe_n,
                cr_seed,
                parameters,
            }
        }

        pub(in super::super) fn ni_rgsw_cts_for_self_leader_lwe_index(
            &self,
            lwe_index: usize,
        ) -> &M {
            let self_segment =
                multi_party_user_id_lwe_segment(self.user_id, self.total_users, self.lwe_n);
            assert!(lwe_index >= self_segment.0 && lwe_index < self_segment.1);
            &self.self_leader_ni_rgsw_cts[lwe_index - self_segment.0]
        }

        pub(in super::super) fn ni_rgsw_cts_for_self_not_leader_lwe_index(
            &self,
            lwe_index: usize,
        ) -> &M {
            let self_segment =
                multi_party_user_id_lwe_segment(self.user_id, self.total_users, self.lwe_n);
            // Non-interactive RGSW cts when self is not leader are stored in
            // sorted-order. For ex, if self is the leader for indices (5, 6]
            // then self stores NI-RGSW cts for rest of indices like [0, 1, 2,
            // 3, 4, 6, 7, 8, 9]
            assert!(lwe_index < self.lwe_n);
            assert!(lwe_index < self_segment.0 || lwe_index >= self_segment.1);
            if lwe_index < self_segment.0 {
                &self.not_self_leader_ni_rgsw_cts[lwe_index]
            } else {
                &self.not_self_leader_ni_rgsw_cts[lwe_index - (self_segment.1 - self_segment.0)]
            }
        }

        pub(in super::super) fn ni_rgsw_zero_enc_for_lwe_index(&self, lwe_index: usize) -> &M {
            &self.ni_rgsw_zero_encs[lwe_index]
        }

        pub(in super::super) fn ui_to_s_ksk(&self) -> &M {
            &self.ui_to_s_ksk
        }

        pub(in super::super) fn user_index(&self) -> usize {
            self.user_id
        }

        pub(in super::super) fn auto_keys_share(&self) -> &HashMap<usize, M> {
            &self.auto_keys_share
        }

        pub(in super::super) fn lwe_ksk_share(&self) -> &M::R {
            &self.lwe_ksk_share
        }

        pub(in super::super) fn ui_to_s_ksk_zero_encs_for_user_i(&self, user_i: usize) -> &M {
            assert!(user_i != self.user_id);
            if user_i < self.user_id {
                &self.ksk_zero_encs_for_others[user_i]
            } else {
                &self.ksk_zero_encs_for_others[user_i - 1]
            }
        }

        pub(in super::super) fn cr_seed(&self) -> &S {
            &self.cr_seed
        }

        pub(in super::super) fn parameters(&self) -> &P {
            &self.parameters
        }
    }
}

/// Stores both normal and shoup representation of elements in the container
/// (for ex, a matrix).
///
/// To access normal representation borrow self as a `self.as_ref()`. To access
/// shoup representation call `self.shoup_repr()`
pub(crate) struct NormalAndShoup<M>(M, M);

impl<M: ToShoup> NormalAndShoup<M> {
    fn new_with_modulus(value: M, modulus: <M as ToShoup>::Modulus) -> Self {
        let value_shoup = M::to_shoup(&value, modulus);
        NormalAndShoup(value, value_shoup)
    }
}

impl<M> AsRef<M> for NormalAndShoup<M> {
    fn as_ref(&self) -> &M {
        &self.0
    }
}

impl<M> WithShoupRepr for NormalAndShoup<M> {
    type M = M;
    fn shoup_repr(&self) -> &Self::M {
        &self.1
    }
}

#[cfg(test)]
pub(crate) mod key_size {
    use num_traits::{FromPrimitive, PrimInt};

    use crate::{backend::Modulus, decomposer::NumInfo, SizeInBitsWithLogModulus};

    use super::*;

    /// Size of the Key in Bits
    pub(crate) trait KeySize {
        /// Returns size of the key in bits
        fn size(&self) -> usize;
    }

    impl<M: Matrix, El, S> KeySize
        for CommonReferenceSeededInteractiveMultiPartyServerKeyShare<M, BoolParameters<El>, S>
    where
        M: SizeInBitsWithLogModulus,
        M::R: SizeInBitsWithLogModulus,
        El: PrimInt + NumInfo + FromPrimitive,
    {
        fn size(&self) -> usize {
            let mut total = 0;

            let log_rlweq = self.parameters().rlwe_q().log_q();
            self.self_leader_rgsws
                .iter()
                .for_each(|v| total += v.size(log_rlweq));
            self.not_self_leader_rgsws
                .iter()
                .for_each(|v| total += v.size(log_rlweq));
            self.auto_keys
                .values()
                .for_each(|v| total += v.size(log_rlweq));

            let log_lweq = self.parameters().lwe_q().log_q();
            total += self.lwe_ksk.size(log_lweq);
            total
        }
    }

    impl<M: Matrix, El, S> KeySize
        for CommonReferenceSeededNonInteractiveMultiPartyServerKeyShare<M, BoolParameters<El>, S>
    where
        M: SizeInBitsWithLogModulus,
        M::R: SizeInBitsWithLogModulus,
        El: PrimInt + NumInfo + FromPrimitive,
    {
        fn size(&self) -> usize {
            let mut total = 0;

            let log_rlweq = self.parameters.rlwe_q().log_q();
            self.self_leader_ni_rgsw_cts
                .iter()
                .for_each(|v| total += v.size(log_rlweq));
            self.not_self_leader_ni_rgsw_cts
                .iter()
                .for_each(|v| total += v.size(log_rlweq));
            self.ni_rgsw_zero_encs
                .iter()
                .for_each(|v| total += v.size(log_rlweq));
            total += self.ui_to_s_ksk.size(log_rlweq);
            self.ksk_zero_encs_for_others
                .iter()
                .for_each(|v| total += v.size(log_rlweq));
            self.auto_keys_share
                .values()
                .for_each(|v| total += v.size(log_rlweq));

            let log_lweq = self.parameters.lwe_q().log_q();
            total += self.lwe_ksk_share.size(log_lweq);

            total
        }
    }
}

pub(super) mod tests {
    use itertools::izip;
    use num_traits::{FromPrimitive, PrimInt, Zero};

    use crate::{
        backend::GetModulus, bool::ClientKey, decomposer::NumInfo, lwe::decrypt_lwe,
        parameters::CiphertextModulus, utils::TryConvertFrom1, ArithmeticOps, Row,
    };

    use super::SinglePartyClientKey;

    pub(crate) fn ideal_sk_rlwe(cks: &[ClientKey]) -> Vec<i32> {
        let mut ideal_rlwe_sk = cks[0].sk_rlwe();
        cks.iter().skip(1).for_each(|k| {
            let sk_rlwe = k.sk_rlwe();
            izip!(ideal_rlwe_sk.iter_mut(), sk_rlwe.iter()).for_each(|(a, b)| {
                *a = *a + b;
            });
        });
        ideal_rlwe_sk
    }

    pub(crate) fn ideal_sk_lwe(cks: &[ClientKey]) -> Vec<i32> {
        let mut ideal_rlwe_sk = cks[0].sk_lwe();
        cks.iter().skip(1).for_each(|k| {
            let sk_rlwe = k.sk_lwe();
            izip!(ideal_rlwe_sk.iter_mut(), sk_rlwe.iter()).for_each(|(a, b)| {
                *a = *a + b;
            });
        });
        ideal_rlwe_sk
    }

    pub(crate) fn measure_noise_lwe<
        R: Row,
        S,
        Modop: ArithmeticOps<Element = R::Element>
            + GetModulus<M = CiphertextModulus<R::Element>, Element = R::Element>,
    >(
        lwe_ct: &R,
        m_expected: R::Element,
        sk: &[S],
        modop: &Modop,
    ) -> R::Element
    where
        R: TryConvertFrom1<[S], CiphertextModulus<R::Element>>,
        R::Element: Zero + FromPrimitive + PrimInt + NumInfo,
    {
        let noisy_m = decrypt_lwe(lwe_ct, &sk, modop);
        let noise = modop.sub(&m_expected, &noisy_m);
        noise
    }
    // #[test]
    // fn trial() {
    //     let parameters = I_2P;
    //     let ck = ClientKey::new(parameters);
    //     let lwe = ck.sk_lwe();
    //     dbg!(lwe);
    // }
}
