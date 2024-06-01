use std::{collections::HashMap, hash::Hash, marker::PhantomData};

use crate::{
    backend::{ModInit, VectorOps},
    lwe::LweSecret,
    random::{NewWithSeed, RandomFillUniformInModulus},
    rgsw::RlweSecret,
    utils::WithLocal,
    Decryptor, Encryptor, Matrix, MatrixEntity, MatrixMut, MultiPartyDecryptor, RowEntity, RowMut,
};

use super::{parameters, BoolEvaluator, BoolParameters, CiphertextModulus};

/// Client key with RLWE and LWE secrets
#[derive(Clone)]
pub struct ClientKey {
    sk_rlwe: RlweSecret,
    sk_lwe: LweSecret,
}

mod impl_ck {
    use super::*;

    // Client key
    impl ClientKey {
        pub(in super::super) fn random() -> Self {
            let sk_rlwe = RlweSecret::random(0, 0);
            let sk_lwe = LweSecret::random(0, 0);
            Self { sk_rlwe, sk_lwe }
        }

        pub(in super::super) fn new(sk_rlwe: RlweSecret, sk_lwe: LweSecret) -> Self {
            Self { sk_rlwe, sk_lwe }
        }

        pub(in super::super) fn sk_rlwe(&self) -> &RlweSecret {
            &self.sk_rlwe
        }

        pub(in super::super) fn sk_lwe(&self) -> &LweSecret {
            &self.sk_lwe
        }
    }

    impl Encryptor<bool, Vec<u64>> for ClientKey {
        fn encrypt(&self, m: &bool) -> Vec<u64> {
            BoolEvaluator::with_local(|e| e.sk_encrypt(*m, self))
        }
    }

    impl Decryptor<bool, Vec<u64>> for ClientKey {
        fn decrypt(&self, c: &Vec<u64>) -> bool {
            BoolEvaluator::with_local(|e| e.sk_decrypt(c, self))
        }
    }

    impl MultiPartyDecryptor<bool, Vec<u64>> for ClientKey {
        type DecryptionShare = u64;

        fn gen_decryption_share(&self, c: &Vec<u64>) -> Self::DecryptionShare {
            BoolEvaluator::with_local(|e| e.multi_party_decryption_share(c, &self))
        }

        fn aggregate_decryption_shares(
            &self,
            c: &Vec<u64>,
            shares: &[Self::DecryptionShare],
        ) -> bool {
            BoolEvaluator::with_local(|e| e.multi_party_decrypt(shares, c))
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

    impl<Rng, ModOp> Encryptor<bool, Vec<u64>> for PublicKey<Vec<Vec<u64>>, Rng, ModOp> {
        fn encrypt(&self, m: &bool) -> Vec<u64> {
            BoolEvaluator::with_local(|e| e.pk_encrypt(&self.key, *m))
        }
    }

    impl<Rng, ModOp> Encryptor<[bool], Vec<Vec<u64>>> for PublicKey<Vec<Vec<u64>>, Rng, ModOp> {
        fn encrypt(&self, m: &[bool]) -> Vec<Vec<u64>> {
            BoolEvaluator::with_local(|e| e.pk_encrypt_batched(&self.key, m))
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
pub struct CommonReferenceSeededCollectivePublicKeyShare<Ro, S, P> {
    share: Ro,
    cr_seed: S,
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

/// CRS seeded Multi-party server key share
pub struct CommonReferenceSeededMultiPartyServerKeyShare<M: Matrix, P, S> {
    rgsw_cts: Vec<M>,
    /// Auto keys. Key corresponding to g^{k} is at index `k`. Key corresponding
    /// to -g is at 0
    auto_keys: HashMap<usize, M>,
    lwe_ksk: M::R,
    /// Common reference seed
    cr_seed: S,
    parameters: P,
}

impl<M: Matrix, P, S> CommonReferenceSeededMultiPartyServerKeyShare<M, P, S> {
    pub(super) fn new(
        rgsw_cts: Vec<M>,
        auto_keys: HashMap<usize, M>,
        lwe_ksk: M::R,
        cr_seed: S,
        parameters: P,
    ) -> Self {
        CommonReferenceSeededMultiPartyServerKeyShare {
            rgsw_cts,
            auto_keys,
            lwe_ksk,
            cr_seed,
            parameters,
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

    pub(super) fn rgsw_cts(&self) -> &[M] {
        &self.rgsw_cts
    }

    pub(super) fn lwe_ksk(&self) -> &M::R {
        &self.lwe_ksk
    }
}

/// CRS seeded MultiParty server key
pub struct SeededMultiPartyServerKey<M: Matrix, S, P> {
    rgsw_cts: Vec<M>,
    /// Auto keys. Key corresponding to g^{k} is at index `k`. Key corresponding
    /// to -g is at 0
    auto_keys: HashMap<usize, M>,
    lwe_ksk: M::R,
    cr_seed: S,
    parameters: P,
}

impl<M: Matrix, S, P> SeededMultiPartyServerKey<M, S, P> {
    pub(super) fn new(
        rgsw_cts: Vec<M>,
        auto_keys: HashMap<usize, M>,
        lwe_ksk: M::R,
        cr_seed: S,
        parameters: P,
    ) -> Self {
        SeededMultiPartyServerKey {
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
pub struct SeededServerKey<M: Matrix, P, S> {
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
impl<M: Matrix, S> SeededServerKey<M, BoolParameters<M::MatElement>, S> {
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

        SeededServerKey {
            rgsw_cts,
            auto_keys,
            lwe_ksk,
            parameters,
            seed,
        }
    }
}

/// Server key in evaluation domain
pub(crate) struct ServerKeyEvaluationDomain<M, R, N> {
    /// Rgsw cts of LWE secret elements
    rgsw_cts: Vec<M>,
    /// Auto keys. Key corresponding to g^{k} is at index `k`. Key corresponding
    /// to -g is at 0
    galois_keys: HashMap<usize, M>,
    /// LWE ksk to key switching LWE ciphertext from RLWE secret to LWE secret
    lwe_ksk: M,
    _phanton: PhantomData<(R, N)>,
}

pub(super) mod impl_server_key_eval_domain {
    use itertools::{izip, Itertools};

    use crate::{
        ntt::{Ntt, NttInit},
        pbs::PbsKey,
    };

    use super::*;

    impl<M, R, N> ServerKeyEvaluationDomain<M, R, N> {
        pub(in super::super) fn rgsw_cts(&self) -> &[M] {
            &self.rgsw_cts
        }
    }

    impl<
            M: MatrixMut + MatrixEntity,
            R: RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>>
                + NewWithSeed,
            N: NttInit<CiphertextModulus<M::MatElement>> + Ntt<Element = M::MatElement>,
        > From<&SeededServerKey<M, BoolParameters<M::MatElement>, R::Seed>>
        for ServerKeyEvaluationDomain<M, R, N>
    where
        <M as Matrix>::R: RowMut,
        M::MatElement: Copy,
        R::Seed: Clone,
    {
        fn from(value: &SeededServerKey<M, BoolParameters<M::MatElement>, R::Seed>) -> Self {
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
                _phanton: PhantomData,
            }
        }
    }

    impl<
            M: MatrixMut + MatrixEntity,
            Rng: NewWithSeed,
            N: NttInit<CiphertextModulus<M::MatElement>> + Ntt<Element = M::MatElement>,
        > From<&SeededMultiPartyServerKey<M, Rng::Seed, BoolParameters<M::MatElement>>>
        for ServerKeyEvaluationDomain<M, Rng, N>
    where
        <M as Matrix>::R: RowMut,
        Rng::Seed: Copy,
        Rng: RandomFillUniformInModulus<[M::MatElement], CiphertextModulus<M::MatElement>>,
        M::MatElement: Copy,
    {
        fn from(
            value: &SeededMultiPartyServerKey<M, Rng::Seed, BoolParameters<M::MatElement>>,
        ) -> Self {
            let g = value.parameters.g() as isize;
            let rlwe_n = value.parameters.rlwe_n().0;
            let lwe_n = value.parameters.lwe_n().0;
            let rlwe_q = value.parameters.rlwe_q();
            let lwe_q = value.parameters.lwe_q();

            let mut main_prng = Rng::new_with_seed(value.cr_seed);

            let rlwe_nttop = N::new(rlwe_q, rlwe_n);

            // auto keys
            let mut auto_keys = HashMap::new();
            let auto_d_count = value.parameters.auto_decomposition_count().0;
            let auto_element_dlogs = value.parameters.auto_element_dlogs();
            for i in auto_element_dlogs.into_iter() {
                let mut key = M::zeros(auto_d_count * 2, rlwe_n);

                // sample a
                key.iter_rows_mut().take(auto_d_count).for_each(|ri| {
                    RandomFillUniformInModulus::random_fill(&mut main_prng, &rlwe_q, ri.as_mut())
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
            let d_lwe = value.parameters.lwe_decomposition_count().0;
            let mut lwe_ksk = M::zeros(rlwe_n * d_lwe, lwe_n + 1);
            izip!(lwe_ksk.iter_rows_mut(), value.lwe_ksk.as_ref().iter()).for_each(
                |(lwe_i, bi)| {
                    RandomFillUniformInModulus::random_fill(
                        &mut main_prng,
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
                _phanton: PhantomData,
            }
        }
    }

    impl<M: Matrix, R, N> PbsKey for ServerKeyEvaluationDomain<M, R, N> {
        type M = M;
        fn galois_key_for_auto(&self, k: usize) -> &Self::M {
            self.galois_keys.get(&k).unwrap()
        }
        fn rgsw_ct_lwe_si(&self, si: usize) -> &Self::M {
            &self.rgsw_cts[si]
        }

        fn lwe_ksk(&self) -> &Self::M {
            &self.lwe_ksk
        }
    }
}
