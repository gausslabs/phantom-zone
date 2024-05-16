use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{Debug, Display},
    marker::PhantomData,
};

use itertools::{izip, partition, Itertools};
use num_traits::{FromPrimitive, Num, One, PrimInt, ToPrimitive, WrappingSub, Zero};

use crate::{
    backend::{ArithmeticOps, ModInit, ModularOpsU64, VectorOps},
    decomposer::{Decomposer, DefaultDecomposer, NumInfo},
    lwe::{decrypt_lwe, encrypt_lwe, lwe_key_switch, lwe_ksk_keygen, measure_noise_lwe, LweSecret},
    multi_party::public_key_share,
    ntt::{self, Ntt, NttBackendU64, NttInit},
    random::{DefaultSecureRng, NewWithSeed, RandomGaussianDist, RandomUniformDist},
    rgsw::{
        decrypt_rlwe, galois_auto, galois_key_gen, generate_auto_map, public_key_encrypt_rgsw,
        rgsw_by_rgsw_inplace, rlwe_by_rgsw, secret_key_encrypt_rgsw, IsTrivial, RlweCiphertext,
        RlweSecret,
    },
    utils::{
        fill_random_ternary_secret_with_hamming_weight, generate_prime, mod_exponent,
        TryConvertFrom, WithLocal,
    },
    Matrix, MatrixEntity, MatrixMut, Row, RowEntity, RowMut, Secret,
};

use super::parameters::{self, BoolParameters};

// thread_local! {
//     pub(crate) static CLIENT_KEY: RefCell<ClientKey> =
// RefCell::new(ClientKey::random()); }

trait PbsKey {
    type M: Matrix;

    /// RGSW ciphertext of LWE secret elements
    fn rgsw_ct_lwe_si(&self, si: usize) -> &Self::M;
    /// Key for automorphism
    fn galois_key_for_auto(&self, k: isize) -> &Self::M;
    /// LWE ksk to key switch from RLWE secret to LWE secret
    fn lwe_ksk(&self) -> &Self::M;
}

trait PbsParameters {
    type Element;
    type D: Decomposer<Element = Self::Element>;
    fn rlwe_q(&self) -> Self::Element;
    fn lwe_q(&self) -> Self::Element;
    fn br_q(&self) -> usize;
    fn d_rgsw(&self) -> usize;
    fn d_lwe(&self) -> usize;
    fn rlwe_n(&self) -> usize;
    fn lwe_n(&self) -> usize;
    /// Embedding fator for ring X^{q}+1 inside
    fn embedding_factor(&self) -> usize;
    /// generator g
    fn g(&self) -> isize;
    fn decomoposer_lwe(&self) -> &Self::D;
    fn decomoposer_rlwe(&self) -> &Self::D;

    /// Maps a \in Z^*_{q} to discrete log k, with generator g (i.e. g^k =
    /// a). Returned vector is of size q that stores dlog of a at `vec[a]`.
    /// For any a, if k is s.t. a = g^{k}, then k is expressed as k. If k is s.t
    /// a = -g^{k}, then k is expressed as k=k+q/2
    fn g_k_dlog_map(&self) -> &[usize];
    fn rlwe_auto_map(&self, k: isize) -> &(Vec<usize>, Vec<bool>);
}

#[derive(Clone)]
struct ClientKey {
    sk_rlwe: RlweSecret,
    sk_lwe: LweSecret,
}

impl ClientKey {
    fn random() -> Self {
        let sk_rlwe = RlweSecret::random(0, 0);
        let sk_lwe = LweSecret::random(0, 0);
        Self { sk_rlwe, sk_lwe }
    }
}

// impl WithLocal for ClientKey {
//     fn with_local<F, R>(func: F) -> R
//     where
//         F: Fn(&Self) -> R,
//     {
//         CLIENT_KEY.with_borrow(|client_key| func(client_key))
//     }

//     fn with_local_mut<F, R>(func: F) -> R
//     where
//         F: Fn(&mut Self) -> R,
//     {
//         CLIENT_KEY.with_borrow_mut(|client_key| func(client_key))
//     }
// }

// fn set_client_key(key: &ClientKey) {
//     ClientKey::with_local_mut(|k| *k = key.clone())
// }

struct MultiPartyDecryptionShare<E> {
    share: E,
}

struct CommonReferenceSeededCollectivePublicKeyShare<R, S, P> {
    share: R,
    cr_seed: S,
    parameters: P,
}

struct PublicKey<M, R, O> {
    key: M,
    _phantom: PhantomData<(R, O)>,
}

impl<
        M: MatrixMut + MatrixEntity,
        Rng: NewWithSeed + RandomUniformDist<[M::MatElement], Parameters = M::MatElement>,
        ModOp: VectorOps<Element = M::MatElement> + ModInit<Element = M::MatElement>,
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
        let mut key = M::zeros(2, parameters.rlwe_n);

        // sample A
        let seed = value[0].cr_seed;
        let mut main_rng = Rng::new_with_seed(seed);
        RandomUniformDist::random_fill(&mut main_rng, &parameters.rlwe_q, key.get_row_mut(0));

        // Sum all Bs
        let rlweq_modop = ModOp::new(parameters.rlwe_q);
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

struct CommonReferenceSeededMultiPartyServerKeyShare<M: Matrix, P, S> {
    rgsw_cts: Vec<M>,
    auto_keys: HashMap<isize, M>,
    lwe_ksk: M::R,
    /// Common reference seed
    cr_seed: S,
    parameters: P,
}
struct SeededMultiPartyServerKey<M: Matrix, S, P> {
    rgsw_cts: Vec<M>,
    auto_keys: HashMap<isize, M>,
    lwe_ksk: M::R,
    cr_seed: S,
    parameters: P,
}

fn aggregate_multi_party_server_key_shares<
    M: MatrixMut + MatrixEntity,
    S: Copy + PartialEq,
    D: Decomposer<Element = M::MatElement>,
    ModOp: VectorOps<Element = M::MatElement> + ModInit<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement> + NttInit<Element = M::MatElement>,
>(
    shares: &[CommonReferenceSeededMultiPartyServerKeyShare<M, BoolParameters<M::MatElement>, S>],
    d_rgsw_decomposer: &D,
) -> SeededMultiPartyServerKey<M, S, BoolParameters<M::MatElement>>
where
    <M as Matrix>::R: RowMut + RowEntity,
    M::MatElement: Copy + PartialEq + Zero,
    M: Clone,
{
    assert!(shares.len() > 0);
    let parameters = shares[0].parameters.clone();
    let cr_seed = shares[0].cr_seed;

    let rlwe_n = parameters.rlwe_n;
    let g = parameters.g as isize;
    let d_rgsw = parameters.d_rgsw;
    let d_lwe = parameters.d_lwe;
    let rlwe_q = parameters.rlwe_q;
    let lwe_q = parameters.lwe_q;

    // sanity checks
    shares.iter().skip(1).for_each(|s| {
        assert!(s.parameters == parameters);
        assert!(s.cr_seed == cr_seed);
    });

    let rlweq_modop = ModOp::new(rlwe_q);
    let rlweq_nttop = NttOp::new(rlwe_q, rlwe_n);

    // auto keys
    let mut auto_keys = HashMap::new();
    for i in [g, -g] {
        let mut key = M::zeros(d_rgsw, rlwe_n);

        shares.iter().for_each(|s| {
            let auto_key_share_i = s.auto_keys.get(&i).expect("Auto key {i} missing");
            assert!(auto_key_share_i.dimension() == (d_rgsw, rlwe_n));
            izip!(key.iter_rows_mut(), auto_key_share_i.iter_rows()).for_each(
                |(partb_out, partb_share)| {
                    rlweq_modop.elwise_add_mut(partb_out.as_mut(), partb_share.as_ref());
                },
            );
        });

        auto_keys.insert(i, key);
    }

    // rgsw ciphertext (most expensive part!)
    let lwe_n = parameters.lwe_n;
    let mut scratch_d_plus_rgsw_by_ring = M::zeros(d_rgsw + (d_rgsw * 4), rlwe_n);
    let mut tmp_rgsw = M::zeros(d_rgsw * 2 * 2, rlwe_n);
    let rgsw_cts = (0..lwe_n)
        .into_iter()
        .map(|index| {
            // copy over rgsw ciphertext for index^th secret element from first share and
            // treat it as accumulating rgsw ciphertext
            let mut rgsw_i = shares[0].rgsw_cts[index].clone();

            shares.iter().skip(1).for_each(|si| {
                // copy over si's RGSW[index] ciphertext and send to evaluation domain
                izip!(tmp_rgsw.iter_rows_mut(), si.rgsw_cts[index].iter_rows()).for_each(
                    |(to_ri, from_ri)| {
                        to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                        rlweq_nttop.forward(to_ri.as_mut())
                    },
                );

                rgsw_by_rgsw_inplace(
                    &mut rgsw_i,
                    &tmp_rgsw,
                    d_rgsw_decomposer,
                    &mut scratch_d_plus_rgsw_by_ring,
                    &rlweq_nttop,
                    &rlweq_modop,
                );
            });

            rgsw_i
        })
        .collect_vec();

    // LWE ksks
    let mut lwe_ksk = M::R::zeros(rlwe_n * d_lwe);
    let lweq_modop = ModOp::new(lwe_q);
    shares.iter().for_each(|si| {
        assert!(si.lwe_ksk.as_ref().len() == rlwe_n * d_lwe);
        lweq_modop.elwise_add_mut(lwe_ksk.as_mut(), si.lwe_ksk.as_ref())
    });

    SeededMultiPartyServerKey {
        rgsw_cts,
        auto_keys,
        lwe_ksk,
        cr_seed,
        parameters: parameters,
    }
}

struct SeededServerKey<M: Matrix, P, S> {
    /// Rgsw cts of LWE secret elements
    pub(crate) rgsw_cts: Vec<M>,
    /// Auto keys
    pub(crate) auto_keys: HashMap<isize, M>,
    /// LWE ksk to key switching LWE ciphertext from RLWE secret to LWE secret
    pub(crate) lwe_ksk: M::R,
    /// Parameters
    pub(crate) parameters: P,
    /// Main seed
    pub(crate) seed: S,
}

impl<M: Matrix, S> SeededServerKey<M, BoolParameters<M::MatElement>, S> {
    pub(crate) fn from_raw(
        auto_keys: HashMap<isize, M>,
        rgsw_cts: Vec<M>,
        lwe_ksk: M::R,
        parameters: BoolParameters<M::MatElement>,
        seed: S,
    ) -> Self {
        // sanity checks
        auto_keys
            .iter()
            .for_each(|v| assert!(v.1.dimension() == (parameters.d_rgsw, parameters.rlwe_n)));
        rgsw_cts
            .iter()
            .for_each(|v| assert!(v.dimension() == (parameters.d_rgsw * 3, parameters.rlwe_n)));
        assert!(lwe_ksk.as_ref().len() == (parameters.d_lwe * parameters.rlwe_n));

        SeededServerKey {
            rgsw_cts,
            auto_keys,
            lwe_ksk,
            parameters,
            seed,
        }
    }
}

struct ServerKeyEvaluationDomain<M, R, N> {
    /// Rgsw cts of LWE secret elements
    rgsw_cts: Vec<M>,
    /// Galois keys
    galois_keys: HashMap<isize, M>,
    /// LWE ksk to key switching LWE ciphertext from RLWE secret to LWE secret
    lwe_ksk: M,
    _phanton: PhantomData<(R, N)>,
}

impl<
        M: MatrixMut + MatrixEntity,
        R: RandomUniformDist<[M::MatElement], Parameters = M::MatElement> + NewWithSeed,
        N: NttInit<Element = M::MatElement> + Ntt<Element = M::MatElement>,
    > From<&SeededServerKey<M, BoolParameters<M::MatElement>, R::Seed>>
    for ServerKeyEvaluationDomain<M, R, N>
where
    <M as Matrix>::R: RowMut,
    M::MatElement: Copy,
    R::Seed: Clone,
{
    fn from(value: &SeededServerKey<M, BoolParameters<M::MatElement>, R::Seed>) -> Self {
        let mut main_prng = R::new_with_seed(value.seed.clone());

        let g = value.parameters.g as isize;
        let ring_size = value.parameters.rlwe_n;
        let lwe_n = value.parameters.lwe_n;
        let d_rgsw = value.parameters.d_rgsw;
        let d_lwe = value.parameters.d_lwe;
        let rlwe_q = value.parameters.rlwe_q;
        let lwq_q = value.parameters.lwe_q;

        let nttop = N::new(rlwe_q, ring_size);

        // galois keys
        let mut auto_keys = HashMap::new();
        for i in [g, -g] {
            let seeded_auto_key = value.auto_keys.get(&i).unwrap();
            assert!(seeded_auto_key.dimension() == (d_rgsw, ring_size));

            let mut data = M::zeros(d_rgsw * 2, ring_size);

            // sample RLWE'_A(-s(X^k))
            data.iter_rows_mut().take(d_rgsw).for_each(|ri| {
                RandomUniformDist::random_fill(&mut main_prng, &rlwe_q, ri.as_mut())
            });

            // copy over RLWE'B_(-s(X^k))
            izip!(
                data.iter_rows_mut().skip(d_rgsw),
                seeded_auto_key.iter_rows()
            )
            .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

            // Send to Evaluation domain
            data.iter_rows_mut()
                .for_each(|ri| nttop.forward(ri.as_mut()));

            auto_keys.insert(i, data);
        }

        // RGSW ciphertexts
        let rgsw_cts = value
            .rgsw_cts
            .iter()
            .map(|seeded_rgsw_si| {
                assert!(seeded_rgsw_si.dimension() == (3 * d_rgsw, ring_size));

                let mut data = M::zeros(d_rgsw * 4, ring_size);

                // copy over RLWE'(-sm)
                izip!(
                    data.iter_rows_mut().take(d_rgsw * 2),
                    seeded_rgsw_si.iter_rows().take(d_rgsw * 2)
                )
                .for_each(|(to_ri, from_ri)| to_ri.as_mut().copy_from_slice(from_ri.as_ref()));

                // sample RLWE'_A(m)
                data.iter_rows_mut()
                    .skip(2 * d_rgsw)
                    .take(d_rgsw)
                    .for_each(|ri| {
                        RandomUniformDist::random_fill(&mut main_prng, &rlwe_q, ri.as_mut())
                    });

                // copy over RLWE'_B(m)
                izip!(
                    data.iter_rows_mut().skip(d_rgsw * 3),
                    seeded_rgsw_si.iter_rows().skip(d_rgsw * 2)
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
            assert!(value.lwe_ksk.as_ref().len() == d_lwe * ring_size);

            let mut data = M::zeros(d_lwe * ring_size, lwe_n + 1);
            izip!(data.iter_rows_mut(), value.lwe_ksk.as_ref().iter()).for_each(|(lwe_i, bi)| {
                RandomUniformDist::random_fill(&mut main_prng, &lwq_q, &mut lwe_i.as_mut()[1..]);
                lwe_i.as_mut()[0] = *bi;
            });

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
        N: NttInit<Element = M::MatElement> + Ntt<Element = M::MatElement>,
    > From<&SeededMultiPartyServerKey<M, Rng::Seed, BoolParameters<M::MatElement>>>
    for ServerKeyEvaluationDomain<M, Rng, N>
where
    <M as Matrix>::R: RowMut,
    Rng::Seed: Copy,
    Rng: RandomUniformDist<[M::MatElement], Parameters = M::MatElement>,
    M::MatElement: Copy,
{
    fn from(
        value: &SeededMultiPartyServerKey<M, Rng::Seed, BoolParameters<M::MatElement>>,
    ) -> Self {
        let g = value.parameters.g as isize;
        let rlwe_n = value.parameters.rlwe_n;
        let lwe_n = value.parameters.lwe_n;
        let rlwe_q = value.parameters.rlwe_q;
        let lwe_q = value.parameters.lwe_q;
        let d_rgsw = value.parameters.d_rgsw;

        let mut main_prng = Rng::new_with_seed(value.cr_seed);

        let rlwe_nttop = N::new(rlwe_q, rlwe_n);

        // auto keys
        let mut auto_keys = HashMap::new();
        for i in [g, -g] {
            let mut key = M::zeros(value.parameters.d_rgsw * 2, rlwe_n);

            // sample a
            key.iter_rows_mut().take(d_rgsw).for_each(|ri| {
                RandomUniformDist::random_fill(&mut main_prng, &rlwe_q, ri.as_mut())
            });

            let key_part_b = value.auto_keys.get(&i).unwrap();
            assert!(key_part_b.dimension() == (d_rgsw, rlwe_n));
            izip!(key.iter_rows_mut().skip(d_rgsw), key_part_b.iter_rows()).for_each(
                |(to_ri, from_ri)| {
                    to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                },
            );

            // send to evaluation domain
            key.iter_rows_mut()
                .for_each(|ri| rlwe_nttop.forward(ri.as_mut()));

            auto_keys.insert(i, key);
        }

        // rgsw cts
        let rgsw_cts = value
            .rgsw_cts
            .iter()
            .map(|ct_i| {
                let mut eval_ct_i = M::zeros(d_rgsw * 4, rlwe_n);

                izip!(eval_ct_i.iter_rows_mut(), ct_i.iter_rows()).for_each(|(to_ri, from_ri)| {
                    to_ri.as_mut().copy_from_slice(from_ri.as_ref());
                    rlwe_nttop.forward(to_ri.as_mut());
                });

                eval_ct_i
            })
            .collect_vec();

        // lwe ksk
        let d_lwe = value.parameters.d_lwe;
        let mut lwe_ksk = M::zeros(rlwe_n * d_lwe, lwe_n + 1);
        izip!(lwe_ksk.iter_rows_mut(), value.lwe_ksk.as_ref().iter()).for_each(|(lwe_i, bi)| {
            RandomUniformDist::random_fill(&mut main_prng, &lwe_q, &mut lwe_i.as_mut()[1..]);
            lwe_i.as_mut()[0] = *bi;
        });

        ServerKeyEvaluationDomain {
            rgsw_cts,
            galois_keys: auto_keys,
            lwe_ksk,
            _phanton: PhantomData,
        }
    }
}

//FIXME(Jay): Figure out a way for BoolEvaluator to have access to ServerKey
// via a pointer and implement PbsKey for BoolEvaluator instead of ServerKey
// directly
impl<M: Matrix, R, N> PbsKey for ServerKeyEvaluationDomain<M, R, N> {
    type M = M;
    fn galois_key_for_auto(&self, k: isize) -> &Self::M {
        self.galois_keys.get(&k).unwrap()
    }
    fn rgsw_ct_lwe_si(&self, si: usize) -> &Self::M {
        &self.rgsw_cts[si]
    }

    fn lwe_ksk(&self) -> &Self::M {
        &self.lwe_ksk
    }
}

struct BoolEvaluator<M, E, Ntt, ModOp>
where
    M: Matrix,
{
    parameters: BoolParameters<E>,
    decomposer_rlwe: DefaultDecomposer<E>,
    decomposer_lwe: DefaultDecomposer<E>,
    g_k_dlog_map: Vec<usize>,
    rlwe_nttop: Ntt,
    rlwe_modop: ModOp,
    lwe_modop: ModOp,
    embedding_factor: usize,
    nand_test_vec: M::R,
    rlweq_by8: M::MatElement,
    rlwe_qby4: M::MatElement,
    rlwe_auto_maps: Vec<(Vec<usize>, Vec<bool>)>,
    _phantom: PhantomData<M>,
}

impl<M, NttOp, ModOp> BoolEvaluator<M, M::MatElement, NttOp, ModOp>
where
    NttOp: NttInit<Element = M::MatElement> + Ntt<Element = M::MatElement>,
    ModOp: ModInit<Element = M::MatElement>
        + ArithmeticOps<Element = M::MatElement>
        + VectorOps<Element = M::MatElement>,
    M::MatElement: PrimInt + Debug + Display + NumInfo + FromPrimitive + WrappingSub,
    M: MatrixEntity + MatrixMut,
    M::R: TryConvertFrom<[i32], Parameters = M::MatElement> + RowEntity + Debug,
    M: TryConvertFrom<[i32], Parameters = M::MatElement>,
    <M as Matrix>::R: RowMut,
    DefaultSecureRng: RandomGaussianDist<[M::MatElement], Parameters = M::MatElement>
        + RandomGaussianDist<M::MatElement, Parameters = M::MatElement>
        + RandomUniformDist<[M::MatElement], Parameters = M::MatElement>
        + NewWithSeed,
{
    fn new(parameters: BoolParameters<M::MatElement>) -> Self {
        //TODO(Jay): Run sanity checks for modulus values in parameters
        assert!(parameters.br_q.is_power_of_two());

        let decomposer_rlwe =
            DefaultDecomposer::new(parameters.rlwe_q, parameters.logb_rgsw, parameters.d_rgsw);
        let decomposer_lwe =
            DefaultDecomposer::new(parameters.lwe_q, parameters.logb_lwe, parameters.d_lwe);

        // generatr dlog map s.t. g^{k} % q = a, for all a \in Z*_{q}
        let g = parameters.g;
        let q = parameters.br_q;
        let mut g_k_dlog_map = vec![0usize; q];
        for i in 0..q / 2 {
            let v = mod_exponent(g as u64, i as u64, q as u64) as usize;
            // g^i
            g_k_dlog_map[v] = i;
            // -(g^i)
            g_k_dlog_map[q - v] = i + (q / 2);
        }

        let embedding_factor = (2 * parameters.rlwe_n) / q;

        let rlwe_nttop = NttOp::new(parameters.rlwe_q, parameters.rlwe_n);
        let rlwe_modop = ModInit::new(parameters.rlwe_q);
        let lwe_modop = ModInit::new(parameters.lwe_q);

        // set test vectors
        let el_one = M::MatElement::one();
        let nand_map = |index: usize, qby8: usize| {
            if index < (3 * qby8) {
                true
            } else {
                false
            }
        };

        let q = parameters.br_q;
        let qby2 = q >> 1;
        let qby8 = q >> 3;
        let mut nand_test_vec = M::R::zeros(qby2);
        // Q/8 (Q: rlwe_q)
        let rlwe_qby8 =
            M::MatElement::from_f64((parameters.rlwe_q.to_f64().unwrap() / 8.0).round()).unwrap();
        let true_m_el = rlwe_qby8;
        // -Q/8
        let false_m_el = parameters.rlwe_q - rlwe_qby8;
        for i in 0..qby2 {
            let v = nand_map(i, qby8);
            if v {
                nand_test_vec.as_mut()[i] = true_m_el;
            } else {
                nand_test_vec.as_mut()[i] = false_m_el;
            }
        }
        // // Rotate and negate by q/8
        // let mut tmp = M::R::zeros(qby2);
        // tmp.as_mut()[..qby2 - qby8].copy_from_slice(&nand_test_vec.as_ref()[qby8..]);
        // tmp.as_mut()[qby2 - qby8..].copy_from_slice(&nand_test_vec.as_ref()[..qby8]);
        // tmp.as_mut()[qby2 - qby8..].iter_mut().for_each(|v| {
        //     *v = parameters.rlwe_q - *v;
        // });
        // let nand_test_vec = tmp;

        // v(X) -> v(X^{-g})
        let (auto_map_index, auto_map_sign) = generate_auto_map(qby2, -(g as isize));
        let mut nand_test_vec_autog = M::R::zeros(qby2);
        izip!(
            nand_test_vec.as_ref().iter(),
            auto_map_index.iter(),
            auto_map_sign.iter()
        )
        .for_each(|(v, to_index, to_sign)| {
            if !to_sign {
                // negate
                nand_test_vec_autog.as_mut()[*to_index] = parameters.rlwe_q - *v;
            } else {
                nand_test_vec_autog.as_mut()[*to_index] = *v;
            }
        });

        // auto map indices and sign
        let mut rlwe_auto_maps = vec![];
        let ring_size = parameters.rlwe_n;
        let g = parameters.g as isize;
        for i in [g, -g] {
            rlwe_auto_maps.push(generate_auto_map(ring_size, i))
        }

        let rlwe_qby4 =
            M::MatElement::from_f64((parameters.rlwe_q.to_f64().unwrap() / 4.0).round()).unwrap();

        BoolEvaluator {
            parameters: parameters,
            decomposer_lwe,
            decomposer_rlwe,
            g_k_dlog_map,
            embedding_factor,
            lwe_modop,
            rlwe_modop,
            rlwe_nttop,
            nand_test_vec: nand_test_vec_autog,
            rlweq_by8: rlwe_qby8,
            rlwe_qby4: rlwe_qby4,
            rlwe_auto_maps,

            _phantom: PhantomData,
        }
    }

    fn client_key(&self) -> ClientKey {
        let sk_lwe = LweSecret::random(self.parameters.lwe_n >> 1, self.parameters.lwe_n);
        let sk_rlwe = RlweSecret::random(self.parameters.rlwe_n >> 1, self.parameters.rlwe_n);
        ClientKey { sk_rlwe, sk_lwe }
    }

    fn server_key(
        &self,
        client_key: &ClientKey,
    ) -> SeededServerKey<M, BoolParameters<M::MatElement>, [u8; 32]> {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut main_seed = [0u8; 32];
            rng.fill_bytes(&mut main_seed);

            let mut main_prng = DefaultSecureRng::new_seeded(main_seed);

            let sk_rlwe = &client_key.sk_rlwe;
            let sk_lwe = &client_key.sk_lwe;

            let d_rgsw_gadget_vec = self.decomposer_rlwe.gadget_vector();

            // generate auto keys -g, g
            let mut auto_keys = HashMap::new();
            let g = self.parameters.g as isize;
            for i in [g, -g] {
                let mut gk = M::zeros(self.parameters.d_rgsw, self.parameters.rlwe_n);
                galois_key_gen(
                    &mut gk,
                    sk_rlwe.values(),
                    i,
                    &d_rgsw_gadget_vec,
                    &self.rlwe_modop,
                    &self.rlwe_nttop,
                    &mut main_prng,
                    rng,
                );
                auto_keys.insert(i, gk);
            }

            // generate rgsw ciphertexts RGSW(si) where si is i^th LWE secret element
            let ring_size = self.parameters.rlwe_n;
            let rlwe_q = self.parameters.rlwe_q;
            let rgsw_cts = sk_lwe
                .values()
                .iter()
                .map(|si| {
                    // X^{si}; assume |emebedding_factor * si| < N
                    let mut m = M::R::zeros(ring_size);
                    let si = (self.embedding_factor as i32) * si;
                    // dbg!(si);
                    if si < 0 {
                        // X^{-i} = X^{2N - i} = -X^{N-i}
                        m.as_mut()[ring_size - (si.abs() as usize)] = rlwe_q - M::MatElement::one();
                    } else {
                        // X^{i}
                        m.as_mut()[si.abs() as usize] = M::MatElement::one();
                    }

                    let mut rgsw_si = M::zeros(self.parameters.d_rgsw * 3, ring_size);
                    secret_key_encrypt_rgsw(
                        &mut rgsw_si,
                        m.as_ref(),
                        &d_rgsw_gadget_vec,
                        sk_rlwe.values(),
                        &self.rlwe_modop,
                        &self.rlwe_nttop,
                        &mut main_prng,
                        rng,
                    );

                    rgsw_si
                })
                .collect_vec();

            // LWE KSK from RLWE secret s -> LWE secret z
            let d_lwe_gadget = self.decomposer_lwe.gadget_vector();

            let mut lwe_ksk = M::R::zeros(self.parameters.d_lwe * ring_size);
            lwe_ksk_keygen(
                &sk_rlwe.values(),
                &sk_lwe.values(),
                &mut lwe_ksk,
                &d_lwe_gadget,
                &self.lwe_modop,
                &mut main_prng,
                rng,
            );

            SeededServerKey::from_raw(
                auto_keys,
                rgsw_cts,
                lwe_ksk,
                self.parameters.clone(),
                main_seed,
            )
        })
    }

    fn multi_party_sever_key_share(
        &self,
        cr_seed: [u8; 32],
        collective_pk: &M,
        client_key: &ClientKey,
    ) -> CommonReferenceSeededMultiPartyServerKeyShare<M, BoolParameters<M::MatElement>, [u8; 32]>
    {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut main_prng = DefaultSecureRng::new_seeded(cr_seed);

            let sk_rlwe = &client_key.sk_rlwe;
            let sk_lwe = &client_key.sk_lwe;

            let g = self.parameters.g as isize;
            let ring_size = self.parameters.rlwe_n;
            let d_rgsw = self.parameters.d_rgsw;
            let d_lwe = self.parameters.d_lwe;
            let rlwe_q = self.parameters.rlwe_q;
            let lwe_q = self.parameters.lwe_q;

            let d_rgsw_gadget_vec = self.decomposer_rlwe.gadget_vector();

            let rlweq_modop = ModOp::new(rlwe_q);
            let rlweq_nttop = NttOp::new(rlwe_q, ring_size);

            // sanity check
            assert!(sk_rlwe.values().len() == ring_size);
            assert!(sk_lwe.values().len() == self.parameters.lwe_n);

            // auto keys
            let mut auto_keys = HashMap::new();
            for i in [g, -g] {
                let mut ksk_out = M::zeros(d_rgsw, ring_size);
                galois_key_gen(
                    &mut ksk_out,
                    sk_rlwe.values(),
                    i,
                    &d_rgsw_gadget_vec,
                    &rlweq_modop,
                    &rlweq_nttop,
                    &mut main_prng,
                    rng,
                );
                auto_keys.insert(i, ksk_out);
            }

            // rgsw ciphertexts of lwe secret elements
            let rgsw_cts = sk_lwe
                .values()
                .iter()
                .map(|si| {
                    let mut m = M::R::zeros(ring_size);
                    //TODO(Jay): It will be nice to have a function that returns polynomial
                    // (monomial infact!) corresponding to secret element embedded in ring X^{2N+1}.
                    // Save lots of mistakes where one forgest to emebed si in bigger ring.
                    let si = *si * (self.embedding_factor as i32);
                    if si < 0 {
                        // X^{-si} = X^{2N-si} = -X^{N-si}, assuming abs(si) < N
                        // (which it is given si is secret element)
                        m.as_mut()[ring_size - (si.abs() as usize)] = rlwe_q - M::MatElement::one();
                    } else {
                        m.as_mut()[si as usize] = M::MatElement::one();
                    }

                    // public key RGSW encryption has no part that can be seeded, unlike secret key
                    // RGSW encryption where RLWE'_A(m) is seeded
                    let mut out_rgsw = M::zeros(d_rgsw * 4, ring_size);
                    public_key_encrypt_rgsw(
                        &mut out_rgsw,
                        &m.as_ref(),
                        collective_pk,
                        &d_rgsw_gadget_vec,
                        &rlweq_modop,
                        &rlweq_nttop,
                        rng,
                    );

                    out_rgsw
                })
                .collect_vec();

            // LWE ksk
            let mut lwe_ksk = M::R::zeros(d_lwe * ring_size);
            let lwe_modop = ModOp::new(lwe_q);
            let d_lwe_gadget_vec = self.decomposer_lwe.gadget_vector();
            lwe_ksk_keygen(
                sk_rlwe.values(),
                sk_lwe.values(),
                &mut lwe_ksk,
                &d_lwe_gadget_vec,
                &lwe_modop,
                &mut main_prng,
                rng,
            );

            CommonReferenceSeededMultiPartyServerKeyShare {
                auto_keys,
                rgsw_cts,
                lwe_ksk,
                cr_seed,
                parameters: self.parameters.clone(),
            }
        })
    }

    fn multi_party_public_key_share(
        &self,
        cr_seed: [u8; 32],
        client_key: &ClientKey,
    ) -> CommonReferenceSeededCollectivePublicKeyShare<
        <M as Matrix>::R,
        [u8; 32],
        BoolParameters<<M as Matrix>::MatElement>,
    > {
        DefaultSecureRng::with_local_mut(|rng| {
            let mut share_out = M::R::zeros(self.parameters.rlwe_n);
            let modop = ModOp::new(self.parameters.rlwe_q);
            let nttop = NttOp::new(self.parameters.rlwe_q, self.parameters.rlwe_n);
            let mut main_prng = DefaultSecureRng::new_seeded(cr_seed);
            public_key_share(
                &mut share_out,
                client_key.sk_rlwe.values(),
                &modop,
                &nttop,
                &mut main_prng,
                rng,
            );

            CommonReferenceSeededCollectivePublicKeyShare {
                share: share_out,
                cr_seed: cr_seed,
                parameters: self.parameters.clone(),
            }
        })
    }

    fn multi_party_decryption_share(
        &self,
        lwe_ct: &M::R,
        client_key: &ClientKey,
    ) -> MultiPartyDecryptionShare<<M as Matrix>::MatElement> {
        assert!(lwe_ct.as_ref().len() == self.parameters.rlwe_n + 1);
        let modop = &self.rlwe_modop;
        let mut neg_s =
            M::R::try_convert_from(client_key.sk_rlwe.values(), &self.parameters.rlwe_q);
        modop.elwise_neg_mut(neg_s.as_mut());

        let mut neg_sa = M::MatElement::zero();
        izip!(lwe_ct.as_ref().iter().skip(1), neg_s.as_ref().iter()).for_each(|(ai, nsi)| {
            neg_sa = modop.add(&neg_sa, &modop.mul(ai, nsi));
        });

        let e = DefaultSecureRng::with_local_mut(|rng| {
            let mut e = M::MatElement::zero();
            RandomGaussianDist::random_fill(rng, &self.parameters.rlwe_q, &mut e);
            e
        });
        let share = modop.add(&neg_sa, &e);

        MultiPartyDecryptionShare { share }
    }

    pub(crate) fn multi_party_decrypt(
        &self,
        shares: &[MultiPartyDecryptionShare<M::MatElement>],
        lwe_ct: &M::R,
    ) -> bool {
        let modop = &self.rlwe_modop;
        let mut sum_a = M::MatElement::zero();
        shares
            .iter()
            .for_each(|share_i| sum_a = modop.add(&sum_a, &share_i.share));

        let encoded_m = modop.add(&lwe_ct.as_ref()[0], &sum_a);

        let m = (((encoded_m + self.rlweq_by8).to_f64().unwrap() * 4f64)
            / self.parameters.rlwe_q.to_f64().unwrap())
        .round() as usize
            % 4usize;

        if m == 0 {
            return false;
        } else if m == 1 {
            return true;
        } else {
            panic!("Bool decryption failure. Expected m to be either 1 or 0, but m={m} ");
        }
    }

    /// First encrypt as RLWE(m) with m as constant polynomial and extract it as
    /// LWE ciphertext
    pub(crate) fn pk_encrypt(&self, pk: &M, m: bool) -> M::R {
        DefaultSecureRng::with_local_mut(|rng| {
            let modop = &self.rlwe_modop;
            let nttop = &self.rlwe_nttop;

            // RLWE(0)
            // sample ephemeral key u
            let ring_size = self.parameters.rlwe_n;
            let mut u = vec![0i32; ring_size];
            fill_random_ternary_secret_with_hamming_weight(u.as_mut(), ring_size >> 1, rng);
            let mut u = M::R::try_convert_from(&u, &self.parameters.rlwe_q);
            nttop.forward(u.as_mut());

            let mut ua = M::R::zeros(ring_size);
            ua.as_mut().copy_from_slice(pk.get_row_slice(0));
            let mut ub = M::R::zeros(ring_size);
            ub.as_mut().copy_from_slice(pk.get_row_slice(1));

            // a*u
            nttop.forward(ua.as_mut());
            modop.elwise_mul_mut(ua.as_mut(), u.as_ref());
            nttop.backward(ua.as_mut());

            // b*u
            nttop.forward(ub.as_mut());
            modop.elwise_mul_mut(ub.as_mut(), u.as_ref());
            nttop.backward(ub.as_mut());

            let mut rlwe = M::zeros(2, ring_size);
            // sample error
            rlwe.iter_rows_mut().for_each(|ri| {
                RandomGaussianDist::random_fill(rng, &self.parameters.rlwe_q, ri.as_mut());
            });

            // a*u + e0
            modop.elwise_add_mut(rlwe.get_row_mut(0), ua.as_ref());
            // b*u + e1
            modop.elwise_add_mut(rlwe.get_row_mut(1), ub.as_ref());

            let m = if m {
                // Q/8
                self.rlweq_by8
            } else {
                // -Q/8
                self.parameters.rlwe_q - self.rlweq_by8
            };

            // b*u + e1 + m, where m is constant polynomial
            rlwe.set(1, 0, modop.add(rlwe.get(1, 0), &m));

            // sample extract index 0
            let mut lwe_out = M::R::zeros(ring_size + 1);
            sample_extract(&mut lwe_out, &rlwe, modop, 0);

            lwe_out
        })
    }

    /// TODO(Jay): Fetch client key from thread local
    pub fn sk_encrypt(&self, m: bool, client_key: &ClientKey) -> M::R {
        let m = if m {
            // Q/8
            self.rlweq_by8
        } else {
            // -Q/8
            self.parameters.rlwe_q - self.rlweq_by8
        };

        DefaultSecureRng::with_local_mut(|rng| {
            let mut lwe_out = M::R::zeros(self.parameters.rlwe_n + 1);
            encrypt_lwe(
                &mut lwe_out,
                &m,
                client_key.sk_rlwe.values(),
                &self.rlwe_modop,
                rng,
            );
            lwe_out
        })
    }

    pub fn sk_decrypt(&self, lwe_ct: &M::R, client_key: &ClientKey) -> bool {
        let m = decrypt_lwe(lwe_ct, client_key.sk_rlwe.values(), &self.rlwe_modop);
        let m = {
            // m + q/8 => {0,q/4 1}
            (((m + self.rlweq_by8).to_f64().unwrap() * 4.0)
                / self.parameters.rlwe_q.to_f64().unwrap())
            .round()
            .to_usize()
            .unwrap()
                % 4
        };

        if m == 0 {
            false
        } else if m == 1 {
            true
        } else {
            panic!("Incorrect bool decryption. Got m={m} but expected m to be 0 or 1")
        }
    }

    // TODO(Jay): scratch spaces must be thread local. Don't pass them as arguments
    pub fn nand(
        &self,
        c0: &M::R,
        c1: &M::R,
        server_key: &ServerKeyEvaluationDomain<M, DefaultSecureRng, NttOp>,
        scratch_lwen_plus1: &mut M::R,
        scratch_matrix_dplus2_ring: &mut M,
    ) -> M::R {
        // ClientKey::with_local(|ck| {
        //     let c0_noise = measure_noise_lwe(
        //         c0,
        //         ck.sk_rlwe.values(),
        //         &self.rlwe_modop,
        //         &(self.rlwe_q() - self.rlweq_by8),
        //     );
        //     let c1_noise =
        //         measure_noise_lwe(c1, ck.sk_rlwe.values(), &self.rlwe_modop,
        // &(self.rlweq_by8));     println!(
        //         "c0 noise: {c0_noise}; c1 noise:
        // {c1_noise}"
        //     );
        // });

        let mut c_out = M::R::zeros(c0.as_ref().len());
        let modop = &self.rlwe_modop;
        izip!(
            c_out.as_mut().iter_mut(),
            c0.as_ref().iter(),
            c1.as_ref().iter()
        )
        .for_each(|(o, i0, i1)| {
            *o = modop.add(i0, i1);
        });
        // +Q/8
        c_out.as_mut()[0] = modop.add(&c_out.as_ref()[0], &self.rlwe_qby4);

        // ClientKey::with_local(|ck| {
        //     let noise = measure_noise_lwe(
        //         &c_out,
        //         ck.sk_rlwe.values(),
        //         &self.rlwe_modop,
        //         &(self.rlweq_by8),
        //     );
        //     println!("cout_noise: {noise}");
        // });

        // PBS
        pbs(
            self,
            &self.nand_test_vec,
            &mut c_out,
            scratch_lwen_plus1,
            scratch_matrix_dplus2_ring,
            &self.lwe_modop,
            &self.rlwe_modop,
            &self.rlwe_nttop,
            server_key,
        );

        c_out
    }
}

impl<M: Matrix, NttOp, ModOp> PbsParameters for BoolEvaluator<M, M::MatElement, NttOp, ModOp>
where
    M::MatElement: PrimInt + WrappingSub + Debug,
{
    type Element = M::MatElement;
    type D = DefaultDecomposer<M::MatElement>;
    fn rlwe_auto_map(&self, k: isize) -> &(Vec<usize>, Vec<bool>) {
        let g = self.parameters.g as isize;
        if k == g {
            &self.rlwe_auto_maps[0]
        } else if k == -g {
            &self.rlwe_auto_maps[1]
        } else {
            panic!("RLWE auto map only supports k in [-g, g], but got k={k}");
        }
    }

    fn br_q(&self) -> usize {
        self.parameters.br_q
    }
    fn d_lwe(&self) -> usize {
        self.parameters.d_lwe
    }
    fn d_rgsw(&self) -> usize {
        self.parameters.d_rgsw
    }
    fn decomoposer_lwe(&self) -> &Self::D {
        &self.decomposer_lwe
    }
    fn decomoposer_rlwe(&self) -> &Self::D {
        &self.decomposer_rlwe
    }
    fn embedding_factor(&self) -> usize {
        self.embedding_factor
    }
    fn g(&self) -> isize {
        self.parameters.g as isize
    }
    fn g_k_dlog_map(&self) -> &[usize] {
        &self.g_k_dlog_map
    }
    fn lwe_n(&self) -> usize {
        self.parameters.lwe_n
    }
    fn lwe_q(&self) -> Self::Element {
        self.parameters.lwe_q
    }
    fn rlwe_n(&self) -> usize {
        self.parameters.rlwe_n
    }
    fn rlwe_q(&self) -> Self::Element {
        self.parameters.rlwe_q
    }
}

/// LMKCY+ Blind rotation
///
/// gk_to_si: [-g^0, -g^1, .., -g^{q/2-1}, g^0, ..., g^{q/2-1}]
fn blind_rotation<
    MT: IsTrivial + MatrixMut,
    Mmut: MatrixMut<MatElement = MT::MatElement> + Matrix,
    D: Decomposer<Element = MT::MatElement>,
    NttOp: Ntt<Element = MT::MatElement>,
    ModOp: ArithmeticOps<Element = MT::MatElement> + VectorOps<Element = MT::MatElement>,
    K: PbsKey<M = Mmut>,
    P: PbsParameters<Element = MT::MatElement>,
>(
    trivial_rlwe_test_poly: &mut MT,
    scratch_matrix_dplus2_ring: &mut Mmut,
    g: isize,
    w: usize,
    q: usize,
    gk_to_si: &[Vec<usize>],
    decomposer: &D,
    ntt_op: &NttOp,
    mod_op: &ModOp,
    parameters: &P,
    pbs_key: &K,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut::MatElement: Copy + Zero,
    <MT as Matrix>::R: RowMut,
{
    let q_by_2 = q / 2;

    // -(g^k)
    for i in (1..q_by_2).rev() {
        gk_to_si[q_by_2 + i].iter().for_each(|s_index| {
            rlwe_by_rgsw(
                trivial_rlwe_test_poly,
                pbs_key.rgsw_ct_lwe_si(*s_index),
                scratch_matrix_dplus2_ring,
                decomposer,
                ntt_op,
                mod_op,
            );
        });

        let (auto_map_index, auto_map_sign) = parameters.rlwe_auto_map(g);
        galois_auto(
            trivial_rlwe_test_poly,
            pbs_key.galois_key_for_auto(g),
            scratch_matrix_dplus2_ring,
            &auto_map_index,
            &auto_map_sign,
            mod_op,
            ntt_op,
            decomposer,
        );
    }

    // -(g^0)
    gk_to_si[q_by_2].iter().for_each(|s_index| {
        rlwe_by_rgsw(
            trivial_rlwe_test_poly,
            pbs_key.rgsw_ct_lwe_si(*s_index),
            scratch_matrix_dplus2_ring,
            decomposer,
            ntt_op,
            mod_op,
        );
    });
    let (auto_map_index, auto_map_sign) = parameters.rlwe_auto_map(-g);
    galois_auto(
        trivial_rlwe_test_poly,
        pbs_key.galois_key_for_auto(-g),
        scratch_matrix_dplus2_ring,
        &auto_map_index,
        &auto_map_sign,
        mod_op,
        ntt_op,
        decomposer,
    );

    // +(g^k)
    for i in (1..q_by_2).rev() {
        gk_to_si[i].iter().for_each(|s_index| {
            rlwe_by_rgsw(
                trivial_rlwe_test_poly,
                pbs_key.rgsw_ct_lwe_si(*s_index),
                scratch_matrix_dplus2_ring,
                decomposer,
                ntt_op,
                mod_op,
            );
        });

        let (auto_map_index, auto_map_sign) = parameters.rlwe_auto_map(g);
        galois_auto(
            trivial_rlwe_test_poly,
            pbs_key.galois_key_for_auto(g),
            scratch_matrix_dplus2_ring,
            &auto_map_index,
            &auto_map_sign,
            mod_op,
            ntt_op,
            decomposer,
        );
    }

    // +(g^0)
    gk_to_si[0].iter().for_each(|s_index| {
        rlwe_by_rgsw(
            trivial_rlwe_test_poly,
            pbs_key.rgsw_ct_lwe_si(gk_to_si[q_by_2][*s_index]),
            scratch_matrix_dplus2_ring,
            decomposer,
            ntt_op,
            mod_op,
        );
    });
}

/// - Mod down
/// - key switching
/// - mod down
/// - blind rotate
fn pbs<
    M: Matrix + MatrixMut + MatrixEntity,
    P: PbsParameters<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement>,
    ModOp: ArithmeticOps<Element = M::MatElement> + VectorOps<Element = M::MatElement>,
    K: PbsKey<M = M>,
>(
    parameters: &P,
    test_vec: &M::R,
    lwe_in: &mut M::R,
    scratch_lwen_plus1: &mut M::R,
    scratch_matrix_dplus2_ring: &mut M,
    modop_lweq: &ModOp,
    modop_rlweq: &ModOp,
    nttop_rlweq: &NttOp,
    pbs_key: &K,
) where
    <M as Matrix>::R: RowMut,
    M::MatElement: PrimInt + ToPrimitive + FromPrimitive + One + Copy + Zero + Display,
{
    let rlwe_q = parameters.rlwe_q();
    let lwe_q = parameters.lwe_q();
    let br_q = parameters.br_q();
    let rlwe_qf64 = rlwe_q.to_f64().unwrap();
    let lwe_qf64 = lwe_q.to_f64().unwrap();
    let br_qf64 = br_q.to_f64().unwrap();
    let rlwe_n = parameters.rlwe_n();

    PBSTracer::with_local_mut(|t| {
        let out = lwe_in
            .as_ref()
            .iter()
            .map(|v| v.to_u64().unwrap())
            .collect_vec();
        t.ct_rlwe_q_mod = out;
    });

    // moddown Q -> Q_ks
    lwe_in.as_mut().iter_mut().for_each(|v| {
        *v =
            M::MatElement::from_f64(((v.to_f64().unwrap() * lwe_qf64) / rlwe_qf64).round()).unwrap()
    });

    PBSTracer::with_local_mut(|t| {
        let out = lwe_in
            .as_ref()
            .iter()
            .map(|v| v.to_u64().unwrap())
            .collect_vec();
        t.ct_lwe_q_mod = out;
    });

    // key switch RLWE secret to LWE secret
    scratch_lwen_plus1.as_mut().fill(M::MatElement::zero());
    lwe_key_switch(
        scratch_lwen_plus1,
        lwe_in,
        pbs_key.lwe_ksk(),
        modop_lweq,
        parameters.decomoposer_lwe(),
    );

    PBSTracer::with_local_mut(|t| {
        let out = scratch_lwen_plus1
            .as_ref()
            .iter()
            .map(|v| v.to_u64().unwrap())
            .collect_vec();
        t.ct_lwe_q_mod_after_ksk = out;
    });

    // odd mowdown Q_ks -> q
    let g_k_dlog_map = parameters.g_k_dlog_map();
    let mut g_k_si = vec![vec![]; br_q];
    scratch_lwen_plus1
        .as_ref()
        .iter()
        .skip(1)
        .enumerate()
        .for_each(|(index, v)| {
            let odd_v = mod_switch_odd(v.to_f64().unwrap(), lwe_qf64, br_qf64);
            let k = g_k_dlog_map[odd_v];
            g_k_si[k].push(index);
        });

    PBSTracer::with_local_mut(|t| {
        let out = scratch_lwen_plus1
            .as_ref()
            .iter()
            .map(|v| mod_switch_odd(v.to_f64().unwrap(), lwe_qf64, br_qf64) as u64)
            .collect_vec();
        t.ct_br_q_mod = out;
    });

    // handle b and set trivial test RLWE
    let g = parameters.g() as usize;
    let g_times_b = (g * mod_switch_odd(
        scratch_lwen_plus1.as_ref()[0].to_f64().unwrap(),
        lwe_qf64,
        br_qf64,
    )) % (br_q);
    // v = (v(X) * X^{g*b}) mod X^{q/2}+1
    let br_qby2 = br_q / 2;
    let mut gb_monomial_sign = true;
    let mut gb_monomial_exp = g_times_b;
    // X^{g*b} mod X^{q/2}+1
    if gb_monomial_exp > br_qby2 {
        gb_monomial_exp -= br_qby2;
        gb_monomial_sign = false
    }
    // monomial mul
    let mut trivial_rlwe_test_poly =
        RlweCiphertext::<_, DefaultSecureRng>::from_raw(M::zeros(2, rlwe_n), true);
    if parameters.embedding_factor() == 1 {
        monomial_mul(
            test_vec.as_ref(),
            trivial_rlwe_test_poly.get_row_mut(1).as_mut(),
            gb_monomial_exp,
            gb_monomial_sign,
            br_qby2,
            modop_rlweq,
        );
    } else {
        // use lwe_in to store the `t = v(X) * X^{g*2} mod X^{q/2}+1` temporarily. This
        // works because q/2 <= N (where N is lwe_in LWE dimension) always.
        monomial_mul(
            test_vec.as_ref(),
            &mut lwe_in.as_mut()[..br_qby2],
            gb_monomial_exp,
            gb_monomial_sign,
            br_qby2,
            modop_rlweq,
        );

        // emebed poly `t` in ring X^{q/2}+1 inside the bigger ring X^{N}+1
        let embed_factor = parameters.embedding_factor();
        let partb_trivial_rlwe = trivial_rlwe_test_poly.get_row_mut(1);
        lwe_in.as_ref()[..br_qby2]
            .iter()
            .enumerate()
            .for_each(|(index, v)| {
                partb_trivial_rlwe[embed_factor * index] = *v;
            });
    }

    // blind rotate
    blind_rotation(
        &mut trivial_rlwe_test_poly,
        scratch_matrix_dplus2_ring,
        parameters.g(),
        1,
        br_q,
        &g_k_si,
        parameters.decomoposer_rlwe(),
        nttop_rlweq,
        modop_rlweq,
        parameters,
        pbs_key,
    );

    // ClientKey::with_local(|ck| {
    //     let ring_size = parameters.rlwe_n();
    //     let mut rlwe_ct = vec![vec![0u64; ring_size]; 2];
    //     izip!(
    //         rlwe_ct[0].iter_mut(),
    //         trivial_rlwe_test_poly.0.get_row_slice(0)
    //     )
    //     .for_each(|(t, f)| {
    //         *t = f.to_u64().unwrap();
    //     });
    //     izip!(
    //         rlwe_ct[1].iter_mut(),
    //         trivial_rlwe_test_poly.0.get_row_slice(1)
    //     )
    //     .for_each(|(t, f)| {
    //         *t = f.to_u64().unwrap();
    //     });
    //     let mut m_out = vec![vec![0u64; ring_size]];
    //     let modop = ModularOpsU64::new(rlwe_q.to_u64().unwrap());
    //     let nttop = NttBackendU64::new(rlwe_q.to_u64().unwrap(), ring_size);
    //     decrypt_rlwe(&rlwe_ct, ck.sk_rlwe.values(), &mut m_out, &nttop, &modop);

    //     println!("RLWE post PBS message: {:?}", m_out[0]);
    // });

    // sample extract
    sample_extract(lwe_in, &trivial_rlwe_test_poly, modop_rlweq, 0);
}

fn mod_switch_odd(v: f64, from_q: f64, to_q: f64) -> usize {
    let odd_v = (((v * to_q) / (from_q)).floor()).to_usize().unwrap();
    //TODO(Jay): check correctness of this
    odd_v + ((odd_v & 1) ^ 1)
}

// TODO(Jay): Add tests for sample extract
fn sample_extract<M: Matrix + MatrixMut, ModOp: ArithmeticOps<Element = M::MatElement>>(
    lwe_out: &mut M::R,
    rlwe_in: &M,
    mod_op: &ModOp,
    index: usize,
) where
    <M as Matrix>::R: RowMut,
    M::MatElement: Copy,
{
    let ring_size = rlwe_in.dimension().1;

    // index..=0
    let to = &mut lwe_out.as_mut()[1..];
    let from = rlwe_in.get_row_slice(0);
    for i in 0..index + 1 {
        to[i] = from[index - i];
    }

    // -(N..index)
    for i in index + 1..ring_size {
        to[i] = mod_op.neg(&from[ring_size + index - i]);
    }

    // set b
    lwe_out.as_mut()[0] = *rlwe_in.get(1, index);
}

/// TODO(Jay): Write tests for monomial mul
fn monomial_mul<El, ModOp: ArithmeticOps<Element = El>>(
    p_in: &[El],
    p_out: &mut [El],
    mon_exp: usize,
    mon_sign: bool,
    ring_size: usize,
    mod_op: &ModOp,
) where
    El: Copy,
{
    debug_assert!(p_in.as_ref().len() == ring_size);
    debug_assert!(p_in.as_ref().len() == p_out.as_ref().len());
    debug_assert!(mon_exp < ring_size);

    p_in.as_ref().iter().enumerate().for_each(|(index, v)| {
        let mut to_index = index + mon_exp;
        let mut to_sign = mon_sign;
        if to_index >= ring_size {
            to_index = to_index - ring_size;
            to_sign = !to_sign;
        }

        if !to_sign {
            p_out.as_mut()[to_index] = mod_op.neg(v);
        } else {
            p_out.as_mut()[to_index] = *v;
        }
    });
}

thread_local! {
    static PBS_TRACER: RefCell<PBSTracer<Vec<Vec<u64>>>> = RefCell::new(PBSTracer::default());
}

#[derive(Default)]
struct PBSTracer<M>
where
    M: Matrix + Default,
{
    pub(crate) ct_rlwe_q_mod: M::R,
    pub(crate) ct_lwe_q_mod: M::R,
    pub(crate) ct_lwe_q_mod_after_ksk: M::R,
    pub(crate) ct_br_q_mod: Vec<u64>,
}

impl PBSTracer<Vec<Vec<u64>>> {
    fn trace(&self, parameters: &BoolParameters<u64>, sk_lwe: &[i32], sk_rlwe: &[i32]) {
        assert!(parameters.rlwe_n == sk_rlwe.len());
        assert!(parameters.lwe_n == sk_lwe.len());

        let modop_rlweq = ModularOpsU64::new(parameters.rlwe_q as u64);
        // noise after mod down Q -> Q_ks
        let m_back0 = decrypt_lwe(&self.ct_rlwe_q_mod, sk_rlwe, &modop_rlweq);

        let modop_lweq = ModularOpsU64::new(parameters.lwe_q as u64);
        // noise after mod down Q -> Q_ks
        let m_back1 = decrypt_lwe(&self.ct_lwe_q_mod, sk_rlwe, &modop_lweq);
        // noise after key switch from RLWE -> LWE
        let m_back2 = decrypt_lwe(&self.ct_lwe_q_mod_after_ksk, sk_lwe, &modop_lweq);

        // noise after mod down odd from Q_ks -> q
        let modop_br_q = ModularOpsU64::new(parameters.br_q as u64);
        let m_back3 = decrypt_lwe(&self.ct_br_q_mod, sk_lwe, &modop_br_q);

        println!(
            "
            M initial mod Q: {m_back0},
            M after mod down Q -> Q_ks: {m_back1},
            M after key switch from RLWE -> LWE: {m_back2},
            M after mod dwon Q_ks -> q: {m_back3}
        "
        );
    }
}

impl WithLocal for PBSTracer<Vec<Vec<u64>>> {
    fn with_local<F, R>(func: F) -> R
    where
        F: Fn(&Self) -> R,
    {
        PBS_TRACER.with_borrow(|t| func(t))
    }

    fn with_local_mut<F, R>(func: F) -> R
    where
        F: Fn(&mut Self) -> R,
    {
        PBS_TRACER.with_borrow_mut(|t| func(t))
    }
}

#[cfg(test)]
mod tests {
    use std::iter::Sum;

    use rand::{thread_rng, Rng};
    use rand_distr::Uniform;

    use crate::{
        backend::ModularOpsU64,
        bool,
        ntt::NttBackendU64,
        random::DEFAULT_RNG,
        rgsw::{
            self, measure_noise, public_key_encrypt_rlwe, secret_key_encrypt_rlwe,
            tests::{_measure_noise_rgsw, _secret_encrypt_rlwe},
            RgswCiphertext, RgswCiphertextEvaluationDomain, SeededRgswCiphertext,
            SeededRlweCiphertext,
        },
        utils::negacyclic_mul,
    };

    use self::parameters::{MP_BOOL_PARAMS, SP_BOOL_PARAMS};

    use super::*;

    // #[test]
    // fn trial() {
    //     dbg!(generate_prime(28, 1 << 11, 1 << 28));
    // }

    #[test]
    fn bool_encrypt_decrypt_works() {
        // let prime = generate_prime(32, 2 * 1024, 1 << 32);
        // dbg!(prime);
        let bool_evaluator =
            BoolEvaluator::<Vec<Vec<u64>>, u64, NttBackendU64, ModularOpsU64>::new(SP_BOOL_PARAMS);
        let client_key = bool_evaluator.client_key();
        // let sever_key = bool_evaluator.server_key(&client_key);

        let mut m = true;
        for _ in 0..1000 {
            let lwe_ct = bool_evaluator.sk_encrypt(m, &client_key);
            let m_back = bool_evaluator.sk_decrypt(&lwe_ct, &client_key);
            assert_eq!(m, m_back);
            m = !m;
        }
    }

    #[test]
    fn bool_nand() {
        DefaultSecureRng::with_local_mut(|r| {
            let rng = DefaultSecureRng::new_seeded([19u8; 32]);
            *r = rng;
        });

        let bool_evaluator =
            BoolEvaluator::<Vec<Vec<u64>>, u64, NttBackendU64, ModularOpsU64>::new(SP_BOOL_PARAMS);

        // println!("{:?}", bool_evaluator.nand_test_vec);
        let client_key = bool_evaluator.client_key();
        let seeded_server_key = bool_evaluator.server_key(&client_key);
        let server_key_eval_domain =
            ServerKeyEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(
                &seeded_server_key,
            );

        let mut scratch_lwen_plus1 = vec![0u64; bool_evaluator.parameters.lwe_n + 1];
        let mut scratch_matrix_dplus2_ring = vec![
            vec![0u64; bool_evaluator.parameters.rlwe_n];
            bool_evaluator.parameters.d_rgsw + 2
        ];

        let mut m0 = false;
        let mut m1 = true;
        let mut ct0 = bool_evaluator.sk_encrypt(m0, &client_key);
        let mut ct1 = bool_evaluator.sk_encrypt(m1, &client_key);

        for _ in 0..1000 {
            let ct_back = bool_evaluator.nand(
                &ct0,
                &ct1,
                &server_key_eval_domain,
                &mut scratch_lwen_plus1,
                &mut scratch_matrix_dplus2_ring,
            );

            let m_out = !(m0 && m1);

            // Trace and measure PBS noise
            {
                let noise0 = {
                    let ideal = if m0 {
                        bool_evaluator.rlweq_by8
                    } else {
                        bool_evaluator.rlwe_q() - bool_evaluator.rlweq_by8
                    };
                    let n = measure_noise_lwe(
                        &ct0,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                        &ideal,
                    );
                    let v = decrypt_lwe(
                        &ct0,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                    );
                    (n, v)
                };
                let noise1 = {
                    let ideal = if m1 {
                        bool_evaluator.rlweq_by8
                    } else {
                        bool_evaluator.rlwe_q() - bool_evaluator.rlweq_by8
                    };
                    let n = measure_noise_lwe(
                        &ct1,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                        &ideal,
                    );
                    let v = decrypt_lwe(
                        &ct1,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                    );
                    (n, v)
                };

                // Trace PBS
                PBSTracer::with_local(|t| {
                    t.trace(
                        &SP_BOOL_PARAMS,
                        &client_key.sk_lwe.values(),
                        client_key.sk_rlwe.values(),
                    )
                });

                // Calculate noise in ciphertext post PBS
                let noise_out = {
                    let ideal = if m_out {
                        bool_evaluator.rlweq_by8
                    } else {
                        bool_evaluator.rlwe_q() - bool_evaluator.rlweq_by8
                    };
                    let n = measure_noise_lwe(
                        &ct_back,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                        &ideal,
                    );
                    let v = decrypt_lwe(
                        &ct_back,
                        client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                    );
                    (n, v)
                };
                dbg!(m0, m1, m_out);
                println!(
                    "ct0 (noise, message): {:?}  \n ct1 (noise, message): {:?} \n PBS (noise, message): {:?}", noise0, noise1, noise_out
                );
            }
            let m_back = bool_evaluator.sk_decrypt(&ct_back, &client_key);
            assert!(m_out == m_back, "Expected {m_out}, got {m_back}");
            println!("----------");

            m1 = m0;
            m0 = m_out;

            ct1 = ct0;
            ct0 = ct_back;
        }
    }

    #[test]
    fn multi_party_encryption_decryption() {
        let bool_evaluator =
            BoolEvaluator::<Vec<Vec<u64>>, u64, NttBackendU64, ModularOpsU64>::new(MP_BOOL_PARAMS);

        let no_of_parties = 500;
        let parties = (0..no_of_parties)
            .map(|_| bool_evaluator.client_key())
            .collect_vec();

        let mut ideal_rlwe_sk = vec![0i32; bool_evaluator.rlwe_n()];
        parties.iter().for_each(|k| {
            izip!(ideal_rlwe_sk.iter_mut(), k.sk_rlwe.values()).for_each(|(ideal_i, s_i)| {
                *ideal_i = *ideal_i + s_i;
            });
        });

        println!("{:?}", &ideal_rlwe_sk);

        let mut m = true;
        for i in 0..100 {
            let pk_cr_seed = [0u8; 32];

            let public_key_share = parties
                .iter()
                .map(|k| bool_evaluator.multi_party_public_key_share(pk_cr_seed, k))
                .collect_vec();

            let collective_pk = PublicKey::<Vec<Vec<u64>>, DefaultSecureRng, ModularOpsU64>::from(
                public_key_share.as_slice(),
            );
            let lwe_ct = bool_evaluator.pk_encrypt(&collective_pk.key, m);

            let decryption_shares = parties
                .iter()
                .map(|k| bool_evaluator.multi_party_decryption_share(&lwe_ct, k))
                .collect_vec();

            let m_back = bool_evaluator.multi_party_decrypt(&decryption_shares, &lwe_ct);

            {
                let ideal_m = if m {
                    bool_evaluator.rlweq_by8
                } else {
                    bool_evaluator.parameters.rlwe_q - bool_evaluator.rlweq_by8
                };
                let noise = measure_noise_lwe(
                    &lwe_ct,
                    &ideal_rlwe_sk,
                    &bool_evaluator.rlwe_modop,
                    &ideal_m,
                );
                println!("Noise: {noise}");
            }

            assert_eq!(m_back, m);
            m = !m;
        }
    }

    #[test]
    fn ms() {
        let logbig_q = 50;
        let logsmall_q = 20;
        let big_q = 1 << logbig_q;
        let small_q = 1 << logsmall_q;
        let lwe_n = 493;

        let no_of_parties = 10;
        let parties_lwe_sk = (0..no_of_parties)
            .map(|_| LweSecret::random(lwe_n >> 1, lwe_n))
            .collect_vec();

        // Ideal secrets
        let mut ideal_lwe_sk = vec![0i32; lwe_n];
        parties_lwe_sk.iter().for_each(|k| {
            izip!(ideal_lwe_sk.iter_mut(), k.values()).for_each(|(ideal_i, s_i)| {
                *ideal_i = *ideal_i + s_i;
            });
        });

        let mut rng = DefaultSecureRng::new();

        let logp = 3;
        let modop_bigq = ModularOpsU64::new(big_q);
        let modop_smallq = ModularOpsU64::new(small_q);

        for i in 0..100 {
            let m = thread_rng().sample(Uniform::new(0u64, (1u64 << logp)));
            let bigq_m = m << (logbig_q - logp);
            let smallq_m = m << (logsmall_q - logp);

            // encrypt
            let mut lwe_ct = vec![0u64; lwe_n + 1];
            encrypt_lwe(&mut lwe_ct, &bigq_m, &ideal_lwe_sk, &modop_bigq, &mut rng);

            let noise = measure_noise_lwe(&lwe_ct, &ideal_lwe_sk, &modop_bigq, &bigq_m);
            println!("Noise Before: {noise}");

            // mod switch
            let lwe_ct_ms = lwe_ct
                .iter()
                .map(|v| (((*v as f64) * small_q as f64) / (big_q as f64)).round() as u64)
                .collect_vec();

            let noise = measure_noise_lwe(&lwe_ct_ms, &ideal_lwe_sk, &modop_smallq, &smallq_m);
            println!("Noise After: {noise}");
        }
    }

    #[test]
    fn multi_party_lwe_keyswitch() {
        let lwe_logq = 18;
        let lwe_q = 1 << lwe_logq;
        let d_lwe = 1;
        let logb_lwe = 6;
        let lweq_modop = ModularOpsU64::new(lwe_q);

        let decomposer = DefaultDecomposer::new(lwe_q, logb_lwe, d_lwe);
        let lwe_gadgect_vec = decomposer.gadget_vector();
        let logp = 2;

        let from_lwe_n = 2048;
        let to_lwe_n = 500;

        let no_of_parties = 10;
        let parties_from_lwe_sk = (0..no_of_parties)
            .map(|_| LweSecret::random(from_lwe_n >> 1, from_lwe_n))
            .collect_vec();
        let parties_to_lwe_sk = (0..no_of_parties)
            .map(|_| LweSecret::random(to_lwe_n >> 1, to_lwe_n))
            .collect_vec();

        // Ideal secrets
        let mut ideal_from_lwe_sk = vec![0i32; from_lwe_n];
        parties_from_lwe_sk.iter().for_each(|k| {
            izip!(ideal_from_lwe_sk.iter_mut(), k.values()).for_each(|(ideal_i, s_i)| {
                *ideal_i = *ideal_i + s_i;
            });
        });
        let mut ideal_to_lwe_sk = vec![0i32; to_lwe_n];
        parties_to_lwe_sk.iter().for_each(|k| {
            izip!(ideal_to_lwe_sk.iter_mut(), k.values()).for_each(|(ideal_i, s_i)| {
                *ideal_i = *ideal_i + s_i;
            });
        });

        // Generate Lwe KSK share
        let mut rng = DefaultSecureRng::new();
        let mut ksk_seed = [0u8; 32];
        rng.fill_bytes(&mut ksk_seed);
        let lwe_ksk_shares = izip!(parties_from_lwe_sk.iter(), parties_to_lwe_sk.iter())
            .map(|(from_sk, to_sk)| {
                let mut ksk_out = vec![0u64; from_lwe_n * d_lwe];
                let mut p_rng = DefaultSecureRng::new_seeded(ksk_seed);
                lwe_ksk_keygen(
                    from_sk.values(),
                    to_sk.values(),
                    &mut ksk_out,
                    &lwe_gadgect_vec,
                    &lweq_modop,
                    &mut p_rng,
                    &mut rng,
                );
                ksk_out
            })
            .collect_vec();

        // Create collective LWE ksk
        let mut sum_partb = vec![0u64; d_lwe * from_lwe_n];
        lwe_ksk_shares.iter().for_each(|share| {
            lweq_modop.elwise_add_mut(sum_partb.as_mut_slice(), share.as_slice())
        });
        let mut lwe_ksk = vec![vec![0u64; to_lwe_n + 1]; d_lwe * from_lwe_n];
        let mut p_rng = DefaultSecureRng::new_seeded(ksk_seed);
        izip!(lwe_ksk.iter_mut(), sum_partb.iter()).for_each(|(lwe_i, part_bi)| {
            RandomUniformDist::random_fill(&mut p_rng, &lwe_q, &mut lwe_i.as_mut_slice()[1..]);
            lwe_i[0] = *part_bi;
        });

        for i in 0..128 {
            println!("############## ITERATION {i} ##############");

            // Encrypt m
            let m = 1;
            let mut lwe_ct = vec![0u64; from_lwe_n + 1];
            encrypt_lwe(&mut lwe_ct, &m, &ideal_from_lwe_sk, &lweq_modop, &mut rng);

            let noise = measure_noise_lwe(&lwe_ct, &ideal_from_lwe_sk, &lweq_modop, &m);
            println!("Noise before key switch: {noise}");

            // Key switch
            let lwe_ct_key_switched = {
                let mut lwe_ct_key_switched = vec![0u64; to_lwe_n + 1];
                lwe_key_switch(
                    &mut lwe_ct_key_switched,
                    &lwe_ct,
                    &lwe_ksk,
                    &lweq_modop,
                    &decomposer,
                );
                lwe_ct_key_switched
            };

            let noise = measure_noise_lwe(&lwe_ct_key_switched, &ideal_to_lwe_sk, &lweq_modop, &m);
            println!("Noise after key switch: {noise}");
        }
    }

    fn _collecitve_public_key_gen(rlwe_q: u64, parties_rlwe_sk: &[RlweSecret]) -> Vec<Vec<u64>> {
        let ring_size = parties_rlwe_sk[0].values.len();
        assert!(ring_size.is_power_of_two());
        let mut rng = DefaultSecureRng::new();
        let nttop = NttBackendU64::new(rlwe_q, ring_size);
        let modop = ModularOpsU64::new(rlwe_q);

        // Generate Pk shares
        let pk_seed = [0u8; 32];
        let pk_shares = parties_rlwe_sk.iter().map(|sk| {
            let mut p_rng = DefaultSecureRng::new_seeded(pk_seed);
            let mut share_out = vec![0u64; ring_size];
            public_key_share(
                &mut share_out,
                sk.values(),
                &modop,
                &nttop,
                &mut p_rng,
                &mut rng,
            );
            share_out
        });

        let mut pk_part_b = vec![0u64; ring_size];
        pk_shares.for_each(|share| modop.elwise_add_mut(&mut pk_part_b, &share));
        let mut pk_part_a = vec![0u64; ring_size];
        let mut p_rng = DefaultSecureRng::new_seeded(pk_seed);
        RandomUniformDist::random_fill(&mut p_rng, &rlwe_q, pk_part_a.as_mut_slice());

        vec![pk_part_a, pk_part_b]
    }

    fn _multi_party_all_keygen(
        bool_evaluator: &BoolEvaluator<Vec<Vec<u64>>, u64, NttBackendU64, ModularOpsU64>,
        no_of_parties: usize,
    ) -> (
        Vec<ClientKey>,
        PublicKey<Vec<Vec<u64>>, DefaultSecureRng, ModularOpsU64>,
        Vec<
            CommonReferenceSeededMultiPartyServerKeyShare<
                Vec<Vec<u64>>,
                BoolParameters<u64>,
                [u8; 32],
            >,
        >,
        SeededMultiPartyServerKey<Vec<Vec<u64>>, [u8; 32], BoolParameters<u64>>,
        ServerKeyEvaluationDomain<Vec<Vec<u64>>, DefaultSecureRng, NttBackendU64>,
        ClientKey,
    ) {
        let parties = (0..no_of_parties)
            .map(|_| bool_evaluator.client_key())
            .collect_vec();

        // Collective public key
        let pk_cr_seed = [0u8; 32];
        let public_key_share = parties
            .iter()
            .map(|k| bool_evaluator.multi_party_public_key_share(pk_cr_seed, k))
            .collect_vec();
        let collective_pk = PublicKey::<Vec<Vec<u64>>, DefaultSecureRng, ModularOpsU64>::from(
            public_key_share.as_slice(),
        );

        // Server key
        let pbs_cr_seed = [1u8; 32];
        let server_key_shares = parties
            .iter()
            .map(|k| bool_evaluator.multi_party_sever_key_share(pbs_cr_seed, &collective_pk.key, k))
            .collect_vec();
        let seeded_server_key =
            aggregate_multi_party_server_key_shares::<_, _, _, ModularOpsU64, NttBackendU64>(
                &server_key_shares,
                &bool_evaluator.decomposer_rlwe,
            );
        let server_key_eval = ServerKeyEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(
            &seeded_server_key,
        );

        // construct ideal rlwe sk for meauring noise
        let ideal_client_key = {
            let mut ideal_rlwe_sk = vec![0i32; bool_evaluator.rlwe_n()];
            parties.iter().for_each(|k| {
                izip!(ideal_rlwe_sk.iter_mut(), k.sk_rlwe.values()).for_each(|(ideal_i, s_i)| {
                    *ideal_i = *ideal_i + s_i;
                });
            });
            let mut ideal_lwe_sk = vec![0i32; bool_evaluator.lwe_n()];
            parties.iter().for_each(|k| {
                izip!(ideal_lwe_sk.iter_mut(), k.sk_lwe.values()).for_each(|(ideal_i, s_i)| {
                    *ideal_i = *ideal_i + s_i;
                });
            });

            ClientKey {
                sk_lwe: LweSecret {
                    values: ideal_lwe_sk,
                },
                sk_rlwe: RlweSecret {
                    values: ideal_rlwe_sk,
                },
            }
        };

        (
            parties,
            collective_pk,
            server_key_shares,
            seeded_server_key,
            server_key_eval,
            ideal_client_key,
        )
    }

    #[test]
    fn mp_key_correcntess() {
        let bool_evaluator =
            BoolEvaluator::<Vec<Vec<u64>>, u64, NttBackendU64, ModularOpsU64>::new(MP_BOOL_PARAMS);

        let (_, collective_pk, _, _, server_key_eval, ideal_client_key) =
            _multi_party_all_keygen(&bool_evaluator, 20);

        let lwe_q = bool_evaluator.parameters.lwe_q;
        let rlwe_q = bool_evaluator.parameters.rlwe_q;
        let d_rgsw = bool_evaluator.parameters.d_rgsw;
        let lwe_logq = bool_evaluator.parameters.lwe_logq;
        let lwe_n = bool_evaluator.parameters.lwe_n;
        let rlwe_n = bool_evaluator.parameters.rlwe_n;
        let lwe_modop = &bool_evaluator.lwe_modop;
        let rlwe_nttop = &bool_evaluator.rlwe_nttop;
        let rlwe_modop = &bool_evaluator.rlwe_modop;
        let rlwe_decomposer = &bool_evaluator.decomposer_rlwe;

        // test LWE ksk from RLWE -> LWE
        if false {
            let logp = 2;
            let mut rng = DefaultSecureRng::new();

            let m = 1;
            let encoded_m = m << (lwe_logq - logp);

            // Encrypt
            let mut lwe_ct = vec![0u64; rlwe_n + 1];
            encrypt_lwe(
                &mut lwe_ct,
                &encoded_m,
                ideal_client_key.sk_rlwe.values(),
                lwe_modop,
                &mut rng,
            );

            // key switch
            let lwe_decomposer = &bool_evaluator.decomposer_lwe;
            let mut lwe_out = vec![0u64; lwe_n + 1];
            lwe_key_switch(
                &mut lwe_out,
                &lwe_ct,
                &server_key_eval.lwe_ksk,
                lwe_modop,
                lwe_decomposer,
            );

            let encoded_m_back = decrypt_lwe(&lwe_out, ideal_client_key.sk_lwe.values(), lwe_modop);
            let m_back =
                ((encoded_m_back as f64 * (1 << logp) as f64) / (lwe_q as f64)).round() as u64;
            dbg!(m_back, m);

            let noise = measure_noise_lwe(
                &lwe_out,
                ideal_client_key.sk_lwe.values(),
                lwe_modop,
                &encoded_m,
            );

            println!("Noise: {noise}");
        }

        // Measure noise in RGSW ciphertexts of ideal LWE secrets
        if true {
            let gadget_vec = rlwe_decomposer.gadget_vector();
            for i in 0..20 {
                // measure noise in RGSW(s[i])
                let si =
                    ideal_client_key.sk_lwe.values[i] * (bool_evaluator.embedding_factor as i32);
                let mut si_poly = vec![0u64; rlwe_n];
                if si < 0 {
                    si_poly[rlwe_n - (si.abs() as usize)] = rlwe_q - 1;
                } else {
                    si_poly[(si.abs() as usize)] = 1;
                }

                let mut rgsw_si = server_key_eval.rgsw_cts[i].clone();
                rgsw_si
                    .iter_mut()
                    .for_each(|ri| rlwe_nttop.backward(ri.as_mut()));

                println!("####### Noise in RGSW(X^s_{i}) #######");
                _measure_noise_rgsw(
                    &rgsw_si,
                    &si_poly,
                    ideal_client_key.sk_rlwe.values(),
                    &gadget_vec,
                    rlwe_q,
                );
                println!("####### ##################### #######");
            }
        }

        // measure noise grwoth in RLWExRGSW
        if true {
            let mut rng = DefaultSecureRng::new();
            let mut carry_m = vec![0u64; rlwe_n];
            RandomUniformDist::random_fill(&mut rng, &rlwe_q, carry_m.as_mut_slice());

            // RGSW(carrym)
            let trivial_rlwect = vec![vec![0u64; rlwe_n], carry_m.clone()];
            let mut rlwe_ct = RlweCiphertext::<_, DefaultSecureRng>::from_raw(trivial_rlwect, true);

            let mut scratch_matrix_dplus2_ring = vec![vec![0u64; rlwe_n]; d_rgsw + 2];
            let mul_mod =
                |v0: &u64, v1: &u64| (((*v0 as u128 * *v1 as u128) % (rlwe_q as u128)) as u64);

            for i in 0..bool_evaluator.parameters.lwe_n {
                rlwe_by_rgsw(
                    &mut rlwe_ct,
                    server_key_eval.rgsw_ct_lwe_si(i),
                    &mut scratch_matrix_dplus2_ring,
                    rlwe_decomposer,
                    rlwe_nttop,
                    rlwe_modop,
                );

                // carry_m[X] * s_i[X]
                let si =
                    ideal_client_key.sk_lwe.values[i] * (bool_evaluator.embedding_factor as i32);
                let mut si_poly = vec![0u64; rlwe_n];
                if si < 0 {
                    si_poly[rlwe_n - (si.abs() as usize)] = rlwe_q - 1;
                } else {
                    si_poly[(si.abs() as usize)] = 1;
                }
                carry_m = negacyclic_mul(&carry_m, &si_poly, mul_mod, rlwe_q);

                let noise = measure_noise(
                    &rlwe_ct,
                    &carry_m,
                    rlwe_nttop,
                    rlwe_modop,
                    ideal_client_key.sk_rlwe.values(),
                );
                println!("Noise RLWE(carry_m) accumulating {i}^th secret monomial: {noise}");
            }
        }

        // Check galois keys
        if false {
            let g = bool_evaluator.g() as isize;
            let mut rng = DefaultSecureRng::new();
            let mut scratch_matrix_dplus2_ring = vec![vec![0u64; rlwe_n]; d_rgsw + 2];
            for i in [g, -g] {
                let mut m = vec![0u64; rlwe_n];
                RandomUniformDist::random_fill(&mut rng, &rlwe_q, m.as_mut_slice());
                let mut rlwe_ct = {
                    let mut data = vec![vec![0u64; rlwe_n]; 2];
                    public_key_encrypt_rlwe(
                        &mut data,
                        &collective_pk.key,
                        &m,
                        rlwe_modop,
                        rlwe_nttop,
                        &mut rng,
                    );
                    RlweCiphertext::<_, DefaultSecureRng>::from_raw(data, false)
                };

                let auto_key = server_key_eval.galois_key_for_auto(i);
                let (auto_map_index, auto_map_sign) = generate_auto_map(rlwe_n, i);
                galois_auto(
                    &mut rlwe_ct,
                    auto_key,
                    &mut scratch_matrix_dplus2_ring,
                    &auto_map_index,
                    &auto_map_sign,
                    rlwe_modop,
                    rlwe_nttop,
                    rlwe_decomposer,
                );

                // send m(X) -> m(X^i)
                let mut m_k = vec![0u64; rlwe_n];
                izip!(m.iter(), auto_map_index.iter(), auto_map_sign.iter()).for_each(
                    |(mi, to_index, to_sign)| {
                        if !to_sign {
                            m_k[*to_index] = rlwe_q - *mi;
                        } else {
                            m_k[*to_index] = *mi;
                        }
                    },
                );

                // measure noise
                let noise = measure_noise(
                    &rlwe_ct,
                    &m_k,
                    rlwe_nttop,
                    rlwe_modop,
                    ideal_client_key.sk_rlwe.values(),
                );

                println!("Noise after auto k={i}: {noise}");
            }
        }
    }

    #[test]
    fn multi_party_nand() {
        let bool_evaluator =
            BoolEvaluator::<Vec<Vec<u64>>, u64, NttBackendU64, ModularOpsU64>::new(MP_BOOL_PARAMS);

        let (parties, collective_pk, _, _, server_key_eval, ideal_client_key) =
            _multi_party_all_keygen(&bool_evaluator, 50);

        // PBS
        let mut scratch_lwen_plus1 = vec![0u64; bool_evaluator.parameters.lwe_n + 1];
        let mut scratch_matrix_dplus2_ring = vec![
            vec![0u64; bool_evaluator.parameters.rlwe_n];
            bool_evaluator.parameters.d_rgsw + 2
        ];

        let mut m0 = true;
        let mut m1 = false;

        let mut lwe0 = bool_evaluator.pk_encrypt(&collective_pk.key, m0);
        let mut lwe1 = bool_evaluator.pk_encrypt(&collective_pk.key, m1);

        for _ in 0..2000 {
            let lwe_out = bool_evaluator.nand(
                &lwe0,
                &lwe1,
                &server_key_eval,
                &mut scratch_lwen_plus1,
                &mut scratch_matrix_dplus2_ring,
            );

            let m_expected = !(m0 & m1);

            // measure noise
            {
                let noise0 = {
                    let ideal = if m0 {
                        bool_evaluator.rlweq_by8
                    } else {
                        bool_evaluator.rlwe_q() - bool_evaluator.rlweq_by8
                    };
                    let n = measure_noise_lwe(
                        &lwe0,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                        &ideal,
                    );
                    let v = decrypt_lwe(
                        &lwe0,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                    );
                    (n, v)
                };
                let noise1 = {
                    let ideal = if m1 {
                        bool_evaluator.rlweq_by8
                    } else {
                        bool_evaluator.rlwe_q() - bool_evaluator.rlweq_by8
                    };
                    let n = measure_noise_lwe(
                        &lwe1,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                        &ideal,
                    );
                    let v = decrypt_lwe(
                        &lwe1,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                    );
                    (n, v)
                };

                // Trace PBS
                PBSTracer::with_local(|t| {
                    t.trace(
                        &MP_BOOL_PARAMS,
                        &ideal_client_key.sk_lwe.values(),
                        &ideal_client_key.sk_rlwe.values(),
                    )
                });

                let noise_out = {
                    let ideal_m = if m_expected {
                        bool_evaluator.rlweq_by8
                    } else {
                        bool_evaluator.parameters.rlwe_q - bool_evaluator.rlweq_by8
                    };
                    let n = measure_noise_lwe(
                        &lwe_out,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                        &ideal_m,
                    );
                    let v = decrypt_lwe(
                        &lwe_out,
                        ideal_client_key.sk_rlwe.values(),
                        &bool_evaluator.rlwe_modop,
                    );
                    (n, v)
                };
                dbg!(m0, m1, m_expected);
                println!(
                    "ct0 (noise, message): {:?}  \n ct1 (noise, message): {:?} \n PBS (noise, message): {:?}", noise0, noise1, noise_out
                );
            }

            // multi-party decrypt
            let decryption_shares = parties
                .iter()
                .map(|k| bool_evaluator.multi_party_decryption_share(&lwe_out, k))
                .collect_vec();
            let m_back = bool_evaluator.multi_party_decrypt(&decryption_shares, &lwe_out);

            // let m_back = bool_evaluator.sk_decrypt(&lwe_out, &ideal_client_key);

            assert!(m_expected == m_back, "Expected {m_expected}, got {m_back}");
            m1 = m0;
            m0 = m_expected;

            lwe1 = lwe0;
            lwe0 = lwe_out;
        }
    }

    struct Stats<T> {
        samples: Vec<T>,
    }

    impl<T: PrimInt + FromPrimitive + Debug> Stats<T>
    where
        // T: for<'a> Sum<&'a T>,
        T: for<'a> std::iter::Sum<&'a T> + std::iter::Sum<T>,
    {
        fn mean(&self) -> f64 {
            self.samples.iter().sum::<T>().to_f64().unwrap() / (self.samples.len() as f64)
        }

        fn std_dev(&self) -> f64 {
            let mean = self.mean();

            // diff
            let diff_sq = self
                .samples
                .iter()
                .map(|v| {
                    let t = v.to_f64().unwrap() - mean;
                    t * t
                })
                .into_iter()
                .sum::<f64>();

            (diff_sq / (self.samples.len() as f64)).sqrt()
        }

        fn add_more(&mut self, values: &[T]) {
            self.samples.extend(values.iter());
        }
    }

    #[test]
    fn tester() {
        // pub(super) const TEST_MP_BOOL_PARAMS: BoolParameters<u64> =
        // BoolParameters::<u64> {     rlwe_q: 1152921504606830593,
        //     rlwe_logq: 60,
        //     lwe_q: 1 << 20,
        //     lwe_logq: 20,
        //     br_q: 1 << 11,
        //     rlwe_n: 1 << 11,
        //     lwe_n: 500,
        //     d_rgsw: 4,
        //     logb_rgsw: 12,
        //     d_lwe: 5,
        //     logb_lwe: 4,
        //     g: 5,
        //     w: 1,
        // };

        let bool_evaluator =
            BoolEvaluator::<Vec<Vec<u64>>, u64, NttBackendU64, ModularOpsU64>::new(MP_BOOL_PARAMS);

        // let (_, collective_pk, _, _, server_key_eval, ideal_client_key) =
        //     _multi_party_all_keygen(&bool_evaluator, 20);
        let no_of_parties = 2;
        let lwe_q = bool_evaluator.parameters.lwe_q;
        let rlwe_q = bool_evaluator.parameters.rlwe_q;
        let d_rgsw = bool_evaluator.parameters.d_rgsw;
        let lwe_logq = bool_evaluator.parameters.lwe_logq;
        let lwe_n = bool_evaluator.parameters.lwe_n;
        let rlwe_n = bool_evaluator.parameters.rlwe_n;
        let lwe_modop = &bool_evaluator.lwe_modop;
        let rlwe_nttop = &bool_evaluator.rlwe_nttop;
        let rlwe_modop = &bool_evaluator.rlwe_modop;
        let rlwe_decomposer = &bool_evaluator.decomposer_rlwe;
        let rlwe_gadget_vector = rlwe_decomposer.gadget_vector();

        let parties = (0..no_of_parties)
            .map(|_| bool_evaluator.client_key())
            .collect_vec();

        let ideal_client_key = {
            let mut ideal_rlwe_sk = vec![0i32; bool_evaluator.rlwe_n()];
            parties.iter().for_each(|k| {
                izip!(ideal_rlwe_sk.iter_mut(), k.sk_rlwe.values()).for_each(|(ideal_i, s_i)| {
                    *ideal_i = *ideal_i + s_i;
                });
            });
            let mut ideal_lwe_sk = vec![0i32; bool_evaluator.lwe_n()];
            parties.iter().for_each(|k| {
                izip!(ideal_lwe_sk.iter_mut(), k.sk_lwe.values()).for_each(|(ideal_i, s_i)| {
                    *ideal_i = *ideal_i + s_i;
                });
            });

            ClientKey {
                sk_lwe: LweSecret {
                    values: ideal_lwe_sk,
                },
                sk_rlwe: RlweSecret {
                    values: ideal_rlwe_sk,
                },
            }
        };

        // check noise in freshly encrypted RLWE ciphertext (ie var_fresh)
        if false {
            let mut rng = DefaultSecureRng::new();
            let mut check = Stats { samples: vec![] };
            for _ in 0..10 {
                // generate a new collective public key
                let mut pk_cr_seed = [0u8; 32];
                rng.fill_bytes(&mut pk_cr_seed);
                let public_key_share = parties
                    .iter()
                    .map(|k| bool_evaluator.multi_party_public_key_share(pk_cr_seed, k))
                    .collect_vec();
                let collective_pk =
                    PublicKey::<Vec<Vec<u64>>, DefaultSecureRng, ModularOpsU64>::from(
                        public_key_share.as_slice(),
                    );

                let m = vec![0u64; rlwe_n];
                let mut rlwe_ct = vec![vec![0u64; rlwe_n]; 2];
                public_key_encrypt_rlwe(
                    &mut rlwe_ct,
                    &collective_pk.key,
                    &m,
                    rlwe_modop,
                    rlwe_nttop,
                    &mut rng,
                );

                let mut m_back = vec![0u64; rlwe_n];
                decrypt_rlwe(
                    &rlwe_ct,
                    ideal_client_key.sk_rlwe.values(),
                    &mut m_back,
                    rlwe_nttop,
                    rlwe_modop,
                );

                rlwe_modop.elwise_sub_mut(m_back.as_mut_slice(), m.as_slice());

                check.add_more(Vec::<i64>::try_convert_from(&m_back, &rlwe_q).as_slice());
            }

            println!("Public key Std: {}", check.std_dev().abs().log2());
        }

        if true {
            // Generate server key shares
            let mut rng = DefaultSecureRng::new();
            let mut pk_cr_seed = [0u8; 32];
            rng.fill_bytes(&mut pk_cr_seed);
            let public_key_share = parties
                .iter()
                .map(|k| bool_evaluator.multi_party_public_key_share(pk_cr_seed, k))
                .collect_vec();
            let collective_pk = PublicKey::<Vec<Vec<u64>>, DefaultSecureRng, ModularOpsU64>::from(
                public_key_share.as_slice(),
            );
            let pbs_cr_seed = [0u8; 32];
            rng.fill_bytes(&mut pk_cr_seed);
            let server_key_shares = parties
                .iter()
                .map(|k| {
                    bool_evaluator.multi_party_sever_key_share(pbs_cr_seed, &collective_pk.key, k)
                })
                .collect_vec();

            let seeded_server_key =
                aggregate_multi_party_server_key_shares::<_, _, _, ModularOpsU64, NttBackendU64>(
                    &server_key_shares,
                    rlwe_decomposer,
                );

            // Check noise in RGSW ciphertexts of ideal LWE secret elements
            if true {
                let mut check = Stats { samples: vec![] };
                izip!(
                    ideal_client_key.sk_lwe.values.iter(),
                    seeded_server_key.rgsw_cts.iter()
                )
                .for_each(|(s_i, rgsw_ct_i)| {
                    // X^{s[i]}
                    let mut m_si = vec![0u64; rlwe_n];
                    let s_i = *s_i * (bool_evaluator.embedding_factor as i32);
                    if s_i < 0 {
                        m_si[rlwe_n - (s_i.abs() as usize)] = rlwe_q - 1;
                    } else {
                        m_si[s_i as usize] = 1;
                    }

                    _measure_noise_rgsw(
                        &rgsw_ct_i,
                        &m_si,
                        ideal_client_key.sk_rlwe.values(),
                        &rlwe_gadget_vector,
                        rlwe_q,
                    );

                    // RLWE(-sm)
                    let mut neg_s_eval =
                        Vec::<u64>::try_convert_from(ideal_client_key.sk_rlwe.values(), &rlwe_q);
                    rlwe_modop.elwise_neg_mut(&mut neg_s_eval);
                    rlwe_nttop.forward(&mut neg_s_eval);
                    for j in 0..rlwe_decomposer.d() {
                        // -s[X]*X^{s_lwe[i]}*B_j
                        let mut m_ideal = m_si.clone();
                        rlwe_nttop.forward(m_ideal.as_mut_slice());
                        rlwe_modop.elwise_mul_mut(m_ideal.as_mut_slice(), neg_s_eval.as_slice());
                        rlwe_nttop.backward(m_ideal.as_mut_slice());
                        rlwe_modop
                            .elwise_scalar_mul_mut(m_ideal.as_mut_slice(), &rlwe_gadget_vector[j]);

                        // RLWE(-s*X^{s_lwe[i]}*B_j)
                        let mut rlwe_ct = vec![vec![0u64; rlwe_n]; 2];
                        rlwe_ct[0].copy_from_slice(&rgsw_ct_i[j]);
                        rlwe_ct[1].copy_from_slice(&rgsw_ct_i[j + rlwe_decomposer.d()]);

                        let mut m_back = vec![0u64; rlwe_n];
                        decrypt_rlwe(
                            &rlwe_ct,
                            ideal_client_key.sk_rlwe.values(),
                            &mut m_back,
                            rlwe_nttop,
                            rlwe_modop,
                        );

                        // diff
                        rlwe_modop.elwise_sub_mut(&mut m_back, &m_ideal);
                        check.add_more(&Vec::<i64>::try_convert_from(&m_back, &rlwe_q));
                    }

                    // RLWE'(m)
                    for j in 0..rlwe_decomposer.d() {
                        // X^{s_lwe[i]}*B_j
                        let mut m_ideal = m_si.clone();
                        rlwe_modop
                            .elwise_scalar_mul_mut(m_ideal.as_mut_slice(), &rlwe_gadget_vector[j]);

                        // RLWE(X^{s_lwe[i]}*B_j)
                        let mut rlwe_ct = vec![vec![0u64; rlwe_n]; 2];
                        rlwe_ct[0].copy_from_slice(&rgsw_ct_i[j + (2 * rlwe_decomposer.d())]);
                        rlwe_ct[1].copy_from_slice(&rgsw_ct_i[j + (3 * rlwe_decomposer.d())]);

                        let mut m_back = vec![0u64; rlwe_n];
                        decrypt_rlwe(
                            &rlwe_ct,
                            ideal_client_key.sk_rlwe.values(),
                            &mut m_back,
                            rlwe_nttop,
                            rlwe_modop,
                        );

                        // diff
                        rlwe_modop.elwise_sub_mut(&mut m_back, &m_ideal);
                        check.add_more(&Vec::<i64>::try_convert_from(&m_back, &rlwe_q));
                    }
                });
                println!(
                    "RGSW Std: {} {} ;; max={}",
                    check.mean(),
                    check.std_dev().abs().log2(),
                    check.samples.iter().max().unwrap()
                );
            }

            // check noise in RLWE x RGSW(X^{s_i}) where RGSW is accunulated RGSW ciphertext
            if false {
                let mut check = Stats { samples: vec![] };
                // server key in Evaluation domain
                let server_key_eval_domain =
                    ServerKeyEvaluationDomain::<_, DefaultSecureRng, NttBackendU64>::from(
                        &seeded_server_key,
                    );
                izip!(
                    ideal_client_key.sk_lwe.values(),
                    seeded_server_key.rgsw_cts.iter()
                )
                .for_each(|(s_i, rgsw_ct_i)| {
                    let mut rgsw_ct_i = rgsw_ct_i.clone();
                    rgsw_ct_i
                        .iter_mut()
                        .for_each(|ri| rlwe_nttop.forward(ri.as_mut()));

                    let mut m = vec![0u64; rlwe_n];
                    RandomUniformDist::random_fill(&mut rng, &rlwe_q, m.as_mut_slice());
                    let mut rlwe_ct = vec![vec![0u64; rlwe_n]; 2];
                    public_key_encrypt_rlwe(
                        &mut rlwe_ct,
                        &collective_pk.key,
                        &m,
                        rlwe_modop,
                        rlwe_nttop,
                        &mut rng,
                    );

                    // RLWE(m*X^{s[i]}) = RLWE(m) x RGSW(X^{s[i]})
                    let mut rlwe_after = RlweCiphertext::<_, DefaultSecureRng>::from_raw(
                        vec![vec![0u64; rlwe_n], m.clone()],
                        true,
                    );
                    // let mut rlwe_after =
                    //     RlweCiphertext::<_, DefaultSecureRng>::from_raw(rlwe_ct.clone(), false);
                    let mut scratch = vec![vec![0u64; rlwe_n]; rlwe_decomposer.d() + 2];
                    rlwe_by_rgsw(
                        &mut rlwe_after,
                        &rgsw_ct_i,
                        &mut scratch,
                        rlwe_decomposer,
                        rlwe_nttop,
                        rlwe_modop,
                    );

                    // m1 = X^{s[i]}
                    let mut m1 = vec![0u64; rlwe_n];
                    let s_i = *s_i * (bool_evaluator.embedding_factor as i32);
                    if s_i < 0 {
                        m1[rlwe_n - (s_i.abs() as usize)] = rlwe_q - 1;
                    } else {
                        m1[s_i as usize] = 1;
                    }

                    // (m+e) * m1
                    let mut m_plus_e_times_m1 = m.clone();
                    // decrypt_rlwe(
                    //     &rlwe_ct,
                    //     ideal_client_key.sk_rlwe.values(),
                    //     &mut m_plus_e_times_m1,
                    //     rlwe_nttop,
                    //     rlwe_modop,
                    // );
                    rlwe_nttop.forward(m_plus_e_times_m1.as_mut_slice());
                    rlwe_nttop.forward(m1.as_mut_slice());
                    rlwe_modop.elwise_mul_mut(m_plus_e_times_m1.as_mut_slice(), m1.as_slice());
                    rlwe_nttop.backward(m_plus_e_times_m1.as_mut_slice());

                    // Resulting RLWE ciphertext will equal: (m0m1 + em1) + e_{rlsw x rgsw}.
                    // Hence, resulting rlwe ciphertext will have error em1 + e_{rlwe x rgsw}.
                    // Here we're only concerned with e_{rlwe x rgsw}, that is noise caused due to
                    // RLWExRGSW. Also note, in practice m1 is a monomial, for ex, X^{s_{i}}, for
                    // some i and var(em1) = var(e).
                    let mut m_plus_e_times_m1_more_e = vec![0u64; rlwe_n];
                    decrypt_rlwe(
                        &rlwe_after,
                        ideal_client_key.sk_rlwe.values(),
                        &mut m_plus_e_times_m1_more_e,
                        rlwe_nttop,
                        rlwe_modop,
                    );

                    // diff
                    rlwe_modop.elwise_sub_mut(
                        m_plus_e_times_m1_more_e.as_mut_slice(),
                        m_plus_e_times_m1.as_slice(),
                    );

                    let noise = measure_noise(
                        &rlwe_after,
                        &m_plus_e_times_m1,
                        rlwe_nttop,
                        rlwe_modop,
                        ideal_client_key.sk_rlwe.values(),
                    );
                    print!("NOISE: {}", noise);

                    check.add_more(&Vec::<i64>::try_convert_from(
                        &m_plus_e_times_m1_more_e,
                        &rlwe_q,
                    ));
                });
                println!(
                    "RLWE x RGSW, where RGSW has noise var_brk, std: {} {}",
                    check.std_dev(),
                    check.std_dev().abs().log2()
                )
            }
        }

        // Check noise in fresh RGSW ciphertexts, ie X^{s_j[i]}, must equal noise in
        // fresh RLWE ciphertext
        if true {}
        // test LWE ksk from RLWE -> LWE
        // if false {
        //     let logp = 2;
        //     let mut rng = DefaultSecureRng::new();

        //     let m = 1;
        //     let encoded_m = m << (lwe_logq - logp);

        //     // Encrypt
        //     let mut lwe_ct = vec![0u64; rlwe_n + 1];
        //     encrypt_lwe(
        //         &mut lwe_ct,
        //         &encoded_m,
        //         ideal_client_key.sk_rlwe.values(),
        //         lwe_modop,
        //         &mut rng,
        //     );

        //     // key switch
        //     let lwe_decomposer = &bool_evaluator.decomposer_lwe;
        //     let mut lwe_out = vec![0u64; lwe_n + 1];
        //     lwe_key_switch(
        //         &mut lwe_out,
        //         &lwe_ct,
        //         &server_key_eval.lwe_ksk,
        //         lwe_modop,
        //         lwe_decomposer,
        //     );

        //     let encoded_m_back = decrypt_lwe(&lwe_out,
        // ideal_client_key.sk_lwe.values(), lwe_modop);     let m_back
        // =         ((encoded_m_back as f64 * (1 << logp) as f64) /
        // (lwe_q as f64)).round() as u64;     dbg!(m_back, m);

        //     let noise = measure_noise_lwe(
        //         &lwe_out,
        //         ideal_client_key.sk_lwe.values(),
        //         lwe_modop,
        //         &encoded_m,
        //     );

        //     println!("Noise: {noise}");
        // }

        // Measure noise in RGSW ciphertexts of ideal LWE secrets
        // if true {
        //     let gadget_vec = gadget_vector(
        //         bool_evaluator.parameters.rlwe_logq,
        //         bool_evaluator.parameters.logb_rgsw,
        //         bool_evaluator.parameters.d_rgsw,
        //     );

        //     for i in 0..20 {
        //         // measure noise in RGSW(s[i])
        //         let si =
        //             ideal_client_key.sk_lwe.values[i] *
        // (bool_evaluator.embedding_factor as i32);         let mut
        // si_poly = vec![0u64; rlwe_n];         if si < 0 {
        //             si_poly[rlwe_n - (si.abs() as usize)] = rlwe_q - 1;
        //         } else {
        //             si_poly[(si.abs() as usize)] = 1;
        //         }

        //         let mut rgsw_si = server_key_eval.rgsw_cts[i].clone();
        //         rgsw_si
        //             .iter_mut()
        //             .for_each(|ri| rlwe_nttop.backward(ri.as_mut()));

        //         println!("####### Noise in RGSW(X^s_{i}) #######");
        //         _measure_noise_rgsw(
        //             &rgsw_si,
        //             &si_poly,
        //             ideal_client_key.sk_rlwe.values(),
        //             &gadget_vec,
        //             rlwe_q,
        //         );
        //         println!("####### ##################### #######");
        //     }
        // }

        // // measure noise grwoth in RLWExRGSW
        // if true {
        //     let mut rng = DefaultSecureRng::new();
        //     let mut carry_m = vec![0u64; rlwe_n];
        //     RandomUniformDist::random_fill(&mut rng, &rlwe_q,
        // carry_m.as_mut_slice());

        //     // RGSW(carrym)
        //     let trivial_rlwect = vec![vec![0u64; rlwe_n], carry_m.clone()];
        //     let mut rlwe_ct = RlweCiphertext::<_,
        // DefaultSecureRng>::from_raw(trivial_rlwect, true);

        //     let mut scratch_matrix_dplus2_ring = vec![vec![0u64; rlwe_n];
        // d_rgsw + 2];     let mul_mod =
        //         |v0: &u64, v1: &u64| (((*v0 as u128 * *v1 as u128) % (rlwe_q as u128)) as u64);

        //     for i in 0..bool_evaluator.parameters.lwe_n {
        //         rlwe_by_rgsw(
        //             &mut rlwe_ct,
        //             server_key_eval.rgsw_ct_lwe_si(i),
        //             &mut scratch_matrix_dplus2_ring,
        //             rlwe_decomposer,
        //             rlwe_nttop,
        //             rlwe_modop,
        //         );

        //         // carry_m[X] * s_i[X]
        //         let si =
        //             ideal_client_key.sk_lwe.values[i] *
        // (bool_evaluator.embedding_factor as i32);         let mut
        // si_poly = vec![0u64; rlwe_n];         if si < 0 {
        //             si_poly[rlwe_n - (si.abs() as usize)] = rlwe_q - 1;
        //         } else {
        //             si_poly[(si.abs() as usize)] = 1;
        //         }
        //         carry_m = negacyclic_mul(&carry_m, &si_poly, mul_mod,
        // rlwe_q);

        //         let noise = measure_noise(
        //             &rlwe_ct,
        //             &carry_m,
        //             rlwe_nttop,
        //             rlwe_modop,
        //             ideal_client_key.sk_rlwe.values(),
        //         );
        //         println!("Noise RLWE(carry_m) accumulating {i}^th secret
        // monomial: {noise}");     }
        // }

        // // Check galois keys
        // if false {
        //     let g = bool_evaluator.g() as isize;
        //     let mut rng = DefaultSecureRng::new();
        //     let mut scratch_matrix_dplus2_ring = vec![vec![0u64; rlwe_n];
        // d_rgsw + 2];     for i in [g, -g] {
        //         let mut m = vec![0u64; rlwe_n];
        //         RandomUniformDist::random_fill(&mut rng, &rlwe_q,
        // m.as_mut_slice());         let mut rlwe_ct = {
        //             let mut data = vec![vec![0u64; rlwe_n]; 2];
        //             public_key_encrypt_rlwe(
        //                 &mut data,
        //                 &collective_pk.key,
        //                 &m,
        //                 rlwe_modop,
        //                 rlwe_nttop,
        //                 &mut rng,
        //             );
        //             RlweCiphertext::<_, DefaultSecureRng>::from_raw(data,
        // false)         };

        //         let auto_key = server_key_eval.galois_key_for_auto(i);
        //         let (auto_map_index, auto_map_sign) =
        // generate_auto_map(rlwe_n, i);         galois_auto(
        //             &mut rlwe_ct,
        //             auto_key,
        //             &mut scratch_matrix_dplus2_ring,
        //             &auto_map_index,
        //             &auto_map_sign,
        //             rlwe_modop,
        //             rlwe_nttop,
        //             rlwe_decomposer,
        //         );

        //         // send m(X) -> m(X^i)
        //         let mut m_k = vec![0u64; rlwe_n];
        //         izip!(m.iter(), auto_map_index.iter(),
        // auto_map_sign.iter()).for_each(             |(mi, to_index, to_sign)|
        // {                 if !to_sign {
        //                     m_k[*to_index] = rlwe_q - *mi;
        //                 } else {
        //                     m_k[*to_index] = *mi;
        //                 }
        //             },
        //         );

        //         // measure noise
        //         let noise = measure_noise(
        //             &rlwe_ct,
        //             &m_k,
        //             rlwe_nttop,
        //             rlwe_modop,
        //             ideal_client_key.sk_rlwe.values(),
        //         );

        //         println!("Noise after auto k={i}: {noise}");
        //     }
        // }
    }

    fn test_2() {
        let bool_evaluator =
            BoolEvaluator::<Vec<Vec<u64>>, u64, NttBackendU64, ModularOpsU64>::new(SP_BOOL_PARAMS);
    }
}
