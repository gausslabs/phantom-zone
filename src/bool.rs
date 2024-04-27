use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
    marker::PhantomData,
    thread::panicking,
};

use itertools::{izip, partition, Itertools};
use num_traits::{FromPrimitive, Num, One, PrimInt, ToPrimitive, WrappingSub, Zero};

use crate::{
    backend::{ArithmeticOps, ModInit, ModularOpsU64, VectorOps},
    decomposer::{gadget_vector, Decomposer, DefaultDecomposer, NumInfo},
    lwe::{decrypt_lwe, encrypt_lwe, lwe_key_switch, lwe_ksk_keygen, measure_noise_lwe, LweSecret},
    ntt::{Ntt, NttBackendU64, NttInit},
    random::{DefaultSecureRng, RandomGaussianDist, RandomUniformDist},
    rgsw::{
        decrypt_rlwe, encrypt_rgsw, galois_auto, galois_key_gen, generate_auto_map, rlwe_by_rgsw,
        IsTrivial, RlweCiphertext, RlweSecret,
    },
    utils::{generate_prime, mod_exponent, TryConvertFrom, WithLocal},
    Matrix, MatrixEntity, MatrixMut, Row, RowEntity, RowMut, Secret,
};

thread_local! {
    pub(crate) static CLIENT_KEY: RefCell<ClientKey> = RefCell::new(ClientKey::random());
}

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

impl WithLocal for ClientKey {
    fn with_local<F, R>(func: F) -> R
    where
        F: Fn(&Self) -> R,
    {
        CLIENT_KEY.with_borrow(|client_key| func(client_key))
    }

    fn with_local_mut<F, R>(func: F) -> R
    where
        F: Fn(&mut Self) -> R,
    {
        CLIENT_KEY.with_borrow_mut(|client_key| func(client_key))
    }
}

fn set_client_key(key: &ClientKey) {
    ClientKey::with_local_mut(|k| *k = key.clone())
}

struct ServerKey<M> {
    /// Rgsw cts of LWE secret elements
    rgsw_cts: Vec<M>,
    /// Galois keys
    galois_keys: HashMap<isize, M>,
    /// LWE ksk to key switching LWE ciphertext from RLWE secret to LWE secret
    lwe_ksk: M,
}

//FIXME(Jay): Figure out a way for BoolEvaluator to have access to ServerKey
// via a pointer and implement PbsKey for BoolEvaluator instead of ServerKey
// directly
impl<M: Matrix> PbsKey for ServerKey<M> {
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

struct BoolParameters<El> {
    rlwe_q: El,
    rlwe_logq: usize,
    lwe_q: El,
    lwe_logq: usize,
    br_q: usize,
    rlwe_n: usize,
    lwe_n: usize,
    d_rgsw: usize,
    logb_rgsw: usize,
    d_lwe: usize,
    logb_lwe: usize,
    g: usize,
    w: usize,
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
    rlwe_auto_maps: Vec<(Vec<usize>, Vec<bool>)>,
    scratch_lwen_plus1: M::R,
    scratch_dplus2_ring: M,
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
    M::R: TryConvertFrom<[i32], Parameters = M::MatElement> + RowEntity,
    M: TryConvertFrom<[i32], Parameters = M::MatElement>,
    <M as Matrix>::R: RowMut,
    DefaultSecureRng: RandomGaussianDist<[M::MatElement], Parameters = M::MatElement>
        + RandomGaussianDist<M::MatElement, Parameters = M::MatElement>
        + RandomUniformDist<[M::MatElement], Parameters = M::MatElement>,
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
        let qby16 = q >> 4;
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
        // Rotate and negate by q/16
        let mut tmp = M::R::zeros(qby2);
        tmp.as_mut()[..qby2 - qby16].copy_from_slice(&nand_test_vec.as_ref()[qby16..]);
        tmp.as_mut()[qby2 - qby16..].copy_from_slice(&nand_test_vec.as_ref()[..qby16]);
        tmp.as_mut()[qby2 - qby16..].iter_mut().for_each(|v| {
            *v = parameters.rlwe_q - *v;
        });
        let nand_test_vec = tmp;

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

        // create srcatch spaces
        let scratch_lwen_plus1 = M::R::zeros(parameters.lwe_n + 1);
        let scratch_dplus2_ring = M::zeros(parameters.d_rgsw + 2, parameters.rlwe_n);

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
            rlwe_auto_maps,
            scratch_lwen_plus1,
            scratch_dplus2_ring,
            _phantom: PhantomData,
        }
    }

    fn client_key(&self) -> ClientKey {
        let sk_lwe = LweSecret::random(self.parameters.lwe_n >> 1, self.parameters.lwe_n);
        let sk_rlwe = RlweSecret::random(self.parameters.rlwe_n >> 1, self.parameters.rlwe_n);
        ClientKey { sk_rlwe, sk_lwe }
    }

    fn server_key(&self, client_key: &ClientKey) -> ServerKey<M> {
        let sk_rlwe = &client_key.sk_rlwe;
        let sk_lwe = &client_key.sk_lwe;

        let d_rgsw_gadget_vec = gadget_vector(
            self.parameters.rlwe_logq,
            self.parameters.logb_rgsw,
            self.parameters.d_rgsw,
        );

        // generate galois key -g, g
        let mut galois_keys = HashMap::new();
        let g = self.parameters.g as isize;
        for i in [g, -g] {
            let gk = DefaultSecureRng::with_local_mut(|rng| {
                let mut ksk_out = M::zeros(self.parameters.d_rgsw * 2, self.parameters.rlwe_n);
                galois_key_gen(
                    &mut ksk_out,
                    sk_rlwe,
                    i,
                    &d_rgsw_gadget_vec,
                    &self.rlwe_modop,
                    &self.rlwe_nttop,
                    rng,
                );
                ksk_out
            });

            galois_keys.insert(i, gk);
        }

        // generate rgsw ciphertexts RGSW(si) where si is i^th LWE secret element
        let ring_size = self.parameters.rlwe_n;
        let rlwe_q = self.parameters.rlwe_q;
        let rgsw_cts = sk_lwe
            .values()
            .iter()
            .map(|si| {
                // X^{si}; assume |emebedding_factor * si| < N
                let mut m = M::zeros(1, ring_size);
                let si = (self.embedding_factor as i32) * si;
                // dbg!(si);
                if si < 0 {
                    // X^{-i} = X^{2N - i} = -X^{N-i}
                    m.set(
                        0,
                        ring_size - (si.abs() as usize),
                        rlwe_q - M::MatElement::one(),
                    );
                } else {
                    // X^{i}
                    m.set(0, (si.abs() as usize), M::MatElement::one());
                }
                self.rlwe_nttop.forward(m.get_row_mut(0));

                let rgsw_si = DefaultSecureRng::with_local_mut(|rng| {
                    let mut rgsw_si = M::zeros(self.parameters.d_rgsw * 4, ring_size);
                    encrypt_rgsw(
                        &mut rgsw_si,
                        &m,
                        &d_rgsw_gadget_vec,
                        sk_rlwe,
                        &self.rlwe_modop,
                        &self.rlwe_nttop,
                        rng,
                    );
                    rgsw_si
                });
                rgsw_si
            })
            .collect_vec();

        // LWE KSK from RLWE secret s -> LWE secret z
        let d_lwe_gadget = gadget_vector(
            self.parameters.lwe_logq,
            self.parameters.logb_lwe,
            self.parameters.d_lwe,
        );
        let mut lwe_ksk = DefaultSecureRng::with_local_mut(|rng| {
            let mut out = M::zeros(self.parameters.d_lwe * ring_size, self.parameters.lwe_n + 1);
            lwe_ksk_keygen(
                &sk_rlwe.values(),
                &sk_lwe.values(),
                &mut out,
                &d_lwe_gadget,
                &self.lwe_modop,
                rng,
            );
            out
        });

        ServerKey {
            rgsw_cts,
            galois_keys,
            lwe_ksk,
        }
    }

    /// TODO(Jay): Fetch client key from thread local
    pub fn encrypt(&self, m: bool, client_key: &ClientKey) -> M::R {
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

    pub fn decrypt(&self, lwe_ct: &M::R, client_key: &ClientKey) -> bool {
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

    pub fn nand(
        &self,
        c0: &M::R,
        c1: &M::R,
        server_key: &ServerKey<M>,
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
        // &(self.rlweq_by8));     println!("c0 noise: {c0_noise}; c1 noise:
        // {c1_noise}"); });

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
        c_out.as_mut()[0] = modop.add(&c_out.as_ref()[0], &self.rlweq_by8);

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
    // FIXME(Jay):  TryConvertFrom<[i32], Parameters = M::MatElement> are only needed for
    // debugging purposes
    <M as Matrix>::R: RowMut + TryConvertFrom<[i32], Parameters = M::MatElement>,
    M::MatElement: PrimInt + ToPrimitive + FromPrimitive + One + Copy + Zero + Display,
{
    let rlwe_q = parameters.rlwe_q();
    let lwe_q = parameters.lwe_q();
    let br_q = parameters.br_q();
    let rlwe_qf64 = rlwe_q.to_f64().unwrap();
    let lwe_qf64 = lwe_q.to_f64().unwrap();
    let br_qf64 = br_q.to_f64().unwrap();
    let rlwe_n = parameters.rlwe_n();

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
    let mut trivial_rlwe_test_poly = RlweCiphertext(M::zeros(2, rlwe_n), true);
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
    // println!("v: {v}, odd_v: {odd_v}, lwe_q:{lwe_q}, br_q:{br_q}");
    let odd_v = (((v * to_q) / (from_q)).floor()).to_usize().unwrap();
    // println!(
    //     "v: {v}, odd_v: {odd_v}, returned_oddv: {},lwe_q:{from_q}, br_q:{to_q}",
    //     odd_v + ((odd_v & 1) ^ 1)
    // );
    //TODO(Jay): check correctness of this
    odd_v + ((odd_v & 1) ^ 1)
}

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
    pub(crate) ct_lwe_q_mod: M::R,
    pub(crate) ct_lwe_q_mod_after_ksk: M::R,
    pub(crate) ct_br_q_mod: Vec<u64>,
}

impl PBSTracer<Vec<Vec<u64>>> {
    fn trace(&self, parameters: &BoolParameters<u64>, client_key: &ClientKey, expected_m: bool) {
        let lwe_q = parameters.lwe_q;
        let lwe_qby8 = ((lwe_q as f64) / 8.0).round() as u64;
        let expected_m_lweq = if expected_m {
            lwe_qby8
        } else {
            lwe_q - lwe_qby8
        };
        let modop_lweq = ModularOpsU64::new(lwe_q);
        // noise after mod down Q -> Q_ks
        let noise0 = {
            measure_noise_lwe(
                &self.ct_lwe_q_mod,
                client_key.sk_rlwe.values(),
                &modop_lweq,
                &expected_m_lweq,
            )
        };

        // noise after key switch from RLWE -> LWE
        let noise1 = {
            measure_noise_lwe(
                &self.ct_lwe_q_mod_after_ksk,
                client_key.sk_lwe.values(),
                &modop_lweq,
                &expected_m_lweq,
            )
        };

        // noise after mod down odd from Q_ks -> q
        let br_q = parameters.br_q as u64;
        let expected_m_brq = if expected_m {
            br_q >> 3
        } else {
            br_q - (br_q >> 3)
        };
        let modop_br_q = ModularOpsU64::new(br_q);
        let noise2 = {
            measure_noise_lwe(
                &self.ct_br_q_mod,
                client_key.sk_lwe.values(),
                &modop_br_q,
                &expected_m_brq,
            )
        };

        println!(
            "
            m: {expected_m},
            Noise after mod down Q -> Q_ks: {noise0},
            Noise after key switch from RLWE -> LWE: {noise1},
            Noise after mod dwon Q_ks -> q: {noise2}
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
    use crate::{backend::ModularOpsU64, ntt::NttBackendU64, random::DEFAULT_RNG};

    use super::*;

    const SP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
        rlwe_q: 268369921u64,
        rlwe_logq: 28,
        lwe_q: 1 << 16,
        lwe_logq: 16,
        br_q: 1 << 10,
        rlwe_n: 1 << 10,
        lwe_n: 493,
        d_rgsw: 3,
        logb_rgsw: 8,
        d_lwe: 3,
        logb_lwe: 4,
        g: 5,
        w: 1,
    };

    // #[test]
    // fn trial() {
    //     dbg!(generate_prime(28, 1 << 11, 1 << 28));
    // }

    #[test]
    fn encrypt_decrypt_works() {
        // let prime = generate_prime(32, 2 * 1024, 1 << 32);
        // dbg!(prime);
        let bool_evaluator =
            BoolEvaluator::<Vec<Vec<u64>>, u64, NttBackendU64, ModularOpsU64>::new(SP_BOOL_PARAMS);
        let client_key = bool_evaluator.client_key();
        // let sever_key = bool_evaluator.server_key(&client_key);

        let mut m = true;
        for _ in 0..1000 {
            let lwe_ct = bool_evaluator.encrypt(m, &client_key);
            let m_back = bool_evaluator.decrypt(&lwe_ct, &client_key);
            assert_eq!(m, m_back);
            m = !m;
        }
    }

    #[test]
    fn trial12() {
        // DefaultSecureRng::with_local_mut(|r| {
        //     let rng = DefaultSecureRng::new_seeded([19u8; 32]);
        //     *r = rng;
        // });

        let bool_evaluator =
            BoolEvaluator::<Vec<Vec<u64>>, u64, NttBackendU64, ModularOpsU64>::new(SP_BOOL_PARAMS);
        // println!("{:?}", bool_evaluator.nand_test_vec);
        let client_key = bool_evaluator.client_key();
        set_client_key(&client_key);

        let server_key = bool_evaluator.server_key(&client_key);

        let mut scratch_lwen_plus1 = vec![0u64; bool_evaluator.parameters.lwe_n + 1];
        let mut scratch_matrix_dplus2_ring = vec![
            vec![0u64; bool_evaluator.parameters.rlwe_n];
            bool_evaluator.parameters.d_rgsw + 2
        ];

        let mut m0 = false;
        let mut m1 = true;
        let mut ct0 = bool_evaluator.encrypt(m0, &client_key);
        let mut ct1 = bool_evaluator.encrypt(m1, &client_key);
        for _ in 0..4 {
            let ct_back = bool_evaluator.nand(
                &ct0,
                &ct1,
                &server_key,
                &mut scratch_lwen_plus1,
                &mut scratch_matrix_dplus2_ring,
            );

            let m_out = !(m0 && m1);

            // Trace and measure PBS noise
            {
                // Trace PBS
                PBSTracer::with_local(|t| t.trace(&SP_BOOL_PARAMS, &client_key, m_out));

                // Calculate nosie in ciphertext post PBS
                let ideal = if m_out {
                    bool_evaluator.rlweq_by8
                } else {
                    bool_evaluator.rlwe_q() - bool_evaluator.rlweq_by8
                };
                let noise = measure_noise_lwe(
                    &ct_back,
                    client_key.sk_rlwe.values(),
                    &bool_evaluator.rlwe_modop,
                    &ideal,
                );
                println!("PBS noise: {noise}");
            }
            let m_back = bool_evaluator.decrypt(&ct_back, &client_key);
            assert_eq!(m_out, m_back);
            println!("----------");

            m1 = m0;
            m0 = m_out;

            ct1 = ct0;
            ct0 = ct_back;
        }
    }
}
