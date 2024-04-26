use std::{collections::HashMap, fmt::Debug, marker::PhantomData};

use itertools::Itertools;
use num_traits::{FromPrimitive, Num, One, PrimInt, ToPrimitive, Zero};

use crate::{
    backend::{ArithmeticOps, ModInit, VectorOps},
    decomposer::{gadget_vector, Decomposer, DefaultDecomposer, NumInfo},
    lwe::{decrypt_lwe, encrypt_lwe, lwe_key_switch, lwe_ksk_keygen, LweSecret},
    ntt::{Ntt, NttInit},
    random::{DefaultSecureRng, RandomGaussianDist, RandomUniformDist},
    rgsw::{encrypt_rgsw, galois_auto, galois_key_gen, rlwe_by_rgsw, IsTrivial, RlweSecret},
    utils::{generate_prime, mod_exponent, TryConvertFrom, WithLocal},
    Matrix, MatrixEntity, MatrixMut, Row, RowEntity, RowMut, Secret,
};

trait PbsKey {
    type M: Matrix;

    fn rgsw_ct_secret_el(&self, si: usize) -> &Self::M;
    fn galois_key_for_auto(&self, k: isize) -> &Self::M;
    fn auto_map_index(&self, k: isize) -> &[usize];
    fn auto_map_sign(&self, k: isize) -> &[bool];
}
trait Parameters {
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
}
struct ClientKey {
    sk_rlwe: RlweSecret,
    sk_lwe: LweSecret,
}

struct ServerKey<M> {
    /// Rgsw cts of LWE secret elements
    rgsw_cts: Vec<M>,
    /// Galois keys
    galois_keys: HashMap<isize, M>,
    /// LWE ksk to key switching LWE ciphertext from RLWE secret to LWE secret
    lwe_ksk: M,
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

struct BoolEvaluator<M, E, Ntt, ModOp> {
    parameters: BoolParameters<E>,
    decomposer_rlwe: DefaultDecomposer<E>,
    decomposer_lwe: DefaultDecomposer<E>,
    g_k_dlog_map: Vec<usize>,
    rlwe_nttop: Ntt,
    rlwe_modop: ModOp,
    lwe_modop: ModOp,
    embedding_factor: usize,

    _phantom: PhantomData<M>,
}

impl<M, NttOp, ModOp> BoolEvaluator<M, M::MatElement, NttOp, ModOp>
where
    NttOp: NttInit<Element = M::MatElement> + Ntt<Element = M::MatElement>,
    ModOp: ModInit<Element = M::MatElement>
        + ArithmeticOps<Element = M::MatElement>
        + VectorOps<Element = M::MatElement>,
    M::MatElement: PrimInt + Debug + NumInfo + FromPrimitive,
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

        BoolEvaluator {
            parameters: parameters,
            decomposer_lwe,
            decomposer_rlwe,
            g_k_dlog_map,
            embedding_factor,
            lwe_modop,
            rlwe_modop,
            rlwe_nttop,

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

    pub fn encrypt(&self, m: bool, client_key: &ClientKey) -> M::R {
        let rlwe_q_by8 =
            M::MatElement::from_f64((self.parameters.rlwe_q.to_f64().unwrap() / 8.0).round())
                .unwrap();
        let m = if m {
            // Q/8
            rlwe_q_by8
        } else {
            // -Q/8
            self.parameters.rlwe_q - rlwe_q_by8
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
            let rlwe_q_by8 =
                M::MatElement::from_f64((self.parameters.rlwe_q.to_f64().unwrap() / 8.0).round())
                    .unwrap();
            (((m + rlwe_q_by8).to_f64().unwrap() * 4.0) / self.parameters.rlwe_q.to_f64().unwrap())
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
            panic!("Incorrect bool decryption. Got m={m} expected m to be 0 or 1")
        }
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
    pbs_key: &K,
) where
    <Mmut as Matrix>::R: RowMut,
    Mmut::MatElement: Copy + Zero,
    <MT as Matrix>::R: RowMut,
{
    let q_by_2 = q / 2;

    // -(g^k)
    for i in 1..q_by_2 {
        gk_to_si[q_by_2 + i].iter().for_each(|s_index| {
            rlwe_by_rgsw(
                trivial_rlwe_test_poly,
                pbs_key.rgsw_ct_secret_el(*s_index),
                scratch_matrix_dplus2_ring,
                decomposer,
                ntt_op,
                mod_op,
            );
        });

        galois_auto(
            trivial_rlwe_test_poly,
            pbs_key.galois_key_for_auto(g),
            scratch_matrix_dplus2_ring,
            pbs_key.auto_map_index(g),
            pbs_key.auto_map_sign(g),
            mod_op,
            ntt_op,
            decomposer,
        );
    }

    // -(g^0)
    gk_to_si[q_by_2].iter().for_each(|s_index| {
        rlwe_by_rgsw(
            trivial_rlwe_test_poly,
            pbs_key.rgsw_ct_secret_el(*s_index),
            scratch_matrix_dplus2_ring,
            decomposer,
            ntt_op,
            mod_op,
        );
    });
    galois_auto(
        trivial_rlwe_test_poly,
        pbs_key.galois_key_for_auto(-g),
        scratch_matrix_dplus2_ring,
        pbs_key.auto_map_index(-g),
        pbs_key.auto_map_sign(-g),
        mod_op,
        ntt_op,
        decomposer,
    );

    // +(g^k)
    for i in 1..q_by_2 {
        gk_to_si[i].iter().for_each(|s_index| {
            rlwe_by_rgsw(
                trivial_rlwe_test_poly,
                pbs_key.rgsw_ct_secret_el(*s_index),
                scratch_matrix_dplus2_ring,
                decomposer,
                ntt_op,
                mod_op,
            );
        });

        galois_auto(
            trivial_rlwe_test_poly,
            pbs_key.galois_key_for_auto(g),
            scratch_matrix_dplus2_ring,
            pbs_key.auto_map_index(g),
            pbs_key.auto_map_sign(g),
            mod_op,
            ntt_op,
            decomposer,
        );
    }

    // +(g^0)
    gk_to_si[0].iter().for_each(|s_index| {
        rlwe_by_rgsw(
            trivial_rlwe_test_poly,
            pbs_key.rgsw_ct_secret_el(gk_to_si[q_by_2][*s_index]),
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
    MT: MatrixMut<MatElement = M::MatElement, R = M::R> + IsTrivial + MatrixEntity,
    P: Parameters<Element = M::MatElement>,
    NttOp: Ntt<Element = M::MatElement>,
    ModOp: ArithmeticOps<Element = M::MatElement> + VectorOps<Element = M::MatElement>,
    K: PbsKey<M = M>,
>(
    parameters: &P,
    test_vec: &M::R,
    lwe_in: &mut M::R,
    lwe_ksk: &M,
    scratch_lwen_plus1: &mut M::R,
    scratch_matrix_dplus2_ring: &mut M,
    modop_lweq: &ModOp,
    modop_rlweq: &ModOp,
    nttop_rlweq: &NttOp,
    pbs_key: K,
) where
    <M as Matrix>::R: RowMut,
    <MT as Matrix>::R: RowMut,
    M::MatElement: PrimInt + ToPrimitive + FromPrimitive + One + Copy + Zero,
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

    // key switch
    // let mut lwe_out = M::zeros(1, parameters.lwe_n() + 1);
    scratch_lwen_plus1.as_mut().fill(M::MatElement::zero());
    lwe_key_switch(
        scratch_lwen_plus1,
        lwe_in,
        lwe_ksk,
        modop_lweq,
        parameters.decomoposer_lwe(),
    );

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
    // X^{g*b} mod X^{q}+1
    if gb_monomial_exp > br_qby2 {
        gb_monomial_exp -= br_qby2;
        gb_monomial_sign = false
    }
    // monomial mul
    let mut trivial_rlwe_test_poly = MT::zeros(2, rlwe_n);
    if parameters.embedding_factor() == 1 {
        monomial_mul(
            test_vec.as_ref(),
            trivial_rlwe_test_poly.get_row_mut(1).as_mut(),
            gb_monomial_exp,
            gb_monomial_sign,
            br_q,
            modop_rlweq,
        );
    } else {
        // use lwe_in to store the `t = v(X) * X^{g*2} mod X^{q/2}+1` temporarily. This
        // works because q/2 < N (where N is lwe_in LWE dimension) always.
        monomial_mul(
            test_vec.as_ref(),
            &mut lwe_in.as_mut()[..br_qby2],
            gb_monomial_exp,
            gb_monomial_sign,
            br_q,
            modop_rlweq,
        );

        // emebed poly `t` in ring X^{q/2}+1 inside the bigger ring X^{N}+1
        let partb_trivial_rlwe = trivial_rlwe_test_poly.get_row_mut(1);
        lwe_in.as_ref()[..br_qby2]
            .iter()
            .enumerate()
            .for_each(|(index, v)| {
                partb_trivial_rlwe[2 * index] = *v;
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
        &pbs_key,
    );

    // sample extract
    sample_extract(lwe_in, &trivial_rlwe_test_poly, modop_rlweq, 0);
}

fn mod_switch_odd(v: f64, from_q: f64, to_q: f64) -> usize {
    let odd_v = (((v.to_f64().unwrap() * to_q) / (from_q)).floor())
        .to_usize()
        .unwrap();
    //TODO(Jay): check correctness of this
    odd_v + (odd_v ^ (usize::one()))
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

#[cfg(test)]
mod tests {
    use crate::{backend::ModularOpsU64, ntt::NttBackendU64};

    use super::*;

    const SP_BOOL_PARAMS: BoolParameters<u64> = BoolParameters::<u64> {
        rlwe_q: 4294957057u64,
        rlwe_logq: 32,
        lwe_q: 1 << 16,
        lwe_logq: 16,
        br_q: 1 << 9,
        rlwe_n: 1 << 10,
        lwe_n: 490,
        d_rgsw: 4,
        logb_rgsw: 7,
        d_lwe: 4,
        logb_lwe: 4,
        g: 5,
        w: 1,
    };

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
}
