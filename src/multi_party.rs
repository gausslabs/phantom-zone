use crate::{
    backend::VectorOps,
    ntt::{self, Ntt},
    random::{NewWithSeed, RandomGaussianDist, RandomUniformDist},
    utils::TryConvertFrom,
    Matrix, Row, RowEntity, RowMut,
};

fn public_key_share<
    R: Row + RowMut + RowEntity,
    S,
    ModOp: VectorOps<Element = R::Element>,
    NttOp: Ntt<Element = R::Element>,
    Rng: RandomGaussianDist<[R::Element], Parameters = R::Element>
        + NewWithSeed
        + RandomUniformDist<[R::Element], Parameters = R::Element>,
>(
    share_out: &mut R,
    s_i: &[S],
    modop: &ModOp,
    nttop: &NttOp,
    crp_seed: Rng::Seed,
    rng: &mut Rng,
) where
    R: TryConvertFrom<[S], Parameters = R::Element>,
{
    let ring_size = share_out.as_ref().len();
    assert!(s_i.len() == ring_size);

    let q = modop.modulus();

    // a*s
    let mut a = R::zeros(ring_size);
    let mut p_rng = Rng::new_with_seed(crp_seed);
    RandomUniformDist::random_fill(&mut p_rng, &q, a.as_mut());
    nttop.forward(a.as_mut());
    let mut s = R::try_convert_from(s_i, &q);
    nttop.forward(s.as_mut());
    modop.elwise_mul_mut(s.as_mut(), a.as_ref());
    nttop.backward(s.as_mut());

    RandomGaussianDist::random_fill(rng, &q, share_out.as_mut());
    modop.elwise_add_mut(share_out.as_mut(), s.as_ref()); // s*e + e
}

fn rlwe_galois_auto_key_share<M: Matrix>() {}
