use crate::{
    backend::{GetModulus, VectorOps},
    ntt::Ntt,
    random::{NewWithSeed, RandomFillGaussianInModulus, RandomFillUniformInModulus},
    utils::TryConvertFrom1,
    Matrix, Row, RowEntity, RowMut,
};

pub(crate) fn public_key_share<
    R: Row + RowMut + RowEntity,
    S,
    ModOp: VectorOps<Element = R::Element> + GetModulus<Element = R::Element>,
    NttOp: Ntt<Element = R::Element>,
    Rng: RandomFillGaussianInModulus<[R::Element], ModOp::M>,
    PRng: RandomFillUniformInModulus<[R::Element], ModOp::M>,
>(
    share_out: &mut R,
    s_i: &[S],
    modop: &ModOp,
    nttop: &NttOp,
    p_rng: &mut PRng,
    rng: &mut Rng,
) where
    R: TryConvertFrom1<[S], ModOp::M>,
{
    let ring_size = share_out.as_ref().len();
    assert!(s_i.len() == ring_size);

    let q = modop.modulus();

    // sample a
    let mut a = {
        let mut a = R::zeros(ring_size);
        RandomFillUniformInModulus::random_fill(p_rng, &q, a.as_mut());
        a
    };

    // s*a
    nttop.forward(a.as_mut());
    let mut s = R::try_convert_from(s_i, &q);
    nttop.forward(s.as_mut());
    modop.elwise_mul_mut(s.as_mut(), a.as_ref());
    nttop.backward(s.as_mut());

    RandomFillGaussianInModulus::random_fill(rng, &q, share_out.as_mut());
    modop.elwise_add_mut(share_out.as_mut(), s.as_ref()); // s*e + e
}
