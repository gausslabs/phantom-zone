use crate::{misc::as_slice::AsSlice, ring::SliceOps};
use core::array::from_fn;
use itertools::izip;

pub fn nega_cyclic_karatsuba_mul<T: SliceOps>(
    arith: &T,
    c: &mut [T::Elem],
    a: &[T::Elem],
    b: &[T::Elem],
) {
    nega_cyclic_karatsuba_mul_inner(arith, c, a, Some(b), |c, t0, t1| arith.slice_sub(c, t0, t1));
}

pub fn nega_cyclic_karatsuba_mul_assign<T: SliceOps>(arith: &T, b: &mut [T::Elem], a: &[T::Elem]) {
    nega_cyclic_karatsuba_mul_inner(arith, b, a, None, |c, t0, t1| arith.slice_sub(c, t0, t1));
}

pub fn nega_cyclic_karatsuba_fma<T: SliceOps>(
    arith: &T,
    c: &mut [T::Elem],
    a: &[T::Elem],
    b: &[T::Elem],
) {
    nega_cyclic_karatsuba_mul_inner(arith, c, a, Some(b), |c, t0, t1| {
        arith.slice_op(c, t0, t1, |c, t0, t1| *c = arith.add(c, &arith.sub(t0, t1)))
    });
}

fn nega_cyclic_karatsuba_mul_inner<T: SliceOps>(
    arith: &T,
    c: &mut [T::Elem],
    a: &[T::Elem],
    b: Option<&[T::Elem]>,
    f: impl Fn(&mut [T::Elem], &[T::Elem], &[T::Elem]),
) {
    assert!(a.len().is_power_of_two());
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), b.map(<[_]>::len).unwrap_or(a.len()));

    if a.len() == 1 {
        c[0] = arith.mul(&a[0], b.map(|b| &b[0]).unwrap_or_else(|| &c[0]));
        return;
    }

    let n = a.len();
    let m = n / 2;

    let (al, ar) = a.split_at_mid();
    let (bl, br) = b.unwrap_or(c).split_at_mid();

    let [t0, t1, t2] = &mut from_fn(|_| vec![Default::default(); n]);
    let [alr, blr] = &mut from_fn(|_| vec![Default::default(); m]);
    arith.slice_add(alr, al, ar);
    arith.slice_add(blr, bl, br);

    recurse(arith, t0, al, bl);
    recurse(arith, t1, ar, br);
    recurse(arith, t2, alr, blr);

    f(c, t0, t1);
    arith.slice_add_assign(&mut c[..m], &t0[m..]);
    arith.slice_add_assign(&mut c[..m], &t1[m..]);
    arith.slice_sub_assign(&mut c[..m], &t2[m..]);
    arith.slice_sub_assign(&mut c[m..], &t0[..m]);
    arith.slice_sub_assign(&mut c[m..], &t1[..m]);
    arith.slice_add_assign(&mut c[m..], &t2[..m]);

    fn recurse<T: SliceOps>(arith: &T, c: &mut [T::Elem], a: &[T::Elem], b: &[T::Elem]) {
        if a.len() <= 64 {
            izip!(0.., a).for_each(|(i, a)| {
                izip!(&mut c[i..], b).for_each(|(c, b)| *c = arith.add(c, &arith.mul(a, b)))
            })
        } else {
            let n = c.len();
            let m = n / 2;
            let q = n / 4;
            let (al, ar) = a.split_at_mid();
            let (bl, br) = b.split_at_mid();

            let [t0, t1, t2] = &mut from_fn(|_| vec![Default::default(); m]);
            let [alr, blr] = &mut from_fn(|_| vec![Default::default(); q]);
            arith.slice_add(alr, al, ar);
            arith.slice_add(blr, bl, br);

            recurse(arith, t0, al, bl);
            recurse(arith, t1, ar, br);
            recurse(arith, t2, alr, blr);

            arith.slice_sub(&mut c[q..m + q], t2, t0);
            arith.slice_sub_assign(&mut c[q..m + q], t1);
            arith.slice_add_assign(&mut c[..m], t0);
            arith.slice_add_assign(&mut c[m..], t1);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        distribution::Sampler,
        modulus::Modulus,
        poly::{
            karatsuba::{
                nega_cyclic_karatsuba_fma, nega_cyclic_karatsuba_mul,
                nega_cyclic_karatsuba_mul_assign,
            },
            test::nega_cyclic_schoolbook_mul,
        },
        ring::{NativeRing, RingOps},
    };
    use rand::thread_rng;

    #[test]
    fn karatsuba() {
        let mut rng = thread_rng();
        for n in (0..10).map(|log_n| 1 << log_n) {
            let arith = NativeRing::new(Modulus::native(), 1);
            let a = arith.sample_uniform_vec(n, &mut rng);
            let b = arith.sample_uniform_vec(n, &mut rng);

            let mut c = vec![0; n];
            nega_cyclic_schoolbook_mul(&arith, &mut c, &a, &b);

            let mut d = vec![0; n];
            nega_cyclic_karatsuba_mul(&arith, &mut d, &a, &b);
            assert_eq!(c, d);

            let mut d = b.clone();
            nega_cyclic_karatsuba_mul_assign(&arith, &mut d, &a);
            assert_eq!(c, d);

            let mut d = vec![0; n];
            nega_cyclic_karatsuba_fma(&arith, &mut d, &a, &b);
            assert_eq!(c, d);
        }
    }
}
