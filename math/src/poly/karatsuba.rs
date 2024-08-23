use crate::{modulus::ModulusOps, util::as_slice::AsSlice};
use core::array::from_fn;
use itertools::izip;

pub fn nega_cyclic_karatsuba_mul<M: ModulusOps>(
    modulus: &M,
    c: &mut [M::Elem],
    a: &[M::Elem],
    b: &[M::Elem],
) {
    nega_cyclic_karatsuba_mul_inner(modulus, c, a, Some(b), |c, t0, t1| {
        modulus.slice_sub(c, t0, t1)
    });
}

pub fn nega_cyclic_karatsuba_mul_assign<M: ModulusOps>(
    modulus: &M,
    b: &mut [M::Elem],
    a: &[M::Elem],
) {
    nega_cyclic_karatsuba_mul_inner(modulus, b, a, None, |c, t0, t1| {
        modulus.slice_sub(c, t0, t1)
    });
}

pub fn nega_cyclic_karatsuba_fma<M: ModulusOps>(
    modulus: &M,
    c: &mut [M::Elem],
    a: &[M::Elem],
    b: &[M::Elem],
) {
    nega_cyclic_karatsuba_mul_inner(modulus, c, a, Some(b), |c, t0, t1| {
        modulus.slice_op(c, t0, t1, |c, t0, t1| {
            *c = modulus.add(c, &modulus.sub(t0, t1))
        })
    });
}

fn nega_cyclic_karatsuba_mul_inner<M: ModulusOps>(
    modulus: &M,
    c: &mut [M::Elem],
    a: &[M::Elem],
    b: Option<&[M::Elem]>,
    f: impl Fn(&mut [M::Elem], &[M::Elem], &[M::Elem]),
) {
    assert!(a.len().is_power_of_two());
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), b.map(<[_]>::len).unwrap_or(a.len()));

    if a.len() == 1 {
        c[0] = modulus.mul(&a[0], b.map(|b| &b[0]).unwrap_or_else(|| &c[0]));
        return;
    }

    let n = a.len();
    let m = n / 2;

    let (al, ar) = a.split_at_mid();
    let (bl, br) = b.unwrap_or(c).split_at_mid();

    let [t0, t1, t2] = &mut from_fn(|_| vec![Default::default(); n]);
    let [alr, blr] = &mut from_fn(|_| vec![Default::default(); m]);
    modulus.slice_add(alr, al, ar);
    modulus.slice_add(blr, bl, br);

    recurse(modulus, t0, al, bl);
    recurse(modulus, t1, ar, br);
    recurse(modulus, t2, alr, blr);

    f(c, t0, t1);
    modulus.slice_add_assign(&mut c[..m], &t0[m..]);
    modulus.slice_add_assign(&mut c[..m], &t1[m..]);
    modulus.slice_sub_assign(&mut c[..m], &t2[m..]);
    modulus.slice_sub_assign(&mut c[m..], &t0[..m]);
    modulus.slice_sub_assign(&mut c[m..], &t1[..m]);
    modulus.slice_add_assign(&mut c[m..], &t2[..m]);

    fn recurse<M: ModulusOps>(modulus: &M, c: &mut [M::Elem], a: &[M::Elem], b: &[M::Elem]) {
        if a.len() <= 64 {
            izip!(0.., a).for_each(|(i, a)| {
                izip!(&mut c[i..], b).for_each(|(c, b)| *c = modulus.add(c, &modulus.mul(a, b)))
            })
        } else {
            let n = c.len();
            let m = n / 2;
            let q = n / 4;
            let (al, ar) = a.split_at_mid();
            let (bl, br) = b.split_at_mid();

            let [t0, t1, t2] = &mut from_fn(|_| vec![Default::default(); m]);
            let [alr, blr] = &mut from_fn(|_| vec![Default::default(); q]);
            modulus.slice_add(alr, al, ar);
            modulus.slice_add(blr, bl, br);

            recurse(modulus, t0, al, bl);
            recurse(modulus, t1, ar, br);
            recurse(modulus, t2, alr, blr);

            modulus.slice_sub(&mut c[q..m + q], t2, t0);
            modulus.slice_sub_assign(&mut c[q..m + q], t1);
            modulus.slice_add_assign(&mut c[..m], t0);
            modulus.slice_add_assign(&mut c[m..], t1);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        distribution::Sampler,
        modulus::Native,
        poly::{
            karatsuba::{
                nega_cyclic_karatsuba_fma, nega_cyclic_karatsuba_mul,
                nega_cyclic_karatsuba_mul_assign,
            },
            test::nega_cyclic_schoolbook_mul,
        },
    };
    use rand::thread_rng;

    #[test]
    fn karatsuba() {
        let mut rng = thread_rng();
        for n in (0..10).map(|log_n| 1 << log_n) {
            let modulus = Native::native();
            let a = modulus.sample_uniform_vec(n, &mut rng);
            let b = modulus.sample_uniform_vec(n, &mut rng);

            let mut c = vec![0; n];
            nega_cyclic_schoolbook_mul(&modulus, &mut c, &a, &b);

            let mut d = vec![0; n];
            nega_cyclic_karatsuba_mul(&modulus, &mut d, &a, &b);
            assert_eq!(c, d);

            let mut d = b.clone();
            nega_cyclic_karatsuba_mul_assign(&modulus, &mut d, &a);
            assert_eq!(c, d);

            let mut d = vec![0; n];
            nega_cyclic_karatsuba_fma(&modulus, &mut d, &a, &b);
            assert_eq!(c, d);
        }
    }
}
