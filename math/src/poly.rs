pub mod automorphism;
pub mod ffnt;
pub mod karatsuba;
pub mod ntt;

#[cfg(test)]
pub(crate) mod test {
    use crate::ring::SliceOps;
    use itertools::izip;

    pub fn nega_cyclic_schoolbook_mul<T: SliceOps>(
        arith: &T,
        c: &mut [T::Elem],
        a: &[T::Elem],
        b: &[T::Elem],
    ) {
        let n = a.len();
        izip!(&mut *c, b).for_each(|(c, b)| *c = arith.mul(&a[0], b));
        izip!(0.., a).skip(1).for_each(|(i, a)| {
            izip!(0.., b).for_each(|(j, b)| {
                if i + j < n {
                    c[i + j] = arith.add(&c[i + j], &arith.mul(a, b));
                } else {
                    c[i + j - n] = arith.sub(&c[i + j - n], &arith.mul(a, b));
                }
            })
        });
    }
}
