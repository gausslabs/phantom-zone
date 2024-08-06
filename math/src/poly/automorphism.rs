use crate::{
    izip_eq,
    misc::{
        as_slice::{AsMutSlice, AsSlice},
        scratch::Scratch,
    },
};
use core::slice::Iter;
use phantom_zone_derive::AsSliceWrapper;

#[derive(Clone, Debug, AsSliceWrapper)]
pub struct AutomorphismMap<S: AsSlice<Elem = usize>> {
    #[as_slice]
    map: S,
    k: usize,
}

impl<S: AsSlice<Elem = usize>> AutomorphismMap<S> {
    pub fn ring_size(&self) -> usize {
        self.map.len()
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn apply<'a, T, F>(&'a self, poly: &'a [T], neg: F) -> AutomorphismIter<'a, T, F>
    where
        F: Fn(&T) -> T,
    {
        debug_assert_eq!(self.map.len(), poly.len());
        AutomorphismIter {
            iter: self.map.as_ref().iter(),
            poly,
            neg,
        }
    }

    pub fn apply_in_place<T: 'static + Copy, F>(&self, poly: &mut [T], neg: F, mut scratch: Scratch)
    where
        F: Fn(&T) -> T,
    {
        let tmp = scratch.copy_slice(poly);
        izip_eq!(poly, self.apply(tmp, neg)).for_each(|(a, b)| *a = b);
    }
}

impl AutomorphismMapOwned {
    pub fn new(ring_size: usize, k: i64) -> Self {
        debug_assert!(ring_size.is_power_of_two());
        let mask = ring_size - 1;
        let log_n = ring_size.ilog2();
        let k = k.rem_euclid(2 * ring_size as i64) as usize;
        let mut map = vec![0; ring_size];
        (0..ring_size).for_each(|i| {
            let j = i * k;
            map[j & mask] = (i << 1) | ((j >> log_n) & 1)
        });
        Self { map, k }
    }
}

#[derive(Clone, Debug)]
pub struct AutomorphismIter<'a, T, F> {
    iter: Iter<'a, usize>,
    poly: &'a [T],
    neg: F,
}

impl<'a, T, F> Iterator for AutomorphismIter<'a, T, F>
where
    T: Clone,
    F: Fn(&T) -> T,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|idx| {
            if (idx & 1) == 1 {
                (self.neg)(&self.poly[idx >> 1])
            } else {
                self.poly[idx >> 1].clone()
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[cfg(test)]
mod test {
    use crate::{modulus::powers_mod, poly::automorphism::AutomorphismMap};
    use core::ops::Neg;
    use itertools::Itertools;

    fn automorphism<T: Copy + Default + Neg<Output = T>>(input: &[T], k: i64) -> Vec<T> {
        assert!(input.len().is_power_of_two());
        let n = input.len();
        let k = k.rem_euclid(2 * n as i64) as usize;
        let mut out = vec![T::default(); n];
        (0..n)
            .map(|i| (i, (i * k) % (2 * n)))
            .for_each(|(i, j)| out[j % n] = if j < n { input[i] } else { -input[i] });
        out
    }

    #[test]
    fn automorphism_iter() {
        for log_n in 0..10 {
            let n = 1 << log_n;
            let indices = (0..n as i64).collect_vec();
            for k in powers_mod(5, 2 * n as u64).take(n / 2) {
                let auto_map = AutomorphismMap::new(n, k as _);
                assert_eq!(
                    auto_map.apply(&indices, |i| -i).collect_vec(),
                    automorphism(&indices, k as _)
                );
            }
        }
    }
}
