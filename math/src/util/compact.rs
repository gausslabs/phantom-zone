use crate::{izip_eq, modulus::ModulusOps, util::as_slice::AsSlice};
use core::iter::repeat_with;

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Compact(#[cfg_attr(feature = "serde", serde(with = "serde_bytes"))] Vec<u8>);

impl Compact {
    pub fn from_elems<M: ModulusOps>(modulus: &M, elems: &impl AsSlice<Elem = M::Elem>) -> Self {
        let bits = modulus.modulus().bits();
        if bits % 8 == 0 {
            let mut bytes = vec![0; elems.as_ref().len() * (bits / 8)];
            izip_eq!(bytes.chunks_exact_mut(bits / 8), elems.as_ref()).for_each(|(bytes, elem)| {
                bytes.copy_from_slice(&modulus.to_u64(*elem).to_le_bytes()[..bytes.len()])
            });
            Self(bytes)
        } else {
            Self::from_elems_unaligned(modulus, elems)
        }
    }

    pub fn to_elems<M: ModulusOps>(&self, modulus: &M) -> Vec<M::Elem> {
        let bits = modulus.modulus().bits();
        if bits % 8 == 0 {
            let to_elem = |bytes: &[_]| {
                let mut buf = [0; 8];
                buf[..bytes.len()].copy_from_slice(bytes);
                modulus.elem_from(u64::from_le_bytes(buf))
            };
            self.0.chunks_exact(bits / 8).map(to_elem).collect()
        } else {
            self.to_elems_unaligned(modulus)
        }
    }

    fn from_elems_unaligned<M: ModulusOps>(
        modulus: &M,
        elems: &impl AsSlice<Elem = M::Elem>,
    ) -> Self {
        let bits = modulus.modulus().bits();
        let (body, tail) = (bits / 8, bits % 8);
        let mut compact = Vec::with_capacity((elems.len() * bits).div_ceil(8));
        let mut leftover = 0;
        elems.as_ref().iter().for_each(|elem| {
            let mut elem = modulus.to_u64(*elem);
            debug_assert_eq!(elem >> bits, 0);
            if leftover != 0 {
                *compact.last_mut().unwrap() |= (elem as u8) << (8 - leftover);
                elem >>= leftover;
                if leftover < tail {
                    compact.extend(elem.to_le_bytes().into_iter().take(body + 1));
                    leftover += 8 - tail;
                } else {
                    compact.extend(elem.to_le_bytes().into_iter().take(body));
                    leftover -= tail;
                }
            } else {
                compact.extend(elem.to_le_bytes().into_iter().take(body + 1));
                leftover = 8 - tail;
            }
        });
        Self(compact)
    }

    fn to_elems_unaligned<M: ModulusOps>(&self, modulus: &M) -> Vec<M::Elem> {
        struct ReadUnaligned<'a>(&'a [u8]);

        impl<'a> ReadUnaligned<'a> {
            #[inline(always)]
            fn advance_by(&mut self, bytes: usize) {
                self.0 = &self.0[bytes..];
            }

            #[inline(always)]
            fn read_head(&mut self, leftover: usize) -> u8 {
                let head = self.0[0] >> (8 - leftover);
                self.advance_by(1);
                head
            }

            #[inline(always)]
            fn read(&mut self, body: usize, tail: usize) -> u64 {
                const MASKS: [u8; 9] = [0, 1, 3, 7, 15, 31, 63, 127, 255];
                let mut buf = [0; 8];
                buf[..body + 1].copy_from_slice(&self.0[..body + 1]);
                buf[body] &= MASKS[tail];
                self.advance_by(body + (tail >> 3));
                u64::from_le_bytes(buf)
            }
        }

        let bits = modulus.modulus().bits();
        let (body, tail) = (bits / 8, bits % 8);
        let mut compact = ReadUnaligned(&self.0);
        let mut leftover = 0;
        repeat_with(|| {
            let value;
            if leftover != 0 {
                let head = compact.read_head(leftover);
                if leftover < tail {
                    value = compact.read(body, tail - leftover) << leftover | head as u64;
                    leftover += 8 - tail;
                } else {
                    value = compact.read(body - 1, 8 - leftover + tail) << leftover | head as u64;
                    leftover -= tail;
                }
            } else {
                value = compact.read(body, tail);
                leftover = 8 - tail;
            }
            modulus.elem_from(value)
        })
        .take((self.0.len() * 8) / bits)
        .collect()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        modulus::{ModulusOps, Native, NonNativePowerOfTwo, Prime},
        util::compact::Compact,
    };
    use rand::{thread_rng, Rng};

    #[test]
    fn from_elems_to_elems() {
        fn run(modulus: impl ModulusOps) {
            for _ in 0..100 {
                let len = thread_rng().gen_range(0..1 << 10);
                let elems = modulus.sample_uniform_vec(len, thread_rng());
                assert_eq!(
                    Compact::from_elems(&modulus, &elems).to_elems(&modulus),
                    elems
                );
            }
        }

        run(Native::native());
        (48..64).for_each(|bits| run(NonNativePowerOfTwo::new(bits)));
        (48..62).for_each(|bits| run(Prime::gen(bits, 0)));
    }
}
