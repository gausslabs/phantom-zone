use core::{any::type_name, array::from_fn, mem::size_of, slice};

#[derive(Clone, Debug, Default)]
pub struct ScratchOwned(Vec<u8>);

impl ScratchOwned {
    pub fn allocate(bytes: usize) -> Self {
        Self(vec![0; bytes])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn borrow_mut(&mut self) -> Scratch {
        Scratch(&mut self.0)
    }
}

#[derive(Debug, Default)]
pub struct Scratch<'a>(&'a mut [u8]);

impl<'a> Scratch<'a> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reborrow(&mut self) -> Scratch {
        Scratch(&mut *self.0)
    }

    pub fn take_slice<T>(&mut self, count: usize) -> &'a mut [T] {
        let ptr = self.0.as_mut_ptr();
        let len = self.0.len();

        let aligned_offset = ptr.align_offset(size_of::<T>());
        let aligned_len = len - aligned_offset;

        let taken_len = count * size_of::<T>();
        let Some(remaining_len) = aligned_len.checked_sub(taken_len) else {
            panic!(
                "{count} {} ({taken_len} bytes) taken from sractch with remaining {len} bytes",
                type_name::<T>()
            )
        };

        unsafe {
            let aligned_ptr = ptr.add(aligned_offset);
            let remaining_ptr = aligned_ptr.add(taken_len);

            self.0 = slice::from_raw_parts_mut(remaining_ptr, remaining_len);
            slice::from_raw_parts_mut(aligned_ptr as *mut T, count)
        }
    }

    pub fn take_slice_array<T, const N: usize>(&mut self, count: usize) -> [&'a mut [T]; N] {
        from_fn(|_| self.take_slice(count))
    }
}
