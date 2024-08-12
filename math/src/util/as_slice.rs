pub trait AsSlice: AsRef<[Self::Elem]> {
    type Elem;

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn split_at_mid(&self) -> (&[Self::Elem], &[Self::Elem]) {
        let mid = self.len() / 2;
        debug_assert_eq!(2 * mid, self.len());
        self.as_ref().split_at(mid)
    }
}

macro_rules! impl_as_slice {
    ($t:ty) => {
        impl<T> AsSlice for $t {
            type Elem = T;
        }
    };
}

impl_as_slice!([T]);
impl_as_slice!(&[T]);
impl_as_slice!(&mut [T]);
impl_as_slice!(Vec<T>);

pub trait AsMutSlice: AsSlice + AsMut<[Self::Elem]> {
    fn split_at_mid_mut(&mut self) -> (&mut [Self::Elem], &mut [Self::Elem]) {
        let mid = self.len() / 2;
        debug_assert_eq!(2 * mid, self.len());
        self.as_mut().split_at_mut(mid)
    }
}

impl<T> AsMutSlice for [T] {}
impl<T> AsMutSlice for &mut [T] {}
impl<T> AsMutSlice for Vec<T> {}
