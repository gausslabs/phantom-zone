pub trait AsSlice: AsRef<[Self::Elem]> {
    type Elem;

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
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

pub trait AsMutSlice: AsSlice + AsMut<[Self::Elem]> {}

impl<T: AsSlice + AsMut<[Self::Elem]>> AsMutSlice for T {}
