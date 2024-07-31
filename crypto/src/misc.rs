use core::fmt::Debug;

#[derive(Clone, Copy, Debug)]
pub enum SecretKeyDistribution {
    Gaussian(f64),
    Ternary(usize),
}

pub trait AsSlice: AsRef<[Self::Elem]> {
    type Elem;

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait AsMutSlice: AsSlice + AsMut<[Self::Elem]> {}

macro_rules! impl_as_slice {
    (@ $t:ty) => {
        impl<T> AsSlice for $t {
            type Elem = T;
        }
    };
    ($($t:ty),* $(,)?) => {
        $(impl_as_slice!(@ $t);)*
    };
}

macro_rules! impl_as_mut_slice {
    (@ $t:ty) => {
        impl<T> AsMutSlice for $t {}
    };
    ($($t:ty),* $(,)?) => {
        $(impl_as_mut_slice!(@ $t);)*
    };
}

impl_as_slice!([T], &[T], &mut [T], Vec<T>);
impl_as_mut_slice!([T], &mut [T], Vec<T>);
