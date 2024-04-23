use num_traits::{Num, PrimInt, WrappingShl, WrappingShr, Zero};

pub trait UnsignedInteger: Zero + Num {}
