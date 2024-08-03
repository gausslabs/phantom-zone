pub mod as_slice;
pub mod automorphism;

pub fn bit_reverse<T, V: AsMut<[T]>>(mut values: V) -> V {
    let n = values.as_mut().len();
    if n > 2 {
        assert!(n.is_power_of_two());
        let log_n = n.ilog2();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - log_n);
            if i < j {
                values.as_mut().swap(i, j)
            }
        }
    }
    values
}

#[macro_export]
macro_rules! izip_eq {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::izip_eq!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        core::iter::IntoIterator::into_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {{
        #[cfg(debug_assertions)]
        { itertools::Itertools::zip_eq($crate::izip_eq!($first), $second) }
        #[cfg(not(debug_assertions))]
        { Iterator::zip($crate::izip_eq!($first), $second) }
    }};
    ($first:expr $(, $rest:expr)* $(,)*) => {{
        let t = $crate::izip_eq!($first);
        $(let t = $crate::izip_eq!(t, $rest);)*
        t.map($crate::izip_eq!(@closure a => (a) $(, $rest)*))
    }};
}

#[cfg(test)]
pub(crate) mod test {
    macro_rules! assert_precision {
        ($a:expr, $b:expr, $prec_loss:expr $(,)?) => {{
            let a = $a.clone();
            let b = $b.clone();
            let distance = ((a.wrapping_sub(b)) as i64).abs();
            assert!(
                distance < (1 << $prec_loss),
                "log2(|{a} - {b}|) = {} >= {}",
                (distance as f64).log2(),
                $prec_loss
            );
        }};
    }

    pub(crate) use assert_precision;
}
