pub(crate) fn bit_reverse<T, V: AsMut<[T]>>(mut values: V) -> V {
    let n = values.as_mut().len();
    if n > 2 {
        debug_assert!(n.is_power_of_two());
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
