pub mod as_slice;
pub mod compact;
pub mod scratch;
pub mod serde;

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

#[cfg(any(test, feature = "dev"))]
pub mod dev {
    use core::iter::Sum;
    use num_traits::AsPrimitive;
    use rand::{rngs::StdRng, SeedableRng};
    use std::{env, time::Instant};

    // Number of repetition for time-consuming tests taking more than 1 second.
    pub fn time_consuming_test_repetition() -> usize {
        env::var("PZ_TIME_CONSUMING_TEST_REPETITION")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10)
    }

    #[derive(Clone, Copy, Debug)]
    pub struct StatsSampler {
        sample_size: usize,
        timeout: u64,
    }

    impl Default for StatsSampler {
        fn default() -> Self {
            Self {
                sample_size: env::var("PZ_STATS_SAMPLE_SIZE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1000000),
                timeout: env::var("PZ_STATS_TIMEOUT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(30),
            }
        }
    }

    impl StatsSampler {
        pub fn sample_size(mut self, sample_size: usize) -> Self {
            self.sample_size = sample_size;
            self
        }

        pub fn timeout(mut self, timeout: u64) -> Self {
            self.timeout = timeout;
            self
        }

        pub fn without_timeout(mut self) -> Self {
            self.timeout = u64::MAX;
            self
        }

        pub fn sample<T, I: IntoIterator<Item = T>>(
            self,
            mut f: impl FnMut(&mut StdRng) -> I,
        ) -> Stats<T> {
            let mut rng = StdRng::from_entropy();
            let mut stats = Stats::with_capacity(self.sample_size);
            let start = Instant::now();
            while start.elapsed().as_secs() < self.timeout && stats.samples.len() < self.sample_size
            {
                stats.extend(f(&mut rng))
            }
            stats
        }
    }

    #[derive(Clone)]
    pub struct Stats<T> {
        samples: Vec<T>,
    }

    impl<T> Stats<T> {
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                samples: Vec::with_capacity(capacity),
            }
        }

        pub fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
            self.samples.extend(iter);
        }
    }

    impl<T: AsPrimitive<f64> + for<'a> Sum<&'a T>> Stats<T> {
        pub fn mean(&self) -> f64 {
            T::sum(self.samples.iter()).as_() / self.samples.len() as f64
        }

        pub fn variance(&self) -> f64 {
            let mean = self.mean();
            let diff_sq = |v: &T| {
                let diff = v.as_() - mean;
                diff * diff
            };
            f64::sum(self.samples.iter().map(diff_sq)) / self.samples.len() as f64
        }

        pub fn std_dev(&self) -> f64 {
            self.variance().sqrt()
        }

        pub fn log2_std_dev(&self) -> f64 {
            self.std_dev().log2()
        }
    }

    impl<T> FromIterator<T> for Stats<T> {
        fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
            Self {
                samples: iter.into_iter().collect(),
            }
        }
    }
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
