pub mod interactive;

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        core::rgsw::RgswDecompositionParam,
        scheme::blind_rotation::lmkcdey::multi_party::interactive::LmkcdeyInteractiveParam,
    };
    use phantom_zone_math::{
        decomposer::DecompositionParam, distribution::DistributionVariance, modulus::Modulus,
    };

    pub struct LmkcdeyNoiseAnalysis {
        param: LmkcdeyInteractiveParam,
    }

    impl LmkcdeyNoiseAnalysis {
        pub fn new(param: LmkcdeyInteractiveParam) -> Self {
            Self { param }
        }

        fn total_shares(&self) -> f64 {
            self.param.total_shares as _
        }

        fn ring_size(&self) -> f64 {
            self.param.ring_size as _
        }

        fn var_noise(&self) -> f64 {
            self.param.noise_distribution.variance()
        }

        fn var_u(&self) -> f64 {
            self.param.u_distribution.variance()
        }

        fn var_sk_ast(&self) -> f64 {
            self.total_shares() * self.param.sk_distribution.variance()
        }

        fn var_noise_pk_ast(&self) -> f64 {
            self.total_shares() * self.var_noise()
        }

        fn var_noise_ct_pk_ast(&self) -> f64 {
            let ring_size = self.ring_size();
            let var_noise = self.var_noise();
            let var_u = self.var_u();
            let var_sk_ast = self.var_sk_ast();
            let var_noise_pk_ast = self.var_noise_pk_ast();
            ring_size * (var_u * var_noise_pk_ast + var_sk_ast * var_noise) + var_noise
        }

        fn var_noise_ks_key(&self) -> f64 {
            self.total_shares() * self.param.lwe_noise_distribution.variance()
        }

        fn var_noise_ak(&self) -> f64 {
            self.total_shares() * self.var_noise()
        }

        fn var_noise_brk(&self) -> f64 {
            let var_noise_rgsw_by_rgsw = var_noise_rgsw_by_rgsw(
                self.param.ring_size,
                self.param.modulus,
                self.param.rgsw_by_rgsw_decomposition_param,
                self.var_sk_ast(),
                self.var_noise_ct_pk_ast(),
            );
            (self.param.total_shares - 1) as f64 * var_noise_rgsw_by_rgsw
        }

        fn var_noise_ct_ks(&self) -> f64 {
            var_noise_key_switch(
                self.param.ring_size,
                self.param.lwe_modulus,
                self.param.lwe_ks_decomposition_param,
                self.var_sk_ast(),
                self.var_noise_ks_key(),
            )
        }

        fn var_noise_ct_auto(&self) -> f64 {
            var_noise_key_switch(
                self.param.ring_size,
                self.param.modulus,
                self.param.auto_decomposition_param,
                self.var_sk_ast(),
                self.var_noise_ak(),
            )
        }

        pub fn log2_std_dev_noise_ks_key(&self) -> f64 {
            self.var_noise_ks_key().sqrt().log2()
        }

        pub fn log2_std_dev_noise_ak(&self) -> f64 {
            self.var_noise_ak().sqrt().log2()
        }

        pub fn log2_std_dev_noise_brk(&self) -> f64 {
            self.var_noise_brk().sqrt().log2()
        }

        pub fn log2_std_dev_noise_ct_ks(&self) -> f64 {
            self.var_noise_ct_ks().sqrt().log2()
        }

        pub fn log2_std_dev_noise_ct_auto(&self) -> f64 {
            self.var_noise_ct_auto().sqrt().log2()
        }
    }

    fn var_noise_rgsw_by_rgsw(
        ring_size: usize,
        modulus: Modulus,
        decomposition_param: RgswDecompositionParam,
        var_sk: f64,
        var_noise_ct: f64,
    ) -> f64 {
        let ring_size = ring_size as f64;
        let level_a = decomposition_param.level_a;
        let level_b = decomposition_param.level_b;
        let log_base = decomposition_param.log_base;
        let log_ignored_a = modulus.bits().saturating_sub(log_base * level_a);
        let log_ignored_b = modulus.bits().saturating_sub(log_base * level_b);
        let var_uniform_base = (0..1 << log_base).variance();
        let var_uniform_ignored_a = (0..1u64 << log_ignored_a).variance();
        let var_uniform_ignored_b = (0..1u64 << log_ignored_b).variance();
        let var_noise_a = level_a as f64 * ring_size * var_uniform_base * var_noise_ct;
        let var_noise_b = level_b as f64 * ring_size * var_uniform_base * var_noise_ct;
        let var_noise_ignored_a = ring_size * var_uniform_ignored_a * var_sk;
        let var_noise_ignored_b = var_uniform_ignored_b;
        var_noise_a + var_noise_b + var_noise_ignored_a + var_noise_ignored_b
    }

    fn var_noise_key_switch(
        ring_size: usize,
        modulus: Modulus,
        decomposition_param: DecompositionParam,
        var_sk: f64,
        var_noise_ks_key: f64,
    ) -> f64 {
        let ring_size = ring_size as f64;
        let level = decomposition_param.level;
        let log_base = decomposition_param.log_base;
        let log_ignored = modulus.bits().saturating_sub(log_base * level);
        let var_uniform_base = (0..1u64 << log_base).variance();
        let var_uniform_ignored = (0..1u64 << log_ignored).variance();
        let var_noise = level as f64 * ring_size * var_uniform_base * var_noise_ks_key;
        let var_noise_ignored = ring_size * var_uniform_ignored * var_sk;
        var_noise + var_noise_ignored
    }
}
