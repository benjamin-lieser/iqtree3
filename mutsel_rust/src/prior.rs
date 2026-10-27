use crate::model::Vector20;

/// Prior on the log equilibrium frequencies log_pi of a single site.
pub trait Prior: Sync {
    /// Log density (up to an additive constant) and its gradient with respect to log_pi.
    fn log_density(&self, log_pi: &Vector20) -> (f64, Vector20);
}

/// Independent gaussian on every entry of log_pi.
#[derive(Debug, Clone)]
pub struct Gaussian {
    pub mean: Vector20,
    pub sigma: f64,
}

impl Gaussian {
    pub fn new(mean: Vector20, sigma: f64) -> Self {
        assert!(sigma > 0.0);
        Gaussian { mean, sigma }
    }
}

impl Prior for Gaussian {
    fn log_density(&self, log_pi: &Vector20) -> (f64, Vector20) {
        let diff = (log_pi - self.mean) / self.sigma;
        let grad = -diff / self.sigma;
        (-0.5 * diff.norm_squared(), grad)
    }
}
