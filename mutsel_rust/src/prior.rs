//! The prior of the exchangeabilities: -mu_reg * sum_{i > j} (log S_ij - log S_prior_ij)^2 around the
//! exchangeabilities S_prior of prior v1.20 (data::PRIOR_EXCHANGEABILITIES).

use crate::data::PRIOR_EXCHANGEABILITIES;

#[derive(Clone)]
pub struct ExchangeabilityPrior {
    /// Log exchangeabilities of the prior, the entries (i, j), i > j, in row-major order
    pub log_exchangeabilities: Vec<f64>,
    pub mu_reg: f64,
}

impl ExchangeabilityPrior {
    pub fn new(mu_reg: f64) -> ExchangeabilityPrior {
        ExchangeabilityPrior { log_exchangeabilities: PRIOR_EXCHANGEABILITIES.iter().map(|s| s.ln()).collect(), mu_reg }
    }

    /// Log prior of the log exchangeabilities log_S and its gradient
    pub fn evaluate(&self, log_S: &[f64]) -> (f64, Vec<f64>) {
        let difference: Vec<f64> = log_S.iter().zip(&self.log_exchangeabilities).map(|(l, l0)| l - l0).collect();
        (
            -self.mu_reg * difference.iter().map(|d| d * d).sum::<f64>(),
            difference.iter().map(|d| -2.0 * self.mu_reg * d).collect(),
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Some log exchangeabilities
    pub(crate) fn test_log_exchangeabilities() -> Vec<f64> {
        (0..190).map(|k| 0.8 * ((k as f64) * 0.61).sin()).collect()
    }

    #[test]
    fn prior_is_zero_at_the_prior_exchangeabilities() {
        let prior = ExchangeabilityPrior::new(3.0);
        assert_eq!(prior.log_exchangeabilities.len(), 190);
        assert!(PRIOR_EXCHANGEABILITIES.iter().all(|s| *s > 0.0 && s.is_finite()));
        assert_eq!(prior.evaluate(&prior.log_exchangeabilities.clone()), (0.0, vec![0.0; 190]));
        let shifted: Vec<f64> = prior.log_exchangeabilities.iter().map(|l| l + 0.1).collect();
        let (log_prior, grad) = prior.evaluate(&shifted);
        assert!((log_prior + 3.0 * 190.0 * 0.01).abs() < 1e-10 && grad.iter().all(|g| (g + 0.6).abs() < 1e-12));
    }
}
