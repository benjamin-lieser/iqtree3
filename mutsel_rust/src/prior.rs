//! The prior of the exchangeabilities: -mu_reg * sum_{i > j} (log S_ij - log S_prior_ij)^2 around the
//! exchangeabilities S_prior ([20, 20], symmetric, the diagonal is ignored) of the npz file in MUTSEL_PRIOR, by
//! default those of the prior compiled into the library (priors/prior_v1.20.npz). Other arrays of the file are
//! not used.

use std::path::Path;

use crate::{Result, npz::Npz};

const DEFAULT_PRIOR_NAME: &str = "prior_v1.20.npz";
const DEFAULT_PRIOR: &[u8] = include_bytes!("../priors/prior_v1.20.npz");

#[derive(Clone)]
pub struct ExchangeabilityPrior {
    /// Log exchangeabilities of the prior, the entries (i, j), i > j, in row-major order
    pub log_exchangeabilities: Vec<f64>,
    pub mu_reg: f64,
}

impl ExchangeabilityPrior {
    /// Log prior of the log exchangeabilities log_S and its gradient
    pub fn evaluate(&self, log_S: &[f64]) -> (f64, Vec<f64>) {
        let difference: Vec<f64> = log_S.iter().zip(&self.log_exchangeabilities).map(|(l, l0)| l - l0).collect();
        (
            -self.mu_reg * difference.iter().map(|d| d * d).sum::<f64>(),
            difference.iter().map(|d| -2.0 * self.mu_reg * d).collect(),
        )
    }

    /// The exchangeabilities of the file in MUTSEL_PRIOR, or of the default prior if it is not set
    pub fn from_env(mu_reg: f64) -> ExchangeabilityPrior {
        let (npz, source) = match std::env::var("MUTSEL_PRIOR") {
            Ok(path) => (
                Npz::read(Path::new(path.trim())).unwrap_or_else(|e| panic!("Could not read MUTSEL_PRIOR={}: {}", path, e)),
                path,
            ),
            Err(_) => (Npz::from_bytes(DEFAULT_PRIOR, DEFAULT_PRIOR_NAME).unwrap(), format!("the default prior {}", DEFAULT_PRIOR_NAME)),
        };
        let log_exchangeabilities =
            read_log_exchangeabilities(&npz).unwrap_or_else(|e| panic!("Could not read the exchangeabilities of {}: {}", source, e));
        println!("MUTSEL: prior exchangeabilities from {}", source);
        ExchangeabilityPrior { log_exchangeabilities, mu_reg }
    }
}

fn read_log_exchangeabilities(npz: &Npz) -> Result<Vec<f64>> {
    let exchangeabilities = npz.get("exchangeabilities")?.to_matrix::<20, 20>()?;
    if (exchangeabilities - exchangeabilities.transpose()).abs().max() > 1e-12 * exchangeabilities.abs().max() {
        return Err("exchangeabilities is not symmetric".into());
    }
    Ok((1..20).flat_map(|i| (0..i).map(move |j| (i, j))).map(|(i, j)| exchangeabilities[(i, j)].ln()).collect())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Some log exchangeabilities
    pub(crate) fn test_log_exchangeabilities() -> Vec<f64> {
        (0..190).map(|k| 0.8 * ((k as f64) * 0.61).sin()).collect()
    }

    #[test]
    fn default_prior_is_compiled_in() {
        let npz = Npz::from_bytes(DEFAULT_PRIOR, DEFAULT_PRIOR_NAME).unwrap();
        let log_S = read_log_exchangeabilities(&npz).unwrap();
        assert_eq!(log_S.len(), 190);
        assert!(log_S.iter().all(|l| l.is_finite()));
        let prior = ExchangeabilityPrior { log_exchangeabilities: log_S.clone(), mu_reg: 3.0 };
        assert_eq!(prior.evaluate(&log_S), (0.0, vec![0.0; 190]));
    }
}
