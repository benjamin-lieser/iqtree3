//! The prior of the MutSel model, read from the npz file in MUTSEL_PRIOR with
//! - exchangeabilities [20, 20]: the exchangeabilities S of the prior Mu (symmetric, the diagonal is ignored),
//!   whose mutation equilibrium is data::M_EQUI
//! - exchangeability_reg []: the log prior of the log exchangeabilities is
//!   -exchangeability_reg * sum_{i > j} (log S_ij - log S_prior_ij)^2
//! - weights [K], means [K, 19], covs [K, 19, 19]: Gaussian mixture on the Helmert coordinates of the site frequencies
//! - site_rate_sd []: sd of the log site rates (see SiteRatePrior), 0 keeps all site rates 1

use std::path::Path;

use phylo_grad::nalgebra::{SMatrix, SVector};

use crate::{Result, helmert::NUM_COORDINATES, npz::Npz};

pub struct Prior {
    /// Log exchangeabilities of the prior Mu, the entries (i, j), i > j, in row-major order
    pub log_exchangeabilities: Vec<f64>,
    pub exchangeability_reg: f64,
    pub site_frequencies: Gmm,
    /// None if the site rates are all 1
    pub site_rate: Option<SiteRatePrior>,
}

impl Prior {
    /// Log prior of the log exchangeabilities log_S and its gradient
    pub fn evaluate_exchangeabilities(&self, log_S: &[f64]) -> (f64, Vec<f64>) {
        let difference: Vec<f64> = log_S.iter().zip(&self.log_exchangeabilities).map(|(l, l0)| l - l0).collect();
        (
            -self.exchangeability_reg * difference.iter().map(|d| d * d).sum::<f64>(),
            difference.iter().map(|d| -2.0 * self.exchangeability_reg * d).collect(),
        )
    }

    /// Reads the prior file in MUTSEL_PRIOR
    pub fn from_env() -> Prior {
        let path = std::env::var("MUTSEL_PRIOR").expect("MUTSEL needs a prior file in MUTSEL_PRIOR");
        let prior = Prior::read(Path::new(path.trim())).unwrap_or_else(|e| panic!("Could not read MUTSEL_PRIOR={}: {}", path, e));
        println!(
            "MUTSEL: prior from {}, exchangeability_reg {}, Gaussian mixture with {} components, site_rate_sd {}",
            path,
            prior.exchangeability_reg,
            prior.site_frequencies.means.len(),
            prior.site_rate.map_or(0.0, |p| p.sd)
        );
        prior
    }

    pub fn read(path: &Path) -> Result<Prior> {
        let npz = Npz::read(path)?;
        let exchangeabilities = npz.get("exchangeabilities")?.to_matrix::<20, 20>()?;
        if (exchangeabilities - exchangeabilities.transpose()).abs().max() > 1e-12 * exchangeabilities.abs().max() {
            return Err("exchangeabilities is not symmetric".into());
        }
        let site_rate_sd = npz.get("site_rate_sd")?.to_scalar()?;
        if site_rate_sd < 0.0 {
            return Err("site_rate_sd must not be negative".into());
        }
        Ok(Prior {
            log_exchangeabilities: (1..20).flat_map(|i| (0..i).map(move |j| (i, j))).map(|(i, j)| exchangeabilities[(i, j)].ln()).collect(),
            exchangeability_reg: npz.get("exchangeability_reg")?.to_scalar()?,
            site_frequencies: Gmm::new(&npz.get("weights")?.data, &npz.get("means")?.rows()?, &npz.get("covs")?.data)?,
            site_rate: (site_rate_sd > 0.0).then_some(SiteRatePrior { sd: site_rate_sd }),
        })
    }
}

/// Gaussian mixture on the Helmert coordinates
pub struct Gmm {
    /// log weight - log det(cov) / 2 - 19 log(2 pi) / 2
    log_norm: Vec<f64>,
    means: Vec<SVector<f64, NUM_COORDINATES>>,
    precisions: Vec<SMatrix<f64, NUM_COORDINATES, NUM_COORDINATES>>,
}

impl Gmm {
    /// weights [K], means [K], covs [K * 19 * 19] (row-major)
    pub fn new(weights: &[f64], means: &[SVector<f64, NUM_COORDINATES>], covs: &[f64]) -> Result<Gmm> {
        let size = NUM_COORDINATES * NUM_COORDINATES;
        if means.len() != weights.len() || covs.len() != weights.len() * size {
            return Err("weights, means and covs of the Gaussian mixture do not match".into());
        }
        let mut gmm = Gmm { log_norm: vec![], means: means.to_vec(), precisions: vec![] };
        for k in 0..weights.len() {
            let cov = SMatrix::<f64, NUM_COORDINATES, NUM_COORDINATES>::from_row_slice(&covs[k * size..(k + 1) * size]);
            let chol = cov.cholesky().ok_or_else(|| format!("Covariance {} of the Gaussian mixture is not positive definite", k))?;
            let log_det = 2.0 * chol.l().diagonal().map(f64::ln).sum();
            gmm.log_norm.push(weights[k].ln() - 0.5 * log_det - 0.5 * NUM_COORDINATES as f64 * (2.0 * std::f64::consts::PI).ln());
            gmm.precisions.push(chol.inverse());
        }
        Ok(gmm)
    }

    /// Log density and its gradient
    pub fn evaluate(&self, x: &SVector<f64, NUM_COORDINATES>) -> (f64, SVector<f64, NUM_COORDINATES>) {
        let mut log_densities = Vec::with_capacity(self.means.len());
        let mut grads = Vec::with_capacity(self.means.len());
        for k in 0..self.means.len() {
            let grad = -(self.precisions[k] * (x - self.means[k]));
            log_densities.push(self.log_norm[k] + 0.5 * (x - self.means[k]).dot(&grad));
            grads.push(grad);
        }
        let max = log_densities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        let mut grad = SVector::<f64, NUM_COORDINATES>::zeros();
        for (log_density, g) in log_densities.iter().zip(&grads) {
            let r = (log_density - max).exp();
            sum += r;
            grad += g * r;
        }
        (max + sum.ln(), grad / sum)
    }
}

/// Log-normal prior of a site rate r that multiplies the site's MutSel rate matrix: log r ~ N(-sd^2/2, sd^2),
/// so that the prior mean of r is 1 and the rates only correct the rates given by the site frequencies.
#[derive(Debug, Clone, Copy)]
pub struct SiteRatePrior {
    pub sd: f64,
}

impl SiteRatePrior {
    /// M-step of the sd given draws of the log site rates: maximizes the mean of log N(l; -sd^2/2, sd^2)
    /// = -l^2 / (2 sd^2) - l / 2 - sd^2 / 8 - log sd + const, which is at sd^4 / 4 + sd^2 = mean(l^2)
    pub fn fit(log_rates: impl Iterator<Item = f64>) -> SiteRatePrior {
        let (sum, count) = log_rates.fold((0.0, 0usize), |(sum, count), l| (sum + l * l, count + 1));
        let mean_square = sum / count as f64;
        SiteRatePrior { sd: (2.0 * ((1.0 + mean_square).sqrt() - 1.0)).sqrt() }
    }

    pub fn mean(&self) -> f64 {
        -0.5 * self.sd * self.sd
    }

    /// Log density of the log rate and its derivative
    pub fn evaluate(&self, log_rate: f64) -> (f64, f64) {
        let z = (log_rate - self.mean()) / self.sd;
        (
            -0.5 * z * z - self.sd.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln(),
            -z / self.sd,
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Gaussian mixture with means and isotropic standard deviations
    pub(crate) fn isotropic_gmm(weights: &[f64], means: &[SVector<f64, NUM_COORDINATES>], stds: &[f64]) -> Gmm {
        let covs: Vec<f64> = stds
            .iter()
            .flat_map(|std| (SMatrix::<f64, NUM_COORDINATES, NUM_COORDINATES>::identity() * (std * std)).as_slice().to_vec())
            .collect();
        Gmm::new(weights, means, &covs).unwrap()
    }

    /// Prior with some exchangeabilities, a two component Gaussian mixture and no site rates
    pub(crate) fn test_prior() -> Prior {
        Prior {
            log_exchangeabilities: (0..190).map(|k| 0.8 * ((k as f64) * 0.61).sin()).collect(),
            exchangeability_reg: 9.67,
            site_frequencies: isotropic_gmm(
                &[0.3, 0.7],
                &[SVector::from_fn(|i, _| 0.2 * (i as f64).cos()), SVector::from_fn(|i, _| -0.3 * (i as f64).sin())],
                &[1.0, 1.4],
            ),
            site_rate: None,
        }
    }

    #[test]
    fn gmm_gradient_matches_finite_differences() {
        let means: Vec<_> = (0..3).map(|k| SVector::from_fn(|i, _| (i + k) as f64 * 0.1)).collect();
        let covs: Vec<f64> = (0..3)
            .flat_map(|k| {
                let a = SMatrix::<f64, NUM_COORDINATES, NUM_COORDINATES>::from_fn(|i, j| ((i * 7 + j * 3 + k) % 5) as f64 * 0.1);
                (a * a.transpose() + SMatrix::identity()).as_slice().to_vec()
            })
            .collect();
        let gmm = Gmm::new(&[0.2, 0.3, 0.5], &means, &covs).unwrap();
        let x = SVector::<f64, NUM_COORDINATES>::from_fn(|i, _| 0.3 * ((i as f64) * 1.7).sin());
        let (_, grad) = gmm.evaluate(&x);
        let h = 1e-5;
        for k in 0..NUM_COORDINATES {
            let mut e = SVector::<f64, NUM_COORDINATES>::zeros();
            e[k] = h;
            let fd = (gmm.evaluate(&(x + e)).0 - gmm.evaluate(&(x - e)).0) / (2.0 * h);
            assert!((fd - grad[k]).abs() < 1e-6 * (1.0 + grad[k].abs()), "{} {} {}", k, fd, grad[k]);
        }
    }

    #[test]
    fn gmm_is_normalized() {
        // A single standard normal has log density -19/2 log(2 pi) at its mean
        let gmm = isotropic_gmm(&[1.0], &[SVector::zeros()], &[1.0]);
        let expected = -0.5 * NUM_COORDINATES as f64 * (2.0 * std::f64::consts::PI).ln();
        assert!((gmm.evaluate(&SVector::zeros()).0 - expected).abs() < 1e-12);
    }

    #[test]
    fn site_rate_sd_fit_recovers_the_prior() {
        // The mean square of log rates from N(-sd^2/2, sd^2) is sd^2 + sd^4/4, which the fit maps back to sd
        for sd in [0.1, 0.5, 1.0, 2.0] {
            let mean = -0.5 * sd * sd;
            let fitted = SiteRatePrior::fit([mean - sd, mean + sd].into_iter());
            assert!((fitted.sd - sd).abs() < 1e-12, "{} vs {}", fitted.sd, sd);
            // and it maximizes the mean log density of the draws
            let objective = |s: f64| [mean - sd, mean + sd].iter().map(|&l| SiteRatePrior { sd: s }.evaluate(l).0).sum::<f64>();
            assert!(objective(sd) > objective(sd * 1.01) && objective(sd) > objective(sd * 0.99));
        }
    }

    #[test]
    fn prior_file_roundtrip() {
        use crate::npz::{Array, write_npz};
        let dir = std::env::temp_dir().join(format!("mutsel_prior_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("prior.npz");
        let S = SMatrix::<f64, 20, 20>::from_fn(|i, j| 1.0 + ((i * j + i + j) % 7) as f64);
        let mut covs = vec![0.0; 19 * 19];
        for i in 0..19 {
            covs[i * 19 + i] = 2.0;
        }
        write_npz(
            &path,
            &[
                ("exchangeabilities", &Array::from_matrix(&S)),
                ("exchangeability_reg", &Array::scalar(3.0)),
                ("weights", &Array::vector(vec![1.0])),
                ("means", &Array::new(&[1, 19], vec![0.1; 19])),
                ("covs", &Array::new(&[1, 19, 19], covs)),
                ("site_rate_sd", &Array::scalar(0.5)),
            ],
        )
        .unwrap();
        let prior = Prior::read(&path).unwrap();
        assert_eq!(prior.log_exchangeabilities.len(), 190);
        assert_eq!(prior.log_exchangeabilities[0], S[(1, 0)].ln());
        assert_eq!(prior.log_exchangeabilities[189], S[(19, 18)].ln());
        assert_eq!(prior.site_rate.unwrap().sd, 0.5);
        assert_eq!(prior.exchangeability_reg, 3.0);
        assert_eq!(prior.evaluate_exchangeabilities(&prior.log_exchangeabilities), (0.0, vec![0.0; 190]));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
