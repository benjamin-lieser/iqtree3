//! Hierarchical prior of the site frequencies and site rates: every alignment gets its own prior, fitted in the
//! M-step of the Monte Carlo EM (hmc::fit_with_em), with the prior file (MUTSEL_PRIOR) as its hyperprior.
//!
//! The per-alignment prior of the Helmert coordinates is a Gaussian mixture with the components of the prior file
//! and low-rank plus diagonal covariances (mixture of factor analyzers)
//!     x ~ sum_k w_k N(mu_k, W_k W_k^T + diag(psi_k)),   W_k [19, rank]
//! and the site rates keep their log-normal prior with a per-alignment sd.
//!
//! The hyperprior with strength tau acts like tau pseudo-sites drawn from the prior file (weights w0, means mu0,
//! covariances Sigma0, site_rate_sd sd0), a_k = tau w0_k of them from component k with known component:
//!     log p(theta) = sum_k a_k log w_k
//!                  - sum_k a_k / 2 (log det Sigma_k + tr(Sigma_k^-1 Sigma0_k) + (mu_k - mu0_k)^T Sigma_k^-1 (mu_k - mu0_k))
//! i.e. a Dirichlet(1 + a) prior of the weights and a normal-inverse-Wishart prior of (mu_k, Sigma_k) centred on
//! component k of the prior file. tau -> infinity keeps the prior file (as far as rank allows), tau = 0 is the
//! maximum likelihood prior of the alignment.
//!
//! Given the posterior samples of the E-step (every site with total weight 1), the M-step of the mixture maximizes
//! the mean log density of the samples plus the log hyperprior with L-BFGS, separately from the M-step of Mu and the
//! branch lengths (the two parts of the EM objective share no parameters). The sd of the site rates has a closed
//! form M-step.

use phylo_grad::nalgebra::{DMatrix, DVector, SMatrix, SVector};
use rayon::prelude::*;

use crate::{
    Result,
    helmert::NUM_COORDINATES,
    hmc::env_or,
    lbfgs::lbfgs,
    prior::{Gmm, Prior, SiteRatePrior},
};

const DIM: usize = NUM_COORDINATES;
type Matrix = SMatrix<f64, DIM, DIM>;
type Coordinates = SVector<f64, DIM>;

/// Settings of the hierarchical prior, None (MUTSEL_PRIOR_LEARN=0) keeps the prior file fixed
#[derive(Debug, Clone, Copy)]
pub struct LearnSettings {
    /// tau, the number of pseudo-sites of the hyperprior
    pub strength: f64,
    /// Rank of the low-rank part of the covariances, 0 to 19
    pub rank: usize,
    /// Maximum number of L-BFGS iterations of the M-step of the mixture
    pub max_iterations: usize,
}

impl LearnSettings {
    /// Reads MUTSEL_PRIOR_LEARN (0 or 1), MUTSEL_PRIOR_STRENGTH, MUTSEL_PRIOR_RANK and MUTSEL_PRIOR_MSTEP_ITERATIONS
    pub fn from_env() -> Option<LearnSettings> {
        if env_or("MUTSEL_PRIOR_LEARN", 0u8) == 0 {
            return None;
        }
        let settings = LearnSettings {
            strength: env_or("MUTSEL_PRIOR_STRENGTH", 200.0),
            rank: env_or("MUTSEL_PRIOR_RANK", 3),
            max_iterations: env_or("MUTSEL_PRIOR_MSTEP_ITERATIONS", 200),
        };
        assert!(settings.strength > 0.0, "MUTSEL_PRIOR_STRENGTH must be positive");
        assert!(settings.rank <= DIM, "MUTSEL_PRIOR_RANK must be at most {}", DIM);
        Some(settings)
    }
}

/// The prior file as hyperprior with strength tau
pub struct Hyperprior {
    weights: Vec<f64>,
    means: Vec<Coordinates>,
    covs: Vec<Matrix>,
    /// sd0 of the site rates, None if the site rates are all 1
    site_rate_sd: Option<f64>,
    strength: f64,
}

impl Hyperprior {
    pub fn new(prior: &Prior, strength: f64) -> Hyperprior {
        let gmm = &prior.site_frequencies;
        Hyperprior {
            weights: gmm.weights().to_vec(),
            means: gmm.means().to_vec(),
            covs: gmm.covs().to_vec(),
            site_rate_sd: prior.site_rate.map(|p| p.sd),
            strength,
        }
    }

    /// Pseudo-sites of component k
    fn pseudo_sites(&self, k: usize) -> f64 {
        self.strength * self.weights[k]
    }

    /// M-step of the sd of the site rates: maximizes the log prior of the log site rates log_rates, each with the
    /// weight point_weight, plus that of the tau pseudo-sites with log rates from N(-sd0^2/2, sd0^2). As in
    /// SiteRatePrior::fit, only the mean of l^2 enters, sd^4 / 4 + sd^2 = mean(l^2), with the pseudo-sites
    /// contributing E[l^2] = sd0^2 + sd0^4 / 4.
    pub fn fit_site_rate(&self, log_rates: impl Iterator<Item = f64>, point_weight: f64) -> Option<SiteRatePrior> {
        let sd0 = self.site_rate_sd?;
        let (sum, count) = log_rates.fold((0.0, 0usize), |(sum, count), l| (sum + l * l, count + 1));
        let pseudo_mean_square = sd0 * sd0 + sd0.powi(4) / 4.0;
        let mean_square = (point_weight * sum + self.strength * pseudo_mean_square) / (point_weight * count as f64 + self.strength);
        Some(SiteRatePrior { sd: (2.0 * ((1.0 + mean_square).sqrt() - 1.0)).sqrt() })
    }
}

/// Gaussian mixture with low-rank plus diagonal covariances in the parametrization of the L-BFGS M-step
#[derive(Debug, Clone)]
pub struct LowRankGmm {
    rank: usize,
    /// Weights are softmax(logits)
    logits: Vec<f64>,
    means: Vec<Coordinates>,
    /// W_k [19, rank]
    factors: Vec<DMatrix<f64>>,
    log_psi: Vec<Coordinates>,
}

/// Lower bound of psi relative to the mean variance of the initial covariance
const MIN_RELATIVE_PSI: f64 = 1e-6;

impl LowRankGmm {
    /// The low-rank plus diagonal approximation of a Gaussian mixture: per component the probabilistic PCA of the
    /// covariance (the rank leading eigenvectors, scaled to their eigenvalues minus the mean of the others), with
    /// psi the remaining diagonal
    pub fn from_gmm(gmm: &Gmm, rank: usize) -> LowRankGmm {
        let mut factors = Vec::new();
        let mut log_psi = Vec::new();
        for cov in gmm.covs() {
            let eigen = cov.symmetric_eigen();
            let mut order: Vec<usize> = (0..DIM).collect();
            order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));
            let noise = if rank < DIM { order[rank..].iter().map(|&i| eigen.eigenvalues[i]).sum::<f64>() / (DIM - rank) as f64 } else { 0.0 };
            let W = DMatrix::from_fn(DIM, rank, |i, j| {
                let e = order[j];
                eigen.eigenvectors[(i, e)] * (eigen.eigenvalues[e] - noise).max(0.0).sqrt()
            });
            let WWt = &W * W.transpose();
            let min_psi = MIN_RELATIVE_PSI * cov.trace() / DIM as f64;
            log_psi.push(Coordinates::from_fn(|i, _| (cov[(i, i)] - WWt[(i, i)]).max(min_psi).ln()));
            factors.push(W);
        }
        LowRankGmm { rank, logits: gmm.weights().iter().map(|w| w.ln()).collect(), means: gmm.means().to_vec(), factors, log_psi }
    }

    fn num_components(&self) -> usize {
        self.logits.len()
    }

    /// Number of parameters per component: logit, mean, W, log psi
    fn component_size(&self) -> usize {
        1 + DIM + DIM * self.rank + DIM
    }

    pub fn weights(&self) -> Vec<f64> {
        softmax(&self.logits)
    }

    pub fn covariance(&self, k: usize) -> Matrix {
        low_rank_covariance(&self.factors[k], &self.log_psi[k])
    }

    pub fn to_gmm(&self) -> Result<Gmm> {
        let covs: Vec<Matrix> = (0..self.num_components()).map(|k| self.covariance(k)).collect();
        Gmm::from_covs(&self.weights(), &self.means, &covs)
    }

    /// W [K, 19, rank] (row-major per component, as numpy) and psi [K, 19]
    pub fn factor_data(&self) -> (Vec<f64>, Vec<f64>) {
        let factors = self.factors.iter().flat_map(|W| W.transpose().as_slice().to_vec()).collect();
        let psi = self.log_psi.iter().flat_map(|l| l.iter().map(|v| v.exp()).collect::<Vec<_>>()).collect();
        (factors, psi)
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    fn to_params(&self) -> DVector<f64> {
        let mut params = Vec::with_capacity(self.num_components() * self.component_size());
        for k in 0..self.num_components() {
            params.push(self.logits[k]);
            params.extend(self.means[k].iter());
            params.extend(self.factors[k].as_slice());
            params.extend(self.log_psi[k].iter());
        }
        DVector::from_vec(params)
    }

    fn from_params(&self, params: &[f64]) -> LowRankGmm {
        let size = self.component_size();
        let rank = self.rank;
        let mut out = self.clone();
        for (k, p) in params.chunks_exact(size).enumerate() {
            out.logits[k] = p[0];
            out.means[k] = Coordinates::from_column_slice(&p[1..1 + DIM]);
            out.factors[k] = DMatrix::from_column_slice(DIM, rank, &p[1 + DIM..1 + DIM + DIM * rank]);
            out.log_psi[k] = Coordinates::from_column_slice(&p[1 + DIM + DIM * rank..]);
        }
        out
    }

    /// M-step: maximizes objective() over the parameters with L-BFGS, starting from the current ones
    pub fn fit(&mut self, points: &[Coordinates], point_weight: f64, hyper: &Hyperprior, max_iterations: usize) -> FitReport {
        // Relative improvement of the objective per L-BFGS iteration below which the M-step stops
        const REL_IMPROVEMENT: f64 = 1e-10;
        let start = std::time::Instant::now();
        let before = self.objective(points, point_weight, hyper).expect("the starting covariances are positive definite");
        let mut num_evaluations = 0;
        let optimum = lbfgs(
            |params| {
                num_evaluations += 1;
                let (value, gradient) = self.from_params(params.as_slice()).objective_and_gradient(points, point_weight, hyper)?;
                value.total().is_finite().then(|| (value.total(), DVector::from_vec(gradient)))
            },
            self.to_params(),
            max_iterations,
            0.0,
            REL_IMPROVEMENT,
        );
        *self = self.from_params(optimum.as_slice());
        let after = self.objective(points, point_weight, hyper).expect("L-BFGS only accepts finite points");
        FitReport { before, after, num_evaluations, seconds: start.elapsed().as_secs_f64() }
    }

    pub fn objective(&self, points: &[Coordinates], point_weight: f64, hyper: &Hyperprior) -> Option<Objective> {
        self.objective_and_gradient(points, point_weight, hyper).map(|(value, _)| value)
    }

    /// Weighted log density of the points (each with point_weight) plus the log hyperprior (up to constants), and
    /// its gradient in the layout of to_params. None if a covariance is not positive definite.
    fn objective_and_gradient(&self, points: &[Coordinates], point_weight: f64, hyper: &Hyperprior) -> Option<(Objective, Vec<f64>)> {
        let num_components = self.num_components();
        let weights = self.weights();
        let mut precisions = Vec::with_capacity(num_components);
        let mut log_dets = Vec::with_capacity(num_components);
        let mut log_norms = Vec::with_capacity(num_components);
        for k in 0..num_components {
            let cov = self.covariance(k);
            let chol = cov.cholesky()?;
            let log_det = 2.0 * chol.l().diagonal().map(f64::ln).sum();
            log_norms.push(weights[k].ln() - 0.5 * log_det - 0.5 * DIM as f64 * (2.0 * std::f64::consts::PI).ln());
            precisions.push(chol.inverse());
            log_dets.push(log_det);
        }

        // Weighted sufficient statistics of the responsibilities gamma_nk (point_weight times the posterior of
        // component k given point n): N_k = sum gamma, s1_k = sum gamma x, s2_k = sum gamma x x^T. The gradient of
        // the log density of the mixture is that of sum_k sum_n gamma_nk log(w_k N(x_n; mu_k, Sigma_k)) with gamma fixed.
        let stats = points
            .par_chunks(256)
            .fold(
                || Statistics::new(num_components),
                |mut stats, chunk| {
                    let mut log_densities = vec![0.0; num_components];
                    for x in chunk {
                        for k in 0..num_components {
                            let d = x - self.means[k];
                            log_densities[k] = log_norms[k] - 0.5 * d.dot(&(precisions[k] * d));
                        }
                        let max = log_densities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let sum: f64 = log_densities.iter().map(|l| (l - max).exp()).sum();
                        let log_density = max + sum.ln();
                        stats.log_density += point_weight * log_density;
                        for k in 0..num_components {
                            let gamma = point_weight * (log_densities[k] - log_density).exp();
                            // Negligible contributions only cost time
                            if gamma > 1e-14 * point_weight {
                                stats.n[k] += gamma;
                                stats.s1[k] += x * gamma;
                                stats.s2[k].ger(gamma, x, x, 1.0);
                            }
                        }
                    }
                    stats
                },
            )
            .reduce(|| Statistics::new(num_components), Statistics::add);

        let total_points: f64 = stats.n.iter().sum();
        let size = self.component_size();
        let mut gradient = vec![0.0; num_components * size];
        let mut log_hyperprior = 0.0;
        for k in 0..num_components {
            let a = hyper.pseudo_sites(k);
            let (n, s1, mu, precision) = (stats.n[k], &stats.s1[k], &self.means[k], &precisions[k]);
            let d = mu - hyper.means[k];
            log_hyperprior += a * weights[k].ln()
                - 0.5 * a * (log_dets[k] + (precision * hyper.covs[k]).trace() + d.dot(&(precision * d)));

            // Scatter of the points and the pseudo-sites around mu_k
            let scatter = stats.s2[k] - s1 * mu.transpose() - mu * s1.transpose() + mu * mu.transpose() * n
                + (hyper.covs[k] + d * d.transpose()) * a;
            // d objective / d Sigma_k (as a symmetric matrix)
            let grad_cov = (precision * scatter * precision - precision * (n + a)) * 0.5;

            let g = &mut gradient[k * size..(k + 1) * size];
            g[0] = n + a - weights[k] * (total_points + hyper.strength);
            g[1..1 + DIM].copy_from_slice((precision * (s1 - mu * n - d * a)).as_slice());
            let W = &self.factors[k];
            let grad_W = DMatrix::from_fn(DIM, DIM, |i, j| grad_cov[(i, j)]) * W * 2.0;
            g[1 + DIM..1 + DIM + DIM * self.rank].copy_from_slice(grad_W.as_slice());
            for i in 0..DIM {
                g[1 + DIM + DIM * self.rank + i] = grad_cov[(i, i)] * self.log_psi[k][i].exp();
            }
        }
        Some((Objective { log_density: stats.log_density, log_hyperprior }, gradient))
    }
}

fn low_rank_covariance(W: &DMatrix<f64>, log_psi: &Coordinates) -> Matrix {
    let WWt = W * W.transpose();
    Matrix::from_fn(|i, j| WWt[(i, j)] + if i == j { log_psi[i].exp() } else { 0.0 })
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|e| e / sum).collect()
}

struct Statistics {
    log_density: f64,
    n: Vec<f64>,
    s1: Vec<Coordinates>,
    s2: Vec<Matrix>,
}

impl Statistics {
    fn new(num_components: usize) -> Statistics {
        Statistics {
            log_density: 0.0,
            n: vec![0.0; num_components],
            s1: vec![Coordinates::zeros(); num_components],
            s2: vec![Matrix::zeros(); num_components],
        }
    }

    fn add(mut self, other: Statistics) -> Statistics {
        self.log_density += other.log_density;
        for k in 0..self.n.len() {
            self.n[k] += other.n[k];
            self.s1[k] += other.s1[k];
            self.s2[k] += other.s2[k];
        }
        self
    }
}

/// The two parts of the objective of the M-step of the mixture
#[derive(Debug, Clone, Copy)]
pub struct Objective {
    /// Weighted log density of the samples under the mixture
    pub log_density: f64,
    /// Log hyperprior up to a constant
    pub log_hyperprior: f64,
}

impl Objective {
    pub fn total(&self) -> f64 {
        self.log_density + self.log_hyperprior
    }
}

pub struct FitReport {
    pub before: Objective,
    pub after: Objective,
    pub num_evaluations: usize,
    pub seconds: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::tests::isotropic_gmm;

    /// Two component hyperprior with full covariances, and points drawn around shifted means
    fn setup() -> (Prior, Vec<Coordinates>) {
        let means = [SVector::from_fn(|i, _| 0.5 * (i as f64).cos()), SVector::from_fn(|i, _| -0.4 * (i as f64 * 0.7).sin())];
        let covs: Vec<Matrix> = (0..2)
            .map(|k| {
                let a = Matrix::from_fn(|i, j| (((i * 5 + j * 3 + k) % 7) as f64 - 3.0) * 0.08);
                a * a.transpose() + Matrix::identity() * (0.3 + 0.2 * k as f64)
            })
            .collect();
        let gmm = Gmm::from_covs(&[0.35, 0.65], &means, &covs).unwrap();
        let prior = Prior { site_frequencies: gmm, site_rate: Some(SiteRatePrior { sd: 0.8 }), ..crate::prior::tests::test_prior() };
        let points = (0..300)
            .map(|n| {
                let k = n % 2;
                means[k] + SVector::from_fn(|i, _| ((n * 31 + i * 17) as f64 * 0.913).sin() * (1.0 + 0.5 * k as f64) + 0.3)
            })
            .collect();
        (prior, points)
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let (prior, points) = setup();
        let hyper = Hyperprior::new(&prior, 37.0);
        let mut model = LowRankGmm::from_gmm(&prior.site_frequencies, 2);
        // Away from the initialization, so that no part of the gradient vanishes by symmetry
        let mut params = model.to_params();
        for (i, p) in params.iter_mut().enumerate() {
            *p += 0.05 * ((i as f64) * 1.3).sin();
        }
        model = model.from_params(params.as_slice());
        let (_, gradient) = model.objective_and_gradient(&points, 0.25, &hyper).unwrap();
        let h = 1e-5;
        for i in 0..params.len() {
            let mut plus = params.clone();
            plus[i] += h;
            let mut minus = params.clone();
            minus[i] -= h;
            let f = |p: &DVector<f64>| model.from_params(p.as_slice()).objective(&points, 0.25, &hyper).unwrap().total();
            let fd = (f(&plus) - f(&minus)) / (2.0 * h);
            assert!((fd - gradient[i]).abs() < 1e-5 * (1.0 + gradient[i].abs()), "{}: {} vs {}", i, fd, gradient[i]);
        }
    }

    #[test]
    fn full_rank_initialization_reproduces_the_gmm() {
        let (prior, _) = setup();
        let model = LowRankGmm::from_gmm(&prior.site_frequencies, DIM);
        // up to the lower bound of psi, as the full-rank factor leaves a zero diagonal
        for k in 0..2 {
            assert!((model.covariance(k) - prior.site_frequencies.covs()[k]).abs().max() < 2e-6);
        }
        assert!((model.weights()[0] - 0.35).abs() < 1e-14);
    }

    #[test]
    fn fit_improves_the_objective_and_moves_towards_the_data() {
        let (prior, points) = setup();
        let mut moved = Vec::new();
        for strength in [1.0, 1e7] {
            let hyper = Hyperprior::new(&prior, strength);
            let mut model = LowRankGmm::from_gmm(&prior.site_frequencies, 3);
            let report = model.fit(&points, 0.5, &hyper, 500);
            assert!(report.after.total() >= report.before.total());
            let shift: f64 = (0..2).map(|k| (model.means[k] - prior.site_frequencies.means()[k]).norm()).sum();
            moved.push(shift);
        }
        // The data shifted every coordinate by 0.3: a weak hyperprior follows, a strong one keeps the means
        assert!(moved[0] > 0.5, "{:?}", moved);
        assert!(moved[1] < 1e-3, "{:?}", moved);
    }

    #[test]
    fn strong_hyperprior_keeps_the_low_rank_approximation_of_a_low_rank_prior() {
        // A hyperprior whose covariances are exactly low rank plus diagonal is kept by a strong hyperprior
        let means = [SVector::zeros(), SVector::from_element(0.5)];
        let gmm = isotropic_gmm(&[0.5, 0.5], &means, &[1.0, 0.7]);
        let prior = Prior { site_frequencies: gmm, ..crate::prior::tests::test_prior() };
        let hyper = Hyperprior::new(&prior, 1e8);
        let mut model = LowRankGmm::from_gmm(&prior.site_frequencies, 2);
        let points: Vec<Coordinates> = (0..50).map(|n| SVector::from_fn(|i, _| ((n + 3 * i) as f64).sin())).collect();
        model.fit(&points, 1.0, &hyper, 500);
        for k in 0..2 {
            assert!((model.covariance(k) - prior.site_frequencies.covs()[k]).abs().max() < 1e-4);
        }
    }

    #[test]
    fn unregularized_weights_follow_the_responsibilities() {
        // With a tiny hyperprior, the weights maximize the log density: at the optimum the gradient
        // N_k - w_k N vanishes, so the weights are the mean responsibilities
        let (prior, points) = setup();
        let hyper = Hyperprior::new(&prior, 1e-9);
        let mut model = LowRankGmm::from_gmm(&prior.site_frequencies, 3);
        model.fit(&points, 1.0, &hyper, 1000);
        let (_, gradient) = model.objective_and_gradient(&points, 1.0, &hyper).unwrap();
        for k in 0..2 {
            assert!(gradient[k * model.component_size()].abs() < 1e-3 * points.len() as f64, "{}", gradient[k * model.component_size()]);
        }
    }

    #[test]
    fn site_rate_sd_is_pulled_to_the_hyperprior() {
        let (prior, _) = setup();
        // Log rates with mean square 4 alone give sd about 1.287, the hyperprior sd is 0.8
        let log_rates = vec![2.0, -2.0];
        let free = SiteRatePrior::fit(log_rates.iter().copied()).sd;
        let weak = Hyperprior::new(&prior, 1e-9).fit_site_rate(log_rates.iter().copied(), 1.0).unwrap().sd;
        let strong = Hyperprior::new(&prior, 1e9).fit_site_rate(log_rates.iter().copied(), 1.0).unwrap().sd;
        let half = Hyperprior::new(&prior, 2.0).fit_site_rate(log_rates.iter().copied(), 1.0).unwrap().sd;
        assert!((weak - free).abs() < 1e-8);
        assert!((strong - 0.8).abs() < 1e-8);
        assert!(half > 0.8 && half < free);
    }
}
