//! Per-site posterior sampling of the PCA coordinates with NUTS (nuts-rs).
//!
//! Mu, the branch lengths and the global rate normalization are shared and kept
//! constant while sampling. Site rates are fixed to 1. With everything shared fixed, the
//! posterior factorizes over alignment columns, so every column gets its own
//! independent 19-dimensional NUTS chain.
//!
//! Before the final sampling, Mu and the branch lengths are fitted with Monte Carlo EM
//! (see sample_internal).

use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use candle_core::{Tensor, Var};
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
    rand::{SeedableRng, rngs::ChaCha8Rng},
};
use phylo_grad::{
    FelsensteinTree,
    nalgebra::{DVector, SMatrix, SVector},
};
use rayon::prelude::*;

use crate::{
    MutselParams, Verbosity, batch,
    felsenstein::{FelsensteinOp, FelsensteinWithEdgeOp},
    model::{self, calc_rate_matrix},
    optimization::{Convergence, Mu, Mu_penalty, map_estimate},
    pca::PCA,
    utils::tensor_full,
};

const NUM_COMPONENTS: usize = 19;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("Could not parse {}={}", name, value)),
        Err(_) => default,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HmcSettings {
    pub num_tune: u64,
    /// Draws after tuning before the effective sample size is checked for the first time
    pub num_draws: u64,
    /// Sampling continues until every pca coordinate of a site reaches this effective sample size
    pub target_ess: usize,
    /// Upper limit of draws per site, even if target_ess is not reached
    pub max_draws: u64,
    /// Number of (approximately independent) samples per site written to the npz file
    pub num_output_samples: usize,
    pub seed: u64,
    /// Estimate the marginal likelihood of every site (see site_log_marginal)
    pub marginal: bool,
}

impl HmcSettings {
    /// Reads MUTSEL_HMC_TUNE, MUTSEL_HMC_DRAWS, MUTSEL_HMC_ESS, MUTSEL_HMC_MAX_DRAWS,
    /// MUTSEL_HMC_SAMPLES, MUTSEL_HMC_SEED and MUTSEL_HMC_MARGINAL (0 or 1) from the environment.
    pub fn from_env() -> HmcSettings {
        HmcSettings {
            num_tune: env_or("MUTSEL_HMC_TUNE", 400),
            num_draws: env_or("MUTSEL_HMC_DRAWS", 500),
            target_ess: env_or("MUTSEL_HMC_ESS", 100),
            max_draws: env_or("MUTSEL_HMC_MAX_DRAWS", 20000),
            num_output_samples: env_or("MUTSEL_HMC_SAMPLES", 100),
            seed: env_or("MUTSEL_HMC_SEED", 42),
            marginal: env_or("MUTSEL_HMC_MARGINAL", 0u8) != 0,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SiteLogpError {
    #[error("non-finite log likelihood")]
    NonFinite,
}

impl LogpError for SiteLogpError {
    // Treated as a divergence by the sampler.
    fn is_recoverable(&self) -> bool {
        true
    }
}

/// Constants of the per-site posterior which are shared by all sites.
struct SharedModel {
    /// PCA components [19, 20], log_pi = components^T * x (up to a constant)
    components: SMatrix<f64, NUM_COMPONENTS, 20>,
    /// Log of the mutation equilibrium (diagonal of Mu)
    log_mutation_equilibrium: SVector<f64, 20>,
    /// Mu with the fixed global rate normalization already applied
    scaled_Mu: SMatrix<f64, 20, 20>,
    prior_mean: SVector<f64, NUM_COMPONENTS>,
    /// alpha / pi_reg
    prior_scale: SVector<f64, NUM_COMPONENTS>,
    prior_beta: SVector<f64, NUM_COMPONENTS>,
    /// Replaces the generalized normal prior if set
    mixture_prior: Option<Arc<MixturePrior>>,
}

/// Alternative priors on the pca coordinates, see MixturePrior::from_env
pub enum MixturePrior {
    Gaussian(GmmPrior),
    Dirichlet(DirichletMixturePrior),
}

impl MixturePrior {
    /// Reads MUTSEL_PRIOR_GMM (Gaussian mixture on the pca coordinates) or MUTSEL_PRIOR_DM
    /// (Dirichlet mixture on the site frequencies), at most one of them may be set
    pub fn from_env() -> Option<Arc<MixturePrior>> {
        let gmm = std::env::var("MUTSEL_PRIOR_GMM").ok();
        let dm = std::env::var("MUTSEL_PRIOR_DM").ok();
        let prior = match (gmm, dm) {
            (None, None) => return None,
            (Some(path), None) => {
                let prior = GmmPrior::read(Path::new(path.trim()))
                    .unwrap_or_else(|e| panic!("Could not read MUTSEL_PRIOR_GMM={}: {}", path, e));
                println!("MUTSEL HMC: Gaussian mixture prior with {} components from {}", prior.means.len(), path);
                MixturePrior::Gaussian(prior)
            }
            (None, Some(path)) => {
                let prior = DirichletMixturePrior::read(Path::new(path.trim()))
                    .unwrap_or_else(|e| panic!("Could not read MUTSEL_PRIOR_DM={}: {}", path, e));
                println!("MUTSEL HMC: Dirichlet mixture prior with {} components from {}", prior.alphas.len(), path);
                MixturePrior::Dirichlet(prior)
            }
            (Some(_), Some(_)) => panic!("Only one of MUTSEL_PRIOR_GMM and MUTSEL_PRIOR_DM can be set"),
        };
        Some(Arc::new(prior))
    }

    /// Log density on the pca coordinates and its gradient
    fn evaluate(
        &self,
        x: &SVector<f64, NUM_COMPONENTS>,
        components: &SMatrix<f64, NUM_COMPONENTS, 20>,
    ) -> (f64, SVector<f64, NUM_COMPONENTS>) {
        match self {
            MixturePrior::Gaussian(prior) => prior.evaluate(x),
            MixturePrior::Dirichlet(prior) => prior.evaluate(x, components),
        }
    }
}

/// log Gamma(x) for x > 0 (Lanczos approximation, g = 7, relative error ~1e-15)
fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut sum = COEFFICIENTS[0];
    for (i, c) in COEFFICIENTS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Dirichlet mixture prior on the site frequencies, read from an npz file with weights [K] and alphas [K, 20].
/// As a density on the pca coordinates (an orthonormal basis of the sum-zero log frequencies) it is
/// sum_k w_k Dir(pi(x); alpha_k) * prod_i pi_i * sqrt(20).
pub struct DirichletMixturePrior {
    /// log weight + log Gamma(sum alpha) - sum log Gamma(alpha_i) + log(sqrt(20))
    log_norm: Vec<f64>,
    alphas: Vec<SVector<f64, 20>>,
}

impl DirichletMixturePrior {
    pub fn read(path: &Path) -> Result<DirichletMixturePrior, candle_core::Error> {
        let arrays: std::collections::HashMap<String, Tensor> = Tensor::read_npz(path)?.into_iter().collect();
        let get = |name: &str| {
            arrays
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("{} not found in {}", name, path.display())))
        };
        let weights = get("weights")?.to_dtype(candle_core::DType::F64)?.to_vec1::<f64>()?;
        let alphas = get("alphas")?.to_dtype(candle_core::DType::F64)?.to_vec2::<f64>()?;
        let mut prior = DirichletMixturePrior { log_norm: vec![], alphas: vec![] };
        for (weight, alpha) in weights.iter().zip(&alphas) {
            let alpha = SVector::<f64, 20>::from_column_slice(alpha);
            prior.log_norm.push(
                weight.ln() + ln_gamma(alpha.sum()) - alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>() + 0.5 * 20f64.ln(),
            );
            prior.alphas.push(alpha);
        }
        Ok(prior)
    }

    fn evaluate(
        &self,
        x: &SVector<f64, NUM_COMPONENTS>,
        components: &SMatrix<f64, NUM_COMPONENTS, 20>,
    ) -> (f64, SVector<f64, NUM_COMPONENTS>) {
        let unnormalized = components.transpose() * x;
        let max = unnormalized.max();
        let log_sum = unnormalized.map(|l| (l - max).exp()).sum().ln() + max;
        let log_pi = unnormalized.add_scalar(-log_sum);
        let pi = log_pi.map(f64::exp);
        // The Jacobian prod_i pi_i adds 1 to every alpha
        let log_densities: Vec<f64> = self
            .log_norm
            .iter()
            .zip(&self.alphas)
            .map(|(log_norm, alpha)| log_norm + alpha.dot(&log_pi))
            .collect();
        let max_density = log_densities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        let mut weighted_alpha = SVector::<f64, 20>::zeros();
        for (log_density, alpha) in log_densities.iter().zip(&self.alphas) {
            let r = (log_density - max_density).exp();
            sum += r;
            weighted_alpha += alpha * r;
        }
        weighted_alpha /= sum;
        // d log pi_i / dx = components[:, i] - components * pi
        let grad = components * (weighted_alpha - pi * weighted_alpha.sum());
        (max_density + sum.ln(), grad)
    }
}

/// Gaussian mixture prior on the pca coordinates, read from an npz file with
/// weights [K], means [K, 19] and covs [K, 19, 19]
pub struct GmmPrior {
    /// log weight - log det(cov) / 2 - 19 log(2 pi) / 2
    log_norm: Vec<f64>,
    means: Vec<SVector<f64, NUM_COMPONENTS>>,
    precisions: Vec<SMatrix<f64, NUM_COMPONENTS, NUM_COMPONENTS>>,
}

impl GmmPrior {
    pub fn read(path: &Path) -> Result<GmmPrior, candle_core::Error> {
        let arrays: std::collections::HashMap<String, Tensor> = Tensor::read_npz(path)?.into_iter().collect();
        let get = |name: &str| {
            arrays
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("{} not found in {}", name, path.display())))
        };
        let weights = get("weights")?.to_dtype(candle_core::DType::F64)?.to_vec1::<f64>()?;
        let means = get("means")?.to_dtype(candle_core::DType::F64)?.to_vec2::<f64>()?;
        let covs = get("covs")?.to_dtype(candle_core::DType::F64)?.to_vec3::<f64>()?;
        let mut prior = GmmPrior { log_norm: vec![], means: vec![], precisions: vec![] };
        for k in 0..weights.len() {
            let cov = SMatrix::<f64, NUM_COMPONENTS, NUM_COMPONENTS>::from_row_iterator(covs[k].iter().flatten().copied());
            let chol = cov
                .cholesky()
                .ok_or_else(|| candle_core::Error::Msg(format!("Covariance {} of the GMM prior is not positive definite", k)))?;
            let log_det = 2.0 * chol.l().diagonal().map(f64::ln).sum();
            prior.log_norm.push(
                weights[k].ln() - 0.5 * log_det - 0.5 * NUM_COMPONENTS as f64 * (2.0 * std::f64::consts::PI).ln(),
            );
            prior.means.push(SVector::<f64, NUM_COMPONENTS>::from_column_slice(&means[k]));
            prior.precisions.push(chol.inverse());
        }
        Ok(prior)
    }

    /// Log density and its gradient
    fn evaluate(&self, x: &SVector<f64, NUM_COMPONENTS>) -> (f64, SVector<f64, NUM_COMPONENTS>) {
        let mut log_densities = Vec::with_capacity(self.means.len());
        let mut grads = Vec::with_capacity(self.means.len());
        for k in 0..self.means.len() {
            let grad = -(self.precisions[k] * (x - self.means[k]));
            log_densities.push(self.log_norm[k] + 0.5 * (x - self.means[k]).dot(&grad));
            grads.push(grad);
        }
        let max = log_densities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        let mut grad = SVector::<f64, NUM_COMPONENTS>::zeros();
        for (log_density, g) in log_densities.iter().zip(&grads) {
            let r = (log_density - max).exp();
            sum += r;
            grad += g * r;
        }
        (max + sum.ln(), grad / sum)
    }
}

impl SharedModel {
    fn new(Mu: &Tensor, rate_scaling: f64, pca: &PCA, pi_reg: f64) -> SharedModel {
        let Mu_values = Mu.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let Mu = SMatrix::<f64, 20, 20>::from_row_slice(&Mu_values);
        let components = pca.components.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let to_vec = |t: &Tensor| {
            SVector::<f64, NUM_COMPONENTS>::from_column_slice(
                &t.narrow(0, 0, NUM_COMPONENTS).unwrap().to_vec1::<f64>().unwrap(),
            )
        };
        SharedModel {
            components: SMatrix::<f64, NUM_COMPONENTS, 20>::from_row_slice(&components),
            log_mutation_equilibrium: Mu.diagonal().map(f64::ln),
            scaled_Mu: Mu * rate_scaling,
            prior_mean: to_vec(&pca.mean),
            prior_scale: to_vec(&pca.alpha) / pi_reg,
            prior_beta: to_vec(&pca.beta),
            mixture_prior: None,
        }
    }
}

impl SharedModel {
    fn with_mixture_prior(self, mixture_prior: Option<Arc<MixturePrior>>) -> SharedModel {
        SharedModel { mixture_prior, ..self }
    }

    /// Site frequencies for the given pca coordinates
    fn pi(&self, x: &SVector<f64, NUM_COMPONENTS>) -> (SVector<f64, 20>, SVector<f64, 20>) {
        let log_pi = self.components.transpose() * x;
        let max_log_pi = log_pi.max();
        let unnormalized_pi = log_pi.map(|l| (l - max_log_pi).exp());
        (log_pi, unnormalized_pi / unnormalized_pi.sum())
    }

    /// S (upper triangle) and sqrt_pi for log frequencies up to a constant
    fn rate_matrix(&self, log_pi: &SVector<f64, 20>) -> (SMatrix<f64, 20, 20>, SVector<f64, 20>) {
        let max_log_pi = log_pi.max();
        let unnormalized_pi = log_pi.map(|l| (l - max_log_pi).exp());
        let sqrt_pi = (unnormalized_pi / unnormalized_pi.sum()).map(f64::sqrt);
        let fitness = log_pi - self.log_mutation_equilibrium;
        let mut S = SMatrix::<f64, 20, 20>::zeros();
        for i in 0..20 {
            for j in (i + 1)..20 {
                S[(i, j)] = self.scaled_Mu[(i, j)]
                    * (0.5 * (log_pi[i] - log_pi[j])).exp()
                    * fixation(fitness[j] - fitness[i]);
            }
        }
        (S, sqrt_pi)
    }

    /// Log prior, S and sqrt_pi for the pca coordinates x
    fn evaluate(&self, x: &SVector<f64, NUM_COMPONENTS>) -> SiteModelEval {
        // Generalized normal prior on the pca coordinates (negative of PCA::penalty_on_pca_coordinates)
        let mut log_prior = 0.0;
        let mut grad_log_prior = SVector::<f64, NUM_COMPONENTS>::zeros();
        if let Some(mixture_prior) = &self.mixture_prior {
            (log_prior, grad_log_prior) = mixture_prior.evaluate(x, &self.components);
        } else {
            for k in 0..NUM_COMPONENTS {
                let z = (x[k] - self.prior_mean[k]) / self.prior_scale[k];
                let beta = self.prior_beta[k];
                let abs_z_pow = z.abs().powf(beta - 1.0);
                log_prior -= abs_z_pow * z.abs();
                grad_log_prior[k] = -beta * abs_z_pow * z.signum() / self.prior_scale[k];
            }
        }

        // log_pi up to an additive constant, pi = softmax(log_pi)
        let (log_pi, pi) = self.pi(x);
        let sqrt_pi = pi.map(f64::sqrt);
        let fitness = log_pi - self.log_mutation_equilibrium;

        // S_ij = sqrt(pi_i / pi_j) * Mu_ij * g(f_j - f_i), only the upper triangle is read by phylo_grad
        let mut S = SMatrix::<f64, 20, 20>::zeros();
        for i in 0..20 {
            for j in (i + 1)..20 {
                S[(i, j)] = self.scaled_Mu[(i, j)]
                    * (0.5 * (log_pi[i] - log_pi[j])).exp()
                    * fixation(fitness[j] - fitness[i]);
            }
        }

        SiteModelEval {
            log_prior,
            grad_log_prior,
            pi,
            sqrt_pi,
            fitness,
            S,
        }
    }

    /// Gradient of log_likelihood + log_prior w.r.t. x, given the gradients of the log likelihood
    /// w.r.t. the upper triangle of S and w.r.t. sqrt_pi.
    fn backpropagate(
        &self,
        eval: &SiteModelEval,
        grad_S: &SMatrix<f64, 20, 20>,
        grad_sqrt_pi: &SVector<f64, 20>,
    ) -> SVector<f64, NUM_COMPONENTS> {
        let mut grad_log_pi = SVector::<f64, 20>::zeros();
        for i in 0..20 {
            for j in (i + 1)..20 {
                let w = grad_S[(i, j)]
                    * eval.S[(i, j)]
                    * (0.5 - dx_log_fixation(eval.fitness[j] - eval.fitness[i]));
                grad_log_pi[i] += w;
                grad_log_pi[j] -= w;
            }
        }
        // d sqrt_pi_a / d log_pi_b = sqrt_pi_a * (delta_ab - pi_b) / 2
        let weighted = grad_sqrt_pi.component_mul(&eval.sqrt_pi);
        grad_log_pi += (weighted - eval.pi * weighted.sum()) * 0.5;

        eval.grad_log_prior + self.components * grad_log_pi
    }
}

struct SiteModelEval {
    log_prior: f64,
    grad_log_prior: SVector<f64, NUM_COMPONENTS>,
    pi: SVector<f64, 20>,
    sqrt_pi: SVector<f64, 20>,
    fitness: SVector<f64, 20>,
    S: SMatrix<f64, 20, 20>,
}

/// Fixation factor g(x) = x / (1 - exp(-x)), see model::GOp
pub(crate) fn fixation(x: f64) -> f64 {
    if x.abs() < 1e-6 {
        1.0 + x / 2.0 + x * x / 12.0
    } else {
        x / (-(-x).exp_m1())
    }
}

/// d/dx log g(x) = 1/x - 1/(exp(x) - 1)
pub(crate) fn dx_log_fixation(x: f64) -> f64 {
    // The closed form loses about eps / |x| to cancellation, so small arguments use the
    // Bernoulli series 1/2 - sum_k B_2k x^(2k-1) / (2k)!, whose truncation error is below 1e-20 for |x| < 0.1.
    if x.abs() < 0.1 {
        let x2 = x * x;
        0.5 - x * (1.0 / 12.0
            - x2 * (1.0 / 720.0
                - x2 * (1.0 / 30240.0 - x2 * (1.0 / 1209600.0 - x2 / 47900160.0))))
    } else {
        1.0 / x - 1.0 / x.exp_m1()
    }
}

/// Unnormalized log posterior of the PCA coordinates of a single alignment column.
/// Same model as optimization::ModelParameters with site rate 1 and fixed Mu, branch lengths and normalization.
struct SiteLogp {
    shared: Arc<SharedModel>,
    tree: Arc<FelsensteinTree<20>>,
    leaf_pl: Vec<SVector<f64, 20>>,
    pl_buffer: Vec<SVector<f64, 20>>,
}

impl HasDims for SiteLogp {
    fn dim_sizes(&self) -> std::collections::HashMap<String, u64> {
        [("unconstrained_parameter".to_string(), NUM_COMPONENTS as u64)]
            .into_iter()
            .collect()
    }
}

impl CpuLogpFunc for SiteLogp {
    type LogpError = SiteLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        NUM_COMPONENTS
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SiteLogpError> {
        let model = &*self.shared;
        let x = SVector::<f64, NUM_COMPONENTS>::from_column_slice(position);
        let eval = model.evaluate(&x);

        let num_leaves = self.leaf_pl.len();
        self.pl_buffer[..num_leaves].copy_from_slice(&self.leaf_pl);
        let result = self.tree.calculate_gradients_single_side(
            eval.S.as_view(),
            eval.sqrt_pi.as_view(),
            &mut self.pl_buffer,
        );

        let logp = result.log_likelihood + eval.log_prior;
        if !logp.is_finite() {
            return Err(SiteLogpError::NonFinite);
        }

        let grad_x = model.backpropagate(&eval, &result.grad_s, &result.grad_sqrt_pi);
        if grad_x.iter().any(|g| !g.is_finite()) {
            return Err(SiteLogpError::NonFinite);
        }
        gradient.copy_from_slice(grad_x.as_slice());

        Ok(logp)
    }

    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        Ok(array.to_vec())
    }
}

struct SiteSamples {
    /// Evenly thinned draws [num_output_samples * 19]
    thinned_draws: Vec<f64>,
    /// Posterior mean of the site frequencies over all draws
    mean_pi: SVector<f64, 20>,
    num_draws: usize,
    /// Minimum over the pca coordinates of the effective sample size of all draws
    ess: f64,
    num_divergences: usize,
    step_size: f64,
    mean_num_steps: f64,
    /// Mean and covariance of the draws in the first half of the chain, if settings.marginal
    proposal: Option<(SVector<f64, NUM_COMPONENTS>, SMatrix<f64, NUM_COMPONENTS, NUM_COMPONENTS>)>,
    /// Thinned draws from the second half of the chain, which are independent of the proposal
    bridge_draws: Vec<SVector<f64, NUM_COMPONENTS>>,
    /// Log marginal likelihood (log likelihood and log prior integrated over the pca coordinates), NaN if not estimated
    log_marginal: f64,
}

/// Effective sample size of a single chain with Geyer's initial monotone sequence estimator.
/// draws: [n * dim] row-major, returns the minimum over the dimensions.
fn min_effective_sample_size(draws: &[f64], dim: usize) -> f64 {
    let n = draws.len() / dim;
    if n < 4 {
        return 0.0;
    }
    let mut min_ess = f64::INFINITY;
    let mut centered = vec![0.0; n];
    for d in 0..dim {
        let mean = (0..n).map(|t| draws[t * dim + d]).sum::<f64>() / n as f64;
        for t in 0..n {
            centered[t] = draws[t * dim + d] - mean;
        }
        let autocov = |lag: usize| -> f64 {
            centered[..n - lag]
                .iter()
                .zip(&centered[lag..])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / n as f64
        };
        let variance = autocov(0);
        if variance <= 0.0 {
            // A coordinate which does not move at all has no information
            return 0.0;
        }
        // tau = -1 + 2 * sum_k P_k, P_k = rho_{2k} + rho_{2k+1}, truncated at the first negative
        // pair and forced to be monotonically decreasing
        let mut sum_pairs = 0.0;
        let mut previous_pair = f64::INFINITY;
        let mut lag = 0;
        while lag + 1 < n {
            let pair = (autocov(lag) + autocov(lag + 1)) / variance;
            if pair <= 0.0 {
                break;
            }
            let pair = pair.min(previous_pair);
            sum_pairs += pair;
            previous_pair = pair;
            lag += 2;
        }
        let tau = (2.0 * sum_pairs - 1.0).max(1.0 / (n as f64).log10());
        min_ess = min_ess.min(n as f64 / tau);
    }
    min_ess
}

fn sample_site(logp: SiteLogp, init: &[f64], settings: &HmcSettings, site_index: usize) -> SiteSamples {
    let shared = Arc::clone(&logp.shared);

    let mut nuts_settings = DiagNutsSettings::default();
    nuts_settings.num_tune = settings.num_tune;
    nuts_settings.num_draws = settings.max_draws;
    nuts_settings.num_chains = 1;
    nuts_settings.seed = settings.seed;

    let mut rng = ChaCha8Rng::seed_from_u64(
        settings.seed ^ (site_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );

    let math = CpuMath::new(logp);
    let mut chain = nuts_settings.new_chain(site_index as u64, math, &mut rng);
    chain
        .set_position(init)
        .unwrap_or_else(|e| panic!("Site {}: could not initialize HMC chain: {:?}", site_index, e));

    let mut draws = Vec::with_capacity(settings.num_draws as usize * NUM_COMPONENTS);
    let mut num_divergences = 0;
    let mut num_steps = 0u64;
    let mut step_size;
    // Enough draws to pick num_output_samples distinct ones when thinning
    let min_draws = (settings.num_draws as usize).max(settings.num_output_samples);
    let mut next_check = min_draws.min(settings.max_draws as usize);
    let mut ess;
    loop {
        let (draw, progress) = chain
            .draw()
            .unwrap_or_else(|e| panic!("Site {}: HMC sampling failed: {:?}", site_index, e));
        if progress.tuning {
            continue;
        }
        draws.extend_from_slice(&draw);
        num_divergences += progress.diverging as usize;
        num_steps += progress.num_steps;
        step_size = progress.step_size;

        let num_draws = draws.len() / NUM_COMPONENTS;
        if num_draws < next_check {
            continue;
        }
        ess = min_effective_sample_size(&draws, NUM_COMPONENTS);
        if ess >= settings.target_ess as f64 || num_draws as u64 >= settings.max_draws {
            break;
        }
        // Extrapolate the number of draws needed, with some margin
        let needed = (num_draws as f64 * 1.2 * settings.target_ess as f64 / ess.max(1.0)) as usize;
        next_check = needed
            .clamp(num_draws + num_draws / 4 + 1, 2 * num_draws)
            .min(settings.max_draws as usize);
    }
    let num_draws = draws.len() / NUM_COMPONENTS;

    let mut thinned_draws = Vec::with_capacity(settings.num_output_samples * NUM_COMPONENTS);
    for i in 0..settings.num_output_samples {
        // Last draw of each of num_output_samples equally sized blocks
        let draw = ((i + 1) * num_draws) / settings.num_output_samples - 1;
        thinned_draws.extend_from_slice(&draws[draw * NUM_COMPONENTS..(draw + 1) * NUM_COMPONENTS]);
    }

    let mut proposal = None;
    let mut bridge_draws = vec![];
    if settings.marginal {
        let first_half: Vec<_> = draws[..(num_draws / 2) * NUM_COMPONENTS]
            .chunks_exact(NUM_COMPONENTS)
            .map(SVector::<f64, NUM_COMPONENTS>::from_column_slice)
            .collect();
        let mean = first_half.iter().sum::<SVector<f64, NUM_COMPONENTS>>() / first_half.len() as f64;
        let mut cov = SMatrix::<f64, NUM_COMPONENTS, NUM_COMPONENTS>::zeros();
        for x in &first_half {
            cov += (x - mean) * (x - mean).transpose();
        }
        cov /= (first_half.len() - 1) as f64;
        proposal = Some((mean, cov));
        for i in 0..settings.num_output_samples {
            let draw = ((i + 1) * num_draws) / settings.num_output_samples - 1;
            if draw >= num_draws / 2 {
                bridge_draws.push(SVector::from_column_slice(&draws[draw * NUM_COMPONENTS..(draw + 1) * NUM_COMPONENTS]));
            }
        }
    }

    let mut mean_pi = SVector::<f64, 20>::zeros();
    for draw in draws.chunks_exact(NUM_COMPONENTS) {
        mean_pi += shared.pi(&SVector::<f64, NUM_COMPONENTS>::from_column_slice(draw)).1;
    }
    mean_pi /= num_draws as f64;

    SiteSamples {
        thinned_draws,
        mean_pi,
        num_draws,
        ess,
        num_divergences,
        step_size,
        mean_num_steps: num_steps as f64 / num_draws.max(1) as f64,
        proposal,
        bridge_draws,
        log_marginal: f64::NAN,
    }
}

/// Number of draws from the Gaussian proposal of the marginal likelihood estimate
const NUM_PROPOSAL_DRAWS: usize = 200;

/// Log marginal likelihood of a site with bridge sampling (Meng & Wong 1996, as in the bridgesampling R package):
/// a Gaussian fitted to the first half of the chain is bridged with the thinned draws of the second half.
/// The proposal only has to overlap with this site's own posterior, so this works for any prior.
fn site_log_marginal(logp: &mut SiteLogp, samples: &SiteSamples, seed: u64) -> f64 {
    let (mean, cov) = samples.proposal.as_ref().unwrap();
    // A small jitter keeps the proposal proper if a coordinate barely moved
    let jitter = 1e-10 * cov.trace().max(1e-300) / NUM_COMPONENTS as f64;
    let chol = match (cov + SMatrix::<f64, NUM_COMPONENTS, NUM_COMPONENTS>::identity() * jitter).cholesky() {
        Some(chol) => chol,
        None => return f64::NAN,
    };
    let l = chol.l();
    let log_norm = -l.diagonal().map(f64::ln).sum() - 0.5 * NUM_COMPONENTS as f64 * (2.0 * std::f64::consts::PI).ln();
    let log_proposal = |x: &SVector<f64, NUM_COMPONENTS>| {
        let z = l.solve_lower_triangular(&(x - mean)).unwrap();
        log_norm - 0.5 * z.norm_squared()
    };
    let mut gradient = vec![0.0; NUM_COMPONENTS];
    let mut log_posterior = |x: &SVector<f64, NUM_COMPONENTS>| {
        logp.logp(x.as_slice(), &mut gradient).unwrap_or(f64::NEG_INFINITY)
    };

    // Standard normal draws with Box-Muller
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut uniform = || ((nuts_rs::rand::Rng::next_u64(&mut rng) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let mut normal = SVector::<f64, NUM_COMPONENTS>::zeros();
    let mut log_ratio_proposal = Vec::with_capacity(NUM_PROPOSAL_DRAWS);
    for _ in 0..NUM_PROPOSAL_DRAWS {
        for k in (0..NUM_COMPONENTS).step_by(2) {
            let r = (-2.0 * uniform().ln()).sqrt();
            let phi = 2.0 * std::f64::consts::PI * uniform();
            normal[k] = r * phi.cos();
            if k + 1 < NUM_COMPONENTS {
                normal[k + 1] = r * phi.sin();
            }
        }
        let x = mean + l * normal;
        log_ratio_proposal.push(log_posterior(&x) - log_proposal(&x));
    }
    let log_ratio_posterior: Vec<f64> = samples
        .bridge_draws
        .iter()
        .map(|x| log_posterior(x) - log_proposal(x))
        .collect();
    bridge_fixed_point(&log_ratio_posterior, &log_ratio_proposal)
}

fn log_sum_exp(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let max = values.clone().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return max;
    }
    max + values.map(|v| (v - max).exp()).sum::<f64>().ln()
}

/// log(a + b) for log a, log b
fn log_add_exp(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    if max == f64::NEG_INFINITY {
        return max;
    }
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Iterative optimal bridge estimate of log Z, given log(q / g) at the posterior draws (target q / Z)
/// and at the proposal draws (normalized proposal g)
fn bridge_fixed_point(log_ratio_posterior: &[f64], log_ratio_proposal: &[f64]) -> f64 {
    let n1 = log_ratio_posterior.len() as f64;
    let n2 = log_ratio_proposal.len() as f64;
    if n1 == 0.0 {
        return f64::NAN;
    }
    let log_s1 = (n1 / (n1 + n2)).ln();
    let log_s2 = (n2 / (n1 + n2)).ln();
    let mut sorted = log_ratio_posterior.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut log_z = sorted[sorted.len() / 2];
    for _ in 0..1000 {
        let numerator = log_sum_exp(log_ratio_proposal.iter().map(|&l| l - log_add_exp(log_s1 + l, log_s2 + log_z))) - n2.ln();
        let denominator = log_sum_exp(log_ratio_posterior.iter().map(|&l| -log_add_exp(log_s1 + l, log_s2 + log_z))) - n1.ln();
        let next = numerator - denominator;
        if !next.is_finite() {
            return f64::NAN;
        }
        let converged = (next - log_z).abs() < 1e-10;
        log_z = next;
        if converged {
            break;
        }
    }
    log_z
}

fn leaf_partial_likelihoods(alignment: &[u8], site: usize, num_leaves: usize) -> Vec<SVector<f64, 20>> {
    (0..num_leaves)
        .map(|seq| {
            let residue = alignment[site * num_leaves + seq];
            if residue < 20 {
                let mut v = SVector::<f64, 20>::zeros();
                v[residue as usize] = 1.0;
                v
            } else {
                SVector::<f64, 20>::from_element(1.0)
            }
        })
        .collect()
}

/// Settings of the Monte Carlo EM for Mu and the branch lengths
#[derive(Debug, Clone, Copy)]
pub struct EmSettings {
    /// Maximum number of EM iterations (E-step and M-step) before the final sampling, 0 keeps the MAP Mu and
    /// branch lengths
    pub iterations: usize,
    /// EM stops once the tree length changed by less than this fraction over the last `window` iterations,
    /// 0 always runs all iterations
    pub tree_length_tolerance: f64,
    /// Number of iterations the tree length change is measured over. The tree length is noisy from iteration to
    /// iteration because of the Monte Carlo E-step, but drifts slowly and steadily until EM has converged.
    pub window: usize,
    /// Posterior samples per site the M-step averages the log likelihood over
    pub num_samples: usize,
    /// NUTS tuning draws per site in the first E-step
    pub num_tune: u64,
    /// NUTS tuning draws per site in the later E-steps, whose chains start at the previous draws
    pub num_retune: u64,
    /// Maximum number of L-BFGS iterations per M-step
    pub max_mstep_iterations: usize,
    /// Convergence of the MAP estimate EM starts from
    pub map_convergence: Convergence,
    /// pi_reg of the MAP estimate, None uses the model's pi_reg
    pub map_pi_reg: Option<f64>,
}

impl EmSettings {
    /// Reads MUTSEL_EM_ITERATIONS, MUTSEL_EM_TOL, MUTSEL_EM_WINDOW, MUTSEL_EM_SAMPLES, MUTSEL_EM_TUNE, MUTSEL_EM_RETUNE, MUTSEL_EM_MSTEP_ITERATIONS,
    /// MUTSEL_MAP_MIN_ITERATIONS, MUTSEL_MAP_MAX_ITERATIONS, MUTSEL_MAP_REL_IMPROVEMENT and MUTSEL_MAP_PI_REG
    /// from the environment.
    pub fn from_env() -> EmSettings {
        EmSettings {
            iterations: env_or("MUTSEL_EM_ITERATIONS", 50),
            tree_length_tolerance: env_or("MUTSEL_EM_TOL", 0.005),
            window: env_or("MUTSEL_EM_WINDOW", 5),
            num_samples: env_or("MUTSEL_EM_SAMPLES", 8),
            num_tune: env_or("MUTSEL_EM_TUNE", 200),
            num_retune: env_or("MUTSEL_EM_RETUNE", 50),
            max_mstep_iterations: env_or("MUTSEL_EM_MSTEP_ITERATIONS", 50),
            // Same as optimize_internal by default
            map_convergence: Convergence {
                min_iterations: env_or("MUTSEL_MAP_MIN_ITERATIONS", Convergence::MAP.min_iterations),
                max_iterations: env_or("MUTSEL_MAP_MAX_ITERATIONS", Convergence::MAP.max_iterations),
                min_rel_improvement: env_or("MUTSEL_MAP_REL_IMPROVEMENT", Convergence::MAP.min_rel_improvement),
                no_improve_patience: Convergence::MAP.no_improve_patience,
            },
            map_pi_reg: std::env::var("MUTSEL_MAP_PI_REG").ok().map(|value| {
                value
                    .trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("Could not parse MUTSEL_MAP_PI_REG={}", value))
            }),
        }
    }

    /// Relative change of the tree length over the last `window` iterations, None if there are not enough iterations
    fn tree_length_change(&self, tree_lengths: &[f64]) -> Option<f64> {
        let n = tree_lengths.len();
        if self.window == 0 || n <= self.window {
            return None;
        }
        Some((tree_lengths[n - 1] - tree_lengths[n - 1 - self.window]).abs() / tree_lengths[n - 1])
    }

    /// NUTS settings of the E-step: about num_samples independent samples per site
    fn hmc_settings(&self, settings: &HmcSettings, iteration: usize) -> HmcSettings {
        HmcSettings {
            num_tune: if iteration == 0 { self.num_tune } else { self.num_retune },
            num_draws: (10 * self.num_samples).max(50) as u64,
            target_ess: self.num_samples,
            max_draws: settings.max_draws,
            num_output_samples: self.num_samples,
            seed: settings.seed.wrapping_add(iteration as u64 + 1),
            marginal: false,
        }
    }
}

/// Posterior mode of the pca coordinates of every site given the shared model, by L-BFGS from init[site]
fn site_modes(
    shared: &Arc<SharedModel>,
    tree: &Arc<FelsensteinTree<20>>,
    alignment: &[u8],
    num_leaves: usize,
    init: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let num_nodes = tree.num_nodes();
    (0..init.len())
        .into_par_iter()
        .map(|site| {
            let mut logp = SiteLogp {
                shared: Arc::clone(shared),
                tree: Arc::clone(tree),
                leaf_pl: leaf_partial_likelihoods(alignment, site, num_leaves),
                pl_buffer: vec![SVector::<f64, 20>::zeros(); num_nodes],
            };
            maximize_lbfgs(&mut logp, &init[site], 100, 1e-3)
        })
        .collect()
}

/// Maximizes logp with L-BFGS (memory 6) and a backtracking line search, starting at init. Returns init if
/// logp is not finite there.
fn maximize_lbfgs(logp: &mut SiteLogp, init: &[f64], max_iterations: usize, gradient_tolerance: f64) -> Vec<f64> {
    let eval = |x: &DVector<f64>| -> Option<(f64, DVector<f64>)> {
        let mut gradient = [0.0; NUM_COMPONENTS];
        let value = logp.logp(x.as_slice(), &mut gradient).ok()?;
        Some((value, DVector::from_column_slice(&gradient)))
    };
    lbfgs(eval, DVector::from_column_slice(init), max_iterations, gradient_tolerance, 0.0)
        .as_slice()
        .to_vec()
}

/// Maximizes f with L-BFGS (memory 6) and a backtracking line search, starting at init, until the gradient norm
/// is below gradient_tolerance or an iteration improves f by less than value_tolerance relative to |f|. eval
/// returns f and its gradient, None where f is not finite. Returns init if f is not finite there; the result is
/// never worse than init.
fn lbfgs<F: FnMut(&DVector<f64>) -> Option<(f64, DVector<f64>)>>(
    mut eval: F,
    init: DVector<f64>,
    max_iterations: usize,
    gradient_tolerance: f64,
    value_tolerance: f64,
) -> DVector<f64> {
    let mut x = init;
    let Some((mut value, mut gradient)) = eval(&x) else {
        return x;
    };
    // (s, y) pairs of the minimization of -f
    let mut history: std::collections::VecDeque<(DVector<f64>, DVector<f64>)> = std::collections::VecDeque::new();
    for _ in 0..max_iterations {
        if gradient.norm() < gradient_tolerance {
            break;
        }
        // Two-loop recursion for the ascent direction
        let mut q = gradient.clone();
        let mut coefficients = Vec::with_capacity(history.len());
        for (s, y) in history.iter().rev() {
            let rho = 1.0 / y.dot(s);
            let alpha = rho * s.dot(&q);
            q -= y * alpha;
            coefficients.push((alpha, rho));
        }
        let gamma = match history.back() {
            Some((s, y)) => s.dot(y) / y.dot(y),
            None => 1.0f64.min(1.0 / gradient.norm()),
        };
        let mut direction = q * gamma;
        for ((s, y), (alpha, rho)) in history.iter().zip(coefficients.iter().rev()) {
            let beta = rho * y.dot(&direction);
            direction += s * (alpha - beta);
        }
        if gradient.dot(&direction) <= 0.0 {
            history.clear();
            direction = &gradient * 1.0f64.min(1.0 / gradient.norm());
        }

        let slope = gradient.dot(&direction);
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..30 {
            let candidate = &x + &direction * step;
            if let Some((candidate_value, candidate_gradient)) = eval(&candidate) {
                if candidate_value >= value + 1e-4 * step * slope {
                    accepted = Some((candidate, candidate_value, candidate_gradient));
                    break;
                }
            }
            step *= 0.5;
        }
        let Some((next, next_value, next_gradient)) = accepted else {
            break;
        };
        let improvement = next_value - value;
        let s = &next - &x;
        let y = &gradient - &next_gradient;
        if s.dot(&y) > 1e-12 {
            history.push_back((s, y));
            if history.len() > 6 {
                history.pop_front();
            }
        }
        x = next;
        value = next_value;
        gradient = next_gradient;
        if improvement <= value_tolerance * value.abs() {
            break;
        }
    }
    x
}

/// Runs an independent NUTS chain for every site, starting at init[site]
fn sample_sites(
    shared: &Arc<SharedModel>,
    tree: &Arc<FelsensteinTree<20>>,
    alignment: &[u8],
    num_leaves: usize,
    init: &[Vec<f64>],
    settings: &HmcSettings,
    verbosity: Verbosity,
    label: &str,
) -> Vec<SiteSamples> {
    let num_sites = init.len();
    let num_nodes = tree.num_nodes();
    let start = std::time::Instant::now();
    let num_done = AtomicUsize::new(0);
    let report_every = (num_sites / 10).max(1);

    let site_samples: Vec<SiteSamples> = (0..num_sites)
        .into_par_iter()
        .map(|site| {
            let logp = SiteLogp {
                shared: Arc::clone(shared),
                tree: Arc::clone(tree),
                leaf_pl: leaf_partial_likelihoods(alignment, site, num_leaves),
                pl_buffer: vec![SVector::<f64, 20>::zeros(); num_nodes],
            };
            let mut samples = sample_site(logp, &init[site], settings, site);
            if settings.marginal {
                let mut logp = SiteLogp {
                    shared: Arc::clone(shared),
                    tree: Arc::clone(tree),
                    leaf_pl: leaf_partial_likelihoods(alignment, site, num_leaves),
                    pl_buffer: vec![SVector::<f64, 20>::zeros(); num_nodes],
                };
                samples.log_marginal = site_log_marginal(&mut logp, &samples, settings.seed ^ (site as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            }

            let done = num_done.fetch_add(1, Ordering::Relaxed) + 1;
            if verbosity.should_print(Verbosity::Min) && (done % report_every == 0 || done == num_sites) {
                println!(
                    "MUTSEL HMC {}: {}/{} sites sampled ({:.1}s)",
                    label,
                    done,
                    num_sites,
                    start.elapsed().as_secs_f64()
                );
            }
            samples
        })
        .collect();

    let total_divergences: usize = site_samples.iter().map(|s| s.num_divergences).sum();
    let sites_with_divergences = site_samples.iter().filter(|s| s.num_divergences > 0).count();
    let total_draws: usize = site_samples.iter().map(|s| s.num_draws).sum();
    let mean_steps = site_samples
        .iter()
        .map(|s| s.mean_num_steps * s.num_draws as f64)
        .sum::<f64>()
        / total_draws.max(1) as f64;
    let min_ess = site_samples.iter().map(|s| s.ess).fold(f64::INFINITY, f64::min);
    let sites_below_target = site_samples
        .iter()
        .filter(|s| s.ess < settings.target_ess as f64)
        .count();
    println!(
        "MUTSEL HMC {}: done in {:.1}s, {} draws per site on average, {} divergent transitions in {} sites, mean {:.1} leapfrog steps per draw",
        label,
        start.elapsed().as_secs_f64(),
        total_draws / num_sites.max(1),
        total_divergences,
        sites_with_divergences,
        mean_steps
    );
    println!(
        "MUTSEL HMC {}: minimum ESS over sites {:.0}, {} sites below the target ESS {} after {} draws",
        label, min_ess, sites_below_target, settings.target_ess, settings.max_draws
    );
    site_samples
}

/// log pi [num_sites, 20] (up to a constant per site) for each of the thinned draws
fn thinned_log_pi(site_samples: &[SiteSamples], num_samples: usize, pca: &PCA) -> Result<Vec<Tensor>, candle_core::Error> {
    (0..num_samples)
        .map(|k| {
            let coordinates: Vec<f64> = site_samples
                .iter()
                .flat_map(|s| s.thinned_draws[k * NUM_COMPONENTS..(k + 1) * NUM_COMPONENTS].iter().copied())
                .collect();
            let coordinates =
                Tensor::from_vec(coordinates, &[site_samples.len(), NUM_COMPONENTS], &candle_core::Device::Cpu)?;
            Ok(pca.pca_coordinates_to_log_freq(&coordinates))
        })
        .collect()
}

/// Average substitution rate over all sites and samples, without normalization
fn mean_substitution_rate(Mu: &Tensor, log_pi_samples: &[Tensor]) -> Result<f64, candle_core::Error> {
    let mut sum = 0.0;
    for log_pi in log_pi_samples {
        let (S, sqrt_pi) = calc_rate_matrix(Mu, log_pi, &tensor_full(1.0, &[]));
        sum += model::substitution_rates_tensor(&S, &sqrt_pi).mean_all()?.to_scalar::<f64>()?;
    }
    Ok(sum / log_pi_samples.len() as f64)
}

/// M-step: maximizes the log likelihood averaged over the posterior samples of the site frequencies
/// minus the penalties on Mu and the branch lengths (as in optimization::ModelParameters),
/// with the rate normalization fixed. Returns the averaged log likelihood at the optimum.
fn m_step(
    op: &FelsensteinWithEdgeOp,
    log_R: &Var,
    init_log_R: &Tensor,
    log_branch_lengths: &Var,
    log_pi_samples: &[Tensor],
    rate_scaling: f64,
    mutsel_params: MutselParams,
    max_iterations: usize,
    verbosity: Verbosity,
) -> Result<f64, candle_core::Error> {
    // L-BFGS on the deterministic objective (given the samples), stopped once an iteration improves it by less
    // than this relative to its value. It starts at the previous Mu and branch lengths and never ends worse.
    const REL_IMPROVEMENT: f64 = 1e-7;

    let num_samples = log_pi_samples.len() as f64;
    let log_pi_samples = log_pi_samples
        .iter()
        .map(batch::rows_from_tensor)
        .collect::<Result<Vec<_>, _>>()?;
    let R_shape = log_R.as_tensor().dims().to_vec();
    let num_R = log_R.as_tensor().elem_count();
    let num_branches = log_branch_lengths.as_tensor().elem_count();

    // Sets the variables to x and returns the expected log likelihood, the objective
    // (expected log likelihood minus penalty) and its gradient with respect to x
    let evaluate = |x: &DVector<f64>| -> Result<(f64, f64, DVector<f64>), candle_core::Error> {
        let device = candle_core::Device::Cpu;
        log_R.set(&Tensor::from_slice(&x.as_slice()[..num_R], R_shape.as_slice(), &device)?)?;
        log_branch_lengths.set(&Tensor::from_slice(&x.as_slice()[num_R..], &[num_branches], &device)?)?;
        let branch_lengths = log_branch_lengths.exp()?;
        let penalty = (Mu_penalty(log_R, init_log_R, mutsel_params.Mu_reg)
            + (branch_lengths.sum_all()? * mutsel_params.branch_length_reg)?)?;

        // Log likelihood averaged over the samples, parallel over the sites
        let Mu_tensor = Mu(log_R);
        let Mu_matrix = batch::smatrix_from_tensor(&Mu_tensor)?;
        let branch_length_values = branch_lengths.to_vec1::<f64>()?;
        let mut log_likelihood = 0.0;
        let mut grad_Mu = SMatrix::<f64, 20, 20>::zeros();
        let mut grad_branch_lengths = vec![0.0; branch_length_values.len()];
        for log_pi in &log_pi_samples {
            let result = batch::log_likelihood_and_gradients(
                op.tree(),
                &Mu_matrix,
                log_pi,
                &branch_length_values,
                batch::RateScaling::Fixed(rate_scaling),
            );
            log_likelihood += result.log_likelihood / num_samples;
            grad_Mu += result.grad_Mu / num_samples;
            for (sum, g) in grad_branch_lengths.iter_mut().zip(&result.grad_branch_lengths) {
                *sum += g / num_samples;
            }
        }
        let likelihood = batch::linear_surrogate(
            log_likelihood,
            &[
                (&Mu_tensor, batch::smatrix_to_tensor(&grad_Mu)?),
                (&branch_lengths, Tensor::from_vec(grad_branch_lengths, &[num_branches], &device)?),
            ],
        )?;
        let grads = (likelihood - &penalty)?.backward()?;
        let mut gradient = Vec::with_capacity(num_R + num_branches);
        for var in [log_R, log_branch_lengths] {
            match grads.get(var.as_tensor()) {
                Some(g) => gradient.extend(g.flatten_all()?.to_vec1::<f64>()?),
                None => gradient.extend(std::iter::repeat_n(0.0, var.as_tensor().elem_count())),
            }
        }
        let objective = log_likelihood - penalty.to_scalar::<f64>()?;
        Ok((log_likelihood, objective, DVector::from_vec(gradient)))
    };

    let mut init = log_R.as_tensor().flatten_all()?.to_vec1::<f64>()?;
    init.extend(log_branch_lengths.as_tensor().to_vec1::<f64>()?);
    let start = std::time::Instant::now();
    let mut num_evaluations = 0;
    let optimum = lbfgs(
        |x| {
            num_evaluations += 1;
            let (_, objective, gradient) = evaluate(x).ok()?;
            objective.is_finite().then_some((objective, gradient))
        },
        DVector::from_vec(init),
        max_iterations,
        0.0,
        REL_IMPROVEMENT,
    );
    let (log_likelihood, objective, _) = evaluate(&optimum)?;
    if verbosity.should_print(Verbosity::Med) {
        println!(
            "MUTSEL HMC M-step: expected log likelihood {:.3}, objective {:.3} after {} evaluations ({:.1}s)",
            log_likelihood,
            objective,
            num_evaluations,
            start.elapsed().as_secs_f64()
        );
    }
    Ok(log_likelihood)
}

/// Writes the log likelihood of every draw ([num_sites, num_draws] as draw_log_likelihood) and of the given
/// frequencies (mean_pi_log_likelihood [num_sites]) on this tree and alignment to path
fn write_draw_log_likelihoods(
    path: &str,
    shared: &Arc<SharedModel>,
    tree: &Arc<FelsensteinTree<20>>,
    alignment: &[u8],
    num_leaves: usize,
    draws: &[Vec<f64>],
    mean_pi: &[SVector<f64, 20>],
) -> Result<(), candle_core::Error> {
    let num_sites = draws.len();
    let num_draws = draws[0].len() / NUM_COMPONENTS;
    let num_nodes = tree.num_nodes();
    let (draw_ll, mean_ll): (Vec<Vec<f64>>, Vec<f64>) = (0..num_sites)
        .into_par_iter()
        .map(|site| {
            let leaf_pl = leaf_partial_likelihoods(alignment, site, num_leaves);
            let mut pl = vec![SVector::<f64, 20>::zeros(); num_nodes];
            let mut log_likelihood = |S: &SMatrix<f64, 20, 20>, sqrt_pi: &SVector<f64, 20>| {
                pl[..num_leaves].copy_from_slice(&leaf_pl);
                tree.calculate_likelihood_single_side(S.as_view(), sqrt_pi.as_view(), &mut pl)
            };
            let draw_ll = draws[site]
                .chunks_exact(NUM_COMPONENTS)
                .map(|x| {
                    let eval = shared.evaluate(&SVector::<f64, NUM_COMPONENTS>::from_column_slice(x));
                    log_likelihood(&eval.S, &eval.sqrt_pi)
                })
                .collect();
            let (S, sqrt_pi) = shared.rate_matrix(&mean_pi[site].map(f64::ln));
            (draw_ll, log_likelihood(&S, &sqrt_pi))
        })
        .unzip();
    Tensor::write_npz(
        &[
            (
                "draw_log_likelihood",
                &Tensor::from_vec(draw_ll.concat(), &[num_sites, num_draws], &candle_core::Device::Cpu)?,
            ),
            ("mean_pi_log_likelihood", &Tensor::from_vec(mean_ll, &[num_sites], &candle_core::Device::Cpu)?),
        ],
        Path::new(path),
    )?;
    println!("MUTSEL HMC: log likelihoods of the draws written to {}", path);
    Ok(())
}

/// MUTSEL_EVAL_SAMPLES: no sampling, the draws (pca_samples) and posterior mean frequencies (mean_pi) of an
/// earlier run, e.g. on a subset of the taxa, are evaluated on this tree and alignment (written to
/// <out_prefix>.draw_ll.npz). Returns the model at those posterior mean frequencies.
fn evaluate_samples(
    path: &Path,
    shared: &Arc<SharedModel>,
    tree: &Arc<FelsensteinTree<20>>,
    alignment: &[u8],
    num_sites: usize,
    num_leaves: usize,
    Mu: &Tensor,
    rate_scaling: f64,
    out_prefix: &str,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let arrays: std::collections::HashMap<String, Tensor> = Tensor::read_npz(path)?.into_iter().collect();
    let get = |name: &str| {
        arrays
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("{} not found in {}", name, path.display())))
    };
    let pca_samples = get("pca_samples")?;
    let mean_pi = get("mean_pi")?;
    if pca_samples.dim(0)? != num_sites {
        return Err(candle_core::Error::Msg(format!("{} does not have {} sites", path.display(), num_sites)));
    }
    println!("MUTSEL HMC: evaluating the draws in {} on this tree and alignment, no sampling", path.display());
    let draws: Vec<Vec<f64>> = (0..num_sites)
        .map(|site| pca_samples.get(site)?.flatten_all()?.to_vec1::<f64>())
        .collect::<Result<_, _>>()?;
    let mean_pi_rows = batch::rows_from_tensor(&mean_pi)?;
    write_draw_log_likelihoods(
        &format!("{}.draw_ll.npz", out_prefix),
        shared,
        tree,
        alignment,
        num_leaves,
        &draws,
        &mean_pi_rows,
    )?;
    Ok(calc_rate_matrix(Mu, &mean_pi.log()?, &tensor_full(rate_scaling, &[])))
}

/// Mu, the global rate normalization and the branch lengths the final sampling is conditioned on
struct FittedModel {
    Mu: Tensor,
    rate_scaling: f64,
    branch_lengths: Tensor,
    /// Initial position of the final chain of every site
    positions: Vec<Vec<f64>>,
    init_pca_tensor: Tensor,
    em_log_likelihood: Vec<f64>,
    em_tree_length: Vec<f64>,
}

impl FittedModel {
    /// Reads Mu, rate_scaling and branch_lengths of an earlier run (its .hmc.npz file), so its posterior can be
    /// sampled again under a different prior. The chains start at the last sample of every site.
    /// With tree_branch_lengths the branch lengths are these instead, e.g. those of a tree with fewer taxa.
    fn read(
        path: &Path,
        num_nodes: usize,
        num_sites: usize,
        tree_branch_lengths: Option<&[f64]>,
    ) -> Result<FittedModel, candle_core::Error> {
        let arrays: std::collections::HashMap<String, Tensor> = Tensor::read_npz(path)?.into_iter().collect();
        let get = |name: &str| {
            arrays
                .get(name)
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("{} not found in {}", name, path.display())))
        };
        let branch_lengths = match tree_branch_lengths {
            Some(lengths) => Tensor::from_slice(lengths, &[lengths.len()], &candle_core::Device::Cpu)?,
            None => get("branch_lengths")?,
        };
        let pca_samples = get("pca_samples")?;
        if branch_lengths.dims() != [num_nodes] || pca_samples.dim(0)? != num_sites {
            return Err(candle_core::Error::Msg(format!(
                "{} does not match the tree ({} nodes) and alignment ({} sites)",
                path.display(),
                num_nodes,
                num_sites
            )));
        }
        let last = pca_samples.dim(1)? - 1;
        if tree_branch_lengths.is_some() {
            println!("MUTSEL HMC: Mu and rate normalization fixed to the ones in {}, branch lengths of the input tree, no MAP estimate and EM", path.display());
        } else {
            println!("MUTSEL HMC: Mu, rate normalization and branch lengths fixed to the ones in {}, no MAP estimate and EM", path.display());
        }
        Ok(FittedModel {
            Mu: get("Mu")?,
            rate_scaling: get("rate_scaling")?.to_scalar::<f64>()?,
            branch_lengths,
            positions: pca_samples.narrow(1, last, 1)?.squeeze(1)?.to_vec2::<f64>()?,
            init_pca_tensor: get("init_pca_coordinates")?,
            em_log_likelihood: vec![],
            em_tree_length: vec![],
        })
    }
}

/// Mu and the branch lengths from the MAP estimate refined with Monte Carlo EM, see sample_internal
fn fit_with_em(
    parents: &[i32],
    branch_lengths: &[f64],
    alignment: &[u8],
    num_sites: usize,
    num_leaves: usize,
    mutsel_params: MutselParams,
    prior_R_file: Option<&Path>,
    verbosity: Verbosity,
    out_prefix: &str,
    settings: &HmcSettings,
    em: &EmSettings,
    pca: &Arc<PCA>,
    mixture_prior: &Option<Arc<MixturePrior>>,
) -> Result<FittedModel, candle_core::Error> {
    let felsenstein = crate::create_felsenstein_tree(parents, branch_lengths, alignment, num_sites, num_leaves);
    let op = FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));
    // The Mu penalty stays relative to the prior Mu
    let (init_log_R, log_R, log_branch_lengths, edge_op, init_pca_tensor, start) = match std::env::var("MUTSEL_INIT_MU") {
        Ok(path) => {
            // Mu from a file (e.g. the average over training alignments), the branch lengths of the input tree times
            // MUTSEL_INIT_TREE_SCALE and the light PMSF site frequencies, which are moved to the site posterior modes below
            let path = Path::new(path.trim());
            let arrays: std::collections::HashMap<String, Tensor> = Tensor::read_npz(path)?.into_iter().collect();
            let init_Mu = arrays
                .get("Mu")
                .ok_or_else(|| candle_core::Error::Msg(format!("Mu not found in {}", path.display())))?;
            let scale: f64 = env_or("MUTSEL_INIT_TREE_SCALE", 1.0);
            let init_log_R = crate::optimization::initial_log_R(prior_R_file)?;
            let log_R = Var::from_tensor(&crate::optimization::log_R_from_Mu(init_Mu, &init_log_R)?)?;
            let scaled: Vec<f64> = branch_lengths.iter().map(|b| b * scale).collect();
            let log_branch_lengths = Var::from_tensor(&Tensor::from_slice(&scaled, &[scaled.len()], &candle_core::Device::Cpu)?.log()?)?;
            let site_freq = crate::optimization::light_pmsf(
                op.into_with_edge_op(),
                crate::data::UDM256,
                crate::data::UDM256_WEIGHTS,
                log_branch_lengths.as_tensor(),
            );
            let init_pca_tensor = pca.log_freq_to_pca_coordinates(&site_freq.log()?);
            let start = format!("Mu from {}, input tree branch lengths x {}", path.display(), scale);
            (init_log_R, log_R, log_branch_lengths, op.into_with_edge_op(), init_pca_tensor, start)
        }
        Err(_) => {
            // The joint MAP estimate of Mu, the site frequencies and the branch lengths,
            // the same estimate as optimization::optimize_internal with the default settings
            let map_params = MutselParams {
                pi_reg: em.map_pi_reg.unwrap_or(mutsel_params.pi_reg),
                ..mutsel_params
            };
            if verbosity.should_print(Verbosity::Min) {
                println!("MUTSEL HMC: MAP estimate with pi_reg {}", map_params.pi_reg);
            }
            let map_start = std::time::Instant::now();
            let map = map_estimate(
                op,
                branch_lengths,
                map_params,
                prior_R_file,
                em.map_convergence,
                verbosity,
                out_prefix,
            )?;
            if verbosity.should_print(Verbosity::Min) {
                println!("MUTSEL HMC: MAP estimate done in {:.1}s", map_start.elapsed().as_secs_f64());
            }
            (
                map.init_log_R.clone(),
                Var::from_tensor(map.log_R.as_tensor())?,
                Var::from_tensor(map.log_branch_lengths.as_tensor())?,
                map.felsenstein_op.clone(),
                map.pca_coordinates.as_tensor().copy()?,
                "the MAP estimate".to_string(),
            )
        }
    };
    let init_log_pi = pca.pca_coordinates_to_log_freq(&init_pca_tensor);

    // Initial normalization: average substitution rate 1 at the initial frequencies, the same as in the
    // MAP estimate, so the branch lengths stay valid.
    let mut rate_scaling = 1.0 / mean_substitution_rate(&Mu(log_R.as_tensor()).detach(), &[init_log_pi])?;
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL HMC: starting from {}, tree length {:.4}, rate normalization 1/{:.5}",
            start,
            log_branch_lengths.exp()?.sum_all()?.to_scalar::<f64>()?,
            1.0 / rate_scaling
        );
    }

    let init_pca_tensor = if std::env::var("MUTSEL_INIT_MU").is_ok() {
        // The light PMSF frequencies are much flatter than the site posteriors and so have a higher average
        // substitution rate, and so do the site modes. The normalization is therefore taken from posterior draws
        // of a subset of the sites (MUTSEL_INIT_PILOT_SITES, evenly spaced), iterated twice because the draws
        // depend on it, so that the branch lengths keep the meaning of the input tree's.
        let pilot_start = std::time::Instant::now();
        let init_Mu = Mu(log_R.as_tensor()).detach();
        let tree = Arc::new(FelsensteinTree::<20>::new(parents, &log_branch_lengths.exp()?.to_vec1::<f64>()?));
        let init_positions = init_pca_tensor.to_vec2::<f64>()?;
        let num_pilot: usize = env_or("MUTSEL_INIT_PILOT_SITES", 2000).clamp(1, num_sites);
        let pilot_sites: Vec<usize> = (0..num_pilot).map(|i| i * num_sites / num_pilot).collect();
        let pilot_alignment: Vec<u8> = pilot_sites
            .iter()
            .flat_map(|&site| alignment[site * num_leaves..(site + 1) * num_leaves].iter().copied())
            .collect();
        let mut pilot_positions: Vec<Vec<f64>> = pilot_sites.iter().map(|&site| init_positions[site].clone()).collect();
        for pass in 0..2 {
            let shared = Arc::new(
                SharedModel::new(&init_Mu, rate_scaling, pca, mutsel_params.pi_reg).with_mixture_prior(mixture_prior.clone()),
            );
            let pilot_settings = em.hmc_settings(&settings, pass);
            let site_samples = sample_sites(
                &shared,
                &tree,
                &pilot_alignment,
                num_leaves,
                &pilot_positions,
                &pilot_settings,
                Verbosity::Quiet,
                "pilot sampling",
            );
            let last = (em.num_samples - 1) * NUM_COMPONENTS;
            pilot_positions = site_samples
                .iter()
                .map(|s| s.thinned_draws[last..last + NUM_COMPONENTS].to_vec())
                .collect();
            rate_scaling = 1.0 / mean_substitution_rate(&init_Mu, &thinned_log_pi(&site_samples, em.num_samples, pca)?)?;
            if verbosity.should_print(Verbosity::Min) {
                println!(
                    "MUTSEL HMC: pilot sampling pass {} of {} sites, rate normalization 1/{:.5}",
                    pass + 1,
                    num_pilot,
                    1.0 / rate_scaling
                );
            }
        }
        let shared = Arc::new(
            SharedModel::new(&init_Mu, rate_scaling, pca, mutsel_params.pi_reg).with_mixture_prior(mixture_prior.clone()),
        );
        let modes = site_modes(&shared, &tree, alignment, num_leaves, &init_positions);
        if verbosity.should_print(Verbosity::Min) {
            println!(
                "MUTSEL HMC: pilot normalization and site posterior modes done in {:.1}s",
                pilot_start.elapsed().as_secs_f64()
            );
        }
        Tensor::from_vec(modes.concat(), &[num_sites, NUM_COMPONENTS], &candle_core::Device::Cpu)?
    } else {
        init_pca_tensor
    };

    let mut positions = init_pca_tensor.to_vec2::<f64>()?;
    let mut em_log_likelihood = Vec::with_capacity(em.iterations);
    let mut em_tree_length = Vec::with_capacity(em.iterations);
    for iteration in 0..em.iterations {
        let label = format!("EM iteration {}", iteration + 1);
        let old_Mu = Mu(log_R.as_tensor()).detach();
        let shared = Arc::new(SharedModel::new(&old_Mu, rate_scaling, pca, mutsel_params.pi_reg).with_mixture_prior(mixture_prior.clone()));
        let tree = Arc::new(FelsensteinTree::<20>::new(
            parents,
            &log_branch_lengths.exp()?.to_vec1::<f64>()?,
        ));
        let em_settings = em.hmc_settings(&settings, iteration);
        let site_samples = sample_sites(&shared, &tree, alignment, num_leaves, &positions, &em_settings, verbosity, &label);

        // Next chains start at the last draw, which is the last thinned draw
        let last = (em.num_samples - 1) * NUM_COMPONENTS;
        positions = site_samples
            .iter()
            .map(|s| s.thinned_draws[last..last + NUM_COMPONENTS].to_vec())
            .collect();

        let log_pi_samples = thinned_log_pi(&site_samples, em.num_samples, pca)?;
        drop(site_samples);
        let log_likelihood = m_step(
            &edge_op,
            &log_R,
            &init_log_R,
            &log_branch_lengths,
            &log_pi_samples,
            rate_scaling,
            mutsel_params,
            em.max_mstep_iterations,
            verbosity,
        )?;

        let new_Mu = Mu(log_R.as_tensor()).detach();
        let new_rate_scaling = 1.0 / mean_substitution_rate(&new_Mu, &log_pi_samples)?;
        // Rescale the branch lengths so the likelihood is unchanged with the new normalization
        log_branch_lengths.set(&(log_branch_lengths.as_tensor() + (rate_scaling / new_rate_scaling).ln())?)?;
        rate_scaling = new_rate_scaling;

        let tree_length = log_branch_lengths.exp()?.sum_all()?.to_scalar::<f64>()?;
        let max_rel_Mu_change = ((&new_Mu - &old_Mu)?.abs()? / &old_Mu)?
            .max_all()?
            .to_scalar::<f64>()?;
        em_log_likelihood.push(log_likelihood);
        em_tree_length.push(tree_length);
        if verbosity.should_print(Verbosity::Min) {
            println!(
                "MUTSEL HMC {}: expected log likelihood {:.3}, tree length {:.4}, rate normalization 1/{:.5}, max relative change of Mu {:.4}",
                label,
                log_likelihood,
                tree_length,
                1.0 / rate_scaling,
                max_rel_Mu_change
            );
        }
        if let Some(change) = em.tree_length_change(&em_tree_length) {
            if change < em.tree_length_tolerance {
                println!(
                    "MUTSEL HMC: EM converged after {} iterations, tree length changed by {:.2}% over the last {} iterations",
                    iteration + 1,
                    100.0 * change,
                    em.window
                );
                break;
            }
        }
        if iteration + 1 == em.iterations && em.tree_length_tolerance > 0.0 {
            println!("MUTSEL HMC: WARNING: EM stopped after the maximum of {} iterations without converging", em.iterations);
        }
    }

    Ok(FittedModel {
        Mu: Mu(log_R.as_tensor()).detach(),
        rate_scaling,
        branch_lengths: log_branch_lengths.exp()?,
        positions,
        init_pca_tensor,
        em_log_likelihood,
        em_tree_length,
    })
}

/// Samples the PCA coordinates of every site from its posterior and returns (S, sqrt_pi)
/// evaluated at the posterior mean of the site frequencies, or with num_tree_draws > 1 at that many
/// evenly spaced posterior draws per site ([num_sites * num_tree_draws], ordered by site), which
/// IQ-TREE uses as an equally weighted mixture per site.
///
/// Mu and the branch lengths start at the joint MAP estimate (optimization::map_estimate, which also
/// gives the initial site positions) and are then refined with Monte Carlo EM: the E-step samples the sites
/// given Mu and the branch lengths, the M-step maximizes the log likelihood averaged over these
/// samples plus the log prior of Mu and the branch lengths. This converges to their MAP estimate with
/// the site frequencies integrated out. The rate normalization is kept fixed within each iteration and
/// afterwards reset to an average substitution rate of 1 over the posterior samples, rescaling the
/// branch lengths so that the likelihood stays the same.
pub fn sample_internal(
    parents: &[i32],
    branch_lengths: &[f64],
    alignment: &[u8],
    num_sites: usize,
    num_leaves: usize,
    mutsel_params: MutselParams,
    prior_R_file: Option<&Path>,
    verbosity: Verbosity,
    out_prefix: &str,
    num_tree_draws: usize,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let settings = HmcSettings::from_env();
    let em = EmSettings::from_env();
    println!(
        "MUTSEL HMC: sampling pca coordinates per site with NUTS ({} tuning, at least {} draws, until ESS >= {} or {} draws, {} thinned samples per site, seed {}), site rates 1",
        settings.num_tune,
        settings.num_draws,
        settings.target_ess,
        settings.max_draws,
        settings.num_output_samples,
        settings.seed
    );
    println!(
        "MUTSEL HMC: Mu and branch lengths from the MAP estimate ({}-{} iterations, relative improvement {:.0e}) refined by Monte Carlo EM until the tree length changes by less than {:.1}% over {} iterations, at most {} iterations ({} samples per site, {} tuning draws in the first and {} in later iterations, at most {} M-step iterations)",
        em.map_convergence.min_iterations,
        em.map_convergence.max_iterations,
        em.map_convergence.min_rel_improvement,
        100.0 * em.tree_length_tolerance,
        em.window,
        em.iterations,
        em.num_samples,
        em.num_tune,
        em.num_retune,
        em.max_mstep_iterations
    );
    assert!(settings.num_output_samples >= 1 && (em.iterations == 0 || em.num_samples >= 1));
    assert!(
        num_tree_draws >= 1 && num_tree_draws <= settings.num_output_samples,
        "MUTSEL_TREE_DRAWS must be between 1 and MUTSEL_HMC_SAMPLES"
    );

    let pca = Arc::new(PCA::new(NUM_COMPONENTS));
    let mixture_prior = MixturePrior::from_env();
    let fitted = match std::env::var("MUTSEL_FIXED_MODEL") {
        Ok(path) => {
            // MUTSEL_FIXED_BRANCH_LENGTHS=tree keeps the branch lengths of the input tree
            let tree_lengths = std::env::var("MUTSEL_FIXED_BRANCH_LENGTHS").map_or(false, |v| v.trim() == "tree");
            FittedModel::read(Path::new(path.trim()), parents.len(), num_sites, tree_lengths.then_some(branch_lengths))?
        }
        Err(_) => fit_with_em(
            parents,
            branch_lengths,
            alignment,
            num_sites,
            num_leaves,
            mutsel_params,
            prior_R_file,
            verbosity,
            out_prefix,
            &settings,
            &em,
            &pca,
            &mixture_prior,
        )?,
    };
    let FittedModel {
        Mu,
        rate_scaling,
        branch_lengths,
        positions,
        init_pca_tensor,
        em_log_likelihood,
        em_tree_length,
    } = fitted;
    let shared = Arc::new(SharedModel::new(&Mu, rate_scaling, &pca, mutsel_params.pi_reg).with_mixture_prior(mixture_prior));
    let tree = Arc::new(FelsensteinTree::<20>::new(parents, &branch_lengths.to_vec1::<f64>()?));
    if let Ok(path) = std::env::var("MUTSEL_EVAL_SAMPLES") {
        return evaluate_samples(Path::new(path.trim()), &shared, &tree, alignment, num_sites, num_leaves, &Mu, rate_scaling, out_prefix);
    }
    let site_samples = sample_sites(&shared, &tree, alignment, num_leaves, &positions, &settings, verbosity, "final");
    let rate_scaling = tensor_full(rate_scaling, &[]);

    // Tensor::write_npz does not support zip entries or files of 4 GiB or more, so pca_samples and
    // pi_samples together are kept below that by writing an evenly spaced subset of the thinned draws.
    const MAX_SAMPLE_BYTES: usize = (1 << 32) - (1 << 26);
    let bytes_per_sample = num_sites * (NUM_COMPONENTS + 20) * std::mem::size_of::<f64>();
    let num_output_samples = settings
        .num_output_samples
        .min(MAX_SAMPLE_BYTES / bytes_per_sample.max(1))
        .max(1);
    if num_output_samples < settings.num_output_samples {
        println!(
            "MUTSEL HMC: WARNING: writing only {} of the {} samples per site, the npz file would exceed 4 GiB",
            num_output_samples, settings.num_output_samples
        );
    }
    // pca_samples [num_sites, num_output_samples, 19]
    let pca_samples: Vec<f64> = site_samples
        .iter()
        .flat_map(|s| {
            (0..num_output_samples).flat_map(move |i| {
                let draw = i * settings.num_output_samples / num_output_samples;
                s.thinned_draws[draw * NUM_COMPONENTS..(draw + 1) * NUM_COMPONENTS].iter().copied()
            })
        })
        .collect();
    let pca_samples = Tensor::from_vec(
        pca_samples,
        &[num_sites, num_output_samples, NUM_COMPONENTS],
        &candle_core::Device::Cpu,
    )?;

    // Posterior mean of the site frequencies over all draws
    let mean_pi: Vec<f64> = site_samples.iter().flat_map(|s| s.mean_pi.iter().copied()).collect();
    let mean_pi = Tensor::from_vec(mean_pi, &[num_sites, 20], &candle_core::Device::Cpu)?;

    let (S, sqrt_pi) = calc_rate_matrix(&Mu, &mean_pi.log()?, &rate_scaling);

    let average_rate_mean_pi = model::substitution_rates_tensor(&S, &sqrt_pi)
        .mean_all()?
        .to_scalar::<f64>()?;
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL HMC: average substitution rate at posterior mean frequencies {:.4}",
            average_rate_mean_pi
        );
    }

    let npz_path = format!("{}.hmc.npz", out_prefix);
    Tensor::write_npz(
        &[
            ("pca_samples", &pca_samples),
            (
                "pi_samples",
                &candle_nn::ops::softmax(
                    &pca.pca_coordinates_to_log_freq(
                        &pca_samples.reshape(&[num_sites * num_output_samples, NUM_COMPONENTS])?,
                    ),
                    1,
                )?
                .reshape(&[num_sites, num_output_samples, 20])?,
            ),
            ("init_pca_coordinates", &init_pca_tensor),
            ("mean_pi", &mean_pi),
            ("Mu", &Mu),
            ("rate_scaling", &rate_scaling),
            ("branch_lengths", &branch_lengths),
            ("em_log_likelihood", &Tensor::from_slice(&em_log_likelihood, &[em_log_likelihood.len()], &candle_core::Device::Cpu)?),
            ("em_tree_length", &Tensor::from_slice(&em_tree_length, &[em_tree_length.len()], &candle_core::Device::Cpu)?),
            (
                "ess",
                &Tensor::from_iter(site_samples.iter().map(|s| s.ess), &candle_core::Device::Cpu)?,
            ),
            (
                "num_draws",
                &Tensor::from_iter(site_samples.iter().map(|s| s.num_draws as f64), &candle_core::Device::Cpu)?,
            ),
            (
                "num_divergences",
                &Tensor::from_iter(site_samples.iter().map(|s| s.num_divergences as f64), &candle_core::Device::Cpu)?,
            ),
            (
                "step_size",
                &Tensor::from_iter(site_samples.iter().map(|s| s.step_size), &candle_core::Device::Cpu)?,
            ),
            (
                "mean_num_steps",
                &Tensor::from_iter(site_samples.iter().map(|s| s.mean_num_steps), &candle_core::Device::Cpu)?,
            ),
            (
                "log_marginal",
                &Tensor::from_iter(site_samples.iter().map(|s| s.log_marginal), &candle_core::Device::Cpu)?,
            ),
            ("parents", &Tensor::from_iter(parents.iter().map(|&p| p as f64), &candle_core::Device::Cpu)?),
        ],
        Path::new(&npz_path),
    )?;
    println!("MUTSEL HMC: samples written to {}", npz_path);

    // MUTSEL_HMC_DRAW_LL=1: log likelihood of every written draw and of the posterior mean frequencies
    if env_or("MUTSEL_HMC_DRAW_LL", 0u8) != 0 {
        let draws: Vec<Vec<f64>> = (0..num_sites)
            .map(|site| pca_samples.get(site)?.flatten_all()?.to_vec1::<f64>())
            .collect::<Result<_, _>>()?;
        let mean_pi_rows = batch::rows_from_tensor(&mean_pi)?;
        write_draw_log_likelihoods(
            &format!("{}.draw_ll.npz", out_prefix),
            &shared,
            &tree,
            alignment,
            num_leaves,
            &draws,
            &mean_pi_rows,
        )?;
    }

    if num_tree_draws == 1 {
        return Ok((S, sqrt_pi));
    }
    println!("MUTSEL HMC: {} posterior draws per site for the tree search", num_tree_draws);
    let draws: Vec<f64> = site_samples
        .iter()
        .flat_map(|s| {
            (0..num_tree_draws).flat_map(move |k| {
                let draw = k * settings.num_output_samples / num_tree_draws;
                s.thinned_draws[draw * NUM_COMPONENTS..(draw + 1) * NUM_COMPONENTS].iter().copied()
            })
        })
        .collect();
    let draws = Tensor::from_vec(draws, &[num_sites * num_tree_draws, NUM_COMPONENTS], &candle_core::Device::Cpu)?;
    // calc_rate_matrix only needs log_pi up to a constant per site
    Ok(calc_rate_matrix(&Mu, &pca.pca_coordinates_to_log_freq(&draws), &rate_scaling))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    const PI_REG: f64 = 0.32;
    const RATE_SCALING: f64 = 1.3;

    fn test_tree() -> FelsensteinTree<20> {
        // ((0,1),(2,3)) plus a fifth leaf at the root
        let parents = [5, 5, 6, 6, 7, 7, 7, -1];
        let branch_lengths = [0.1, 0.3, 0.2, 0.05, 0.4, 0.15, 0.25, 0.0];
        FelsensteinTree::<20>::new(&parents, &branch_lengths)
    }

    fn test_Mu() -> Tensor {
        let init_R = crate::data::load_lower_R_with_equi(crate::data::M_TXT);
        Mu(&init_R.log().unwrap()).detach()
    }

    const ALIGNMENT: [u8; 5] = [0u8, 0, 3, 20, 7];

    fn test_logp() -> SiteLogp {
        let tree = test_tree();
        SiteLogp {
            shared: Arc::new(SharedModel::new(&test_Mu(), RATE_SCALING, &PCA::new(NUM_COMPONENTS), PI_REG)),
            leaf_pl: leaf_partial_likelihoods(&ALIGNMENT, 0, 5),
            pl_buffer: vec![SVector::<f64, 20>::zeros(); tree.num_nodes()],
            tree: Arc::new(tree),
        }
    }

    /// Reference implementation with the candle model code used by the ML optimization.
    fn candle_logp(position: &[f64]) -> (f64, Vec<f64>) {
        let pca = PCA::new(NUM_COMPONENTS);
        let x = Var::from_slice(position, &[1, NUM_COMPONENTS], &candle_core::Device::Cpu).unwrap();
        let log_pi = pca.pca_coordinates_to_log_freq(x.as_tensor());
        let (S, sqrt_pi) = calc_rate_matrix(&test_Mu(), &log_pi, &tensor_full(RATE_SCALING, &[]));
        let log_prior = pca.penalty_on_pca_coordinates(x.as_tensor(), PI_REG).neg().unwrap();

        let tree = test_tree();
        let mut felsenstein = tree.clone();
        felsenstein.bind_leaf_pl(vec![leaf_partial_likelihoods(&ALIGNMENT, 0, 5)]);
        let op = crate::felsenstein::FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));
        let ll = S.apply_op2(&sqrt_pi, op).unwrap().sum_all().unwrap();
        let logp = (ll + log_prior).unwrap();
        let grads = logp.backward().unwrap();
        let grad = grads.get(x.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        (logp.to_scalar::<f64>().unwrap(), grad)
    }

    fn test_point() -> Vec<f64> {
        (0..NUM_COMPONENTS).map(|i| 0.3 * ((i as f64) * 1.7).sin()).collect()
    }

    #[test]
    fn matches_candle_model() {
        let mut logp = test_logp();
        let x = test_point();
        let mut grad = vec![0.0; NUM_COMPONENTS];
        let value = logp.logp(&x, &mut grad).unwrap();
        let (ref_value, ref_grad) = candle_logp(&x);
        assert!((value - ref_value).abs() < 1e-13 * ref_value.abs(), "{} vs {}", value, ref_value);
        for i in 0..NUM_COMPONENTS {
            assert!(
                (grad[i] - ref_grad[i]).abs() < 1e-12 * (1.0 + ref_grad[i].abs()),
                "component {}: {} vs {}",
                i,
                grad[i],
                ref_grad[i]
            );
        }
    }

    #[test]
    fn rate_matrix_matches_evaluate() {
        let logp = test_logp();
        let x = SVector::<f64, NUM_COMPONENTS>::from_column_slice(&test_point());
        let eval = logp.shared.evaluate(&x);
        // log_pi up to a constant
        let (S, sqrt_pi) = logp.shared.rate_matrix(&(logp.shared.components.transpose() * x).add_scalar(0.7));
        assert!((S - eval.S).abs().max() < 1e-13 * eval.S.abs().max());
        assert!((sqrt_pi - eval.sqrt_pi).abs().max() < 1e-14);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let mut logp = test_logp();
        let x = test_point();
        let mut grad = vec![0.0; NUM_COMPONENTS];
        let value = logp.logp(&x, &mut grad).unwrap();
        assert!(value.is_finite());

        let eps = 1e-6;
        let mut dummy = vec![0.0; NUM_COMPONENTS];
        for i in 0..NUM_COMPONENTS {
            let mut xp = x.clone();
            xp[i] += eps;
            let mut xm = x.clone();
            xm[i] -= eps;
            let fd = (logp.logp(&xp, &mut dummy).unwrap() - logp.logp(&xm, &mut dummy).unwrap()) / (2.0 * eps);
            assert!(
                (fd - grad[i]).abs() < 1e-5 * (1.0 + fd.abs()),
                "component {}: finite difference {} vs gradient {}",
                i,
                fd,
                grad[i]
            );
        }
    }

    #[test]
    fn effective_sample_size() {
        use nuts_rs::rand::RngExt;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut normal = || {
            let u1: f64 = rng.random::<f64>().max(1e-300);
            let u2: f64 = rng.random();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let n = 20000;
        // Two dimensions: iid and AR(1) with phi = 0.9, which has ESS = n (1 - phi) / (1 + phi)
        let mut draws = Vec::with_capacity(2 * n);
        let mut ar = 0.0;
        for _ in 0..n {
            ar = 0.9 * ar + normal();
            draws.push(normal());
            draws.push(ar);
        }
        let iid: Vec<f64> = draws.iter().step_by(2).copied().collect();
        let iid_ess = min_effective_sample_size(&iid, 1);
        assert!((iid_ess / n as f64 - 1.0).abs() < 0.1, "iid ESS {}", iid_ess);

        let ar_ess = min_effective_sample_size(&draws, 2);
        let expected = n as f64 * 0.1 / 1.9;
        assert!((ar_ess / expected - 1.0).abs() < 0.2, "AR(1) ESS {} vs {}", ar_ess, expected);
    }

    /// Forward mode dual number for exact directional derivatives
    #[derive(Debug, Clone, Copy)]
    struct Dual {
        v: f64,
        d: f64,
    }

    impl Dual {
        fn constant(v: f64) -> Dual {
            Dual { v, d: 0.0 }
        }
        fn exp(self) -> Dual {
            let e = self.v.exp();
            Dual { v: e, d: e * self.d }
        }
        fn exp_m1(self) -> Dual {
            Dual { v: self.v.exp_m1(), d: self.v.exp() * self.d }
        }
        fn ln(self) -> Dual {
            Dual { v: self.v.ln(), d: self.d / self.v }
        }
        fn sqrt(self) -> Dual {
            let r = self.v.sqrt();
            Dual { v: r, d: self.d / (2.0 * r) }
        }
        fn abs(self) -> Dual {
            if self.v < 0.0 { -self } else { self }
        }
    }
    impl std::ops::Add for Dual {
        type Output = Dual;
        fn add(self, o: Dual) -> Dual {
            Dual { v: self.v + o.v, d: self.d + o.d }
        }
    }
    impl std::ops::Sub for Dual {
        type Output = Dual;
        fn sub(self, o: Dual) -> Dual {
            Dual { v: self.v - o.v, d: self.d - o.d }
        }
    }
    impl std::ops::Mul for Dual {
        type Output = Dual;
        fn mul(self, o: Dual) -> Dual {
            Dual { v: self.v * o.v, d: self.d * o.v + self.v * o.d }
        }
    }
    impl std::ops::Div for Dual {
        type Output = Dual;
        fn div(self, o: Dual) -> Dual {
            Dual { v: self.v / o.v, d: (self.d * o.v - self.v * o.d) / (o.v * o.v) }
        }
    }
    impl std::ops::Neg for Dual {
        type Output = Dual;
        fn neg(self) -> Dual {
            Dual { v: -self.v, d: -self.d }
        }
    }

    /// Independent reference of the per-site model following model::calc_rate_matrix and
    /// PCA::penalty_on_pca_coordinates: returns <G, S(x)> + <b, sqrt_pi(x)> + log_prior(x)
    fn reference_surrogate(
        model: &SharedModel,
        x: &[Dual],
        G: &SMatrix<f64, 20, 20>,
        b: &SVector<f64, 20>,
    ) -> Dual {
        let c = Dual::constant;
        let log_pi: Vec<Dual> = (0..20)
            .map(|a| (0..NUM_COMPONENTS).fold(c(0.0), |acc, k| acc + c(model.components[(k, a)]) * x[k]))
            .collect();
        let max = log_pi.iter().map(|l| l.v).fold(f64::NEG_INFINITY, f64::max);
        let unnormalized: Vec<Dual> = log_pi.iter().map(|&l| (l - c(max)).exp()).collect();
        let total = unnormalized.iter().fold(c(0.0), |acc, &u| acc + u);
        let sqrt_pi: Vec<Dual> = unnormalized.iter().map(|&u| (u / total).sqrt()).collect();
        let fitness: Vec<Dual> = (0..20).map(|a| log_pi[a] - c(model.log_mutation_equilibrium[a])).collect();
        // g(d) = d / (1 - exp(-d)) = 1 + d/2 + (d/2) coth(d/2) - 1; the Taylor series is used for small |d|,
        // since the derivative of the closed form suffers from cancellation (relative error ~ eps / |d|)
        let g = |d: Dual| {
            if d.v.abs() < 1e-2 {
                let d2 = d * d;
                c(1.0) + d * c(0.5)
                    + d2 * (c(1.0 / 12.0)
                        - d2 * (c(1.0 / 720.0) - d2 * (c(1.0 / 30240.0) - d2 * c(1.0 / 1209600.0))))
            } else {
                d / -(-d).exp_m1()
            }
        };

        let mut value = c(0.0);
        for i in 0..20 {
            for j in (i + 1)..20 {
                let S_ij = sqrt_pi[i] / sqrt_pi[j] * c(model.scaled_Mu[(i, j)]) * g(fitness[j] - fitness[i]);
                value = value + c(G[(i, j)]) * S_ij;
            }
        }
        for a in 0..20 {
            value = value + c(b[a]) * sqrt_pi[a];
        }
        for k in 0..NUM_COMPONENTS {
            let z = ((x[k] - c(model.prior_mean[k])) / c(model.prior_scale[k])).abs();
            value = value - (c(model.prior_beta[k]) * z.ln()).exp();
        }
        value
    }

    #[test]
    fn backpropagation_matches_forward_mode_derivatives() {
        let model = SharedModel::new(&test_Mu(), RATE_SCALING, &PCA::new(NUM_COMPONENTS), PI_REG);

        // Arbitrary upstream gradients
        let G = SMatrix::<f64, 20, 20>::from_fn(|i, j| ((i * 20 + j) as f64 * 0.37).sin());
        let b = SVector::<f64, 20>::from_fn(|a, _| ((a as f64) * 1.3).cos());

        // x with fitness differences exactly zero: log_pi equal to the mutation equilibrium
        let neutral = model.components * model.log_mutation_equilibrium;
        let direction: Vec<f64> = test_point();

        let mut points: Vec<Vec<f64>> = vec![];
        for scale in [0.0, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0] {
            points.push((0..NUM_COMPONENTS).map(|k| neutral[k] + scale * direction[k]).collect());
        }
        for scale in [1e-6, 1e-2, 1.0, 3.0] {
            points.push(direction.iter().map(|d| d * scale / 0.3).collect());
        }

        for x in points {
            let x_vec = SVector::<f64, NUM_COMPONENTS>::from_column_slice(&x);
            let eval = model.evaluate(&x_vec);
            let grad = model.backpropagate(&eval, &G, &b);

            let mut max_rel_error = 0.0f64;
            let max_grad = grad.iter().map(|g| g.abs()).fold(0.0, f64::max);
            for k in 0..NUM_COMPONENTS {
                let x_dual: Vec<Dual> = (0..NUM_COMPONENTS)
                    .map(|m| Dual { v: x[m], d: if m == k { 1.0 } else { 0.0 } })
                    .collect();
                let reference = reference_surrogate(&model, &x_dual, &G, &b).d;
                max_rel_error = max_rel_error.max((grad[k] - reference).abs() / max_grad);
            }
            let min_abs_fitness_diff = (0..20)
                .flat_map(|i| ((i + 1)..20).map(move |j| (i, j)))
                .map(|(i, j)| (eval.fitness[j] - eval.fitness[i]).abs())
                .fold(f64::INFINITY, f64::min);
            println!(
                "min |f_j - f_i| {:.1e}: max error relative to max |grad| {:.2e}",
                min_abs_fitness_diff, max_rel_error
            );
            assert!(max_rel_error < 1e-13, "relative error {:e}", max_rel_error);
        }
    }

    #[test]
    fn gmm_prior_gradient_matches_finite_differences() {
        let mut prior = GmmPrior { log_norm: vec![], means: vec![], precisions: vec![] };
        for k in 0..3 {
            let a = SMatrix::<f64, NUM_COMPONENTS, NUM_COMPONENTS>::from_fn(|i, j| ((i * 7 + j * 3 + k) % 5) as f64 * 0.1);
            prior.precisions.push(a * a.transpose() + SMatrix::identity());
            prior.means.push(SVector::from_fn(|i, _| (i + k) as f64 * 0.1));
            prior.log_norm.push(-(k as f64));
        }
        let x = SVector::<f64, NUM_COMPONENTS>::from_column_slice(&test_point());
        let (_, grad) = prior.evaluate(&x);
        let h = 1e-5;
        for k in 0..NUM_COMPONENTS {
            let mut e = SVector::<f64, NUM_COMPONENTS>::zeros();
            e[k] = h;
            let fd = (prior.evaluate(&(x + e)).0 - prior.evaluate(&(x - e)).0) / (2.0 * h);
            assert!((fd - grad[k]).abs() < 1e-6 * (1.0 + grad[k].abs()), "{} {} {}", k, fd, grad[k]);
        }
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        for (x, expected) in [
            (0.1, 2.252712651734206),
            (0.5, 0.5723649429247001),
            (1.0, 0.0),
            (3.7, 1.428072326665388),
            (50.0, 144.56574394634487),
        ] {
            assert!((ln_gamma(x) - expected).abs() < 1e-12 * (1.0 + expected.abs()), "{} {}", x, ln_gamma(x));
        }
    }

    #[test]
    fn dirichlet_prior_gradient_matches_finite_differences() {
        let mut prior = DirichletMixturePrior { log_norm: vec![], alphas: vec![] };
        for k in 0..3 {
            prior.alphas.push(SVector::from_fn(|i, _| 0.05 + ((i * 7 + k * 3) % 11) as f64 * 0.3));
            prior.log_norm.push(-(k as f64));
        }
        let model = SharedModel::new(&test_Mu(), RATE_SCALING, &PCA::new(NUM_COMPONENTS), PI_REG);
        let x = SVector::<f64, NUM_COMPONENTS>::from_column_slice(&test_point());
        let (_, grad) = prior.evaluate(&x, &model.components);
        let h = 1e-5;
        for k in 0..NUM_COMPONENTS {
            let mut e = SVector::<f64, NUM_COMPONENTS>::zeros();
            e[k] = h;
            let fd = (prior.evaluate(&(x + e), &model.components).0 - prior.evaluate(&(x - e), &model.components).0)
                / (2.0 * h);
            assert!((fd - grad[k]).abs() < 1e-6 * (1.0 + grad[k].abs()), "{} {} {}", k, fd, grad[k]);
        }
    }

    /// The bridge estimate of the marginal likelihood of a site against brute force Monte Carlo from a Gaussian prior
    #[test]
    fn site_log_marginal_matches_prior_monte_carlo() {
        let mean = SVector::<f64, NUM_COMPONENTS>::from_column_slice(&test_point());
        let std = 0.7;
        let prior = GmmPrior {
            log_norm: vec![-(NUM_COMPONENTS as f64) * (std as f64).ln() - 0.5 * NUM_COMPONENTS as f64 * (2.0 * std::f64::consts::PI).ln()],
            means: vec![mean],
            precisions: vec![SMatrix::identity() / (std * std)],
        };
        let shared = Arc::new(
            SharedModel::new(&test_Mu(), RATE_SCALING, &PCA::new(NUM_COMPONENTS), PI_REG)
                .with_mixture_prior(Some(Arc::new(MixturePrior::Gaussian(prior)))),
        );
        let logp = || {
            let mut logp = test_logp();
            logp.shared = Arc::clone(&shared);
            logp
        };
        let settings = HmcSettings {
            num_tune: 400,
            num_draws: 500,
            target_ess: 200,
            max_draws: 20000,
            num_output_samples: 100,
            seed: 3,
            marginal: true,
        };
        // A single estimate has a standard deviation of about 0.1, so the bias is checked with the average of several chains
        let num_chains = 8;
        let estimate = (0..num_chains)
            .map(|seed| {
                let samples = sample_site(logp(), mean.as_slice(), &HmcSettings { seed, ..settings }, 0);
                site_log_marginal(&mut logp(), &samples, seed)
            })
            .sum::<f64>()
            / num_chains as f64;

        // log E_prior[likelihood]
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut uniform = || ((nuts_rs::rand::Rng::next_u64(&mut rng) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let mut tree_logp = logp();
        let mut log_likelihoods = vec![];
        let mut gradient = vec![0.0; NUM_COMPONENTS];
        for _ in 0..1 << 17 {
            let x = SVector::<f64, NUM_COMPONENTS>::from_fn(|_, _| {
                (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos() * std
            }) + mean;
            let eval = shared.evaluate(&x);
            log_likelihoods.push(tree_logp.logp(x.as_slice(), &mut gradient).unwrap() - eval.log_prior);
        }
        let reference = log_sum_exp(log_likelihoods.iter().copied()) - (log_likelihoods.len() as f64).ln();
        println!("bridge {:.4}, prior Monte Carlo {:.4}", estimate, reference);
        assert!((estimate - reference).abs() < 0.06, "bridge {} vs Monte Carlo {}", estimate, reference);
    }

    #[test]
    fn bridge_fixed_point_recovers_normalization_of_a_gaussian() {
        // Target q = exp(c) * N(0, 1) in one dimension, proposal g = N(0.3, 1.2^2)
        let c = 2.5;
        let log_n = |x: f64, m: f64, s: f64| -0.5 * ((x - m) / s).powi(2) - s.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
        let quantiles = |n: usize, m: f64, s: f64| -> Vec<f64> {
            // Deterministic stratified "draws" through the inverse CDF of a logistic approximation
            (0..n).map(|i| m + s * 0.5513 * (((i as f64 + 0.5) / n as f64) / (1.0 - (i as f64 + 0.5) / n as f64)).ln()).collect()
        };
        let posterior = quantiles(2000, 0.0, 1.0);
        let proposal = quantiles(2000, 0.3, 1.2);
        let lr = |x: &f64| c + log_n(*x, 0.0, 1.0) - log_n(*x, 0.3, 1.2);
        let log_z = bridge_fixed_point(&posterior.iter().map(lr).collect::<Vec<_>>(), &proposal.iter().map(lr).collect::<Vec<_>>());
        assert!((log_z - c).abs() < 0.02, "{}", log_z);
    }

    #[test]
    fn em_stops_on_the_tree_length_change_over_the_window() {
        let em = EmSettings { tree_length_tolerance: 0.01, window: 3, ..EmSettings::from_env() };
        // Tree lengths of the 32066 convergence test, iterations 10 to 16: stops at the last one
        let tree_lengths = [6.8612, 6.9200, 6.8612, 6.7914, 6.7559, 6.7526, 6.7588];
        let first_stop = (1..=tree_lengths.len())
            .find(|&n| em.tree_length_change(&tree_lengths[..n]).is_some_and(|c| c < em.tree_length_tolerance));
        assert_eq!(first_stop, Some(7));
        assert_eq!(em.tree_length_change(&tree_lengths[..3]), None);
    }

    /// Full gradient including phylo_grad against a fourth order central difference
    #[test]
    fn full_gradient_matches_fourth_order_finite_differences() {
        let mut logp = test_logp();
        let mut dummy = vec![0.0; NUM_COMPONENTS];
        for x in [test_point(), test_point().iter().map(|v| v * 5.0).collect::<Vec<f64>>()] {
            let mut grad = vec![0.0; NUM_COMPONENTS];
            logp.logp(&x, &mut grad).unwrap();
            let max_grad = grad.iter().map(|g| g.abs()).fold(0.0, f64::max);
            let h = 1e-3;
            let mut max_rel_error = 0.0f64;
            for k in 0..NUM_COMPONENTS {
                let mut f = |offset: f64| {
                    let mut xs = x.clone();
                    xs[k] += offset;
                    logp.logp(&xs, &mut dummy).unwrap()
                };
                let fd = (-f(2.0 * h) + 8.0 * f(h) - 8.0 * f(-h) + f(-2.0 * h)) / (12.0 * h);
                max_rel_error = max_rel_error.max((fd - grad[k]).abs() / max_grad);
            }
            println!("full gradient vs 4th order FD: max error relative to max |grad| {:.2e}", max_rel_error);
            assert!(max_rel_error < 1e-9, "relative error {:e}", max_rel_error);
        }
    }

    #[test]
    fn logp_timing() {
        let mut logp = test_logp();
        let x = vec![0.1; NUM_COMPONENTS];
        let mut grad = vec![0.0; NUM_COMPONENTS];
        let n = 2000;
        let start = std::time::Instant::now();
        for _ in 0..n {
            logp.logp(&x, &mut grad).unwrap();
        }
        println!("logp+grad: {:.1} us", start.elapsed().as_secs_f64() * 1e6 / n as f64);
    }

    /// Timing of the parts of logp on a random binary tree with 100 leaves
    #[test]
    #[ignore]
    fn logp_timing_100_leaves() {
        let num_leaves = 100;
        let mut state = 1u64;
        let mut uniform = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        // Random topology by joining random pairs of the remaining subtrees
        let mut parents = vec![-1i32; 2 * num_leaves - 1];
        let mut open: Vec<usize> = (0..num_leaves).collect();
        let mut next = num_leaves;
        while open.len() > 1 {
            let a = open.swap_remove((uniform() * open.len() as f64) as usize);
            let b = open.swap_remove((uniform() * open.len() as f64) as usize);
            parents[a] = next as i32;
            parents[b] = next as i32;
            open.push(next);
            next += 1;
        }
        let branch_lengths: Vec<f64> = (0..parents.len()).map(|_| 0.01 + 0.29 * uniform()).collect();
        let alignment: Vec<u8> = (0..num_leaves).map(|i| [0u8, 3, 3, 7, 20][i % 5]).collect();
        let mut logp = SiteLogp {
            shared: Arc::new(SharedModel::new(&test_Mu(), RATE_SCALING, &PCA::new(NUM_COMPONENTS), PI_REG)),
            tree: Arc::new(FelsensteinTree::<20>::new(&parents, &branch_lengths)),
            leaf_pl: leaf_partial_likelihoods(&alignment, 0, num_leaves),
            pl_buffer: vec![SVector::<f64, 20>::zeros(); parents.len()],
        };
        let x = SVector::<f64, NUM_COMPONENTS>::from_column_slice(&test_point());
        let mut grad = vec![0.0; NUM_COMPONENTS];
        let n = 5000;
        let time = |f: &mut dyn FnMut()| {
            let start = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            start.elapsed().as_secs_f64() * 1e6 / n as f64
        };
        let total = time(&mut || {
            logp.logp(x.as_slice(), &mut grad).unwrap();
        });
        let model = Arc::clone(&logp.shared);
        let eval = model.evaluate(&x);
        let evaluate = time(&mut || {
            std::hint::black_box(model.evaluate(&x));
        });
        let tree = Arc::clone(&logp.tree);
        let mut pl = logp.pl_buffer.clone();
        pl[..num_leaves].copy_from_slice(&logp.leaf_pl);
        let felsenstein = time(&mut || {
            std::hint::black_box(tree.calculate_gradients_single_side(eval.S.as_view(), eval.sqrt_pi.as_view(), &mut pl));
        });
        let likelihood_only = time(&mut || {
            std::hint::black_box(tree.calculate_likelihood_single_side(eval.S.as_view(), eval.sqrt_pi.as_view(), &mut pl));
        });
        let result = tree.calculate_gradients_single_side(eval.S.as_view(), eval.sqrt_pi.as_view(), &mut pl);
        let backpropagate = time(&mut || {
            std::hint::black_box(model.backpropagate(&eval, &result.grad_s, &result.grad_sqrt_pi));
        });
        println!(
            "100 leaves: logp+grad {:.1} us = evaluate {:.1} + phylo_grad {:.1} (likelihood only {:.1}) + backpropagate {:.1}",
            total, evaluate, felsenstein, likelihood_only, backpropagate
        );
    }

    #[test]
    fn log_R_from_Mu_inverts_Mu() {
        let init_log_R = crate::optimization::initial_log_R(None).unwrap();
        let mut values = init_log_R.to_vec2::<f64>().unwrap();
        for i in 0..20 {
            for j in 0..i {
                values[i][j] += 0.5 * ((i * 20 + j) as f64 * 0.73).sin();
            }
        }
        let log_R = Tensor::from_vec(values.concat(), &[20, 20], &candle_core::Device::Cpu).unwrap();
        let expected = Mu(&log_R).to_vec2::<f64>().unwrap();
        let inverted = crate::optimization::log_R_from_Mu(&Mu(&log_R), &init_log_R).unwrap();
        let actual = Mu(&inverted).to_vec2::<f64>().unwrap();
        for i in 0..20 {
            for j in 0..20 {
                assert!((expected[i][j] - actual[i][j]).abs() < 1e-12, "Mu[{}][{}]: {} vs {}", i, j, expected[i][j], actual[i][j]);
            }
        }
    }

    #[test]
    fn lbfgs_finds_the_site_posterior_mode() {
        // Two component Gaussian mixture prior (the generalized normal one is not twice differentiable at its mean)
        let prior = GmmPrior {
            log_norm: vec![(0.3f64).ln(), (0.7f64).ln()],
            means: vec![SVector::from_fn(|i, _| 0.2 * (i as f64).cos()), SVector::from_fn(|i, _| -0.1 * (i as f64).sin())],
            precisions: vec![SMatrix::identity() * 4.0, SMatrix::from_fn(|i, j| if i == j { 2.0 } else { 0.3 / (1.0 + (i + j) as f64) })],
        };
        let mut logp = test_logp();
        logp.shared = Arc::new(
            SharedModel::new(&test_Mu(), RATE_SCALING, &PCA::new(NUM_COMPONENTS), PI_REG)
                .with_mixture_prior(Some(Arc::new(MixturePrior::Gaussian(prior)))),
        );
        let start = test_point();
        let mut gradient = vec![0.0; NUM_COMPONENTS];
        let start_value = logp.logp(&start, &mut gradient).unwrap();
        let mode = maximize_lbfgs(&mut logp, &start, 200, 1e-6);
        let mode_value = logp.logp(&mode, &mut gradient).unwrap();
        let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!(mode_value > start_value, "{} <= {}", mode_value, start_value);
        assert!(norm < 1e-4, "gradient norm {} at the mode", norm);
    }
}
