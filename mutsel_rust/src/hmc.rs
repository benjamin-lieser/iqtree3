//! Per-site posterior sampling of the Helmert coordinates of the site frequencies with NUTS (nuts-rs).
//!
//! Mu, the branch lengths and the global rate normalization are shared and kept
//! constant while sampling. Site rates are fixed to 1, or with site_rate_sd > 0 in the prior sampled as a
//! log-normal correction of the MutSel rate (see SiteRatePrior). With everything shared fixed, the
//! posterior factorizes over alignment columns, so every column gets its own
//! independent 19-dimensional (20 with site rates) NUTS chain.
//!
//! Before the final sampling, Mu and the branch lengths are fitted with Monte Carlo EM
//! (see sample_internal).

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
    rand::{SeedableRng, rngs::ChaCha8Rng},
};
use phylo_grad::{
    Workspace,
    nalgebra::{SMatrix, SVector},
};
use rayon::prelude::*;

use crate::{
    Result, Verbosity, batch,
    helmert::{self, NUM_COORDINATES},
    lbfgs::maximize_finite,
    model,
    npz::{Array, write_npz},
    optimization::{self, Convergence, MapEstimate, Mu, map_estimate},
    prior::{Prior, SiteRatePrior},
};

/// Topologically sorted tree (see lib.rs) with its branch lengths in the same order
pub struct Tree {
    parents: Vec<i32>,
    branch_lengths: Vec<f64>,
}

impl Tree {
    pub fn new(parents: &[i32], branch_lengths: &[f64]) -> Tree {
        assert_eq!(parents.len(), branch_lengths.len());
        Tree { parents: parents.to_vec(), branch_lengths: branch_lengths.to_vec() }
    }

    fn num_nodes(&self) -> usize {
        self.parents.len()
    }
}

/// The leaf states of every site, columns of the alignment [num_sites * num_leaves]
fn alignment_columns(alignment: &[u8], num_leaves: usize) -> Vec<&[u8]> {
    alignment.chunks_exact(num_leaves).collect()
}

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
    /// Sampling continues until every Helmert coordinate of a site reaches this effective sample size
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

/// Appends the prior mean of the log site rate to a position of only Helmert coordinates, if site rates are sampled
fn with_initial_log_rate(mut position: Vec<f64>, site_rate_prior: Option<SiteRatePrior>) -> Vec<f64> {
    if let Some(prior) = site_rate_prior {
        if position.len() == NUM_COORDINATES {
            position.push(prior.mean());
        }
    }
    position
}

/// Constants of the per-site posterior which are shared by all sites.
struct SharedModel {
    /// Helmert basis [19, 20], log_pi = basis^T * x (up to a constant)
    basis: SMatrix<f64, NUM_COORDINATES, 20>,
    /// Mu with the fixed global rate normalization already applied
    scaled_Mu: SMatrix<f64, 20, 20>,
    /// The prior of the site frequencies is read from it
    prior: Arc<Prior>,
    /// If set, the position has the log site rate as an additional last coordinate
    site_rate_prior: Option<SiteRatePrior>,
}

impl SharedModel {
    fn new(Mu: &SMatrix<f64, 20, 20>, rate_scaling: f64, prior: Arc<Prior>) -> SharedModel {
        SharedModel {
            basis: helmert::basis(),
            scaled_Mu: Mu * rate_scaling,
            prior,
            site_rate_prior: None,
        }
    }

    fn with_site_rate_prior(self, site_rate_prior: Option<SiteRatePrior>) -> SharedModel {
        SharedModel { site_rate_prior, ..self }
    }

    /// Dimension of a site's position: the Helmert coordinates and, if sampled, the log site rate
    fn dim(&self) -> usize {
        NUM_COORDINATES + self.site_rate_prior.is_some() as usize
    }

    /// Site rate of a position, 1 if site rates are not sampled
    fn site_rate(&self, position: &[f64]) -> f64 {
        if self.site_rate_prior.is_some() { position[NUM_COORDINATES].exp() } else { 1.0 }
    }

    /// evaluate() at the Helmert coordinates of a position, with S multiplied by the site rate and the log prior of
    /// the log site rate added if site rates are sampled
    fn evaluate_position(&self, position: &[f64]) -> SiteModelEval {
        let mut eval = self.evaluate(&SVector::<f64, NUM_COORDINATES>::from_column_slice(&position[..NUM_COORDINATES]));
        if let Some(prior) = &self.site_rate_prior {
            eval.S *= position[NUM_COORDINATES].exp();
            eval.log_prior += prior.evaluate(position[NUM_COORDINATES]).0;
        }
        eval
    }

    /// Gradient of log_likelihood + log_prior w.r.t. the position (see backpropagate) for an eval of
    /// evaluate_position
    fn backpropagate_position(
        &self,
        position: &[f64],
        eval: &SiteModelEval,
        grad_S: &SMatrix<f64, 20, 20>,
        grad_sqrt_pi: &SVector<f64, 20>,
        gradient: &mut [f64],
    ) {
        // S scales with the site rate, so backpropagate gives the gradient w.r.t. the Helmert coordinates
        // with the scaled S, and d S_ij / d log r = S_ij
        gradient[..NUM_COORDINATES].copy_from_slice(self.backpropagate(eval, grad_S, grad_sqrt_pi).as_slice());
        if let Some(prior) = &self.site_rate_prior {
            let mut grad_log_rate = prior.evaluate(position[NUM_COORDINATES]).1;
            for i in 0..20 {
                for j in (i + 1)..20 {
                    grad_log_rate += grad_S[(i, j)] * eval.S[(i, j)];
                }
            }
            gradient[NUM_COORDINATES] = grad_log_rate;
        }
    }

    /// Site frequencies for the given Helmert coordinates
    fn pi(&self, x: &SVector<f64, NUM_COORDINATES>) -> (SVector<f64, 20>, SVector<f64, 20>) {
        let log_pi = self.basis.transpose() * x;
        let max_log_pi = log_pi.max();
        let unnormalized_pi = log_pi.map(|l| (l - max_log_pi).exp());
        (log_pi, unnormalized_pi / unnormalized_pi.sum())
    }

    /// S and sqrt_pi for log frequencies up to a constant
    fn rate_matrix(&self, log_pi: &SVector<f64, 20>) -> (SMatrix<f64, 20, 20>, SVector<f64, 20>) {
        model::rate_matrix(&self.scaled_Mu, log_pi)
    }

    /// Log prior, S and sqrt_pi for the Helmert coordinates x
    fn evaluate(&self, x: &SVector<f64, NUM_COORDINATES>) -> SiteModelEval {
        let (log_prior, grad_log_prior) = self.prior.site_frequencies.evaluate(x);
        // log_pi up to an additive constant
        let log_pi = self.basis.transpose() * x;
        let (S, sqrt_pi) = self.rate_matrix(&log_pi);
        SiteModelEval { log_prior, grad_log_prior, log_pi, sqrt_pi, S }
    }

    /// Gradient of log_likelihood + log_prior w.r.t. x, given the gradients of the log likelihood
    /// w.r.t. S (of which phylo_grad reads the upper triangle) and sqrt_pi
    fn backpropagate(
        &self,
        eval: &SiteModelEval,
        grad_S: &SMatrix<f64, 20, 20>,
        grad_sqrt_pi: &SVector<f64, 20>,
    ) -> SVector<f64, NUM_COORDINATES> {
        let (_, grad_log_pi) = model::rate_matrix_backward(
            &eval.log_pi,
            &eval.S,
            &eval.sqrt_pi,
            &self.scaled_Mu,
            &grad_S.upper_triangle(),
            grad_sqrt_pi,
        );
        eval.grad_log_prior + self.basis * grad_log_pi
    }
}

struct SiteModelEval {
    log_prior: f64,
    grad_log_prior: SVector<f64, NUM_COORDINATES>,
    log_pi: SVector<f64, 20>,
    sqrt_pi: SVector<f64, 20>,
    S: SMatrix<f64, 20, 20>,
}

/// Unnormalized log posterior of the Helmert coordinates of a single alignment column.
/// Same model as optimization::map_estimate with fixed Mu, branch lengths and normalization.
struct SiteLogp {
    shared: Arc<SharedModel>,
    tree: Arc<Tree>,
    /// Leaf states of the column
    leaves: Vec<u8>,
    workspace: Workspace<20>,
}

impl HasDims for SiteLogp {
    fn dim_sizes(&self) -> std::collections::HashMap<String, u64> {
        [("unconstrained_parameter".to_string(), self.shared.dim() as u64)]
            .into_iter()
            .collect()
    }
}

impl CpuLogpFunc for SiteLogp {
    type LogpError = SiteLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.shared.dim()
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> std::result::Result<f64, SiteLogpError> {
        let model = &*self.shared;
        let eval = model.evaluate_position(position);

        let result = phylo_grad::calculate_gradients_single_column(
            &self.tree.parents,
            &self.leaves,
            &self.tree.branch_lengths,
            eval.S.as_view(),
            eval.sqrt_pi.as_view(),
            &mut self.workspace,
            None,
        );

        let logp = result.log_likelihood + eval.log_prior;
        if !logp.is_finite() {
            return Err(SiteLogpError::NonFinite);
        }

        model.backpropagate_position(position, &eval, &result.grad_s, &result.grad_sqrt_pi, gradient);
        if gradient.iter().any(|g| !g.is_finite()) {
            return Err(SiteLogpError::NonFinite);
        }

        Ok(logp)
    }

    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> std::result::Result<Vec<f64>, CpuMathError> {
        Ok(array.to_vec())
    }
}

struct SiteSamples {
    /// Dimension of a draw, the Helmert coordinates and, if sampled, the log site rate
    dim: usize,
    /// Evenly thinned draws [num_output_samples * dim]
    thinned_draws: Vec<f64>,
    /// Posterior mean of the site frequencies over all draws
    mean_pi: SVector<f64, 20>,
    /// Posterior mean of the site rate over all draws, 1 if site rates are not sampled
    mean_site_rate: f64,
    num_draws: usize,
    /// Minimum over the Helmert coordinates of the effective sample size of all draws
    ess: f64,
    num_divergences: usize,
    step_size: f64,
    mean_num_steps: f64,
    /// Mean and covariance of the draws in the first half of the chain, if settings.marginal
    proposal: Option<(SVector<f64, NUM_COORDINATES>, SMatrix<f64, NUM_COORDINATES, NUM_COORDINATES>)>,
    /// Thinned draws from the second half of the chain, which are independent of the proposal
    bridge_draws: Vec<SVector<f64, NUM_COORDINATES>>,
    /// Log marginal likelihood (log likelihood and log prior integrated over the Helmert coordinates), NaN if not estimated
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

type SiteChain = <DiagNutsSettings as Settings>::Chain<CpuMath<SiteLogp>>;

fn new_site_chain(logp: SiteLogp, init: &[f64], settings: &HmcSettings, site_index: usize) -> SiteChain {
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
    chain
}

fn sample_site(logp: SiteLogp, init: &[f64], settings: &HmcSettings, site_index: usize) -> SiteSamples {
    let mut chain = new_site_chain(logp, init, settings, site_index);
    draw_site(&mut chain, settings, 0, site_index)
}

/// Draws from a chain until the ESS target is reached, after its tuning draws and then burn_in further draws
fn draw_site(chain: &mut SiteChain, settings: &HmcSettings, burn_in: usize, site_index: usize) -> SiteSamples {
    let shared = Arc::clone(&chain.math().logp_func().shared);
    let dim = shared.dim();
    assert!(!settings.marginal || dim == NUM_COORDINATES, "marginal likelihoods are not supported with site rates");
    let mut num_burn_in = 0;
    let mut draws = Vec::with_capacity(settings.num_draws as usize * dim);
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
        if num_burn_in < burn_in {
            num_burn_in += 1;
            continue;
        }
        draws.extend_from_slice(&draw);
        num_divergences += progress.diverging as usize;
        num_steps += progress.num_steps;
        step_size = progress.step_size;

        let num_draws = draws.len() / dim;
        if num_draws < next_check {
            continue;
        }
        ess = min_effective_sample_size(&draws, dim);
        if ess >= settings.target_ess as f64 || num_draws as u64 >= settings.max_draws {
            break;
        }
        // Extrapolate the number of draws needed, with some margin
        let needed = (num_draws as f64 * 1.2 * settings.target_ess as f64 / ess.max(1.0)) as usize;
        next_check = needed
            .clamp(num_draws + num_draws / 4 + 1, 2 * num_draws)
            .min(settings.max_draws as usize);
    }
    let num_draws = draws.len() / dim;

    let mut thinned_draws = Vec::with_capacity(settings.num_output_samples * dim);
    for i in 0..settings.num_output_samples {
        // Last draw of each of num_output_samples equally sized blocks
        let draw = ((i + 1) * num_draws) / settings.num_output_samples - 1;
        thinned_draws.extend_from_slice(&draws[draw * dim..(draw + 1) * dim]);
    }

    let mut proposal = None;
    let mut bridge_draws = vec![];
    if settings.marginal {
        let first_half: Vec<_> = draws[..(num_draws / 2) * NUM_COORDINATES]
            .chunks_exact(NUM_COORDINATES)
            .map(SVector::<f64, NUM_COORDINATES>::from_column_slice)
            .collect();
        let mean = first_half.iter().sum::<SVector<f64, NUM_COORDINATES>>() / first_half.len() as f64;
        let mut cov = SMatrix::<f64, NUM_COORDINATES, NUM_COORDINATES>::zeros();
        for x in &first_half {
            cov += (x - mean) * (x - mean).transpose();
        }
        cov /= (first_half.len() - 1) as f64;
        proposal = Some((mean, cov));
        for i in 0..settings.num_output_samples {
            let draw = ((i + 1) * num_draws) / settings.num_output_samples - 1;
            if draw >= num_draws / 2 {
                bridge_draws.push(SVector::from_column_slice(&draws[draw * NUM_COORDINATES..(draw + 1) * NUM_COORDINATES]));
            }
        }
    }

    let mut mean_pi = SVector::<f64, 20>::zeros();
    let mut mean_site_rate = 0.0;
    for draw in draws.chunks_exact(dim) {
        mean_pi += shared.pi(&SVector::<f64, NUM_COORDINATES>::from_column_slice(&draw[..NUM_COORDINATES])).1;
        mean_site_rate += shared.site_rate(draw);
    }
    mean_pi /= num_draws as f64;
    mean_site_rate /= num_draws as f64;

    SiteSamples {
        dim,
        thinned_draws,
        mean_pi,
        mean_site_rate,
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
    let jitter = 1e-10 * cov.trace().max(1e-300) / NUM_COORDINATES as f64;
    let chol = match (cov + SMatrix::<f64, NUM_COORDINATES, NUM_COORDINATES>::identity() * jitter).cholesky() {
        Some(chol) => chol,
        None => return f64::NAN,
    };
    let l = chol.l();
    let log_norm = -l.diagonal().map(f64::ln).sum() - 0.5 * NUM_COORDINATES as f64 * (2.0 * std::f64::consts::PI).ln();
    let log_proposal = |x: &SVector<f64, NUM_COORDINATES>| {
        let z = l.solve_lower_triangular(&(x - mean)).unwrap();
        log_norm - 0.5 * z.norm_squared()
    };
    let mut gradient = vec![0.0; NUM_COORDINATES];
    let mut log_posterior = |x: &SVector<f64, NUM_COORDINATES>| {
        logp.logp(x.as_slice(), &mut gradient).unwrap_or(f64::NEG_INFINITY)
    };

    // Standard normal draws with Box-Muller
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut uniform = || ((nuts_rs::rand::Rng::next_u64(&mut rng) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let mut normal = SVector::<f64, NUM_COORDINATES>::zeros();
    let mut log_ratio_proposal = Vec::with_capacity(NUM_PROPOSAL_DRAWS);
    for _ in 0..NUM_PROPOSAL_DRAWS {
        for k in (0..NUM_COORDINATES).step_by(2) {
            let r = (-2.0 * uniform().ln()).sqrt();
            let phi = 2.0 * std::f64::consts::PI * uniform();
            normal[k] = r * phi.cos();
            if k + 1 < NUM_COORDINATES {
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
    /// Draws per site discarded in the later E-steps, whose chains continue from the previous E-step with the new
    /// Mu and branch lengths and keep their step size and mass matrix
    pub num_burn_in: usize,
    /// Maximum number of L-BFGS iterations per M-step
    pub max_mstep_iterations: usize,
    /// Number of first EM iterations whose M-step uses fewer samples and L-BFGS iterations (see mstep_schedule)
    pub mstep_ramp: usize,
    /// Convergence of the MAP estimate EM starts from
    pub map_convergence: Convergence,
}

impl EmSettings {
    /// Reads MUTSEL_EM_ITERATIONS, MUTSEL_EM_TOL, MUTSEL_EM_WINDOW, MUTSEL_EM_SAMPLES, MUTSEL_EM_TUNE, MUTSEL_EM_BURNIN, MUTSEL_EM_MSTEP_ITERATIONS,
    /// MUTSEL_MAP_MAX_ITERATIONS and MUTSEL_MAP_REL_IMPROVEMENT from the environment.
    pub fn from_env() -> EmSettings {
        EmSettings {
            iterations: env_or("MUTSEL_EM_ITERATIONS", 50),
            tree_length_tolerance: env_or("MUTSEL_EM_TOL", 0.005),
            window: env_or("MUTSEL_EM_WINDOW", 5),
            num_samples: env_or("MUTSEL_EM_SAMPLES", 8),
            num_tune: env_or("MUTSEL_EM_TUNE", 200),
            num_burn_in: env_or("MUTSEL_EM_BURNIN", 20),
            max_mstep_iterations: env_or("MUTSEL_EM_MSTEP_ITERATIONS", 50),
            mstep_ramp: env_or("MUTSEL_EM_MSTEP_RAMP", 3),
            map_convergence: Convergence {
                max_iterations: env_or("MUTSEL_MAP_MAX_ITERATIONS", Convergence::MAP.max_iterations),
                min_rel_improvement: env_or("MUTSEL_MAP_REL_IMPROVEMENT", Convergence::MAP.min_rel_improvement),
            },
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

    /// Number of samples per site and maximum number of L-BFGS iterations of the M-step. In the first
    /// mstep_ramp iterations Mu and the branch lengths are still far from the optimum and the next E-step
    /// resamples the sites anyway, so both rise linearly from 1/(mstep_ramp+1) of the full values.
    fn mstep_schedule(&self, iteration: usize) -> (usize, usize) {
        if iteration >= self.mstep_ramp {
            return (self.num_samples, self.max_mstep_iterations);
        }
        let fraction = |full: usize| (full * (iteration + 1) / (self.mstep_ramp + 1)).max(1);
        (fraction(self.num_samples).max(2.min(self.num_samples)), fraction(self.max_mstep_iterations))
    }

    /// NUTS settings of the E-step: about num_samples independent samples per site
    fn hmc_settings(&self, settings: &HmcSettings, iteration: usize) -> HmcSettings {
        HmcSettings {
            num_tune: self.num_tune,
            num_draws: (10 * self.num_samples).max(50) as u64,
            target_ess: self.num_samples,
            max_draws: settings.max_draws,
            num_output_samples: self.num_samples,
            seed: settings.seed.wrapping_add(iteration as u64 + 1),
            marginal: false,
        }
    }
}

fn new_site_logp(
    shared: &Arc<SharedModel>,
    tree: &Arc<Tree>,
    alignment: &[u8],
    num_leaves: usize,
    site: usize,
) -> SiteLogp {
    SiteLogp {
        shared: Arc::clone(shared),
        tree: Arc::clone(tree),
        leaves: alignment[site * num_leaves..(site + 1) * num_leaves].to_vec(),
        workspace: Workspace::new(tree.num_nodes(), num_leaves),
    }
}

/// Prints the number of sampled sites every 10% of the sites
struct SamplingProgress<'a> {
    label: &'a str,
    num_sites: usize,
    verbosity: Verbosity,
    start: std::time::Instant,
    num_done: AtomicUsize,
}

impl<'a> SamplingProgress<'a> {
    fn new(label: &'a str, num_sites: usize, verbosity: Verbosity) -> Self {
        SamplingProgress { label, num_sites, verbosity, start: std::time::Instant::now(), num_done: AtomicUsize::new(0) }
    }

    fn site_done(&self) {
        let done = self.num_done.fetch_add(1, Ordering::Relaxed) + 1;
        let report_every = (self.num_sites / 10).max(1);
        if self.verbosity.should_print(Verbosity::Min) && (done % report_every == 0 || done == self.num_sites) {
            println!(
                "MUTSEL HMC {}: {}/{} sites sampled ({:.1}s)",
                self.label,
                done,
                self.num_sites,
                self.start.elapsed().as_secs_f64()
            );
        }
    }

    fn summary(&self, site_samples: &[SiteSamples], settings: &HmcSettings) {
        let num_sites = site_samples.len();
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
            self.label,
            self.start.elapsed().as_secs_f64(),
            total_draws / num_sites.max(1),
            total_divergences,
            sites_with_divergences,
            mean_steps
        );
        println!(
            "MUTSEL HMC {}: minimum ESS over sites {:.0}, {} sites below the target ESS {} after {} draws",
            self.label, min_ess, sites_below_target, settings.target_ess, settings.max_draws
        );
    }
}

/// Runs an independent NUTS chain for every site, starting at init[site]
fn sample_sites(
    shared: &Arc<SharedModel>,
    tree: &Arc<Tree>,
    alignment: &[u8],
    num_leaves: usize,
    init: &[Vec<f64>],
    settings: &HmcSettings,
    verbosity: Verbosity,
    label: &str,
) -> Vec<SiteSamples> {
    let num_sites = init.len();
    let progress = SamplingProgress::new(label, num_sites, verbosity);

    let site_samples: Vec<SiteSamples> = (0..num_sites)
        .into_par_iter()
        .map(|site| {
            let logp = new_site_logp(shared, tree, alignment, num_leaves, site);
            let mut samples = sample_site(logp, &init[site], settings, site);
            if settings.marginal {
                let mut logp = new_site_logp(shared, tree, alignment, num_leaves, site);
                samples.log_marginal = site_log_marginal(&mut logp, &samples, settings.seed ^ (site as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            }
            progress.site_done();
            samples
        })
        .collect();

    progress.summary(&site_samples, settings);
    site_samples
}

/// Samples every site with a NUTS chain that is kept between calls (the E-steps of EM). If chains is empty, the
/// chains start at init[site] and are tuned. Otherwise each chain continues from its last draw under the new shared
/// model and tree, with the step size and mass matrix of its tuning, after burn_in draws. Any fixed step size and
/// mass matrix leave the new posterior invariant, and the posterior moves little between EM iterations.
fn sample_sites_persistent(
    chains: &mut Vec<SiteChain>,
    shared: &Arc<SharedModel>,
    tree: &Arc<Tree>,
    alignment: &[u8],
    num_leaves: usize,
    init: &[Vec<f64>],
    settings: &HmcSettings,
    burn_in: usize,
    verbosity: Verbosity,
    label: &str,
) -> Vec<SiteSamples> {
    let num_sites = init.len();
    let progress = SamplingProgress::new(label, num_sites, verbosity);
    // The workspace is only allocated while a site is sampled, so that the kept chains are small
    let free_buffers = |chain: &mut SiteChain| {
        let mut math = chain.math_mut();
        let logp = math.logp_func_mut();
        logp.leaves = Vec::new();
        logp.workspace = Workspace::new(0, 0);
    };
    let burn_in = if chains.is_empty() {
        *chains = (0..num_sites)
            .into_par_iter()
            .map(|site| {
                let mut chain = new_site_chain(new_site_logp(shared, tree, alignment, num_leaves, site), &init[site], settings, site);
                free_buffers(&mut chain);
                chain
            })
            .collect();
        // The tuning draws already move the chains into the posterior
        0
    } else {
        burn_in
    };
    assert_eq!(chains.len(), num_sites);

    let site_samples: Vec<SiteSamples> = chains
        .par_iter_mut()
        .enumerate()
        .map(|(site, chain)| {
            *chain.math_mut().logp_func_mut() = new_site_logp(shared, tree, alignment, num_leaves, site);
            chain
                .refresh_state()
                .unwrap_or_else(|e| panic!("Site {}: could not continue HMC chain: {:?}", site, e));
            let samples = draw_site(chain, settings, burn_in, site);
            free_buffers(chain);
            progress.site_done();
            samples
        })
        .collect();

    progress.summary(&site_samples, settings);
    site_samples
}

/// log pi [num_sites] (up to a constant per site) for each of the thinned draws
fn thinned_log_pi(site_samples: &[SiteSamples], num_samples: usize) -> Vec<Vec<SVector<f64, 20>>> {
    (0..num_samples)
        .map(|k| {
            site_samples
                .iter()
                .map(|s| {
                    let x = SVector::<f64, NUM_COORDINATES>::from_column_slice(&s.thinned_draws[k * s.dim..k * s.dim + NUM_COORDINATES]);
                    helmert::coordinates_to_log_freq(&x)
                })
                .collect()
        })
        .collect()
}

/// Site rates [num_sites] for each of the thinned draws, None if site rates are not sampled
fn thinned_site_rates(site_samples: &[SiteSamples], num_samples: usize) -> Option<Vec<Vec<f64>>> {
    if site_samples.first().map_or(true, |s| s.dim == NUM_COORDINATES) {
        return None;
    }
    Some(
        (0..num_samples)
            .map(|k| site_samples.iter().map(|s| s.thinned_draws[k * s.dim + NUM_COORDINATES].exp()).collect())
            .collect(),
    )
}

/// Average substitution rate over all sites and samples, without normalization. Only the site frequencies
/// enter, the site rates have prior mean 1 and are a correction on top of the normalized MutSel rates.
fn mean_substitution_rate(Mu: &SMatrix<f64, 20, 20>, log_pi_samples: &[Vec<SVector<f64, 20>>]) -> f64 {
    log_pi_samples.iter().map(|log_pi| batch::mean_substitution_rate(Mu, log_pi)).sum::<f64>() / log_pi_samples.len() as f64
}

/// M-step: maximizes the log likelihood averaged over the posterior samples of the site frequencies (and site
/// rates, if site_rate_samples are given for the same samples) plus the log prior of the exchangeabilities,
/// with the rate normalization fixed. The log exchangeabilities log_S and the
/// branch lengths are updated in place. Returns the averaged log likelihood at the optimum.
fn m_step(
    parents: &[i32],
    columns: &[&[u8]],
    log_S: &mut [f64],
    branch_lengths: &mut [f64],
    log_pi_samples: &[Vec<SVector<f64, 20>>],
    site_rate_samples: Option<&[Vec<f64>]>,
    rate_scaling: f64,
    prior: &Prior,
    max_iterations: usize,
    verbosity: Verbosity,
) -> f64 {
    // L-BFGS on the deterministic objective (given the samples), stopped once an iteration improves it by less
    // than this relative to its value. It starts at the previous Mu and branch lengths and never ends worse.
    const REL_IMPROVEMENT: f64 = 1e-7;

    let num_samples = log_pi_samples.len() as f64;
    let num_branches = branch_lengths.len();

    // Expected log likelihood, objective (expected log likelihood plus log prior) and its gradient at
    // x = [log exchangeabilities, log branch lengths]
    let num_exchangeabilities = optimization::NUM_EXCHANGEABILITIES;
    let evaluate = |x: &[f64]| -> (f64, f64, Vec<f64>) {
        let log_S = &x[..num_exchangeabilities];
        let branch_lengths: Vec<f64> = x[num_exchangeabilities..].iter().map(|l| l.exp()).collect();
        let Mu = Mu(log_S);
        let mut log_likelihood = 0.0;
        let mut grad_Mu = SMatrix::<f64, 20, 20>::zeros();
        let mut grad_branch_lengths = vec![0.0; num_branches];
        for (k, log_pi) in log_pi_samples.iter().enumerate() {
            let result = batch::log_likelihood_and_gradients(
                parents,
                columns,
                &Mu,
                log_pi,
                &branch_lengths,
                batch::RateScaling::Fixed(rate_scaling),
                site_rate_samples.map(|rates| rates[k].as_slice()),
            );
            log_likelihood += result.log_likelihood / num_samples;
            grad_Mu += result.grad_Mu / num_samples;
            for (sum, g) in grad_branch_lengths.iter_mut().zip(&result.grad_branch_lengths) {
                *sum += g / num_samples;
            }
        }
        let (log_prior, grad_log_prior) = prior.evaluate_exchangeabilities(log_S);
        let mut gradient: Vec<f64> =
            optimization::Mu_backward(log_S, &grad_Mu).iter().zip(&grad_log_prior).map(|(g, p)| g + p).collect();
        gradient.extend(branch_lengths.iter().zip(&grad_branch_lengths).map(|(b, g)| g * b));
        let objective = log_likelihood + log_prior;
        (log_likelihood, objective, gradient)
    };

    let mut x = log_S.to_vec();
    x.extend(branch_lengths.iter().map(|b| b.ln()));
    let start = std::time::Instant::now();
    let num_evaluations = maximize_finite(
        |x| {
            let (_, objective, gradient) = evaluate(x);
            Some((objective, gradient))
        },
        &mut x,
        max_iterations,
        REL_IMPROVEMENT,
    );
    let (log_likelihood, objective, _) = evaluate(&x);
    log_S.copy_from_slice(&x[..num_exchangeabilities]);
    for (b, l) in branch_lengths.iter_mut().zip(&x[num_exchangeabilities..]) {
        *b = l.exp();
    }
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL HMC M-step: expected log likelihood {:.3}, objective {:.3} after {} evaluations with {} samples per site ({:.1}s)",
            log_likelihood,
            objective,
            num_evaluations,
            log_pi_samples.len(),
            start.elapsed().as_secs_f64()
        );
    }
    log_likelihood
}

/// S (upper triangle) and sqrt_pi of every site at the given frequencies, S multiplied by the site rate
fn rate_matrices(
    shared: &SharedModel,
    pi: &[SVector<f64, 20>],
    site_rates: &[f64],
) -> (Vec<SMatrix<f64, 20, 20>>, Vec<SVector<f64, 20>>) {
    pi.par_iter()
        .zip(site_rates)
        .map(|(pi, rate)| {
            let (S, sqrt_pi) = shared.rate_matrix(&pi.map(f64::ln));
            (S * *rate, sqrt_pi)
        })
        .unzip()
}

/// Mu, the global rate normalization and the branch lengths the final sampling is conditioned on
struct FittedModel {
    log_S: Vec<f64>,
    rate_scaling: f64,
    branch_lengths: Vec<f64>,
    /// Initial position of the final chain of every site
    positions: Vec<Vec<f64>>,
    init_coordinates: Vec<SVector<f64, NUM_COORDINATES>>,
    em_log_likelihood: Vec<f64>,
    em_tree_length: Vec<f64>,
    /// Prior of the site rates, with the sd fitted by EM if MUTSEL_SITE_RATE_FIT is set
    site_rate_prior: Option<SiteRatePrior>,
}

/// Prints the mean and the quantiles of site rates (draws over all sites)
fn print_site_rate_summary(label: &str, rates: impl Iterator<Item = f64>) {
    let mut rates: Vec<f64> = rates.collect();
    rates.sort_by(|a, b| a.total_cmp(b));
    let quantile = |q: f64| rates[((rates.len() - 1) as f64 * q).round() as usize];
    let log_sd = {
        let logs: Vec<f64> = rates.iter().map(|r| r.ln()).collect();
        let mean = logs.iter().sum::<f64>() / logs.len() as f64;
        (logs.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / logs.len() as f64).sqrt()
    };
    println!(
        "MUTSEL HMC {}: site rates mean {:.3}, quantiles 5/25/50/75/95% {:.3} {:.3} {:.3} {:.3} {:.3}, sd of log rates {:.3}",
        label,
        rates.iter().sum::<f64>() / rates.len() as f64,
        quantile(0.05),
        quantile(0.25),
        quantile(0.5),
        quantile(0.75),
        quantile(0.95),
        log_sd
    );
}

/// Mu and the branch lengths from the MAP estimate refined with Monte Carlo EM, see sample_internal
fn fit_with_em(
    parents: &[i32],
    branch_lengths: &[f64],
    alignment: &[u8],
    num_leaves: usize,
    verbosity: Verbosity,
    settings: &HmcSettings,
    em: &EmSettings,
    prior: &Arc<Prior>,
    fit_site_rate_sd: bool,
) -> FittedModel {
    let mut site_rate_prior = prior.site_rate;
    let shared_model = |Mu: &SMatrix<f64, 20, 20>, rate_scaling: f64, site_rate_prior: Option<SiteRatePrior>| {
        Arc::new(
            SharedModel::new(Mu, rate_scaling, Arc::clone(prior)).with_site_rate_prior(site_rate_prior),
        )
    };
    let columns = alignment_columns(alignment, num_leaves);
    // The joint MAP estimate of Mu, the site frequencies and the branch lengths
    let map_start = std::time::Instant::now();
    let map = map_estimate(parents, &columns, branch_lengths, prior, em.map_convergence, verbosity);
    if verbosity.should_print(Verbosity::Min) {
        println!("MUTSEL HMC: MAP estimate done in {:.1}s", map_start.elapsed().as_secs_f64());
    }
    let MapEstimate { mut log_S, mut branch_lengths, coordinates: init_coordinates } = map;
    let init_log_pi: Vec<SVector<f64, 20>> =
        init_coordinates.iter().map(helmert::coordinates_to_log_freq).collect();

    // Initial normalization: average substitution rate 1 at the initial frequencies, the same as in the
    // MAP estimate, so the branch lengths stay valid.
    let mut rate_scaling = 1.0 / mean_substitution_rate(&Mu(&log_S), &[init_log_pi]);
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL HMC: starting from the MAP estimate, tree length {:.4}, rate normalization 1/{:.5}",
            branch_lengths.iter().sum::<f64>(),
            1.0 / rate_scaling
        );
    }
    let mut positions: Vec<Vec<f64>> = init_coordinates
        .iter()
        .map(|x| with_initial_log_rate(x.as_slice().to_vec(), site_rate_prior))
        .collect();

    let mut chains: Vec<SiteChain> = Vec::new();
    let mut em_log_likelihood = Vec::with_capacity(em.iterations);
    let mut em_tree_length = Vec::with_capacity(em.iterations);
    for iteration in 0..em.iterations {
        let label = format!("EM iteration {}", iteration + 1);
        let old_Mu = Mu(&log_S);
        let shared = shared_model(&old_Mu, rate_scaling, site_rate_prior);
        let tree = Arc::new(Tree::new(parents, &branch_lengths));
        let em_settings = em.hmc_settings(&settings, iteration);
        let site_samples = sample_sites_persistent(
            &mut chains,
            &shared,
            &tree,
            alignment,
            num_leaves,
            &positions,
            &em_settings,
            em.num_burn_in,
            verbosity,
            &label,
        );

        // The last draw, which is the last thinned draw, for the final sampling
        positions = site_samples
            .iter()
            .map(|s| s.thinned_draws[(em.num_samples - 1) * s.dim..].to_vec())
            .collect();

        let log_pi_samples = thinned_log_pi(&site_samples, em.num_samples);
        let site_rate_samples = thinned_site_rates(&site_samples, em.num_samples);
        drop(site_samples);
        if let Some(rates) = &site_rate_samples {
            if verbosity.should_print(Verbosity::Min) {
                print_site_rate_summary(&label, rates.iter().flatten().copied());
            }
            if fit_site_rate_sd {
                let fitted = SiteRatePrior::fit(rates.iter().flatten().map(|r| r.ln()));
                if verbosity.should_print(Verbosity::Min) {
                    println!(
                        "MUTSEL HMC M-step: sd of the log site rates {:.4} (was {:.4})",
                        fitted.sd,
                        site_rate_prior.map_or(0.0, |p| p.sd)
                    );
                }
                site_rate_prior = Some(fitted);
            }
        }
        // The last thinned draws are the furthest from the chain starts
        let (mstep_samples, mstep_iterations) = em.mstep_schedule(iteration);
        let log_likelihood = m_step(
            parents,
            &columns,
            &mut log_S,
            &mut branch_lengths,
            &log_pi_samples[em.num_samples - mstep_samples..],
            site_rate_samples.as_deref().map(|rates| &rates[em.num_samples - mstep_samples..]),
            rate_scaling,
            prior,
            mstep_iterations,
            verbosity,
        );

        let new_Mu = Mu(&log_S);
        let new_rate_scaling = 1.0 / mean_substitution_rate(&new_Mu, &log_pi_samples);
        // Rescale the branch lengths so the likelihood is unchanged with the new normalization
        for b in branch_lengths.iter_mut() {
            *b *= rate_scaling / new_rate_scaling;
        }
        rate_scaling = new_rate_scaling;

        let tree_length: f64 = branch_lengths.iter().sum();
        let max_rel_Mu_change = (0..20)
            .flat_map(|i| (0..20).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| ((new_Mu[(i, j)] - old_Mu[(i, j)]) / old_Mu[(i, j)]).abs())
            .fold(0.0, f64::max);
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

    FittedModel {
        log_S,
        rate_scaling,
        branch_lengths,
        positions,
        init_coordinates,
        em_log_likelihood,
        em_tree_length,
        site_rate_prior,
    }
}

/// Samples the Helmert coordinates of every site from its posterior and returns (S, sqrt_pi) of every site, S with
/// its upper triangle set, evaluated at the posterior mean of the site frequencies, or with num_tree_draws > 1
/// at that many evenly spaced posterior draws per site ([num_sites * num_tree_draws], ordered by site), which
/// IQ-TREE uses as an equally weighted mixture per site.
///
/// Mu and the branch lengths start at the joint MAP estimate (optimization::map_estimate, which also
/// gives the initial site positions) and are then refined with Monte Carlo EM: the E-step samples the sites
/// given Mu and the branch lengths, the M-step maximizes the log likelihood averaged over these
/// samples plus the log prior of the exchangeabilities. This converges to their MAP estimate with
/// the site frequencies integrated out. The rate normalization is kept fixed within each iteration and
/// afterwards reset to an average substitution rate of 1 over the posterior samples, rescaling the
/// branch lengths so that the likelihood stays the same.
pub fn sample_internal(
    parents: &[i32],
    branch_lengths: &[f64],
    alignment: &[u8],
    num_sites: usize,
    num_leaves: usize,
    verbosity: Verbosity,
    out_prefix: &str,
    num_tree_draws: usize,
) -> Result<(Vec<SMatrix<f64, 20, 20>>, Vec<SVector<f64, 20>>)> {
    let settings = HmcSettings::from_env();
    let em = EmSettings::from_env();
    let prior = Arc::new(Prior::from_env());
    let fit_site_rate_sd = env_or("MUTSEL_SITE_RATE_FIT", 0u8) != 0;
    assert!(
        !fit_site_rate_sd || prior.site_rate.is_some(),
        "MUTSEL_SITE_RATE_FIT needs a starting site_rate_sd > 0 in the prior"
    );
    println!(
        "MUTSEL HMC: sampling Helmert coordinates per site with NUTS ({} tuning, at least {} draws, until ESS >= {} or {} draws, {} thinned samples per site, seed {}), {}",
        settings.num_tune,
        settings.num_draws,
        settings.target_ess,
        settings.max_draws,
        settings.num_output_samples,
        settings.seed,
        match prior.site_rate {
            Some(prior) => format!(
                "with a log-normal site rate multiplying the MutSel rate (log rate ~ N(-sd^2/2, sd^2), mean rate 1, sd {}{})",
                prior.sd,
                if fit_site_rate_sd { " at the start, fitted in every M-step" } else { "" }
            ),
            None => "site rates 1".to_string(),
        }
    );
    println!(
        "MUTSEL HMC: Mu and branch lengths from the MAP estimate (L-BFGS, at most {} iterations, relative improvement {:.0e}) refined by Monte Carlo EM until the tree length changes by less than {:.1}% over {} iterations, at most {} iterations ({} samples per site, {} tuning draws in the first iteration, then the chains continue with {} burn-in draws, at most {} M-step iterations, fewer samples and M-step iterations in the first {})",
        em.map_convergence.max_iterations,
        em.map_convergence.min_rel_improvement,
        100.0 * em.tree_length_tolerance,
        em.window,
        em.iterations,
        em.num_samples,
        em.num_tune,
        em.num_burn_in,
        em.max_mstep_iterations,
        em.mstep_ramp
    );
    assert!(settings.num_output_samples >= 1 && (em.iterations == 0 || em.num_samples >= 1));
    assert!(
        num_tree_draws >= 1 && num_tree_draws <= settings.num_output_samples,
        "MUTSEL_TREE_DRAWS must be between 1 and MUTSEL_HMC_SAMPLES"
    );

    let fitted = fit_with_em(parents, branch_lengths, alignment, num_leaves, verbosity, &settings, &em, &prior, fit_site_rate_sd);
    let FittedModel {
        log_S,
        rate_scaling,
        branch_lengths,
        positions,
        init_coordinates,
        em_log_likelihood,
        em_tree_length,
        site_rate_prior,
    } = fitted;
    let shared = Arc::new(
        SharedModel::new(&Mu(&log_S), rate_scaling, prior).with_site_rate_prior(site_rate_prior),
    );
    let tree = Arc::new(Tree::new(parents, &branch_lengths));
    let site_samples = sample_sites(&shared, &tree, alignment, num_leaves, &positions, &settings, verbosity, "final");
    let dim = shared.dim();
    if site_rate_prior.is_some() && verbosity.should_print(Verbosity::Min) {
        print_site_rate_summary(
            "final",
            site_samples
                .iter()
                .flat_map(|s| s.thinned_draws.chunks_exact(dim).map(|draw| draw[NUM_COORDINATES].exp())),
        );
    }

    let num_output_samples = settings.num_output_samples;
    // coordinate_samples [num_sites, num_output_samples, 19]
    let coordinate_samples: Vec<f64> = site_samples
        .iter()
        .flat_map(|s| s.thinned_draws.chunks_exact(dim).flat_map(|draw| draw[..NUM_COORDINATES].iter().copied()))
        .collect();
    let pi_samples: Vec<f64> = coordinate_samples
        .par_chunks_exact(NUM_COORDINATES)
        .flat_map_iter(|x| shared.pi(&SVector::from_column_slice(x)).1.iter().copied().collect::<Vec<_>>())
        .collect();
    let coordinate_samples = Array::new(&[num_sites, num_output_samples, NUM_COORDINATES], coordinate_samples);

    // log_site_rates [num_sites, num_output_samples], if site rates are sampled
    let log_site_rates = site_rate_prior.map(|_| {
        Array::new(
            &[num_sites, num_output_samples],
            site_samples
                .iter()
                .flat_map(|s| s.thinned_draws.chunks_exact(dim).map(|draw| draw[NUM_COORDINATES]))
                .collect(),
        )
    });

    // Posterior mean of the site frequencies and the site rates over all draws
    let mean_pi: Vec<SVector<f64, 20>> = site_samples.iter().map(|s| s.mean_pi).collect();
    let mean_site_rate: Vec<f64> = site_samples.iter().map(|s| s.mean_site_rate).collect();

    let (S, sqrt_pi) = rate_matrices(&shared, &mean_pi, &mean_site_rate);
    let average_rate_mean_pi =
        S.iter().zip(&sqrt_pi).map(|(S, sqrt_pi)| model::substitution_rate(S, sqrt_pi)).sum::<f64>() / num_sites as f64;
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL HMC: average substitution rate at posterior mean frequencies {:.4}",
            average_rate_mean_pi
        );
    }

    let npz_path = format!("{}.hmc.npz", out_prefix);
    let per_site = |f: &dyn Fn(&SiteSamples) -> f64| Array::vector(site_samples.iter().map(f).collect());
    let mut arrays: Vec<(&str, Array)> = vec![
        ("coordinate_samples", coordinate_samples),
        ("pi_samples", Array::new(&[num_sites, num_output_samples, 20], pi_samples)),
        ("init_coordinates", Array::from_rows(&init_coordinates)),
        ("mean_pi", Array::from_rows(&mean_pi)),
        ("Mu", Array::from_matrix(&Mu(&log_S))),
        ("log_exchangeabilities", Array::vector(log_S.clone())),
        ("rate_scaling", Array::scalar(rate_scaling)),
        ("branch_lengths", Array::vector(branch_lengths.clone())),
        ("em_log_likelihood", Array::vector(em_log_likelihood)),
        ("em_tree_length", Array::vector(em_tree_length)),
        ("ess", per_site(&|s| s.ess)),
        ("num_draws", per_site(&|s| s.num_draws as f64)),
        ("num_divergences", per_site(&|s| s.num_divergences as f64)),
        ("step_size", per_site(&|s| s.step_size)),
        ("mean_num_steps", per_site(&|s| s.mean_num_steps)),
        ("log_marginal", per_site(&|s| s.log_marginal)),
        ("parents", Array::vector(parents.iter().map(|&p| p as f64).collect())),
    ];
    if let Some(log_site_rates) = log_site_rates {
        arrays.push(("log_site_rates", log_site_rates));
        arrays.push(("mean_site_rate", Array::vector(mean_site_rate.clone())));
        arrays.push(("site_rate_sd", Array::scalar(site_rate_prior.map_or(0.0, |p| p.sd))));
    }
    write_npz(Path::new(&npz_path), &arrays.iter().map(|(name, array)| (*name, array)).collect::<Vec<_>>())?;
    println!("MUTSEL HMC: samples written to {}", npz_path);

    if num_tree_draws == 1 {
        return Ok((S, sqrt_pi));
    }
    println!("MUTSEL HMC: {} posterior draws per site for the tree search", num_tree_draws);
    let tree_draw = |k: usize| k * settings.num_output_samples / num_tree_draws;
    Ok(site_samples
        .par_iter()
        .flat_map_iter(|s| {
            let shared = &shared;
            (0..num_tree_draws).map(move |k| {
                let position = &s.thinned_draws[tree_draw(k) * dim..(tree_draw(k) + 1) * dim];
                let x = SVector::<f64, NUM_COORDINATES>::from_column_slice(&position[..NUM_COORDINATES]);
                // rate_matrix only needs log_pi up to a constant
                let (S, sqrt_pi) = shared.rate_matrix(&(shared.basis.transpose() * x));
                (S * shared.site_rate(position), sqrt_pi)
            })
        })
        .unzip())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::LOG_M_EQUI,
        prior::tests::{isotropic_gmm, test_prior},
    };

    const RATE_SCALING: f64 = 1.3;

    fn test_tree() -> Tree {
        // ((0,1),(2,3)) plus a fifth leaf at the root
        let parents = [5, 5, 6, 6, 7, 7, 7, -1];
        let branch_lengths = [0.1, 0.3, 0.2, 0.05, 0.4, 0.15, 0.25, 0.0];
        Tree::new(&parents, &branch_lengths)
    }

    fn test_Mu() -> SMatrix<f64, 20, 20> {
        Mu(&test_prior().log_exchangeabilities)
    }

    fn test_shared(rate_scaling: f64) -> SharedModel {
        SharedModel::new(&test_Mu(), rate_scaling, Arc::new(test_prior()))
    }

    /// Shared model whose prior on the site frequencies is a single isotropic Gaussian
    fn gaussian_shared(mean: SVector<f64, NUM_COORDINATES>, std: f64) -> SharedModel {
        let prior = Prior { site_frequencies: isotropic_gmm(&[1.0], &[mean], &[std]), ..test_prior() };
        SharedModel::new(&test_Mu(), RATE_SCALING, Arc::new(prior))
    }

    const ALIGNMENT: [u8; 5] = [0u8, 0, 3, 20, 7];

    fn test_logp() -> SiteLogp {
        let shared = Arc::new(test_shared(RATE_SCALING));
        new_site_logp(&shared, &Arc::new(test_tree()), &ALIGNMENT, 5, 0)
    }

    fn test_point() -> Vec<f64> {
        (0..NUM_COORDINATES).map(|i| 0.3 * ((i as f64) * 1.7).sin()).collect()
    }

    #[test]
    fn rate_matrix_is_invariant_to_a_constant_in_log_pi() {
        let logp = test_logp();
        let x = SVector::<f64, NUM_COORDINATES>::from_column_slice(&test_point());
        let eval = logp.shared.evaluate(&x);
        // log_pi up to a constant
        let (S, sqrt_pi) = logp.shared.rate_matrix(&(logp.shared.basis.transpose() * x).add_scalar(0.7));
        assert!((S - eval.S).abs().max() < 1e-13 * eval.S.abs().max());
        assert!((sqrt_pi - eval.sqrt_pi).abs().max() < 1e-14);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let mut logp = test_logp();
        let x = test_point();
        let mut grad = vec![0.0; NUM_COORDINATES];
        let value = logp.logp(&x, &mut grad).unwrap();
        assert!(value.is_finite());

        let eps = 1e-6;
        let mut dummy = vec![0.0; NUM_COORDINATES];
        for i in 0..NUM_COORDINATES {
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

    fn test_logp_with_site_rates(rate_scaling: f64, sd: f64) -> SiteLogp {
        let mut logp = test_logp();
        logp.shared = Arc::new(
            test_shared(rate_scaling)
                .with_site_rate_prior(Some(SiteRatePrior { sd })),
        );
        logp
    }

    #[test]
    fn site_rate_gradient_matches_finite_differences() {
        let mut logp = test_logp_with_site_rates(RATE_SCALING, 0.5);
        let mut x = test_point();
        x.push(0.3);
        let mut grad = vec![0.0; NUM_COORDINATES + 1];
        logp.logp(&x, &mut grad).unwrap();
        let eps = 1e-6;
        let mut dummy = vec![0.0; NUM_COORDINATES + 1];
        for i in 0..=NUM_COORDINATES {
            let mut xp = x.clone();
            xp[i] += eps;
            let mut xm = x.clone();
            xm[i] -= eps;
            let fd = (logp.logp(&xp, &mut dummy).unwrap() - logp.logp(&xm, &mut dummy).unwrap()) / (2.0 * eps);
            assert!((fd - grad[i]).abs() < 1e-5 * (1.0 + fd.abs()), "component {}: finite difference {} vs gradient {}", i, fd, grad[i]);
        }
    }

    #[test]
    fn site_rate_multiplies_the_rate_scaling() {
        // Without the prior of the log rate, a site rate r is the rate scaling times r
        let log_rate = -0.4;
        let prior = SiteRatePrior { sd: 0.7 };
        let mut with_rate = test_logp_with_site_rates(RATE_SCALING, prior.sd);
        let mut position = test_point();
        position.push(log_rate);
        let mut grad = vec![0.0; NUM_COORDINATES + 1];
        let value = with_rate.logp(&position, &mut grad).unwrap() - prior.evaluate(log_rate).0;

        let mut scaled = test_logp();
        scaled.shared = Arc::new(test_shared(RATE_SCALING * f64::exp(log_rate)));
        let mut scaled_grad = vec![0.0; NUM_COORDINATES];
        let reference = scaled.logp(&test_point(), &mut scaled_grad).unwrap();
        assert!((value - reference).abs() < 1e-12 * reference.abs(), "{} vs {}", value, reference);
        for i in 0..NUM_COORDINATES {
            assert!((grad[i] - scaled_grad[i]).abs() < 1e-10 * (1.0 + scaled_grad[i].abs()), "component {}", i);
        }
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
    fn site_rates_of_an_uninformative_column_follow_the_prior() {
        // An all-gap column has likelihood 1, so the sampled log rates are draws from the prior
        let sd = 0.6;
        let shared = Arc::new(
            test_shared(RATE_SCALING)
                .with_site_rate_prior(Some(SiteRatePrior { sd })),
        );
        let tree = Arc::new(test_tree());
        let alignment = [20u8; 5];
        let settings = HmcSettings {
            num_tune: 400,
            num_draws: 2000,
            target_ess: 1000,
            max_draws: 50000,
            num_output_samples: 1000,
            seed: 3,
            marginal: false,
        };
        let init = vec![with_initial_log_rate(test_point(), Some(SiteRatePrior { sd }))];
        let samples = sample_sites(&shared, &tree, &alignment, 5, &init, &settings, Verbosity::Quiet, "prior");
        let log_rates: Vec<f64> = samples[0].thinned_draws.chunks_exact(NUM_COORDINATES + 1).map(|d| d[NUM_COORDINATES]).collect();
        let n = log_rates.len() as f64;
        let mean = log_rates.iter().sum::<f64>() / n;
        let std = (log_rates.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / n).sqrt();
        // About 1000 effective samples: standard errors 0.019 of the mean and 0.013 of the standard deviation
        assert!((mean + 0.5 * sd * sd).abs() < 0.08, "mean log rate {} instead of {}", mean, -0.5 * sd * sd);
        assert!((std - sd).abs() < 0.06, "sd of the log rates {} instead of {}", std, sd);
        assert!((samples[0].mean_site_rate - 1.0).abs() < 0.1, "mean rate {}", samples[0].mean_site_rate);
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
        fn sqrt(self) -> Dual {
            let r = self.v.sqrt();
            Dual { v: r, d: self.d / (2.0 * r) }
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

    /// Independent reference of the per-site model: returns <G, S(x)> + <b, sqrt_pi(x)>
    fn reference_surrogate(
        model: &SharedModel,
        x: &[Dual],
        G: &SMatrix<f64, 20, 20>,
        b: &SVector<f64, 20>,
    ) -> Dual {
        let c = Dual::constant;
        let log_pi: Vec<Dual> = (0..20)
            .map(|a| (0..NUM_COORDINATES).fold(c(0.0), |acc, k| acc + c(model.basis[(k, a)]) * x[k]))
            .collect();
        let max = log_pi.iter().map(|l| l.v).fold(f64::NEG_INFINITY, f64::max);
        let unnormalized: Vec<Dual> = log_pi.iter().map(|&l| (l - c(max)).exp()).collect();
        let total = unnormalized.iter().fold(c(0.0), |acc, &u| acc + u);
        let sqrt_pi: Vec<Dual> = unnormalized.iter().map(|&u| (u / total).sqrt()).collect();
        let fitness: Vec<Dual> = (0..20).map(|a| log_pi[a] - c(LOG_M_EQUI[a])).collect();
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
        value
    }

    #[test]
    fn backpropagation_matches_forward_mode_derivatives() {
        let model = test_shared(RATE_SCALING);

        // Arbitrary upstream gradients
        let G = SMatrix::<f64, 20, 20>::from_fn(|i, j| ((i * 20 + j) as f64 * 0.37).sin());
        let b = SVector::<f64, 20>::from_fn(|a, _| ((a as f64) * 1.3).cos());

        // x with fitness differences exactly zero: log_pi equal to the mutation equilibrium
        let neutral = model.basis * LOG_M_EQUI;
        let direction: Vec<f64> = test_point();

        let mut points: Vec<Vec<f64>> = vec![];
        for scale in [0.0, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0] {
            points.push((0..NUM_COORDINATES).map(|k| neutral[k] + scale * direction[k]).collect());
        }
        for scale in [1e-6, 1e-2, 1.0, 3.0] {
            points.push(direction.iter().map(|d| d * scale / 0.3).collect());
        }

        for x in points {
            let x_vec = SVector::<f64, NUM_COORDINATES>::from_column_slice(&x);
            let eval = model.evaluate(&x_vec);
            // The prior is checked in prior.rs
            let grad = model.backpropagate(&eval, &G, &b) - eval.grad_log_prior;

            let mut max_rel_error = 0.0f64;
            let max_grad = grad.iter().map(|g| g.abs()).fold(0.0, f64::max);
            for k in 0..NUM_COORDINATES {
                let x_dual: Vec<Dual> = (0..NUM_COORDINATES)
                    .map(|m| Dual { v: x[m], d: if m == k { 1.0 } else { 0.0 } })
                    .collect();
                let reference = reference_surrogate(&model, &x_dual, &G, &b).d;
                max_rel_error = max_rel_error.max((grad[k] - reference).abs() / max_grad);
            }
            let min_abs_fitness_diff = (0..20)
                .flat_map(|i| ((i + 1)..20).map(move |j| (i, j)))
                .map(|(i, j)| {
                    let fitness = eval.log_pi - LOG_M_EQUI;
                    (fitness[j] - fitness[i]).abs()
                })
                .fold(f64::INFINITY, f64::min);
            println!(
                "min |f_j - f_i| {:.1e}: max error relative to max |grad| {:.2e}",
                min_abs_fitness_diff, max_rel_error
            );
            assert!(max_rel_error < 1e-13, "relative error {:e}", max_rel_error);
        }
    }

    /// The bridge estimate of the marginal likelihood of a site against brute force Monte Carlo from a Gaussian prior
    #[test]
    fn site_log_marginal_matches_prior_monte_carlo() {
        let mean = SVector::<f64, NUM_COORDINATES>::from_column_slice(&test_point());
        let std = 0.7;
        let shared = Arc::new(gaussian_shared(mean, std));
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
        let mut gradient = vec![0.0; NUM_COORDINATES];
        for _ in 0..1 << 17 {
            let x = SVector::<f64, NUM_COORDINATES>::from_fn(|_, _| {
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

    #[test]
    fn mstep_schedule_ramps_up_to_the_full_settings() {
        let em = EmSettings { num_samples: 8, max_mstep_iterations: 50, mstep_ramp: 3, ..EmSettings::from_env() };
        let schedule: Vec<_> = (0..5).map(|i| em.mstep_schedule(i)).collect();
        assert_eq!(schedule, vec![(2, 12), (4, 25), (6, 37), (8, 50), (8, 50)]);
        let em = EmSettings { mstep_ramp: 0, ..em };
        assert_eq!(em.mstep_schedule(0), (8, 50));
        let em = EmSettings { num_samples: 1, mstep_ramp: 3, ..em };
        assert_eq!(em.mstep_schedule(0).0, 1);
    }

    /// Full gradient including phylo_grad against a fourth order central difference
    #[test]
    fn full_gradient_matches_fourth_order_finite_differences() {
        let mut logp = test_logp();
        let mut dummy = vec![0.0; NUM_COORDINATES];
        for x in [test_point(), test_point().iter().map(|v| v * 5.0).collect::<Vec<f64>>()] {
            let mut grad = vec![0.0; NUM_COORDINATES];
            logp.logp(&x, &mut grad).unwrap();
            let max_grad = grad.iter().map(|g| g.abs()).fold(0.0, f64::max);
            let h = 1e-3;
            let mut max_rel_error = 0.0f64;
            for k in 0..NUM_COORDINATES {
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
        let x = vec![0.1; NUM_COORDINATES];
        let mut grad = vec![0.0; NUM_COORDINATES];
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
        let shared = Arc::new(test_shared(RATE_SCALING));
        let mut logp = new_site_logp(&shared, &Arc::new(Tree::new(&parents, &branch_lengths)), &alignment, num_leaves, 0);
        let x = SVector::<f64, NUM_COORDINATES>::from_column_slice(&test_point());
        let mut grad = vec![0.0; NUM_COORDINATES];
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
        let mut workspace = Workspace::new(tree.num_nodes(), num_leaves);
        let mut felsenstein = || {
            phylo_grad::calculate_gradients_single_column(
                &tree.parents,
                &alignment,
                &tree.branch_lengths,
                eval.S.as_view(),
                eval.sqrt_pi.as_view(),
                &mut workspace,
                None,
            )
        };
        let result = felsenstein();
        let felsenstein = time(&mut || {
            std::hint::black_box(felsenstein());
        });
        let backpropagate = time(&mut || {
            std::hint::black_box(model.backpropagate(&eval, &result.grad_s, &result.grad_sqrt_pi));
        });
        println!(
            "100 leaves: logp+grad {:.1} us = evaluate {:.1} + phylo_grad {:.1} + backpropagate {:.1}",
            total, evaluate, felsenstein, backpropagate
        );
    }

    #[test]
    fn persistent_chains_follow_a_changed_posterior() {
        // A narrow Gaussian prior dominates the posterior, so moving its mean moves the posterior by about as much
        let gaussian_prior = |shift: f64| {
            let mean = SVector::<f64, NUM_COORDINATES>::from_iterator(test_point().iter().map(|x| x + shift));
            Arc::new(gaussian_shared(mean, 0.3))
        };
        let tree = Arc::new(test_tree());
        let alignment: Vec<u8> = ALIGNMENT.iter().chain(ALIGNMENT.iter()).copied().collect();
        let init = vec![test_point(); 2];
        let settings = HmcSettings {
            num_tune: 200,
            num_draws: 400,
            target_ess: 100,
            max_draws: 20000,
            num_output_samples: 100,
            seed: 5,
            marginal: false,
        };
        let mean_draw = |samples: &SiteSamples| {
            samples
                .thinned_draws
                .chunks_exact(NUM_COORDINATES)
                .map(SVector::<f64, NUM_COORDINATES>::from_column_slice)
                .sum::<SVector<f64, NUM_COORDINATES>>()
                / settings.num_output_samples as f64
        };

        let mut chains = Vec::new();
        let before = sample_sites_persistent(&mut chains, &gaussian_prior(0.0), &tree, &alignment, 5, &init, &settings, 20, Verbosity::Quiet, "before");
        let after = sample_sites_persistent(&mut chains, &gaussian_prior(1.0), &tree, &alignment, 5, &init, &settings, 20, Verbosity::Quiet, "after");
        let fresh = sample_sites(&gaussian_prior(1.0), &tree, &alignment, 5, &init, &settings, Verbosity::Quiet, "fresh");
        for site in 0..2 {
            let shift = (mean_draw(&after[site]) - mean_draw(&before[site])).mean();
            assert!((shift - 1.0).abs() < 0.2, "site {}: the posterior mean moved by {} instead of about 1", site, shift);
            // Each coordinate of a mean has a Monte Carlo error of about 0.03
            let difference = (mean_draw(&after[site]) - mean_draw(&fresh[site])).abs().max();
            assert!(difference < 0.15, "site {}: continued and fresh chains differ by {}", site, difference);
        }
    }
}
