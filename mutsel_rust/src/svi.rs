//! Stochastic variational inference of the site frequencies with a full-rank Gaussian prior on log pi.
//!
//! log pi of a site is only defined up to an additive constant, so it is represented by its 19
//! coordinates x = B log pi in a fixed orthonormal basis B [19, 20] of the sum-zero subspace
//! (Helmert basis), log pi = B^T x up to the constant. The prior is a full-rank Gaussian
//! N(mu, C C^T) on x, trained by variational EM on 19 phyla (data::GAUSSIAN_PRIOR_*). Every site
//! gets a full-rank Gaussian q(x) = N(m, L L^T).
//!
//! Inference for one alignment, jointly with Mu (with the Mu penalty) and the branch lengths:
//! 1. PMSF site frequencies with the branch lengths fitted to them (two_step_light_pmsf),
//! 2. the MAP estimate of x of every site, Mu and the branch lengths (Adam at a large learning
//!    rate on the deterministic objective),
//! 3. the full-rank Laplace approximation of every site at the MAP as initial q,
//! 4. SVI: the importance weighted ELBO
//!        sum_sites log mean_k p(column, x_k) / q(x_k),    x_k = m + L eps_k,
//!    over 2 * pairs antithetic draws per site, maximized over q of all sites, Mu and the branch
//!    lengths with Adam (constant learning rate for the first quarter of the steps, then a
//!    geometric decay; the parameters are averaged over the last quarter; during a warm-up only q
//!    is optimized).
//! The substitution rates of every draw are normalized to an average of 1 over the sites.
//!
//! The site models given to IQ-TREE are K tempered samples per site, m +- t L eps in antithetic
//! pairs (centred on the mode of q, t = MUTSEL_SAMPLE_SCALE, default 0.5), whose likelihoods
//! IQ-TREE averages; with K = 1 the site frequencies at the mode of q.
//!
//! The objective and its gradient are computed with nalgebra per site and phylo_grad over all
//! sites together (per-site branch length gradients), in parallel with rayon; only Mu(log_R) and
//! its penalty use candle.

use std::{
    collections::VecDeque,
    path::Path,
    sync::{Arc, Mutex},
};

use candle_core::{Tensor, Var};
use phylo_grad::{
    FelsensteinTree,
    nalgebra::{SMatrix, SVector},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::{
    MutselParams, Verbosity,
    felsenstein::FelsensteinOp,
    model::{self, calc_rate_matrix},
    optimization::{Mu, Mu_penalty, load_init_R, two_step_light_pmsf},
    utils::tensor_full,
};

pub const DIM: usize = 19;

pub type Coordinates = SVector<f64, DIM>;
pub type Cholesky = SMatrix<f64, DIM, DIM>;
type Matrix20 = SMatrix<f64, 20, 20>;
type Basis = SMatrix<f64, DIM, 20>;

/// Bounds of the diagonal of L of the Laplace initialization
const MIN_SCALE: f64 = 1e-3;
const MAX_SCALE: f64 = 10.0;

const LOG_2PI: f64 = 1.8378770664093453;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("Could not parse {}={}", name, value)),
        Err(_) => default,
    }
}

/// Orthonormal basis of the sum-zero subspace of R^20 (Helmert), one basis vector per row
pub fn basis() -> Basis {
    Basis::from_fn(|k, j| {
        let n = (k + 1) as f64;
        let norm = (n * (n + 1.0)).sqrt();
        if j <= k {
            1.0 / norm
        } else if j == k + 1 {
            -n / norm
        } else {
            0.0
        }
    })
}

/// Coordinates of log pi, which loses its mean
pub fn log_freq_to_coordinates(log_pi: &SVector<f64, 20>) -> Coordinates {
    basis() * log_pi
}

#[derive(Debug, Clone, Copy)]
pub struct SviSettings {
    /// Adam steps of the SVI
    pub steps: usize,
    /// Antithetic pairs of draws per site and step
    pub pairs: usize,
    /// Adam learning rate, constant for the first quarter of the steps, then decaying geometrically to lr_final
    pub lr: f64,
    pub lr_final: f64,
    /// Fraction of the steps (at the start) in which only q is optimized
    pub warmup_fraction: f64,
    /// Maximum L-BFGS iterations per site of the Laplace initialization
    pub laplace_iterations: usize,
    /// Maximum Adam iterations of the MAP stage, 0 to skip it
    pub map_iterations: usize,
    pub map_lr: f64,
    /// Scale t of the tempered samples m +- t L eps given to IQ-TREE
    pub sample_scale: f64,
    pub seed: u64,
}

impl SviSettings {
    /// Reads MUTSEL_VI_STEPS, MUTSEL_VI_PAIRS, MUTSEL_VI_LR, MUTSEL_VI_LR_FINAL, MUTSEL_VI_WARMUP,
    /// MUTSEL_VI_LAPLACE_ITERATIONS, MUTSEL_VI_MAP_ITERATIONS, MUTSEL_VI_MAP_LR, MUTSEL_SAMPLE_SCALE and
    /// MUTSEL_VI_SEED from the environment.
    pub fn from_env() -> SviSettings {
        let settings = SviSettings {
            steps: env_or("MUTSEL_VI_STEPS", 1000),
            pairs: env_or("MUTSEL_VI_PAIRS", 1),
            lr: env_or("MUTSEL_VI_LR", 0.002),
            lr_final: env_or("MUTSEL_VI_LR_FINAL", 0.0002),
            warmup_fraction: env_or("MUTSEL_VI_WARMUP", 0.25),
            laplace_iterations: env_or("MUTSEL_VI_LAPLACE_ITERATIONS", 200),
            map_iterations: env_or("MUTSEL_VI_MAP_ITERATIONS", 500),
            map_lr: env_or("MUTSEL_VI_MAP_LR", 0.03),
            sample_scale: env_or("MUTSEL_SAMPLE_SCALE", 0.5),
            seed: env_or("MUTSEL_VI_SEED", 42),
        };
        assert!(settings.pairs >= 1, "MUTSEL_VI_PAIRS must be at least 1");
        assert!((0.0..=1.0).contains(&settings.warmup_fraction), "MUTSEL_VI_WARMUP must be in [0, 1]");
        assert!(settings.sample_scale >= 0.0, "MUTSEL_SAMPLE_SCALE must not be negative");
        settings
    }
}

/// Full-rank Gaussian N(mean, cholesky cholesky^T) on the coordinates
#[derive(Debug, Clone)]
pub struct GaussianPrior {
    pub mean: Coordinates,
    /// Lower triangular with positive diagonal
    pub cholesky: Cholesky,
}

impl GaussianPrior {
    /// The prior compiled in (data::GAUSSIAN_PRIOR_*), or the one in the npz file of MUTSEL_PRIOR
    pub fn load() -> Result<GaussianPrior, candle_core::Error> {
        match std::env::var("MUTSEL_PRIOR") {
            Ok(path) => GaussianPrior::read_npz(Path::new(&path)),
            Err(_) => Ok(GaussianPrior::builtin()),
        }
    }

    pub fn builtin() -> GaussianPrior {
        let parse = |s: &str| -> Vec<f64> { s.split_whitespace().map(|v| v.parse::<f64>().unwrap()).collect() };
        GaussianPrior::from_leading_block(&parse(crate::data::GAUSSIAN_PRIOR_MEAN), &parse(crate::data::GAUSSIAN_PRIOR_CHOLESKY))
    }

    /// The prior from a mean [n] and a row-major lower triangular Cholesky factor [n, n] with
    /// n >= 19: the marginal of the first 19 coordinates, the leading block of the Cholesky factor
    fn from_leading_block(mean: &[f64], cholesky: &[f64]) -> GaussianPrior {
        let n = mean.len();
        assert!(n >= DIM && cholesky.len() == n * n, "prior of dimension {} with {} Cholesky entries", n, cholesky.len());
        GaussianPrior {
            mean: Coordinates::from_fn(|i, _| mean[i]),
            cholesky: Cholesky::from_fn(|i, j| cholesky[i * n + j]),
        }
    }

    /// From an npz file with the arrays mean and cholesky (row-major)
    pub fn read_npz(path: &Path) -> Result<GaussianPrior, candle_core::Error> {
        let arrays: std::collections::HashMap<String, Tensor> = Tensor::read_npz(path)?.into_iter().collect();
        let get = |name: &str| {
            arrays
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("{} has no array {}", path.display(), name)))
        };
        Ok(GaussianPrior::from_leading_block(&get("mean")?.to_vec1::<f64>()?, &get("cholesky")?.flatten_all()?.to_vec1::<f64>()?))
    }

    pub fn covariance(&self) -> Cholesky {
        self.cholesky * self.cholesky.transpose()
    }

    pub fn log_det(&self) -> f64 {
        2.0 * self.cholesky.diagonal().iter().map(|d| d.ln()).sum::<f64>()
    }
}

/// Constants of the per-site posterior which are shared by all sites of an alignment.
struct SharedModel {
    /// log_pi = basis^T * x (up to a constant)
    basis: Basis,
    /// Log of the mutation equilibrium (diagonal of Mu)
    log_mutation_equilibrium: SVector<f64, 20>,
    /// Mu with the rate normalization already applied
    scaled_Mu: Matrix20,
    prior_mean: Coordinates,
    /// Inverse of the Cholesky factor of the prior
    prior_cholesky_inverse: Cholesky,
    /// log det C + DIM / 2 log(2 pi)
    prior_log_normalization: f64,
}

struct SiteModelEval {
    log_prior: f64,
    grad_log_prior: Coordinates,
    pi: SVector<f64, 20>,
    sqrt_pi: SVector<f64, 20>,
    fitness: SVector<f64, 20>,
    S: Matrix20,
}

impl SharedModel {
    fn new(Mu: &Matrix20, rate_scaling: f64, prior: &GaussianPrior) -> SharedModel {
        SharedModel {
            basis: basis(),
            log_mutation_equilibrium: Mu.diagonal().map(f64::ln),
            scaled_Mu: Mu * rate_scaling,
            prior_mean: prior.mean,
            prior_cholesky_inverse: prior
                .cholesky
                .solve_lower_triangular(&Cholesky::identity())
                .expect("the Cholesky factor of the prior is singular"),
            prior_log_normalization: 0.5 * prior.log_det() + 0.5 * DIM as f64 * LOG_2PI,
        }
    }

    /// log_pi (normalized) and the site frequencies for the coordinates x
    fn pi(&self, x: &Coordinates) -> (SVector<f64, 20>, SVector<f64, 20>) {
        let log_pi = self.basis.transpose() * x;
        let max_log_pi = log_pi.max();
        let unnormalized_pi = log_pi.map(|l| (l - max_log_pi).exp());
        let total = unnormalized_pi.sum();
        (log_pi.map(|l| l - max_log_pi - total.ln()), unnormalized_pi / total)
    }

    /// Log prior, S and sqrt_pi for the coordinates x
    fn evaluate(&self, x: &Coordinates) -> SiteModelEval {
        let z = self.prior_cholesky_inverse * (x - self.prior_mean);
        let log_prior = -0.5 * z.norm_squared() - self.prior_log_normalization;
        let grad_log_prior = -(self.prior_cholesky_inverse.transpose() * z);

        let (log_pi, pi) = self.pi(x);
        let sqrt_pi = pi.map(f64::sqrt);
        let fitness = log_pi - self.log_mutation_equilibrium;

        // S_ij = sqrt(pi_i / pi_j) * Mu_ij * g(f_j - f_i), only the upper triangle is read by phylo_grad
        let mut S = Matrix20::zeros();
        for i in 0..20 {
            for j in (i + 1)..20 {
                S[(i, j)] = self.scaled_Mu[(i, j)] * (0.5 * (log_pi[i] - log_pi[j])).exp() * fixation(fitness[j] - fitness[i]);
            }
        }

        SiteModelEval { log_prior, grad_log_prior, pi, sqrt_pi, fitness, S }
    }

    /// Gradient of log_likelihood + log_prior w.r.t. x, given the gradients of the log likelihood
    /// w.r.t. the upper triangle of S and w.r.t. sqrt_pi.
    fn backpropagate(&self, eval: &SiteModelEval, grad_S: &Matrix20, grad_sqrt_pi: &SVector<f64, 20>) -> Coordinates {
        let mut grad_log_pi = SVector::<f64, 20>::zeros();
        for i in 0..20 {
            for j in (i + 1)..20 {
                let w = grad_S[(i, j)] * eval.S[(i, j)] * (0.5 - dx_log_fixation(eval.fitness[j] - eval.fitness[i]));
                grad_log_pi[i] += w;
                grad_log_pi[j] -= w;
            }
        }
        // d sqrt_pi_a / d log_pi_b = sqrt_pi_a * (delta_ab - pi_b) / 2
        let weighted = grad_sqrt_pi.component_mul(&eval.sqrt_pi);
        grad_log_pi += (weighted - eval.pi * weighted.sum()) * 0.5;

        eval.grad_log_prior + self.basis * grad_log_pi
    }

    /// Substitution rate sum_{i != j} pi_i Mu_ij g(f_j - f_i) at the site frequencies of eval with
    /// scaled_Mu as Mu, and its gradients w.r.t. x and the off-diagonal Mu entries
    fn rate(&self, eval: &SiteModelEval) -> (f64, Coordinates, Matrix20) {
        let (pi, fitness) = (&eval.pi, &eval.fitness);
        let mut row_rates = SVector::<f64, 20>::zeros();
        let mut grad_Mu = Matrix20::zeros();
        let mut grad_log_pi = SVector::<f64, 20>::zeros();
        for i in 0..20 {
            for j in 0..20 {
                if i == j {
                    continue;
                }
                let d = fitness[j] - fitness[i];
                let g = fixation(d);
                row_rates[i] += self.scaled_Mu[(i, j)] * g;
                grad_Mu[(i, j)] = pi[i] * g;
                let t = pi[i] * self.scaled_Mu[(i, j)] * g * dx_log_fixation(d);
                grad_log_pi[j] += t;
                grad_log_pi[i] -= t;
            }
        }
        let rate = pi.dot(&row_rates);
        // Through pi = softmax(log_pi)
        grad_log_pi += pi.component_mul(&row_rates) - pi * rate;
        (rate, self.basis * grad_log_pi, grad_Mu)
    }
}

/// Fixation factor g(x) = x / (1 - exp(-x)), see model::GOp
fn fixation(x: f64) -> f64 {
    if x.abs() < 1e-6 { 1.0 + x / 2.0 + x * x / 12.0 } else { x / (-(-x).exp_m1()) }
}

/// d/dx log g(x) = 1/x - 1/(exp(x) - 1)
fn dx_log_fixation(x: f64) -> f64 {
    // The closed form loses about eps / |x| to cancellation, so small arguments use the Bernoulli series
    if x.abs() < 0.1 {
        let x2 = x * x;
        0.5 - x * (1.0 / 12.0 - x2 * (1.0 / 720.0 - x2 * (1.0 / 30240.0 - x2 * (1.0 / 1209600.0 - x2 / 47900160.0))))
    } else {
        1.0 / x - 1.0 / x.exp_m1()
    }
}

fn Mu_matrix(Mu: &Tensor) -> Result<Matrix20, candle_core::Error> {
    Ok(Matrix20::from_row_slice(&Mu.flatten_all()?.to_vec1::<f64>()?))
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

/// Unnormalized log posterior of the coordinates of a single alignment column with fixed Mu,
/// branch lengths and rate normalization.
struct SiteLogp<'a> {
    shared: &'a SharedModel,
    tree: &'a FelsensteinTree<20>,
    leaf_pl: Vec<SVector<f64, 20>>,
    pl_buffer: Vec<SVector<f64, 20>>,
}

impl<'a> SiteLogp<'a> {
    fn new(shared: &'a SharedModel, tree: &'a FelsensteinTree<20>, alignment: &[u8], site: usize, num_leaves: usize) -> SiteLogp<'a> {
        SiteLogp {
            shared,
            tree,
            leaf_pl: leaf_partial_likelihoods(alignment, site, num_leaves),
            pl_buffer: vec![SVector::<f64, 20>::zeros(); tree.num_nodes()],
        }
    }

    /// Log posterior (up to a constant) and its gradient, None if either is not finite
    fn logp(&mut self, x: &Coordinates) -> Option<(f64, Coordinates)> {
        let eval = self.shared.evaluate(x);
        let num_leaves = self.leaf_pl.len();
        self.pl_buffer[..num_leaves].copy_from_slice(&self.leaf_pl);
        let result = self.tree.calculate_gradients_single_side(eval.S.as_view(), eval.sqrt_pi.as_view(), &mut self.pl_buffer);
        let logp = result.log_likelihood + eval.log_prior;
        let grad_x = self.shared.backpropagate(&eval, &result.grad_s, &result.grad_sqrt_pi);
        if !logp.is_finite() || grad_x.iter().any(|g| !g.is_finite()) {
            return None;
        }
        Some((logp, grad_x))
    }
}

/// Maximizes f with L-BFGS and a backtracking (Armijo) line search, starting at init.
/// f returns the value and gradient, or None outside of its domain.
/// Returns the final point and whether it converged.
fn maximize_lbfgs(mut f: impl FnMut(&Coordinates) -> Option<(f64, Coordinates)>, init: &Coordinates, max_iterations: usize) -> (Coordinates, bool) {
    const HISTORY: usize = 10;
    const GRADIENT_TOLERANCE: f64 = 1e-6;

    // Minimizes -f
    let mut eval = |x: &Coordinates| f(x).map(|(value, grad)| (-value, -grad));
    let mut x = *init;
    let Some((mut value, mut grad)) = eval(&x) else {
        return (x, false);
    };
    // (s, y, 1 / y^T s) of the last iterations
    let mut history: VecDeque<(Coordinates, Coordinates, f64)> = VecDeque::with_capacity(HISTORY);
    for _ in 0..max_iterations {
        if grad.amax() < GRADIENT_TOLERANCE {
            return (x, true);
        }

        // Two loop recursion for the approximate inverse Hessian times the gradient
        let mut q = grad;
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let alpha = rho * s.dot(&q);
            q -= y * alpha;
            alphas.push(alpha);
        }
        let gamma = match history.back() {
            Some((s, y, _)) => s.dot(y) / y.dot(y),
            // The first step has at most unit length
            None => 1.0 / grad.norm().max(1.0),
        };
        let mut r = q * gamma;
        for ((s, y, rho), alpha) in history.iter().zip(alphas.iter().rev()) {
            let beta = rho * y.dot(&r);
            r += s * (alpha - beta);
        }
        let mut direction = -r;
        if direction.dot(&grad) >= 0.0 {
            // Not a descent direction, restart with steepest descent
            history.clear();
            direction = -grad / grad.norm().max(1.0);
        }

        let slope = grad.dot(&direction);
        let mut step = 1.0;
        let (new_x, new_value, new_grad) = loop {
            let candidate = x + direction * step;
            if let Some((candidate_value, candidate_grad)) = eval(&candidate) {
                if candidate_value <= value + 1e-4 * step * slope {
                    break (candidate, candidate_value, candidate_grad);
                }
            }
            step *= 0.5;
            if step < 1e-12 {
                return (x, false);
            }
        };

        let s = new_x - x;
        let y = new_grad - grad;
        let sy = s.dot(&y);
        if sy > 1e-12 {
            if history.len() == HISTORY {
                history.pop_front();
            }
            history.push_back((s, y, 1.0 / sy));
        }
        let converged = (value - new_value).abs() <= 1e-12 * value.abs().max(1.0);
        x = new_x;
        value = new_value;
        grad = new_grad;
        if converged {
            return (x, true);
        }
    }
    (x, grad.amax() < GRADIENT_TOLERANCE)
}

/// Hessian of f at x from central differences of its gradient, symmetrized
fn hessian(f: &mut impl FnMut(&Coordinates) -> Option<(f64, Coordinates)>, x: &Coordinates) -> Option<Cholesky> {
    const H: f64 = 1e-4;
    let mut hessian = Cholesky::zeros();
    for i in 0..DIM {
        let mut forward = *x;
        forward[i] += H;
        let mut backward = *x;
        backward[i] -= H;
        let (_, grad_forward) = f(&forward)?;
        let (_, grad_backward) = f(&backward)?;
        hessian.set_column(i, &((grad_forward - grad_backward) / (2.0 * H)));
    }
    Some((hessian + hessian.transpose()) * 0.5)
}

/// Full-rank Laplace approximation of a log density f: the mode and the Cholesky factor of the
/// inverse of the negative Hessian at the mode, or the fallback if it is not positive definite.
/// Returns the mode, the Cholesky factor, whether the fallback was used and whether L-BFGS converged.
fn laplace(
    mut f: impl FnMut(&Coordinates) -> Option<(f64, Coordinates)>,
    init: &Coordinates,
    max_iterations: usize,
    fallback: &Cholesky,
) -> (Coordinates, Cholesky, bool, bool) {
    let (mode, converged) = maximize_lbfgs(&mut f, init, max_iterations);
    let cholesky = hessian(&mut f, &mode)
        .and_then(|h| (-h).cholesky())
        .and_then(|precision| precision.inverse().cholesky().map(|c| c.l()))
        .filter(|l| l.diagonal().iter().all(|d| d.is_finite() && *d >= MIN_SCALE && *d <= MAX_SCALE));
    match cholesky {
        Some(l) => (mode, l, false, converged),
        None => (mode, *fallback, true, converged),
    }
}

/// One alignment with its tree, the leaf partial likelihoods bound
struct AlignmentProblem {
    parents: Vec<i32>,
    /// [num_sites * num_leaves], the states of a site consecutive, >= 20 for gaps and unknown
    alignment: Vec<u8>,
    num_sites: usize,
    num_leaves: usize,
    /// Initial branch lengths [num_nodes]
    branch_lengths: Vec<f64>,
    felsenstein: Mutex<FelsensteinTree<20>>,
}

impl AlignmentProblem {
    fn new(parents: &[i32], branch_lengths: &[f64], alignment: &[u8], num_sites: usize, num_leaves: usize) -> AlignmentProblem {
        let felsenstein = crate::create_felsenstein_tree(parents, branch_lengths, alignment, num_sites, num_leaves);
        AlignmentProblem {
            parents: parents.to_vec(),
            alignment: alignment.to_vec(),
            num_sites,
            num_leaves,
            branch_lengths: branch_lengths.to_vec(),
            felsenstein: Mutex::new(felsenstein),
        }
    }

    fn tree(&self, branch_lengths: &[f64]) -> FelsensteinTree<20> {
        FelsensteinTree::<20>::new(&self.parents, branch_lengths)
    }
}

/// Gaussian q of every site of the alignment, x = mean + L eps with a lower triangular L, plus the
/// branch lengths and log_R (of optimization::Mu). The same struct holds the gradient of the objective.
#[derive(Debug, Clone)]
struct AlignmentParameters {
    mean: Vec<Coordinates>,
    /// Log of the diagonal of L
    log_scale: Vec<Coordinates>,
    /// Strictly lower triangle of L, the diagonal and upper triangle stay zero
    lower: Vec<Cholesky>,
    /// [i, j] of the tensor at (i, j)
    log_R: Matrix20,
    log_branch_lengths: Vec<f64>,
}

impl AlignmentParameters {
    fn new(mean: &[Coordinates], cholesky: &[Cholesky], log_R: &Tensor, log_branch_lengths: &[f64]) -> Result<Self, candle_core::Error> {
        Ok(AlignmentParameters {
            mean: mean.to_vec(),
            log_scale: cholesky.iter().map(|l| l.diagonal().map(f64::ln)).collect(),
            lower: cholesky.iter().map(|l| l.lower_triangle() - Cholesky::from_diagonal(&l.diagonal())).collect(),
            log_R: Matrix20::from_row_slice(&log_R.flatten_all()?.to_vec1::<f64>()?),
            log_branch_lengths: log_branch_lengths.to_vec(),
        })
    }

    fn zeros_like(&self) -> Self {
        AlignmentParameters {
            mean: vec![Coordinates::zeros(); self.mean.len()],
            log_scale: vec![Coordinates::zeros(); self.mean.len()],
            lower: vec![Cholesky::zeros(); self.mean.len()],
            log_R: Matrix20::zeros(),
            log_branch_lengths: vec![0.0; self.log_branch_lengths.len()],
        }
    }

    fn num_sites(&self) -> usize {
        self.mean.len()
    }

    fn cholesky(&self, site: usize) -> Cholesky {
        let mut l = self.lower[site];
        l.set_diagonal(&self.log_scale[site].map(f64::exp));
        l
    }

    fn log_R_tensor(&self) -> Result<Tensor, candle_core::Error> {
        Tensor::from_vec(self.log_R.transpose().as_slice().to_vec(), &[20, 20], &candle_core::Device::Cpu)
    }

    fn Mu(&self) -> Result<Tensor, candle_core::Error> {
        Ok(Mu(&self.log_R_tensor()?).detach())
    }

    fn branch_lengths(&self) -> Vec<f64> {
        self.log_branch_lengths.iter().map(|b| b.exp()).collect()
    }

    fn tree_length(&self) -> f64 {
        self.branch_lengths().iter().sum()
    }

    /// Number of entries of the q part at the start of flatten
    fn num_q(&self) -> usize {
        self.num_sites() * (2 * DIM + DIM * (DIM - 1) / 2)
    }

    /// q (means, log scales, strictly lower triangles), then log_R and the log branch lengths
    fn flatten(&self) -> Vec<f64> {
        let mut flat = Vec::with_capacity(self.num_q() + 400 + self.log_branch_lengths.len());
        flat.extend(self.mean.iter().flat_map(|m| m.iter().copied()));
        flat.extend(self.log_scale.iter().flat_map(|s| s.iter().copied()));
        for l in &self.lower {
            for i in 0..DIM {
                for j in 0..i {
                    flat.push(l[(i, j)]);
                }
            }
        }
        flat.extend(self.log_R.iter().copied());
        flat.extend(self.log_branch_lengths.iter().copied());
        flat
    }

    fn unflatten(&mut self, flat: &[f64]) {
        let mut values = flat.iter().copied();
        for m in self.mean.iter_mut().chain(self.log_scale.iter_mut()) {
            for v in m.iter_mut() {
                *v = values.next().unwrap();
            }
        }
        for l in &mut self.lower {
            for i in 0..DIM {
                for j in 0..i {
                    l[(i, j)] = values.next().unwrap();
                }
            }
        }
        for v in self.log_R.iter_mut().chain(self.log_branch_lengths.iter_mut()) {
            *v = values.next().unwrap();
        }
        assert!(values.next().is_none());
    }
}

/// One draw of one site in evaluate_alignment
struct DrawEval {
    /// log p(alignment column, x) with the rate normalization of the draw
    log_joint: f64,
    grad_x: Coordinates,
    /// Derivative of the log likelihood w.r.t. the rate normalization factor of the draw
    grad_normalization: f64,
    grad_rate_x: Coordinates,
    /// Gradient of the log likelihood w.r.t. Mu
    grad_Mu: Matrix20,
    grad_branch_lengths: Vec<f64>,
}

/// The Mu penalty: Mu_reg times the squared log difference of the off-diagonal entries to init_log_R
struct MuSettings {
    init_log_R: Tensor,
    Mu_reg: f64,
}

/// The objective and its gradient w.r.t. q, log_R and the branch lengths for the standard normal
/// draws eps [draw][site]: with importance_weighted the bound sum_sites log mean_k w_k with
/// log w_k = log p(column, x_k) - log q(x_k), whose gradient is sum_k w~_k grad log w_k with the
/// normalized weights w~, otherwise the ELBO sum_sites mean_k log p(column, x_k) + entropy; both
/// minus the Mu penalty. The rates of every draw are normalized to an average of 1 over the sites
/// (c = 1 / mean rate), which couples the sites; its gradient goes through all x of the draw.
fn evaluate_alignment(
    problem: &AlignmentProblem,
    params: &AlignmentParameters,
    prior: &GaussianPrior,
    Mu_settings: &MuSettings,
    eps: &[Vec<Coordinates>],
    importance_weighted: bool,
) -> Result<(f64, AlignmentParameters), candle_core::Error> {
    let num_sites = params.num_sites();
    let num_draws = eps.len();
    let shared = SharedModel::new(&Mu_matrix(&params.Mu()?)?, 1.0, prior);
    let choleskys: Vec<Cholesky> = (0..num_sites).map(|site| params.cholesky(site)).collect();
    let branch_lengths = params.branch_lengths();

    // draws[k][site]
    let mut draws: Vec<Vec<DrawEval>> = Vec::with_capacity(num_draws);
    let mut mean_rates = Vec::with_capacity(num_draws);
    // Gradient of the summed rates of every draw w.r.t. Mu
    let mut grad_rate_Mu = Vec::with_capacity(num_draws);
    for eps_k in eps {
        let mut evals: Vec<(SiteModelEval, f64, Coordinates, Matrix20)> = (0..num_sites)
            .into_par_iter()
            .map(|site| {
                let eval = shared.evaluate(&(params.mean[site] + choleskys[site] * eps_k[site]));
                let (rate, grad_rate_x, grad_rate_Mu) = shared.rate(&eval);
                (eval, rate, grad_rate_x, grad_rate_Mu)
            })
            .collect();
        let mean_rate = evals.iter().map(|e| e.1).sum::<f64>() / num_sites as f64;
        grad_rate_Mu.push(evals.iter().map(|e| e.3).sum::<Matrix20>());
        let normalization = 1.0 / mean_rate;
        let raw_S: Vec<Matrix20> = evals.iter().map(|e| e.0.S).collect();
        for e in evals.iter_mut() {
            e.0.S *= normalization;
        }
        let S: Vec<Matrix20> = evals.iter().map(|e| e.0.S).collect();
        let sqrt_pi: Vec<SVector<f64, 20>> = evals.iter().map(|e| e.0.sqrt_pi).collect();
        let result = problem.felsenstein.lock().unwrap().calculate_gradients_with_branch_lengths(&S, &sqrt_pi, &branch_lengths);

        let draws_k: Vec<DrawEval> = evals
            .into_par_iter()
            .zip(raw_S.into_par_iter())
            .zip(result.log_likelihood.into_par_iter())
            .zip(result.grad_s.into_par_iter())
            .zip(result.grad_sqrt_pi.into_par_iter())
            .zip(result.grad_tree.into_par_iter())
            .map(|(((((e, raw_S), log_likelihood), grad_s), grad_sqrt_pi), grad_tree)| {
                let (eval, _, grad_rate_x, _) = e;
                // phylo_grad only reads and differentiates the upper triangle
                let grad_s = grad_s.upper_triangle() - Matrix20::from_diagonal(&grad_s.diagonal());
                DrawEval {
                    log_joint: log_likelihood + eval.log_prior,
                    grad_x: shared.backpropagate(&eval, &grad_s, &grad_sqrt_pi),
                    grad_normalization: grad_s.component_mul(&raw_S).sum(),
                    grad_rate_x,
                    grad_Mu: grad_s.component_mul(&eval.S).component_div(&shared.scaled_Mu.map(|m| if m > 0.0 { m } else { 1.0 })),
                    grad_branch_lengths: grad_tree,
                }
            })
            .collect();
        mean_rates.push(mean_rate);
        draws.push(draws_k);
    }

    // Normalized weights [site][k] and the bound per site
    let (weights, bounds): (Vec<Vec<f64>>, Vec<f64>) = (0..num_sites)
        .map(|site| {
            if importance_weighted {
                let log_weights: Vec<f64> = (0..num_draws).map(|k| draws[k][site].log_joint + 0.5 * eps[k][site].norm_squared()).collect();
                let max = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let unnormalized: Vec<f64> = log_weights.iter().map(|w| (w - max).exp()).collect();
                let total: f64 = unnormalized.iter().sum();
                (unnormalized.iter().map(|w| w / total).collect(), max + (total / num_draws as f64).ln())
            } else {
                (
                    vec![1.0 / num_draws as f64; num_draws],
                    (0..num_draws).map(|k| draws[k][site].log_joint).sum::<f64>() / num_draws as f64,
                )
            }
        })
        .unzip();
    let bound: f64 = bounds.iter().sum();
    if !bound.is_finite() {
        candle_core::bail!("MUTSEL VI: non-finite objective");
    }

    // Gradient w.r.t. the normalization factor of every draw, through the mean rate into all x of the draw:
    // d c / d rate_site = -c^2 / num_sites
    let normalization_coefficients: Vec<f64> = (0..num_draws)
        .map(|k| {
            let grad_normalization: f64 = (0..num_sites).map(|site| weights[site][k] * draws[k][site].grad_normalization).sum();
            -grad_normalization / (mean_rates[k] * mean_rates[k] * num_sites as f64)
        })
        .collect();

    let mut grad = params.zeros_like();
    let per_site: Vec<(Coordinates, Coordinates, Cholesky)> = (0..num_sites)
        .into_par_iter()
        .map(|site| {
            let mut grad_mean = Coordinates::zeros();
            // The entropy, sum log diag(L)
            let mut grad_log_scale = Coordinates::from_element(1.0);
            let mut grad_lower = Cholesky::zeros();
            for k in 0..num_draws {
                let d = &draws[k][site];
                let g = d.grad_x * weights[site][k] + d.grad_rate_x * normalization_coefficients[k];
                grad_mean += g;
                grad_log_scale += g.component_mul(&eps[k][site]).component_mul(&choleskys[site].diagonal());
                grad_lower += g * eps[k][site].transpose();
            }
            (grad_mean, grad_log_scale, grad_lower.lower_triangle() - Cholesky::from_diagonal(&grad_lower.diagonal()))
        })
        .collect();
    for (site, (grad_mean, grad_log_scale, grad_lower)) in per_site.into_iter().enumerate() {
        grad.mean[site] = grad_mean;
        grad.log_scale[site] = grad_log_scale;
        grad.lower[site] = grad_lower;
    }

    // Branch lengths and Mu
    let num_branches = branch_lengths.len();
    let zero = || (Matrix20::zeros(), vec![0.0; num_branches]);
    let (mut grad_Mu, grad_branch_lengths) = (0..num_sites)
        .into_par_iter()
        .fold(zero, |(mut grad_Mu, mut grad_branch_lengths), site| {
            for k in 0..num_draws {
                let d = &draws[k][site];
                let w = weights[site][k];
                grad_Mu += d.grad_Mu * w;
                for (total, g) in grad_branch_lengths.iter_mut().zip(&d.grad_branch_lengths) {
                    *total += w * g;
                }
            }
            (grad_Mu, grad_branch_lengths)
        })
        .reduce(zero, |(a_Mu, mut a_branches), (b_Mu, b_branches)| {
            a_branches.iter_mut().zip(&b_branches).for_each(|(a, b)| *a += b);
            (a_Mu + b_Mu, a_branches)
        });
    grad.log_branch_lengths = grad_branch_lengths.iter().zip(&branch_lengths).map(|(g, b)| g * b).collect();

    // log_R through optimization::Mu, together with the Mu penalty
    for k in 0..num_draws {
        grad_Mu += grad_rate_Mu[k] * normalization_coefficients[k];
    }
    let log_R = Var::from_tensor(&params.log_R_tensor()?)?;
    let penalty = Mu_penalty(&log_R, &Mu_settings.init_log_R, Mu_settings.Mu_reg);
    let grad_Mu = Tensor::from_vec(grad_Mu.transpose().as_slice().to_vec(), &[20, 20], &candle_core::Device::Cpu)?;
    let surrogate = (Mu(&log_R).mul(&grad_Mu)?.sum_all()? - &penalty)?;
    let grad_log_R = match surrogate.backward()?.get(log_R.as_tensor()) {
        Some(g) => g.flatten_all()?.to_vec1::<f64>()?,
        None => vec![0.0; 400],
    };
    // Entries at log R = -inf (unused) have no gradient
    grad.log_R = Matrix20::from_row_slice(&grad_log_R).map(|g| if g.is_finite() { g } else { 0.0 });

    let log_det: f64 = params.log_scale.iter().map(|s| s.sum()).sum();
    let dims = (num_sites * DIM) as f64;
    let log_normalization = 0.5 * LOG_2PI * dims;
    // The ELBO has the expectation of |eps|^2 / 2 in the entropy, the importance weighted bound has it in the weights
    let constant = if importance_weighted { log_normalization } else { log_normalization + 0.5 * dims };
    Ok((bound + log_det + constant - penalty.to_scalar::<f64>()?, grad))
}

/// Adam ascent on a flat parameter vector
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
}

impl Adam {
    fn new(len: usize) -> Adam {
        Adam { m: vec![0.0; len], v: vec![0.0; len], t: 0 }
    }

    fn ascend(&mut self, x: &mut [f64], grad: &[f64], lr: f64) {
        const BETA1: f64 = 0.9;
        const BETA2: f64 = 0.999;
        const EPS: f64 = 1e-8;
        self.t += 1;
        let (c1, c2) = (1.0 - BETA1.powi(self.t), 1.0 - BETA2.powi(self.t));
        for i in 0..x.len() {
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * grad[i];
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * grad[i] * grad[i];
            x[i] += lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + EPS);
        }
    }
}

/// Standard normal draws [2 * pairs][site] in antithetic pairs (eps, -eps)
fn antithetic_draws(num_sites: usize, pairs: usize, rng: &mut impl Rng) -> Vec<Vec<Coordinates>> {
    let mut eps = Vec::with_capacity(2 * pairs);
    for _ in 0..pairs {
        let e: Vec<Coordinates> = (0..num_sites).map(|_| Coordinates::from_fn(|_, _| rng.sample(StandardNormal))).collect();
        let negated = e.iter().map(|e| -e).collect();
        eps.push(e);
        eps.push(negated);
    }
    eps
}

/// Maximizes the importance weighted ELBO over q, log_R and the branch lengths with Adam:
/// settings.steps steps, a constant learning rate for the first quarter, then a geometric decay to
/// lr_final; only q during the warm-up; the parameters averaged over the last quarter of the steps.
/// Returns the objective estimate of every step.
fn maximize(
    problem: &AlignmentProblem,
    params: &mut AlignmentParameters,
    prior: &GaussianPrior,
    Mu_settings: &MuSettings,
    settings: &SviSettings,
    verbosity: Verbosity,
) -> Result<Vec<f64>, candle_core::Error> {
    let steps = settings.steps;
    let mut flat = params.flatten();
    let num_q = params.num_q();
    let mut adam_q = Adam::new(num_q);
    let mut adam_globals = Adam::new(flat.len() - num_q);
    let warmup_steps = (settings.warmup_fraction * steps as f64) as usize;
    let average_from = steps - steps / 4;
    let mut sum = vec![0.0; flat.len()];
    let mut rng = ChaCha8Rng::seed_from_u64(settings.seed);

    let mut trace = Vec::with_capacity(steps);
    let report_every = (steps / 20).max(1);
    let start = std::time::Instant::now();
    for step in 0..steps {
        let progress = step as f64 / steps as f64;
        let lr = settings.lr * (settings.lr_final / settings.lr).powf(((progress - 0.25) / 0.75).max(0.0));

        let eps = antithetic_draws(problem.num_sites, settings.pairs, &mut rng);
        let (value, grad) = evaluate_alignment(problem, params, prior, Mu_settings, &eps, true)?;
        trace.push(value);
        let flat_grad = grad.flatten();
        if step >= warmup_steps {
            adam_globals.ascend(&mut flat[num_q..], &flat_grad[num_q..], lr);
        }
        adam_q.ascend(&mut flat[..num_q], &flat_grad[..num_q], lr);
        params.unflatten(&flat);
        if step >= average_from {
            sum.iter_mut().zip(&flat).for_each(|(s, x)| *s += x);
        }

        if verbosity.should_print(Verbosity::Min) && ((step + 1) % report_every == 0 || step + 1 == steps) {
            let window = &trace[trace.len().saturating_sub(report_every)..];
            let mean_value = window.iter().sum::<f64>() / window.len() as f64;
            println!(
                "MUTSEL VI step {}/{}: mean objective of the last {} steps {:.3} ({:.5} per site), tree length {:.4}, learning rate {:.5} ({:.1}s)",
                step + 1,
                steps,
                window.len(),
                mean_value,
                mean_value / problem.num_sites as f64,
                params.tree_length(),
                lr,
                start.elapsed().as_secs_f64()
            );
        }
    }

    let num_averaged = (steps - average_from) as f64;
    if num_averaged > 0.0 {
        let average: Vec<f64> = sum.iter().map(|s| s / num_averaged).collect();
        params.unflatten(&average);
    }
    Ok(trace)
}

/// PMSF site frequencies of every site with the branch lengths fitted to them (Mu fixed at
/// init_log_R). Returns their coordinates and the log branch lengths.
fn pmsf_initialization(
    problem: &AlignmentProblem,
    init_log_R: &Tensor,
    verbosity: Verbosity,
    out_prefix: &str,
) -> Result<(Vec<Coordinates>, Vec<f64>), candle_core::Error> {
    let felsenstein = crate::create_felsenstein_tree(&problem.parents, &problem.branch_lengths, &problem.alignment, problem.num_sites, problem.num_leaves);
    let op = FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));
    let init_Mu = Mu(init_log_R).detach();
    let (site_freq, log_branch_lengths) = two_step_light_pmsf(
        op,
        crate::data::UDM256,
        crate::data::UDM256_WEIGHTS,
        &Tensor::from_slice(&problem.branch_lengths, &[problem.branch_lengths.len()], &candle_core::Device::Cpu)?.log()?,
        &init_Mu,
        verbosity,
        out_prefix,
    );
    let coordinates: Vec<Coordinates> = site_freq
        .log()?
        .to_vec2::<f64>()?
        .iter()
        .map(|row| log_freq_to_coordinates(&SVector::<f64, 20>::from_column_slice(row)))
        .collect();
    Ok((coordinates, log_branch_lengths.to_vec1::<f64>()?))
}

/// MAP estimate of the coordinates of every site jointly with Mu and the branch lengths, starting
/// from the PMSF estimate: the deterministic objective sum_sites log p(column, x) with the rate
/// normalization minus the Mu penalty (evaluate_alignment at the means, eps = 0), maximized with
/// Adam at learning rate lr until the relative improvement stays below 1e-6 for 5 iterations, at
/// most max_iterations. Returns (coordinates, log branch lengths, log_R, average rate at the MAP).
fn map_stage(
    problem: &AlignmentProblem,
    prior: &GaussianPrior,
    Mu_settings: &MuSettings,
    pmsf: &(Vec<Coordinates>, Vec<f64>),
    max_iterations: usize,
    lr: f64,
    verbosity: Verbosity,
) -> Result<(Vec<Coordinates>, Vec<f64>, Tensor, f64), candle_core::Error> {
    let (pmsf_coordinates, log_branch_lengths) = pmsf;
    let num_sites = pmsf_coordinates.len();
    let identity = vec![Cholesky::identity() * 0.1; num_sites];
    let mut params = AlignmentParameters::new(pmsf_coordinates, &identity, &Mu_settings.init_log_R, log_branch_lengths)?;
    let mut flat = params.flatten();
    let num_q = params.num_q();
    let num_means = num_sites * DIM;
    let mut adam = Adam::new(flat.len());
    let eps = vec![vec![Coordinates::zeros(); num_sites]];
    let start = std::time::Instant::now();
    let (mut best_value, mut best_flat) = (f64::NEG_INFINITY, flat.clone());
    let mut no_improvement = 0;
    let mut iterations = 0;
    for iteration in 0..max_iterations {
        iterations = iteration + 1;
        let (value, grad) = evaluate_alignment(problem, &params, prior, Mu_settings, &eps, false)?;
        let mut flat_grad = grad.flatten();
        // Only the means and the globals move
        flat_grad[num_means..num_q].iter_mut().for_each(|g| *g = 0.0);
        let improvement = (value - best_value) / value.abs().max(1e-12);
        if value > best_value {
            best_value = value;
            best_flat = flat.clone();
        }
        if iteration >= 10 {
            no_improvement = if improvement > 1e-6 { 0 } else { no_improvement + 1 };
            if no_improvement >= 5 {
                break;
            }
        }
        adam.ascend(&mut flat, &flat_grad, lr);
        params.unflatten(&flat);
    }
    params.unflatten(&best_flat);
    let raw = SharedModel::new(&Mu_matrix(&params.Mu()?)?, 1.0, prior);
    let average_rate = params.mean.iter().map(|x| raw.rate(&raw.evaluate(x)).0).sum::<f64>() / num_sites as f64;
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL VI: MAP in {} iterations ({:.1}s), log joint {:.3} per site, tree length {:.4}",
            iterations,
            start.elapsed().as_secs_f64(),
            best_value / num_sites as f64,
            params.tree_length()
        );
    }
    Ok((params.mean.clone(), params.log_branch_lengths.clone(), params.log_R_tensor()?, average_rate))
}

/// Initial q of every site: the full-rank Laplace approximation of its posterior at Mu(log_R), the
/// branch lengths and the rate normalization 1 / average_rate, starting at init
fn laplace_initialization(
    problem: &AlignmentProblem,
    prior: &GaussianPrior,
    log_R: &Tensor,
    init: &[Coordinates],
    log_branch_lengths: &[f64],
    average_rate: f64,
    settings: &SviSettings,
    verbosity: Verbosity,
) -> Result<AlignmentParameters, candle_core::Error> {
    let start = std::time::Instant::now();
    let shared = SharedModel::new(&Mu_matrix(&Mu(log_R).detach())?, 1.0 / average_rate, prior);
    let tree = problem.tree(&log_branch_lengths.iter().map(|b| b.exp()).collect::<Vec<_>>());
    let results: Vec<(Coordinates, Cholesky, bool, bool)> = init
        .par_iter()
        .enumerate()
        .map(|(site, init)| {
            let mut logp = SiteLogp::new(&shared, &tree, &problem.alignment, site, problem.num_leaves);
            laplace(|x| logp.logp(x), init, settings.laplace_iterations, &prior.cholesky)
        })
        .collect();
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL VI: Laplace initialization in {:.1}s, L-BFGS did not converge in {} of {} sites, {} sites without positive definite curvature got the prior covariance",
            start.elapsed().as_secs_f64(),
            results.iter().filter(|r| !r.3).count(),
            problem.num_sites,
            results.iter().filter(|r| r.2).count()
        );
    }
    let mean: Vec<Coordinates> = results.iter().map(|r| r.0).collect();
    let cholesky: Vec<Cholesky> = results.iter().map(|r| r.1).collect();
    AlignmentParameters::new(&mean, &cholesky, log_R, log_branch_lengths)
}

fn site_seed(seed: u64, site: usize) -> u64 {
    seed ^ (site as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Rate normalization for an average rate of 1 over the sites, in expectation over q
fn expected_rate_scaling(Mu: &Matrix20, prior: &GaussianPrior, params: &AlignmentParameters, seed: u64) -> f64 {
    const RATE_DRAWS: usize = 20;
    let raw = SharedModel::new(Mu, 1.0, prior);
    let num_sites = params.num_sites();
    let total: f64 = (0..num_sites)
        .into_par_iter()
        .map(|site| {
            let mut rng = ChaCha8Rng::seed_from_u64(site_seed(seed, site));
            let l = params.cholesky(site);
            (0..RATE_DRAWS)
                .map(|_| {
                    let eps = Coordinates::from_fn(|_, _| rng.sample(StandardNormal));
                    raw.rate(&raw.evaluate(&(params.mean[site] + l * eps))).0
                })
                .sum::<f64>()
        })
        .sum();
    (num_sites * RATE_DRAWS) as f64 / total
}

/// num_samples tempered samples mean +- scale L eps in antithetic pairs (an odd last one without
/// its partner), centred on the mean of q: scale 0 gives num_samples copies of the mode
fn tempered_samples(mean: &Coordinates, cholesky: &Cholesky, num_samples: usize, scale: f64, rng: &mut impl Rng) -> Vec<Coordinates> {
    let l = cholesky * scale;
    let mut samples = Vec::with_capacity(num_samples);
    while samples.len() < num_samples {
        let eps = Coordinates::from_fn(|_, _| rng.sample(StandardNormal));
        samples.push(mean + l * eps);
        if samples.len() < num_samples {
            samples.push(mean - l * eps);
        }
    }
    samples
}

fn coordinates_tensor(x: &[Coordinates]) -> Result<Tensor, candle_core::Error> {
    Tensor::from_iter(x.iter().flat_map(|m| m.iter().copied()), &candle_core::Device::Cpu)?.reshape(&[x.len(), DIM])
}

fn cholesky_tensor(l: &[Cholesky]) -> Result<Tensor, candle_core::Error> {
    // nalgebra is column-major, the npz array [n, 19, 19] row-major
    Tensor::from_vec(l.iter().flat_map(|l| l.transpose().as_slice().to_vec()).collect(), &[l.len(), DIM, DIM], &candle_core::Device::Cpu)
}

/// Fits q of every site jointly with Mu and the branch lengths under the prior (GaussianPrior::load)
/// and returns (S, sqrt_pi) of [num_sites * num_samples] site models, the num_samples tempered
/// samples of a site consecutive (with num_samples = 1 at the mode of q). The rate normalization is
/// an average substitution rate of 1 over the sites in expectation over q.
pub fn infer_internal(
    parents: &[i32],
    num_samples: usize,
    branch_lengths: &[f64],
    alignment: &[u8],
    num_sites: usize,
    num_leaves: usize,
    mutsel_params: MutselParams,
    prior_R_file: Option<&Path>,
    verbosity: Verbosity,
    out_prefix: &str,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    assert!(num_samples >= 1, "at least one site model per site");
    let settings = SviSettings::from_env();
    let prior = GaussianPrior::load()?;
    println!(
        "MUTSEL VI: full-rank Gaussian per site with a full-rank Gaussian prior on log pi (log det {:.4}, trace {:.4}), jointly with Mu and the branch lengths; MAP, Laplace, {} SVI steps with {} antithetic pairs of draws, learning rate {}; {} site models per site{}; seed {}",
        prior.log_det(),
        prior.covariance().trace(),
        settings.steps,
        settings.pairs,
        settings.lr,
        num_samples,
        if num_samples > 1 { format!(" (tempered samples m +- {} L eps)", settings.sample_scale) } else { " (the mode of q)".to_string() },
        settings.seed
    );
    // Zero length branches (log -inf) could never change; the root has no branch (parent -1)
    let branch_lengths: Vec<f64> = branch_lengths.iter().zip(parents).map(|(b, p)| if *p >= 0 { b.max(1e-6) } else { *b }).collect();
    let problem = AlignmentProblem::new(parents, &branch_lengths, alignment, num_sites, num_leaves);
    // The Mu penalty stays relative to the prior Mu
    let init_log_R = load_init_R(prior_R_file)?.log()?;
    let Mu_settings = MuSettings { init_log_R: init_log_R.clone(), Mu_reg: mutsel_params.Mu_reg };

    // PMSF, MAP and Laplace initialization, then SVI
    let pmsf = pmsf_initialization(&problem, &init_log_R, verbosity, out_prefix)?;
    let (map_x, map_log_branch_lengths, map_log_R, map_rate) = if settings.map_iterations > 0 {
        map_stage(&problem, &prior, &Mu_settings, &pmsf, settings.map_iterations, settings.map_lr, verbosity)?
    } else {
        let raw = SharedModel::new(&Mu_matrix(&Mu(&init_log_R).detach())?, 1.0, &prior);
        let rate = pmsf.0.iter().map(|x| raw.rate(&raw.evaluate(x)).0).sum::<f64>() / num_sites as f64;
        (pmsf.0.clone(), pmsf.1.clone(), init_log_R.clone(), rate)
    };
    let mut params = laplace_initialization(&problem, &prior, &map_log_R, &map_x, &map_log_branch_lengths, map_rate, &settings, verbosity)?;
    let trace = maximize(&problem, &mut params, &prior, &Mu_settings, &settings, verbosity)?;

    let Mu = params.Mu()?;
    let Mu_nalgebra = Mu_matrix(&Mu)?;
    let rate_scaling = expected_rate_scaling(&Mu_nalgebra, &prior, &params, settings.seed.wrapping_add(1));
    let shared = SharedModel::new(&Mu_nalgebra, rate_scaling, &prior);

    // The site models: num_samples tempered samples per site, the mode of q if num_samples = 1
    let samples: Vec<Coordinates> = (0..num_sites)
        .into_par_iter()
        .flat_map_iter(|site| {
            if num_samples == 1 {
                return vec![params.mean[site]];
            }
            let mut rng = ChaCha8Rng::seed_from_u64(site_seed(settings.seed.wrapping_add(3), site));
            tempered_samples(&params.mean[site], &params.cholesky(site), num_samples, settings.sample_scale, &mut rng)
        })
        .collect();
    // IQ-TREE requires frequencies above 1e-10
    let log_pi: Vec<f64> = samples
        .iter()
        .flat_map(|x| {
            let pi = shared.pi(x).1.map(|v| v.max(1e-9));
            (pi / pi.sum()).map(f64::ln).iter().copied().collect::<Vec<_>>()
        })
        .collect();
    let log_pi = Tensor::from_vec(log_pi, &[num_sites * num_samples, 20], &candle_core::Device::Cpu)?;
    let rate_scaling = tensor_full(rate_scaling, &[]);
    let (S, sqrt_pi) = calc_rate_matrix(&Mu, &log_pi, &rate_scaling);
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL VI: tree length {:.4}, average substitution rate of the site models {:.4}",
            params.tree_length(),
            model::substitution_rates_tensor(&S, &sqrt_pi).mean_all()?.to_scalar::<f64>()?
        );
    }

    if verbosity.should_print(Verbosity::Med) {
        let choleskys: Vec<Cholesky> = (0..num_sites).map(|site| params.cholesky(site)).collect();
        let npz_path = format!("{}.vi.npz", out_prefix);
        Tensor::write_npz(
            &[
                ("vi_mean", &coordinates_tensor(&params.mean)?),
                ("vi_cholesky", &cholesky_tensor(&choleskys)?),
                ("map_mean", &coordinates_tensor(&map_x)?),
                ("pmsf_coordinates", &coordinates_tensor(&pmsf.0)?),
                ("Mu", &Mu),
                ("rate_scaling", &rate_scaling),
                ("branch_lengths", &Tensor::from_vec(params.branch_lengths(), &[params.log_branch_lengths.len()], &candle_core::Device::Cpu)?),
                ("trace", &Tensor::from_slice(&trace, &[trace.len()], &candle_core::Device::Cpu)?),
            ],
            Path::new(&npz_path),
        )?;
        println!("MUTSEL VI: q written to {}", npz_path);
    }
    Ok((S, sqrt_pi))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARENTS: [i32; 8] = [5, 5, 6, 6, 7, 7, 7, -1];
    const BRANCH_LENGTHS: [f64; 8] = [0.1, 0.3, 0.2, 0.05, 0.4, 0.15, 0.25, 0.0];

    fn test_Mu() -> Matrix20 {
        let init_R = crate::data::load_lower_R_with_equi(crate::data::M_TXT);
        Mu_matrix(&Mu(&init_R.log().unwrap()).detach()).unwrap()
    }

    fn test_prior() -> GaussianPrior {
        // Some correlated covariance
        let a = Cholesky::from_fn(|i, j| 0.3 * ((i * 7 + j * 3) as f64).sin() + if i == j { 1.0 } else { 0.0 });
        let covariance = a * a.transpose() * 0.5;
        GaussianPrior { mean: Coordinates::from_fn(|i, _| 0.2 * (i as f64).cos()), cholesky: covariance.cholesky().unwrap().l() }
    }

    fn test_point() -> Coordinates {
        Coordinates::from_fn(|i, _| 0.3 * ((i as f64) * 1.7).sin())
    }

    fn Mu_settings() -> MuSettings {
        let init_R = crate::data::load_lower_R_with_equi(crate::data::M_TXT);
        MuSettings { init_log_R: init_R.log().unwrap(), Mu_reg: 1.0 }
    }

    #[test]
    fn basis_is_orthonormal_and_sums_to_zero() {
        let b = basis();
        assert!((b * b.transpose() - Cholesky::identity()).amax() < 1e-12);
        assert!((b * SVector::<f64, 20>::from_element(1.0)).amax() < 1e-12);
    }

    #[test]
    fn coordinates_round_trip_the_frequencies() {
        let pi = SVector::<f64, 20>::from_fn(|i, _| 1.0 + (i as f64 * 0.37).sin().abs());
        let pi = pi / pi.sum();
        let shared = SharedModel::new(&test_Mu(), 1.0, &test_prior());
        let (log_pi, recovered) = shared.pi(&log_freq_to_coordinates(&pi.map(f64::ln)));
        assert!((recovered - pi).amax() < 1e-12);
        assert!((log_pi.map(f64::exp) - pi).amax() < 1e-12);
    }

    #[test]
    fn builtin_prior_is_valid() {
        let prior = GaussianPrior::builtin();
        assert!(prior.mean.iter().all(|v| v.is_finite()));
        assert!(prior.cholesky.diagonal().iter().all(|d| *d > 0.0));
        assert!((prior.cholesky.upper_triangle() - Cholesky::from_diagonal(&prior.cholesky.diagonal())).amax() == 0.0);
    }

    #[test]
    fn prior_density_matches_the_gaussian() {
        let prior = test_prior();
        let shared = SharedModel::new(&test_Mu(), 1.0, &prior);
        let x = test_point();
        let covariance = prior.covariance();
        let d = x - prior.mean;
        let expected = -0.5 * d.dot(&(covariance.try_inverse().unwrap() * d)) - 0.5 * covariance.determinant().ln() - 0.5 * DIM as f64 * LOG_2PI;
        assert!((shared.evaluate(&x).log_prior - expected).abs() < 1e-10);
    }

    #[test]
    fn site_gradient_matches_finite_differences() {
        let shared = SharedModel::new(&test_Mu(), 1.3, &test_prior());
        let tree = FelsensteinTree::<20>::new(&PARENTS, &BRANCH_LENGTHS);
        let alignment = [0u8, 0, 3, 20, 7];
        let mut logp = SiteLogp::new(&shared, &tree, &alignment, 0, 5);
        let x = test_point();
        let (value, grad) = logp.logp(&x).unwrap();
        assert!(value.is_finite());
        for i in 0..DIM {
            let h = 1e-6;
            let mut plus = x;
            plus[i] += h;
            let mut minus = x;
            minus[i] -= h;
            let fd = (logp.logp(&plus).unwrap().0 - logp.logp(&minus).unwrap().0) / (2.0 * h);
            assert!((fd - grad[i]).abs() < 1e-5 * (1.0 + fd.abs()), "component {}: {} vs {}", i, fd, grad[i]);
        }
    }

    #[test]
    fn laplace_of_a_gaussian_is_exact() {
        let prior = test_prior();
        let shared = SharedModel::new(&test_Mu(), 1.0, &prior);
        let f = |x: &Coordinates| {
            let eval = shared.evaluate(x);
            Some((eval.log_prior, eval.grad_log_prior))
        };
        let (mode, l, fallback, converged) = laplace(f, &Coordinates::zeros(), 200, &Cholesky::identity());
        assert!(converged && !fallback);
        assert!((mode - prior.mean).amax() < 1e-5);
        assert!((l * l.transpose() - prior.covariance()).amax() < 1e-5);
    }

    fn small_problem() -> AlignmentProblem {
        let columns: [[u8; 5]; 6] = [[0, 0, 3, 20, 7], [1, 1, 1, 2, 1], [5, 6, 5, 5, 20], [9, 10, 9, 9, 9], [15, 15, 16, 15, 15], [18, 18, 18, 19, 18]];
        let alignment: Vec<u8> = columns.iter().flatten().copied().collect();
        AlignmentProblem::new(&PARENTS, &BRANCH_LENGTHS, &alignment, columns.len(), 5)
    }

    fn small_params(problem: &AlignmentProblem) -> AlignmentParameters {
        let init_R = crate::data::load_lower_R_with_equi(crate::data::M_TXT);
        let mean: Vec<Coordinates> = (0..problem.num_sites).map(|s| Coordinates::from_fn(|i, _| 0.2 * ((i + 3 * s) as f64).sin())).collect();
        let cholesky: Vec<Cholesky> = (0..problem.num_sites)
            .map(|s| Cholesky::from_fn(|i, j| if i == j { 0.3 } else if i > j { 0.05 * ((i * j + s) as f64).cos() } else { 0.0 }))
            .collect();
        let log_bl: Vec<f64> = BRANCH_LENGTHS.iter().map(|b| b.max(0.01).ln()).collect();
        AlignmentParameters::new(&mean, &cholesky, &init_R.log().unwrap(), &log_bl).unwrap()
    }

    /// The objective gradient w.r.t. q, the branch lengths and log_R against central differences
    fn check_objective_gradient(importance_weighted: bool) {
        let problem = small_problem();
        let params = small_params(&problem);
        let prior = test_prior();
        let Mu_settings = Mu_settings();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let eps = antithetic_draws(problem.num_sites, 1, &mut rng);
        let (_, grad) = evaluate_alignment(&problem, &params, &prior, &Mu_settings, &eps, importance_weighted).unwrap();
        let flat = params.flatten();
        let flat_grad = grad.flatten();
        let value_at = |flat: &[f64]| {
            let mut p = params.clone();
            p.unflatten(flat);
            evaluate_alignment(&problem, &p, &prior, &Mu_settings, &eps, importance_weighted).unwrap().0
        };
        // A sample of the parameters: q of the first site, some log scales, some log_R entries, the branch lengths
        let num_q = params.num_q();
        let mut indices: Vec<usize> = (0..DIM).chain(problem.num_sites * DIM..problem.num_sites * DIM + 5).collect();
        indices.extend([num_q + 20 + 3, num_q + 5 * 20 + 2, num_q + 19 * 20 + 18]);
        indices.extend(num_q + 400..flat.len() - 1);
        for &i in &indices {
            let h = 1e-6;
            let mut plus = flat.clone();
            plus[i] += h;
            let mut minus = flat.clone();
            minus[i] -= h;
            let fd = (value_at(&plus) - value_at(&minus)) / (2.0 * h);
            assert!((fd - flat_grad[i]).abs() < 1e-4 * (1.0 + fd.abs()), "parameter {}: {} vs {}", i, fd, flat_grad[i]);
        }
    }

    #[test]
    fn objective_gradient_matches_finite_differences_iw() {
        check_objective_gradient(true);
    }

    #[test]
    fn objective_gradient_matches_finite_differences_elbo() {
        check_objective_gradient(false);
    }

    #[test]
    fn svi_increases_the_objective() {
        let problem = small_problem();
        let mut params = small_params(&problem);
        let settings = SviSettings {
            steps: 300,
            pairs: 2,
            lr: 0.02,
            lr_final: 0.002,
            warmup_fraction: 0.1,
            laplace_iterations: 100,
            map_iterations: 0,
            map_lr: 0.03,
            sample_scale: 0.5,
            seed: 3,
        };
        let trace = maximize(&problem, &mut params, &test_prior(), &Mu_settings(), &settings, Verbosity::Quiet).unwrap();
        let first: f64 = trace[..30].iter().sum::<f64>() / 30.0;
        let last: f64 = trace[trace.len() - 30..].iter().sum::<f64>() / 30.0;
        assert!(last > first, "{} -> {}", first, last);
    }

    #[test]
    fn map_stage_increases_the_log_joint() {
        let problem = small_problem();
        let params = small_params(&problem);
        let prior = test_prior();
        let Mu_settings = Mu_settings();
        let eps = vec![vec![Coordinates::zeros(); problem.num_sites]];
        let before = evaluate_alignment(&problem, &params, &prior, &Mu_settings, &eps, false).unwrap().0;
        let (x, log_bl, log_R, rate) = map_stage(&problem, &prior, &Mu_settings, &(params.mean.clone(), params.log_branch_lengths.clone()), 200, 0.03, Verbosity::Quiet).unwrap();
        let mut map = params.clone();
        map.mean = x;
        map.log_branch_lengths = log_bl;
        map.log_R = Matrix20::from_row_slice(&log_R.flatten_all().unwrap().to_vec1::<f64>().unwrap());
        let after = evaluate_alignment(&problem, &map, &prior, &Mu_settings, &eps, false).unwrap().0;
        assert!(after > before, "{} -> {}", before, after);
        assert!(rate.is_finite() && rate > 0.0);
    }

    #[test]
    fn tempered_samples_are_centred_and_scaled() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mean = test_point();
        let l = test_prior().cholesky;
        // Antithetic pairs average to the mean
        let samples = tempered_samples(&mean, &l, 8, 0.5, &mut rng);
        assert_eq!(samples.len(), 8);
        assert!((samples.iter().sum::<Coordinates>() / 8.0 - mean).amax() < 1e-12);
        assert!(samples.chunks(2).all(|p| (p[0] + p[1] - mean * 2.0).amax() < 1e-12));
        // Scale 0 is the mode, the deviation is proportional to the scale
        assert!(tempered_samples(&mean, &l, 3, 0.0, &mut rng).iter().all(|x| (x - mean).amax() == 0.0));
        let mut a = ChaCha8Rng::seed_from_u64(9);
        let mut b = ChaCha8Rng::seed_from_u64(9);
        let half = tempered_samples(&mean, &l, 4, 0.5, &mut a);
        let full = tempered_samples(&mean, &l, 4, 1.0, &mut b);
        for (h, f) in half.iter().zip(&full) {
            assert!(((h - mean) * 2.0 - (f - mean)).amax() < 1e-12);
        }
    }
}
