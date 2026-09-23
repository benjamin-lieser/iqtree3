//! Per-site posterior sampling of the PCA coordinates with NUTS (nuts-rs).
//!
//! Mu, the branch lengths and the global rate normalization are shared and kept
//! constant. Site rates are fixed to 1. With everything shared fixed, the
//! posterior factorizes over alignment columns, so every column gets its own
//! independent 19-dimensional NUTS chain.

use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use candle_core::Tensor;
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
    rand::{SeedableRng, rngs::ChaCha8Rng},
};
use phylo_grad::{
    FelsensteinTree,
    nalgebra::{SMatrix, SVector},
};
use rayon::prelude::*;

use crate::{
    MutselParams, Verbosity,
    felsenstein::FelsensteinOp,
    model::{self, calc_rate_matrix},
    optimization::{Mu, two_step_light_pmsf},
    pca::PCA,
    utils::tensor_full,
};

const NUM_COMPONENTS: usize = 19;

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
}

impl HmcSettings {
    /// Reads MUTSEL_HMC_TUNE, MUTSEL_HMC_DRAWS, MUTSEL_HMC_ESS, MUTSEL_HMC_MAX_DRAWS,
    /// MUTSEL_HMC_SAMPLES and MUTSEL_HMC_SEED from the environment.
    pub fn from_env() -> HmcSettings {
        fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
            match std::env::var(name) {
                Ok(value) => value
                    .trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("Could not parse {}={}", name, value)),
                Err(_) => default,
            }
        }
        HmcSettings {
            num_tune: env_or("MUTSEL_HMC_TUNE", 400),
            num_draws: env_or("MUTSEL_HMC_DRAWS", 500),
            target_ess: env_or("MUTSEL_HMC_ESS", 100),
            max_draws: env_or("MUTSEL_HMC_MAX_DRAWS", 20000),
            num_output_samples: env_or("MUTSEL_HMC_SAMPLES", 100),
            seed: env_or("MUTSEL_HMC_SEED", 42),
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
        }
    }
}

impl SharedModel {
    /// Site frequencies for the given pca coordinates
    fn pi(&self, x: &SVector<f64, NUM_COMPONENTS>) -> (SVector<f64, 20>, SVector<f64, 20>) {
        let log_pi = self.components.transpose() * x;
        let max_log_pi = log_pi.max();
        let unnormalized_pi = log_pi.map(|l| (l - max_log_pi).exp());
        (log_pi, unnormalized_pi / unnormalized_pi.sum())
    }

    /// Log prior, S and sqrt_pi for the pca coordinates x
    fn evaluate(&self, x: &SVector<f64, NUM_COMPONENTS>) -> SiteModelEval {
        // Generalized normal prior on the pca coordinates (negative of PCA::penalty_on_pca_coordinates)
        let mut log_prior = 0.0;
        let mut grad_log_prior = SVector::<f64, NUM_COMPONENTS>::zeros();
        for k in 0..NUM_COMPONENTS {
            let z = (x[k] - self.prior_mean[k]) / self.prior_scale[k];
            let beta = self.prior_beta[k];
            let abs_z_pow = z.abs().powf(beta - 1.0);
            log_prior -= abs_z_pow * z.abs();
            grad_log_prior[k] = -beta * abs_z_pow * z.signum() / self.prior_scale[k];
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
fn fixation(x: f64) -> f64 {
    if x.abs() < 1e-6 {
        1.0 + x / 2.0 + x * x / 12.0
    } else {
        x / (-(-x).exp_m1())
    }
}

/// d/dx log g(x) = 1/x - 1/(exp(x) - 1)
fn dx_log_fixation(x: f64) -> f64 {
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
    let mut next_check = settings.num_draws.min(settings.max_draws) as usize;
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
    }
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

/// Samples the PCA coordinates of every site from its posterior and returns (S, sqrt_pi)
/// evaluated at the posterior mean of the site frequencies.
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
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let settings = HmcSettings::from_env();
    println!(
        "MUTSEL HMC: sampling pca coordinates per site with NUTS ({} tuning, at least {} draws, until ESS >= {} or {} draws, {} thinned samples per site, seed {}), Mu fixed, branch lengths fixed to the two step PMSF estimate, site rates 1",
        settings.num_tune,
        settings.num_draws,
        settings.target_ess,
        settings.max_draws,
        settings.num_output_samples,
        settings.seed
    );
    assert!(settings.num_output_samples >= 1 && settings.num_draws as usize >= settings.num_output_samples);

    let init_R = if let Some(prior_R_file) = prior_R_file {
        let file_content = std::fs::read_to_string(prior_R_file)?;
        crate::data::load_lower_R_with_equi(&file_content)
    } else {
        crate::data::load_lower_R_with_equi(crate::data::M_TXT)
    };
    let Mu = Mu(&init_R.log()?).detach();

    let log_branch_lengths =
        Tensor::from_slice(branch_lengths, &[branch_lengths.len()], &candle_core::Device::Cpu)?.log()?;

    // Branch lengths and initial site frequencies from the two step light PMSF procedure.
    let felsenstein = crate::create_felsenstein_tree(parents, branch_lengths, alignment, num_sites, num_leaves);
    let op = FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));
    let (site_freq, log_branch_lengths) = two_step_light_pmsf(
        op,
        crate::data::UDM256,
        crate::data::UDM256_WEIGHTS,
        &log_branch_lengths,
        mutsel_params,
        verbosity,
        out_prefix,
    );
    let branch_lengths = log_branch_lengths.exp()?.to_vec1::<f64>()?;
    let branch_lengths = branch_lengths.as_slice();

    let pca = Arc::new(PCA::new(NUM_COMPONENTS));
    let init_pca_coordinates = pca.log_freq_to_pca_coordinates(&site_freq.log()?);
    let init_log_pi = pca.pca_coordinates_to_log_freq(&init_pca_coordinates);

    // Fixed global normalization: average substitution rate 1 at the initial frequencies.
    let (S_init, sqrt_pi_init) = calc_rate_matrix(&Mu, &init_log_pi, &tensor_full(1.0, &[]));
    let average_rate = model::substitution_rates_tensor(&S_init, &sqrt_pi_init)
        .mean_all()?
        .to_scalar::<f64>()?;
    let rate_scaling = tensor_full(1.0 / average_rate, &[]);
    if verbosity.should_print(Verbosity::Min) {
        println!("MUTSEL HMC: fixed rate normalization 1/{:.5}", average_rate);
    }

    let shared = Arc::new(SharedModel::new(&Mu, 1.0 / average_rate, &pca, mutsel_params.pi_reg));
    let tree = Arc::new(FelsensteinTree::<20>::new(parents, branch_lengths));
    let num_nodes = tree.num_nodes();
    let init_pca_tensor = init_pca_coordinates;
    let init_pca_coordinates = init_pca_tensor.to_vec2::<f64>()?;

    let start = std::time::Instant::now();
    let num_done = AtomicUsize::new(0);
    let report_every = (num_sites / 10).max(1);

    let site_samples: Vec<SiteSamples> = (0..num_sites)
        .into_par_iter()
        .map(|site| {
            let logp = SiteLogp {
                shared: Arc::clone(&shared),
                tree: Arc::clone(&tree),
                leaf_pl: leaf_partial_likelihoods(alignment, site, num_leaves),
                pl_buffer: vec![SVector::<f64, 20>::zeros(); num_nodes],
            };
            let samples = sample_site(logp, &init_pca_coordinates[site], &settings, site);

            let done = num_done.fetch_add(1, Ordering::Relaxed) + 1;
            if verbosity.should_print(Verbosity::Min) && (done % report_every == 0 || done == num_sites) {
                println!(
                    "MUTSEL HMC: {}/{} sites sampled ({:.1}s)",
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
        "MUTSEL HMC: done in {:.1}s, {} draws per site on average, {} divergent transitions in {} sites, mean {:.1} leapfrog steps per draw",
        start.elapsed().as_secs_f64(),
        total_draws / num_sites.max(1),
        total_divergences,
        sites_with_divergences,
        mean_steps
    );
    println!(
        "MUTSEL HMC: minimum ESS over sites {:.0}, {} sites below the target ESS {} after {} draws",
        min_ess, sites_below_target, settings.target_ess, settings.max_draws
    );

    // pca_samples [num_sites, num_output_samples, 19]
    let num_output_samples = settings.num_output_samples;
    let pca_samples: Vec<f64> = site_samples
        .iter()
        .flat_map(|s| s.thinned_draws.iter().copied())
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
            ("branch_lengths", &Tensor::from_slice(branch_lengths, &[branch_lengths.len()], &candle_core::Device::Cpu)?),
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
        ],
        Path::new(&npz_path),
    )?;
    println!("MUTSEL HMC: samples written to {}", npz_path);

    Ok((S, sqrt_pi))
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
}

