//! The generalized-normal MutSel model (MUTSEL{pi_reg/mu_reg/rate_reg}): the joint MAP estimate of Mu, the site
//! frequencies of every site (their PCA coordinates c, see pca.rs), a rate of every site and the branch lengths, with
//! - a generalized normal prior on the PCA coordinates: log p(c) = -sum_k (|c_k - mean_k| * pi_reg / alpha_k)^beta_k
//!   + const, pi_reg the strength of the prior
//! - the log prior -mu_reg * sum_{i > j} (log S_ij - log S_prior_ij)^2 of the log exchangeabilities (see prior.rs)
//! - a log-normal prior on the site rates r_s, which multiply the MutSel rate matrices of the sites:
//!   log p(log r_s) = -rate_reg * (log r_s)^2 + const, i.e. log r_s ~ N(0, 1 / (2 rate_reg)) (rate_reg inf keeps all
//!   rates 1)
//! The rate matrices r_s S_s are normalized to a mean substitution rate of 1 over the sites, so the site rates only
//! redistribute the rates. Initialized with the light PMSF procedure (optimization::light_pmsf_init).

use phylo_grad::nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::{
    Result, Verbosity, batch,
    lbfgs::maximize_finite,
    model,
    npz::{Array, write_npz},
    optimization::{Convergence, Mu, Mu_backward, NUM_EXCHANGEABILITIES, light_pmsf_init},
    pca::{self, NUM_COORDINATES, PCA},
    prior::ExchangeabilityPrior,
};

/// Parameters of the model string MUTSEL{pi_reg/mu_reg/rate_reg} (case-insensitive name, ',' also separates them).
/// Trailing ones can be left out: MUTSEL, MUTSEL{pi_reg} and MUTSEL{pi_reg/mu_reg} use the defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameters {
    /// Strength of the generalized normal prior of the site frequencies (default 0.32), it divides its scales alpha
    pub pi_reg: f64,
    /// Strength of the prior of the log exchangeabilities (default 9.67)
    pub mu_reg: f64,
    /// Strength of the prior of the log site rates, 1 / (2 sd^2) (default 2, sd 0.5, a placeholder), inf for no site
    /// rates
    pub rate_reg: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters { pi_reg: 0.32, mu_reg: 9.67, rate_reg: 2.0 }
    }
}

impl Parameters {
    fn site_rates(&self) -> bool {
        self.rate_reg.is_finite()
    }

    pub fn parse(model: &str) -> std::result::Result<Parameters, String> {
        let model = model.trim();
        let usage = || {
            format!(
                "invalid model {}: use MUTSEL{{pi_reg/mu_reg/rate_reg}}, trailing ones can be left out, e.g. MUTSEL, \
                 MUTSEL{{0.32}}, MUTSEL{{0.32/9.67/2}} (pi_reg > 0, the strength of the site-frequency prior; mu_reg >= 0, that \
                 of the exchangeability prior; rate_reg > 0, that of the log site rates, inf for no site rates)",
                model
            )
        };
        let mut parameters = Parameters::default();
        if model.len() < 6 || !model[..6].eq_ignore_ascii_case("MUTSEL") {
            return Err(usage());
        }
        let rest = &model[6..];
        if rest.is_empty() {
            return Ok(parameters);
        }
        let values = rest.strip_prefix('{').and_then(|r| r.strip_suffix('}')).ok_or_else(usage)?;
        let values: Vec<&str> = values.split([',', '/']).map(str::trim).collect();
        if values.len() > 3 {
            return Err(usage());
        }
        parameters.pi_reg = values[0].parse().map_err(|_| usage())?;
        if let Some(mu_reg) = values.get(1) {
            parameters.mu_reg = mu_reg.parse().map_err(|_| usage())?;
        }
        if let Some(rate_reg) = values.get(2) {
            parameters.rate_reg = rate_reg.parse().map_err(|_| usage())?;
        }
        if !(parameters.pi_reg > 0.0 && parameters.pi_reg.is_finite())
            || !(parameters.mu_reg >= 0.0 && parameters.mu_reg.is_finite())
            || !(parameters.rate_reg > 0.0)
        {
            return Err(usage());
        }
        Ok(parameters)
    }
}

/// Generalized normal prior of the PCA coordinates: log density (without the normalization) and its gradient
fn site_frequency_log_prior(pi_reg: f64, c: &SVector<f64, NUM_COORDINATES>) -> (f64, SVector<f64, NUM_COORDINATES>) {
    let mut log_prior = 0.0;
    let mut grad = SVector::<f64, NUM_COORDINATES>::zeros();
    for k in 0..NUM_COORDINATES {
        let d = c[k] - PCA.mean[k];
        let scale = PCA.alpha[k] / pi_reg;
        let beta = PCA.beta[k];
        let z = d.abs() / scale;
        log_prior -= z.powf(beta);
        // all beta > 1, so the derivative is continuous (0 at the mean)
        grad[k] = -beta * z.powf(beta - 1.0) * d.signum() / scale;
    }
    (log_prior, grad)
}

/// Log-normal prior of the site rates: -rate_reg (log r)^2 and its derivative
fn site_rate_log_prior(rate_reg: f64, log_rate: f64) -> (f64, f64) {
    (-rate_reg * log_rate * log_rate, -2.0 * rate_reg * log_rate)
}

/// The layout of x = [log exchangeabilities, PCA coordinates of every site, log site rates (if any), log branch
/// lengths]
#[derive(Debug, Clone, Copy)]
struct Layout {
    num_sites: usize,
    site_rates: bool,
}

impl Layout {
    fn coordinates(&self) -> std::ops::Range<usize> {
        NUM_EXCHANGEABILITIES..NUM_EXCHANGEABILITIES + self.num_sites * NUM_COORDINATES
    }

    fn log_site_rates(&self) -> std::ops::Range<usize> {
        let start = self.coordinates().end;
        start..start + if self.site_rates { self.num_sites } else { 0 }
    }

    fn log_branch_lengths(&self) -> std::ops::RangeFrom<usize> {
        self.log_site_rates().end..
    }
}

/// Log posterior (log likelihood with the rate matrices r_s S_s normalized to a mean substitution rate of 1 plus the
/// log priors) and its gradient at x (see Layout). site_weights: the number of alignment columns of every site
/// (pattern), which also multiply its log priors.
fn objective(
    parents: &[i32],
    columns: &[&[u8]],
    mu_prior: &ExchangeabilityPrior,
    parameters: Parameters,
    x: &[f64],
    site_weights: &[f64],
) -> (f64, Vec<f64>) {
    let layout = Layout { num_sites: columns.len(), site_rates: parameters.site_rates() };
    let log_S = &x[..NUM_EXCHANGEABILITIES];
    let coordinates = &x[layout.coordinates()];
    let log_site_rates = &x[layout.log_site_rates()];
    let branch_lengths: Vec<f64> = x[layout.log_branch_lengths()].iter().map(|l| l.exp()).collect();
    let site_rates: Vec<f64> = log_site_rates.iter().map(|l| l.exp()).collect();
    let log_pi: Vec<SVector<f64, 20>> =
        coordinates.par_chunks_exact(NUM_COORDINATES).map(|c| pca::coordinates_to_log_freq(&SVector::from_column_slice(c))).collect();
    let result = batch::log_likelihood_and_gradients(
        parents,
        columns,
        &Mu(log_S),
        &log_pi,
        &branch_lengths,
        layout.site_rates.then_some(&site_rates[..]),
        Some(site_weights),
    );

    let (log_prior_S, grad_log_prior_S) = mu_prior.evaluate(log_S);
    let mut gradient: Vec<f64> =
        Mu_backward(log_S, &result.grad_Mu).iter().zip(&grad_log_prior_S).map(|(g, p)| g + p).collect();

    let (log_prior_pi, grad_coordinates): (Vec<f64>, Vec<SVector<f64, NUM_COORDINATES>>) = coordinates
        .par_chunks_exact(NUM_COORDINATES)
        .zip(result.grad_log_pi.par_iter())
        .zip(site_weights.par_iter())
        .map(|((c, grad_log_pi), &w)| {
            let (log_prior, grad_log_prior) = site_frequency_log_prior(parameters.pi_reg, &SVector::from_column_slice(c));
            (w * log_prior, PCA.components * grad_log_pi + grad_log_prior * w)
        })
        .unzip();
    gradient.extend(grad_coordinates.iter().flat_map(|g| g.iter().copied()));

    let mut log_prior_rates = 0.0;
    for ((l, w), g) in log_site_rates.iter().zip(site_weights).zip(&result.grad_log_site_rates) {
        let (log_prior, grad) = site_rate_log_prior(parameters.rate_reg, *l);
        log_prior_rates += w * log_prior;
        gradient.push(g + w * grad);
    }
    gradient.extend(branch_lengths.iter().zip(&result.grad_branch_lengths).map(|(b, g)| g * b));

    (result.log_likelihood + log_prior_S + log_prior_pi.iter().sum::<f64>() + log_prior_rates, gradient)
}

pub struct Estimate {
    pub log_S: Vec<f64>,
    pub coordinates: Vec<SVector<f64, NUM_COORDINATES>>,
    /// All 0 without site rates
    pub log_site_rates: Vec<f64>,
    pub branch_lengths: Vec<f64>,
    pub log_posterior: f64,
}

impl Estimate {
    /// S (including the site rate and the normalization to a mean substitution rate of 1) and sqrt_pi of every site
    pub fn site_models(&self, site_weights: &[f64]) -> (Vec<SMatrix<f64, 20, 20>>, Vec<SVector<f64, 20>>) {
        let Mu = Mu(&self.log_S);
        let models: Vec<(SMatrix<f64, 20, 20>, SVector<f64, 20>)> =
            self.coordinates.par_iter().map(|c| model::rate_matrix(&Mu, &pca::coordinates_to_log_freq(c))).collect();
        let total: f64 = site_weights.iter().sum();
        let rate_sum: f64 = models
            .iter()
            .zip(&self.log_site_rates)
            .zip(site_weights)
            .map(|(((S, sqrt_pi), l), w)| w * l.exp() * model::substitution_rate(S, sqrt_pi))
            .sum();
        let scale = total / rate_sum;
        models.into_iter().zip(&self.log_site_rates).map(|((S, sqrt_pi), l)| (S * (scale * l.exp()), sqrt_pi)).unzip()
    }
}

/// The joint MAP estimate (see the module documentation), every site with its weight (site_weights: the number of
/// alignment columns of every pattern)
pub fn estimate(
    parents: &[i32],
    columns: &[&[u8]],
    branch_lengths: &[f64],
    mu_prior: &ExchangeabilityPrior,
    parameters: Parameters,
    convergence: Convergence,
    verbosity: Verbosity,
    site_weights: &[f64],
) -> Estimate {
    let num_sites = columns.len();
    let layout = Layout { num_sites, site_rates: parameters.site_rates() };
    let (site_freq, branch_lengths) =
        light_pmsf_init(parents, columns, &mu_prior.log_exchangeabilities, branch_lengths, convergence, Some(site_weights));

    let mut x = mu_prior.log_exchangeabilities.clone();
    x.extend(site_freq.iter().flat_map(|f| pca::log_freq_to_coordinates(&f.map(f64::ln)).iter().copied().collect::<Vec<_>>()));
    x.extend(std::iter::repeat_n(0.0, layout.log_site_rates().len()));
    x.extend(branch_lengths.iter().map(|b| b.ln()));

    let start = std::time::Instant::now();
    let objective = |x: &[f64]| objective(parents, columns, mu_prior, parameters, x, site_weights);
    let num_evaluations = maximize_finite(|x| Some(objective(x)), &mut x, convergence.max_iterations, convergence.min_rel_improvement);
    let log_posterior = objective(&x).0;
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL generalized normal MAP estimate: log posterior {:.3}, {} evaluations ({:.1}s)",
            log_posterior,
            num_evaluations,
            start.elapsed().as_secs_f64()
        );
    }

    Estimate {
        log_S: x[..NUM_EXCHANGEABILITIES].to_vec(),
        coordinates: x[layout.coordinates()].chunks_exact(NUM_COORDINATES).map(SVector::from_column_slice).collect(),
        log_site_rates: if layout.site_rates { x[layout.log_site_rates()].to_vec() } else { vec![0.0; num_sites] },
        branch_lengths: x[layout.log_branch_lengths()].iter().map(|l| l.exp()).collect(),
        log_posterior,
    }
}

/// The distinct columns (patterns) of an alignment: identical columns have the same estimate, so every pattern is
/// estimated once and counts as often as it occurs
struct Patterns {
    /// [num_patterns * num_leaves], in the order of their first column
    alignment: Vec<u8>,
    /// Number of columns of every pattern
    counts: Vec<f64>,
    /// Pattern of every column
    site_to_pattern: Vec<usize>,
}

impl Patterns {
    fn new(alignment: &[u8], num_leaves: usize) -> Patterns {
        let mut index: std::collections::HashMap<&[u8], usize> = std::collections::HashMap::new();
        let mut patterns = Patterns { alignment: Vec::new(), counts: Vec::new(), site_to_pattern: Vec::new() };
        for column in alignment.chunks_exact(num_leaves) {
            let pattern = *index.entry(column).or_insert_with(|| {
                patterns.alignment.extend_from_slice(column);
                patterns.counts.push(0.0);
                patterns.counts.len() - 1
            });
            patterns.counts[pattern] += 1.0;
            patterns.site_to_pattern.push(pattern);
        }
        patterns
    }

    /// Per-pattern values for every column
    fn expand<T: Clone>(&self, values: &[T]) -> Vec<T> {
        self.site_to_pattern.iter().map(|&p| values[p].clone()).collect()
    }
}

/// The site models of the generalized-normal model for IQ-TREE: S, sqrt_pi and the weight of every site and draw
/// (num_tree_draws copies of the MAP site model with weights 1 / num_tree_draws, as there are no posterior draws).
/// Writes the estimate to out_prefix.map.npz.
pub fn run(
    parents: &[i32],
    branch_lengths: &[f64],
    alignment: &[u8],
    num_sites: usize,
    num_leaves: usize,
    verbosity: Verbosity,
    out_prefix: &str,
    parameters: Parameters,
    num_tree_draws: usize,
) -> Result<(Vec<SMatrix<f64, 20, 20>>, Vec<SVector<f64, 20>>, Vec<f64>)> {
    let mu_prior = ExchangeabilityPrior::new(parameters.mu_reg);
    let convergence = Convergence::MAP;
    println!(
        "MUTSEL generalized normal model: joint MAP estimate (L-BFGS, at most {} iterations, relative improvement {:.0e}) of Mu \
         (around the exchangeabilities of prior v1.20, strength mu_reg {}), the site frequencies (generalized normal prior on their PCA \
         coordinates, strength pi_reg {}), {} and the branch lengths",
        convergence.max_iterations,
        convergence.min_rel_improvement,
        parameters.mu_reg,
        parameters.pi_reg,
        if parameters.site_rates() {
            format!("a rate per site (log prior -rate_reg (log r)^2, rate_reg {}, sd {:.4})", parameters.rate_reg, (0.5 / parameters.rate_reg).sqrt())
        } else {
            "no site rates".to_string()
        }
    );

    assert_eq!(alignment.len(), num_sites * num_leaves);
    let patterns = Patterns::new(alignment, num_leaves);
    println!("MUTSEL: {} sites, {} distinct patterns", num_sites, patterns.counts.len());
    let columns: Vec<&[u8]> = patterns.alignment.chunks_exact(num_leaves).collect();
    let counts = &patterns.counts[..];

    let estimate = estimate(parents, &columns, branch_lengths, &mu_prior, parameters, convergence, verbosity, counts);
    let (S, sqrt_pi) = estimate.site_models(counts);

    let pi: Vec<SVector<f64, 20>> = sqrt_pi.iter().map(|s| s.component_mul(s)).collect();
    let site_model_rate: Vec<f64> = S.iter().zip(&sqrt_pi).map(|(S, s)| model::substitution_rate(S, s)).collect();
    let path = format!("{}.map.npz", out_prefix);
    write_npz(
        std::path::Path::new(&path),
        &[
            ("Mu", &Array::from_matrix(&Mu(&estimate.log_S))),
            ("log_exchangeabilities", &Array::vector(estimate.log_S.clone())),
            ("branch_lengths", &Array::vector(estimate.branch_lengths.clone())),
            ("parents", &Array::vector(parents.iter().map(|&p| p as f64).collect())),
            ("pi", &Array::from_rows(&patterns.expand(&pi))),
            ("pca_coordinates", &Array::from_rows(&patterns.expand(&estimate.coordinates))),
            ("log_site_rates", &Array::vector(patterns.expand(&estimate.log_site_rates))),
            ("site_model_rate", &Array::vector(patterns.expand(&site_model_rate))),
            ("log_posterior", &Array::scalar(estimate.log_posterior)),
            ("pi_reg", &Array::scalar(parameters.pi_reg)),
            ("mu_reg", &Array::scalar(parameters.mu_reg)),
            ("rate_reg", &Array::scalar(parameters.rate_reg)),
        ],
    )?;
    println!("MUTSEL: estimate written to {}", path);

    let mut out_S = Vec::with_capacity(num_sites * num_tree_draws);
    let mut out_sqrt_pi = Vec::with_capacity(num_sites * num_tree_draws);
    for &pattern in &patterns.site_to_pattern {
        for _ in 0..num_tree_draws {
            out_S.push(S[pattern]);
            out_sqrt_pi.push(sqrt_pi[pattern]);
        }
    }
    Ok((out_S, out_sqrt_pi, vec![1.0 / num_tree_draws as f64; num_sites * num_tree_draws]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::tests::*;

    #[test]
    fn model_string() {
        let parse = |s| Parameters::parse(s).map(|p| (p.pi_reg, p.mu_reg, p.rate_reg));
        assert_eq!(parse("MUTSEL"), Ok((0.32, 9.67, 2.0)));
        assert_eq!(parse(" mutsel "), Ok((0.32, 9.67, 2.0)));
        assert_eq!(parse("MUTSEL{1.5}"), Ok((1.5, 9.67, 2.0)));
        assert_eq!(parse("MUTSEL{1.5/0}"), Ok((1.5, 0.0, 2.0)));
        assert_eq!(parse("MUTSEL{0.32, 5, 3}"), Ok((0.32, 5.0, 3.0)));
        assert_eq!(parse("MUTSEL{0.32/5/inf}"), Ok((0.32, 5.0, f64::INFINITY)));
        for bad in ["MUTSEL{}", "MUTSEL{0}", "MUTSEL{-1}", "MUTSEL{inf}", "MUTSEL{1/-1}", "MUTSEL{1/inf}", "MUTSEL{1/2/0}",
                    "MUTSEL{1/2/-1}", "MUTSEL{1/2/nan}", "MUTSEL{1/2/3/4}", "MUTSEL{1", "LG"] {
            assert!(parse(bad).is_err(), "{} parsed", bad);
        }
    }

    #[test]
    fn site_frequency_prior_is_the_generalized_normal() {
        let c = SVector::<f64, NUM_COORDINATES>::from_fn(|i, _| 0.8 * ((i as f64) * 1.7).sin());
        let (log_prior, grad) = site_frequency_log_prior(0.7, &c);
        let expected: f64 = (0..NUM_COORDINATES).map(|k| ((c[k] - PCA.mean[k]).abs() * 0.7 / PCA.alpha[k]).powf(PCA.beta[k])).sum();
        assert!((log_prior + expected).abs() < 1e-12 * expected);
        assert_eq!(site_frequency_log_prior(0.7, &PCA.mean), (0.0, SVector::zeros()));
        let h = 1e-6;
        for k in 0..NUM_COORDINATES {
            let mut e = SVector::<f64, NUM_COORDINATES>::zeros();
            e[k] = h;
            let fd = (site_frequency_log_prior(0.7, &(c + e)).0 - site_frequency_log_prior(0.7, &(c - e)).0) / (2.0 * h);
            assert!((fd - grad[k]).abs() < 1e-6 * (1.0 + grad[k].abs()), "{} {} {}", k, fd, grad[k]);
        }
    }

    fn test_mu_prior() -> ExchangeabilityPrior {
        ExchangeabilityPrior { log_exchangeabilities: crate::prior::tests::test_log_exchangeabilities(), mu_reg: 9.67 }
    }

    fn test_x(site_rates: bool) -> Vec<f64> {
        let mut x: Vec<f64> = test_mu_prior().log_exchangeabilities.iter().enumerate().map(|(k, l)| l + 0.2 * (k as f64).sin()).collect();
        x.extend(log_pi().iter().flat_map(|lp| pca::log_freq_to_coordinates(lp).iter().copied().collect::<Vec<_>>()));
        if site_rates {
            x.extend((0..NUM_SITES).map(|site| 0.3 * (site as f64 * 1.9).cos()));
        }
        x.extend(branch_lengths().iter().map(|b| b.ln()));
        x
    }

    #[test]
    fn objective_gradient_matches_finite_differences() {
        let alignment = alignment();
        let columns: Vec<&[u8]> = alignment.chunks_exact(NUM_LEAVES).collect();
        let mu_prior = test_mu_prior();
        let weights: Vec<f64> = (0..NUM_SITES).map(|site| 1.0 + (site % 3) as f64).collect();
        for rate_reg in [2.0, f64::INFINITY] {
            let parameters = Parameters { rate_reg, ..Parameters::default() };
            let x = test_x(parameters.site_rates());
            let f = |x: &[f64]| objective(&PARENTS, &columns, &mu_prior, parameters, x, &weights);
            let (_, gradient) = f(&x);
            let h = 1e-5;
            let max_abs = gradient.iter().fold(0.0f64, |m, g| m.max(g.abs()));
            for k in 0..x.len() {
                let (mut plus, mut minus) = (x.clone(), x.clone());
                plus[k] += h;
                minus[k] -= h;
                let fd = (f(&plus).0 - f(&minus).0) / (2.0 * h);
                assert!((fd - gradient[k]).abs() < 1e-5 * max_abs, "rate_reg {} entry {}: {} vs {}", rate_reg, k, fd, gradient[k]);
            }
        }
    }

    #[test]
    fn site_models_have_mean_rate_one_and_the_site_rates() {
        let x = test_x(true);
        let layout = Layout { num_sites: NUM_SITES, site_rates: true };
        let log_site_rates = x[layout.log_site_rates()].to_vec();
        let estimate = Estimate {
            log_S: x[..NUM_EXCHANGEABILITIES].to_vec(),
            coordinates: x[layout.coordinates()].chunks_exact(NUM_COORDINATES).map(SVector::from_column_slice).collect(),
            log_site_rates: log_site_rates.clone(),
            branch_lengths: branch_lengths(),
            log_posterior: 0.0,
        };
        let weights: Vec<f64> = (0..NUM_SITES).map(|site| 1.0 + (site % 3) as f64).collect();
        let (S, sqrt_pi) = estimate.site_models(&weights);
        let rates: Vec<f64> = S.iter().zip(&sqrt_pi).map(|(S, s)| model::substitution_rate(S, s)).collect();
        let mean = rates.iter().zip(&weights).map(|(r, w)| r * w).sum::<f64>() / weights.iter().sum::<f64>();
        assert!((mean - 1.0).abs() < 1e-12);
        // Relative to the site models without site rates, every rate is multiplied by its site rate
        let plain = Estimate { log_site_rates: vec![0.0; NUM_SITES], ..estimate };
        let (S0, sqrt_pi0) = plain.site_models(&weights);
        let ratio = |site: usize| rates[site] / model::substitution_rate(&S0[site], &sqrt_pi0[site]);
        for site in 1..NUM_SITES {
            assert!((ratio(site) / ratio(0) - (log_site_rates[site] - log_site_rates[0]).exp()).abs() < 1e-12);
        }
    }

    #[test]
    fn patterns_count_identical_columns() {
        let alignment = [0u8, 1, 2, 3, 0, 1, 4, 4, 2, 3];
        let patterns = Patterns::new(&alignment, 2);
        assert_eq!(patterns.alignment, vec![0, 1, 2, 3, 4, 4]);
        assert_eq!(patterns.counts, vec![2.0, 2.0, 1.0]);
        assert_eq!(patterns.site_to_pattern, vec![0, 1, 0, 2, 1]);
        assert_eq!(patterns.expand(&[10, 20, 30]), vec![10, 20, 10, 30, 20]);
    }
}
