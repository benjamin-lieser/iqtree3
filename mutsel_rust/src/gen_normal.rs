//! The generalized-normal MutSel model (MUTSEL{pi_reg/mu_reg/rate_reg}): the joint MAP estimate of Mu, the site
//! frequencies of every site, a rate of every site and the branch lengths, with
//! - a generalized normal prior on the PCA coordinates c of the log site frequencies (data::PCA_*):
//!   log p(c) = -sum_k (|c_k - mean_k| * pi_reg / alpha_k)^beta_k + const, pi_reg the strength of the prior
//! - the log prior -mu_reg * sum_{i > j} (log S_ij - log S_prior_ij)^2 of the log exchangeabilities, S_prior those
//!   of the prior file (see prior.rs; its site-frequency mixture, exchangeability_reg and site rate sd are not used)
//! - a log-normal prior on the site rates r_s, which multiply the MutSel rate matrices of the sites:
//!   log p(log r_s) = -rate_reg * (log r_s)^2 + const, i.e. log r_s ~ N(0, 1 / (2 rate_reg)) (rate_reg inf keeps all
//!   rates 1)
//! The rate matrices r_s S_s are normalized to a mean substitution rate of 1 over the sites, so the site rates only
//! redistribute the rates. Initialized with the light PMSF procedure as optimization::map_estimate.

use phylo_grad::nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::{
    Result, Verbosity, batch, data,
    helmert::{self, NUM_COORDINATES},
    hmc::Patterns,
    lbfgs::maximize_finite,
    model,
    npz::{Array, write_npz},
    optimization::{Convergence, Mu, Mu_backward, NUM_EXCHANGEABILITIES, light_pmsf_init},
    prior::Prior,
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
}

impl Parameters {
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

fn parse_values(text: &str) -> Vec<f64> {
    text.split_whitespace().map(|s| s.parse::<f64>().unwrap()).collect()
}

/// Generalized normal prior of the PCA coordinates c = P log_pi of the site frequencies (P: the first 19 PCA
/// components, orthogonal to the constant direction), evaluated on the Helmert coordinates x: c = P basis^T x
#[derive(Debug, Clone)]
pub struct GeneralizedNormalPrior {
    /// P basis^T, orthogonal
    rotation: SMatrix<f64, NUM_COORDINATES, NUM_COORDINATES>,
    mean: SVector<f64, NUM_COORDINATES>,
    /// alpha / pi_reg
    scale: SVector<f64, NUM_COORDINATES>,
    beta: SVector<f64, NUM_COORDINATES>,
}

impl GeneralizedNormalPrior {
    pub fn new(pi_reg: f64) -> GeneralizedNormalPrior {
        let components = SMatrix::<f64, 20, 20>::from_row_slice(&parse_values(data::PCA_COMPONENTS));
        let P: SMatrix<f64, NUM_COORDINATES, 20> = components.fixed_rows::<NUM_COORDINATES>(0).into_owned();
        let alpha = SVector::<f64, NUM_COORDINATES>::from_vec(parse_values(data::PCA_ALPHA));
        GeneralizedNormalPrior {
            rotation: P * helmert::basis().transpose(),
            mean: SVector::from_vec(parse_values(data::PCA_MEAN)),
            scale: alpha / pi_reg,
            beta: SVector::from_vec(parse_values(data::PCA_BETA)),
        }
    }

    /// PCA coordinates of the Helmert coordinates x
    pub fn pca_coordinates(&self, x: &SVector<f64, NUM_COORDINATES>) -> SVector<f64, NUM_COORDINATES> {
        self.rotation * x
    }

    /// Log density (without the normalization) and its gradient w.r.t. the Helmert coordinates x
    pub fn evaluate(&self, x: &SVector<f64, NUM_COORDINATES>) -> (f64, SVector<f64, NUM_COORDINATES>) {
        let d = self.pca_coordinates(x) - self.mean;
        let mut penalty = 0.0;
        let mut grad_c = SVector::<f64, NUM_COORDINATES>::zeros();
        for k in 0..NUM_COORDINATES {
            let z = d[k].abs() / self.scale[k];
            penalty += z.powf(self.beta[k]);
            // all beta > 1, so the derivative is continuous (0 at the mean)
            grad_c[k] = self.beta[k] * z.powf(self.beta[k] - 1.0) * d[k].signum() / self.scale[k];
        }
        (-penalty, -(self.rotation.transpose() * grad_c))
    }
}

/// Log-normal prior of the site rates: -rate_reg (log r)^2 and its derivative
fn site_rate_log_prior(rate_reg: f64, log_rate: f64) -> (f64, f64) {
    (-rate_reg * log_rate * log_rate, -2.0 * rate_reg * log_rate)
}

/// The layout of x = [log exchangeabilities, Helmert coordinates of every site, log site rates (if any), log
/// branch lengths]
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
/// log priors) and its gradient at x (see Layout). prior: its log exchangeabilities and exchangeability_reg (mu_reg)
/// are the prior of Mu. site_weights: the number of alignment columns of every site (pattern), which also multiply
/// its log priors.
fn objective(
    parents: &[i32],
    columns: &[&[u8]],
    prior: &Prior,
    pi_prior: &GeneralizedNormalPrior,
    rate_reg: f64,
    x: &[f64],
    site_weights: &[f64],
) -> (f64, Vec<f64>) {
    let layout = Layout { num_sites: columns.len(), site_rates: rate_reg.is_finite() };
    let log_S = &x[..NUM_EXCHANGEABILITIES];
    let coordinates = &x[layout.coordinates()];
    let log_site_rates = &x[layout.log_site_rates()];
    let branch_lengths: Vec<f64> = x[layout.log_branch_lengths()].iter().map(|l| l.exp()).collect();
    let site_rates: Vec<f64> = log_site_rates.iter().map(|l| l.exp()).collect();
    let basis = helmert::basis();
    let log_pi: Vec<SVector<f64, 20>> = coordinates
        .par_chunks_exact(NUM_COORDINATES)
        .map(|c| helmert::coordinates_to_log_freq(&SVector::from_column_slice(c)))
        .collect();
    let result = batch::log_likelihood_and_gradients(
        parents,
        columns,
        &Mu(log_S),
        &log_pi,
        &branch_lengths,
        batch::RateScaling::MeanRate,
        layout.site_rates.then_some(&site_rates[..]),
        Some(site_weights),
    );

    let (log_prior_S, grad_log_prior_S) = prior.evaluate_exchangeabilities(log_S);
    let mut gradient: Vec<f64> =
        Mu_backward(log_S, &result.grad_Mu).iter().zip(&grad_log_prior_S).map(|(g, p)| g + p).collect();

    let (log_prior_pi, grad_coordinates): (Vec<f64>, Vec<SVector<f64, NUM_COORDINATES>>) = coordinates
        .par_chunks_exact(NUM_COORDINATES)
        .zip(result.grad_log_pi.par_iter())
        .zip(site_weights.par_iter())
        .map(|((c, grad_log_pi), &w)| {
            let (log_prior, grad_log_prior) = pi_prior.evaluate(&SVector::from_column_slice(c));
            (w * log_prior, basis * grad_log_pi + grad_log_prior * w)
        })
        .unzip();
    gradient.extend(grad_coordinates.iter().flat_map(|g| g.iter().copied()));

    let mut log_prior_rates = 0.0;
    for ((l, w), g) in log_site_rates.iter().zip(site_weights).zip(&result.grad_log_site_rates) {
        let (log_prior, grad) = site_rate_log_prior(rate_reg, *l);
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
            self.coordinates.par_iter().map(|x| model::rate_matrix(&Mu, &helmert::coordinates_to_log_freq(x))).collect();
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
/// alignment columns of every pattern). The prior of Mu is that of prior with mu_reg as its strength.
pub fn estimate(
    parents: &[i32],
    columns: &[&[u8]],
    branch_lengths: &[f64],
    prior: &Prior,
    parameters: Parameters,
    convergence: Convergence,
    verbosity: Verbosity,
    site_weights: &[f64],
) -> Estimate {
    let num_sites = columns.len();
    let prior = &Prior { exchangeability_reg: parameters.mu_reg, ..prior.clone() };
    let pi_prior = GeneralizedNormalPrior::new(parameters.pi_reg);
    let layout = Layout { num_sites, site_rates: parameters.site_rates() };
    let (site_freq, branch_lengths) =
        light_pmsf_init(parents, columns, &prior.log_exchangeabilities, branch_lengths, convergence, Some(site_weights));

    let mut x = prior.log_exchangeabilities.clone();
    x.extend(site_freq.iter().flat_map(|f| helmert::log_freq_to_coordinates(&f.map(f64::ln)).iter().copied().collect::<Vec<_>>()));
    x.extend(std::iter::repeat_n(0.0, layout.log_site_rates().len()));
    x.extend(branch_lengths.iter().map(|b| b.ln()));

    let start = std::time::Instant::now();
    let objective = |x: &[f64]| objective(parents, columns, prior, &pi_prior, parameters.rate_reg, x, site_weights);
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
    let prior = Prior::from_env();
    let convergence = Convergence::MAP;
    println!(
        "MUTSEL generalized normal model: joint MAP estimate (L-BFGS, at most {} iterations, relative improvement {:.0e}) of Mu \
         (around the exchangeabilities of the prior file, strength mu_reg {}), the site frequencies (generalized normal prior \
         on their PCA coordinates, strength pi_reg {}), {} and the branch lengths",
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
    println!("MUTSEL: {} sites, {} distinct patterns", num_sites, patterns.num_patterns());
    let columns: Vec<&[u8]> = patterns.alignment.chunks_exact(num_leaves).collect();
    let counts = &patterns.counts[..];

    let estimate = estimate(parents, &columns, branch_lengths, &prior, parameters, convergence, verbosity, counts);
    let (S, sqrt_pi) = estimate.site_models(counts);

    let pi: Vec<SVector<f64, 20>> = sqrt_pi.iter().map(|s| s.component_mul(s)).collect();
    let path = format!("{}.map.npz", out_prefix);
    write_npz(
        std::path::Path::new(&path),
        &[
            ("Mu", &Array::from_matrix(&Mu(&estimate.log_S))),
            ("log_exchangeabilities", &Array::vector(estimate.log_S.clone())),
            ("branch_lengths", &Array::vector(estimate.branch_lengths.clone())),
            ("parents", &Array::vector(parents.iter().map(|&p| p as f64).collect())),
            ("pi", &Array::from_rows(&patterns.expand(&pi))),
            ("coordinates", &Array::from_rows(&patterns.expand(&estimate.coordinates))),
            ("log_site_rates", &Array::vector(patterns.expand(&estimate.log_site_rates))),
            ("site_model_rate", &Array::vector(patterns.expand(&S.iter().zip(&sqrt_pi).map(|(S, s)| model::substitution_rate(S, s)).collect::<Vec<_>>()))),
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
    fn prior_is_the_generalized_normal_of_the_pca_coordinates() {
        let prior = GeneralizedNormalPrior::new(0.7);
        assert!((prior.rotation * prior.rotation.transpose() - SMatrix::<f64, NUM_COORDINATES, NUM_COORDINATES>::identity()).abs().max() < 1e-12);
        // The PCA coordinates of log frequencies with any offset, as in the old implementation (log_pi P^T)
        let components = SMatrix::<f64, 20, 20>::from_row_slice(&parse_values(data::PCA_COMPONENTS));
        let log_pi = SVector::<f64, 20>::from_fn(|i, _| ((i * 7) % 5) as f64 * 0.4 - 3.0);
        let c = components.fixed_rows::<NUM_COORDINATES>(0) * log_pi;
        let x = helmert::log_freq_to_coordinates(&log_pi);
        assert!((prior.pca_coordinates(&x) - c).abs().max() < 1e-12);
        let alpha = parse_values(data::PCA_ALPHA);
        let expected: f64 = (0..NUM_COORDINATES).map(|k| ((c[k] - prior.mean[k]).abs() / (alpha[k] / 0.7)).powf(prior.beta[k])).sum();
        assert!((prior.evaluate(&x).0 + expected).abs() < 1e-12 * expected);
        // 0 at the mean (up to rounding: with beta < 2 a deviation of 1e-16 still gives a gradient of about 1e-4)
        let (at_mean, _) = prior.evaluate(&(prior.rotation.transpose() * prior.mean));
        assert!(at_mean.abs() < 1e-15);
    }

    #[test]
    fn prior_gradient_matches_finite_differences() {
        let prior = GeneralizedNormalPrior::new(0.32);
        let x = SVector::<f64, NUM_COORDINATES>::from_fn(|i, _| 0.8 * ((i as f64) * 1.7).sin());
        let (_, grad) = prior.evaluate(&x);
        let h = 1e-6;
        for k in 0..NUM_COORDINATES {
            let mut e = SVector::<f64, NUM_COORDINATES>::zeros();
            e[k] = h;
            let fd = (prior.evaluate(&(x + e)).0 - prior.evaluate(&(x - e)).0) / (2.0 * h);
            assert!((fd - grad[k]).abs() < 1e-6 * (1.0 + grad[k].abs()), "{} {} {}", k, fd, grad[k]);
        }
    }

    fn test_x(site_rates: bool) -> Vec<f64> {
        use crate::batch::tests::*;
        let mut x: Vec<f64> = crate::prior::tests::test_prior().log_exchangeabilities.iter().enumerate().map(|(k, l)| l + 0.2 * (k as f64).sin()).collect();
        x.extend(log_pi().iter().flat_map(|lp| helmert::log_freq_to_coordinates(lp).iter().copied().collect::<Vec<_>>()));
        if site_rates {
            x.extend((0..NUM_SITES).map(|site| 0.3 * (site as f64 * 1.9).cos()));
        }
        x.extend(branch_lengths().iter().map(|b| b.ln()));
        x
    }

    #[test]
    fn objective_gradient_matches_finite_differences() {
        use crate::batch::tests::*;
        let alignment = alignment();
        let columns: Vec<&[u8]> = alignment.chunks_exact(NUM_LEAVES).collect();
        let prior = crate::prior::tests::test_prior();
        let pi_prior = GeneralizedNormalPrior::new(0.32);
        let weights: Vec<f64> = (0..NUM_SITES).map(|site| 1.0 + (site % 3) as f64).collect();
        for rate_reg in [2.0, f64::INFINITY] {
            let x = test_x(rate_reg.is_finite());
            let f = |x: &[f64]| objective(&PARENTS, &columns, &prior, &pi_prior, rate_reg, x, &weights);
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
        use crate::batch::tests::*;
        let x = test_x(true);
        let layout = Layout { num_sites: NUM_SITES, site_rates: true };
        let estimate = Estimate {
            log_S: x[..NUM_EXCHANGEABILITIES].to_vec(),
            coordinates: x[layout.coordinates()].chunks_exact(NUM_COORDINATES).map(SVector::from_column_slice).collect(),
            log_site_rates: x[layout.log_site_rates()].to_vec(),
            branch_lengths: branch_lengths(),
            log_posterior: 0.0,
        };
        let weights: Vec<f64> = (0..NUM_SITES).map(|site| 1.0 + (site % 3) as f64).collect();
        let (S, sqrt_pi) = estimate.site_models(&weights);
        let rates: Vec<f64> = S.iter().zip(&sqrt_pi).map(|(S, s)| model::substitution_rate(S, s)).collect();
        let mean = rates.iter().zip(&weights).map(|(r, w)| r * w).sum::<f64>() / weights.iter().sum::<f64>();
        assert!((mean - 1.0).abs() < 1e-12);
        // Without the site rates, the rates are those of the frequencies alone
        let plain = Estimate { log_site_rates: vec![0.0; NUM_SITES], ..estimate };
        let (S0, sqrt_pi0) = plain.site_models(&weights);
        let ratio = |site: usize| rates[site] / model::substitution_rate(&S0[site], &sqrt_pi0[site]);
        for site in 1..NUM_SITES {
            let expected = (x[layout.log_site_rates()][site] - x[layout.log_site_rates()][0]).exp();
            assert!((ratio(site) / ratio(0) - expected).abs() < 1e-12);
        }
    }
}
