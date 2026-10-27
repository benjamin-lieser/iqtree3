//! NUTS sampling of the log_pi of every site, with Mu and the branch lengths fixed.
//!
//! The sites are independent, every site has its own chain and adaptation, and the sites run in parallel.
//! All sites share one tree and only own their partial likelihoods.

use std::collections::HashMap;

use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
    rand::{Rng, SeedableRng, rngs::StdRng},
};
use phylo_grad::FelsensteinTree;
use rayon::prelude::*;

use crate::{
    model::{self, Matrix20, Vector20},
    prior::Prior,
};

/// phylo_grad could not compute the likelihood, NUTS treats this as a divergence
#[derive(Debug)]
pub struct LikelihoodFailed;

impl std::fmt::Display for LikelihoodFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the likelihood is not finite")
    }
}

impl std::error::Error for LikelihoodFailed {}

impl LogpError for LikelihoodFailed {
    fn is_recoverable(&self) -> bool {
        true
    }
}

/// Posterior of log_pi of a single site
pub struct SitePosterior<'a, P: Prior> {
    felsenstein: &'a FelsensteinTree<20>,
    /// Partial likelihoods of all nodes of this site, the leaves come first
    partial_likelihoods: Vec<Vector20>,
    Mu: &'a Matrix20,
    prior: &'a P,
}

impl<'a, P: Prior> SitePosterior<'a, P> {
    /// leaves: partial likelihoods of the leaves of this site, in the order of the leaves in the tree
    pub fn new(felsenstein: &'a FelsensteinTree<20>, mut leaves: Vec<Vector20>, Mu: &'a Matrix20, prior: &'a P) -> Self {
        assert_eq!(leaves.len(), felsenstein.num_leaves());
        leaves.resize(felsenstein.num_nodes(), Vector20::zeros());
        SitePosterior { felsenstein, partial_likelihoods: leaves, Mu, prior }
    }

    /// Log posterior (up to a constant) and its gradient with respect to log_pi
    pub fn log_posterior(&mut self, log_pi: &Vector20) -> Result<(f64, Vector20), LikelihoodFailed> {
        let (S, sqrt_pi) = model::mutsel(self.Mu, log_pi);
        let result =
            self.felsenstein
                .calculate_gradients_single_side(S.as_view(), sqrt_pi.as_view(), &mut self.partial_likelihoods);

        let log_likelihood = result.log_likelihood;
        if !log_likelihood.is_finite() {
            return Err(LikelihoodFailed);
        }
        let grad_likelihood =
            model::mutsel_backward_log_pi(self.Mu, log_pi, &result.grad_s, &result.grad_sqrt_pi);
        let (log_prior, grad_prior) = self.prior.log_density(log_pi);

        Ok((log_likelihood + log_prior, grad_likelihood + grad_prior))
    }
}

impl<P: Prior> HasDims for SitePosterior<'_, P> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("log_pi".to_string(), 20)])
    }
}

impl<P: Prior> CpuLogpFunc for SitePosterior<'_, P> {
    type LogpError = LikelihoodFailed;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        20
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, LikelihoodFailed> {
        let (log_p, grad) = self.log_posterior(&Vector20::from_column_slice(position))?;
        gradient.copy_from_slice(grad.as_slice());
        Ok(log_p)
    }

    fn expand_vector<R: Rng + ?Sized>(&mut self, _rng: &mut R, position: &[f64]) -> Result<Vec<f64>, CpuMathError> {
        Ok(position.to_vec())
    }
}

#[derive(Debug, Clone)]
pub struct NutsResult {
    /// [site][draw], only the draws after tuning
    pub samples: Vec<Vec<Vector20>>,
    /// Number of divergent draws after tuning, per site
    pub num_divergences: Vec<usize>,
    /// Step size after tuning, per site
    pub step_size: Vec<f64>,
}

/// Samples log_pi of every site independently.
/// felsenstein: the tree with the fixed branch lengths, it does not need bound partial likelihoods.
/// leaves: [site][leaf] partial likelihoods of the leaves, usually one hot.
/// init: [site] starting position of the chains.
/// Site i uses the seed `seed + i`, so the result does not depend on the number of threads.
pub fn sample_log_pi(
    felsenstein: &FelsensteinTree<20>,
    leaves: Vec<Vec<Vector20>>,
    Mu: &Matrix20,
    prior: &impl Prior,
    init: &[Vector20],
    settings: &DiagNutsSettings,
    seed: u64,
) -> NutsResult {
    assert_eq!(leaves.len(), init.len());

    let sites: Vec<_> = leaves
        .into_par_iter()
        .zip(init)
        .enumerate()
        .map(|(site, (site_leaves, init))| {
            let posterior = SitePosterior::new(felsenstein, site_leaves, Mu, prior);
            let mut rng = StdRng::seed_from_u64(seed + site as u64);
            let mut chain = settings
                .new_chain(site as u64, CpuMath::new(posterior), &mut rng)
                .expect("Invalid NUTS settings");
            chain
                .set_position(init.as_slice())
                .unwrap_or_else(|e| panic!("Invalid initial log_pi for site {site}: {e}"));

            let mut samples = Vec::with_capacity(settings.num_draws as usize);
            let mut num_divergences = 0;
            let mut step_size = f64::NAN;
            for _ in 0..(settings.num_tune + settings.num_draws) {
                let (draw, progress) = chain
                    .draw()
                    .unwrap_or_else(|e| panic!("NUTS failed for site {site}: {e}"));
                if !progress.tuning {
                    samples.push(Vector20::from_column_slice(&draw));
                    num_divergences += progress.diverging as usize;
                    step_size = progress.step_size;
                }
            }
            (samples, num_divergences, step_size)
        })
        .collect();

    let mut result = NutsResult { samples: vec![], num_divergences: vec![], step_size: vec![] };
    for (samples, num_divergences, step_size) in sites {
        result.samples.push(samples);
        result.num_divergences.push(num_divergences);
        result.step_size.push(step_size);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{leaf_partial_likelihoods, prior::Gaussian};

    /// Mutation matrix with equilibrium on the diagonal, like the output of optimization::Mu
    fn test_Mu() -> Matrix20 {
        let mut Mu = Matrix20::from_fn(|i, j| 0.2 + ((i * 7 + j * 7) % 13) as f64 * 0.1);
        for i in 0..20 {
            Mu[(i, i)] = 0.02 + (i % 5) as f64 * 0.01;
        }
        Mu
    }

    /// Tree ((0,1)4,(2,3)5)6
    fn test_tree() -> FelsensteinTree<20> {
        FelsensteinTree::new(&[4, 4, 5, 5, 6, 6, -1], &[0.1, 0.3, 0.2, 0.15, 0.05, 0.1, 0.0])
    }

    /// alignment: [site][leaf], 255 is a gap
    fn test_leaves(alignment: &[[u8; 4]]) -> Vec<Vec<Vector20>> {
        leaf_partial_likelihoods(alignment.as_flattened(), alignment.len(), 4)
    }

    #[test]
    fn test_log_posterior_finite_differences() {
        let felsenstein = test_tree();
        let Mu = test_Mu();
        let prior = Gaussian::new(Vector20::from_element(0.5), 1.5);
        let log_pis = [
            Vector20::from_fn(|i, _| ((i * 3) % 7) as f64 * 0.2 - 0.5),
            Vector20::from_fn(|i, _| ((i * 5) % 11) as f64 * 0.1),
        ];

        for (leaves, log_pi) in test_leaves(&[[0, 0, 3, 255], [5, 6, 7, 8]]).into_iter().zip(log_pis) {
            let mut posterior = SitePosterior::new(&felsenstein, leaves, &Mu, &prior);
            let (_, grad) = posterior.log_posterior(&log_pi).unwrap();

            let h = 1e-5;
            for k in 0..20 {
                let mut plus = log_pi;
                plus[k] += h;
                let mut minus = log_pi;
                minus[k] -= h;
                let fd = (posterior.log_posterior(&plus).unwrap().0 - posterior.log_posterior(&minus).unwrap().0)
                    / (2.0 * h);
                assert!((fd - grad[k]).abs() < 1e-5 * (1.0 + fd.abs()), "{k}: {fd} vs {}", grad[k]);
            }
        }
    }

    /// Without data the posterior is the prior
    #[test]
    fn test_nuts_samples_prior_without_data() {
        let mean = Vector20::from_fn(|i, _| (i % 4) as f64 * 0.5 - 1.0);
        let sigma = 0.7;
        let prior = Gaussian::new(mean, sigma);
        let mut settings = DiagNutsSettings::default();
        settings.num_draws = 2000;

        let result = sample_log_pi(
            &test_tree(),
            test_leaves(&[[255; 4], [255; 4]]),
            &test_Mu(),
            &prior,
            &[Vector20::zeros(); 2],
            &settings,
            42,
        );

        for site in 0..2 {
            let samples = &result.samples[site];
            assert_eq!(samples.len(), 2000);
            let n = samples.len() as f64;
            let sample_mean = samples.iter().sum::<Vector20>() / n;
            let sample_var = samples.iter().map(|x| (x - sample_mean).map(|d| d * d)).sum::<Vector20>() / n;
            assert!((sample_mean - mean).abs().max() < 0.1, "{}", sample_mean - mean);
            assert!((sample_var.map(f64::sqrt) - Vector20::from_element(sigma)).abs().max() < 0.1);
            assert_eq!(result.num_divergences[site], 0);
        }
    }

    /// The posterior of a column with a conserved residue should put the most weight on that residue
    #[test]
    fn test_nuts_conserved_column() {
        let prior = Gaussian::new(Vector20::zeros(), 1.0);

        let result = sample_log_pi(
            &test_tree(),
            test_leaves(&[[3; 4]]),
            &test_Mu(),
            &prior,
            &[Vector20::zeros()],
            &DiagNutsSettings::default(),
            1,
        );

        let mean = result.samples[0].iter().sum::<Vector20>() / result.samples[0].len() as f64;
        assert_eq!(mean.argmax().0, 3);
    }
}
