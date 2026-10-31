//! Log likelihood of all sites and its gradients w.r.t. Mu, the log site frequencies and the branch lengths.
//!
//! The per-site model computations (S, sqrt_pi, the rate normalization and the backpropagation) run in
//! parallel over the sites, phylo_grad parallelizes the pruning itself.

use phylo_grad::nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::model::{rate_matrix, rate_matrix_backward, substitution_rate};

/// Factor the rate matrices S of all sites are multiplied with
#[derive(Debug, Clone, Copy)]
pub enum RateScaling {
    Fixed(f64),
    /// 1 / the mean substitution rate over the sites
    MeanRate,
}

pub struct BatchGradients {
    pub log_likelihood: f64,
    /// Gradient w.r.t. the upper triangle of Mu
    pub grad_Mu: SMatrix<f64, 20, 20>,
    /// Gradient w.r.t. the unnormalized log frequencies of every site
    pub grad_log_pi: Vec<SVector<f64, 20>>,
    pub grad_branch_lengths: Vec<f64>,
}

/// Mean substitution rate of Mu over the sites with the log frequencies log_pi
pub fn mean_substitution_rate(Mu: &SMatrix<f64, 20, 20>, log_pi: &[SVector<f64, 20>]) -> f64 {
    log_pi
        .par_iter()
        .map(|lp| {
            let (S, sqrt_pi) = rate_matrix(Mu, lp);
            substitution_rate(&S, &sqrt_pi)
        })
        .sum::<f64>()
        / log_pi.len() as f64
}

/// Summed log likelihood of all sites and its gradients. parents is topologically sorted (see lib.rs), columns
/// [num_sites] holds the leaf states of every site, log_pi [num_sites] are the log frequencies up to a constant
/// per site, branch_lengths in the node order of parents. site_rates [num_sites] multiply the rate matrices of
/// the sites on top of the scaling (only with RateScaling::Fixed).
pub fn log_likelihood_and_gradients(
    parents: &[i32],
    columns: &[&[u8]],
    Mu: &SMatrix<f64, 20, 20>,
    log_pi: &[SVector<f64, 20>],
    branch_lengths: &[f64],
    scaling: RateScaling,
    site_rates: Option<&[f64]>,
) -> BatchGradients {
    let num_sites = log_pi.len();
    assert_eq!(columns.len(), num_sites);
    if let Some(rates) = site_rates {
        assert_eq!(rates.len(), num_sites);
        assert!(matches!(scaling, RateScaling::Fixed(_)), "site rates need a fixed rate scaling");
    }
    let site_rate = |site: usize| site_rates.map_or(1.0, |rates| rates[site]);
    // S and sqrt_pi of every site before the scaling
    let models: Vec<(SMatrix<f64, 20, 20>, SVector<f64, 20>)> = log_pi.par_iter().map(|lp| rate_matrix(Mu, lp)).collect();
    let scale = match scaling {
        RateScaling::Fixed(scale) => scale,
        RateScaling::MeanRate => {
            num_sites as f64 / models.par_iter().map(|(S, sqrt_pi)| substitution_rate(S, sqrt_pi)).sum::<f64>()
        }
    };
    let S: Vec<SMatrix<f64, 20, 20>> =
        models.par_iter().enumerate().map(|(site, (S, _))| S * (scale * site_rate(site))).collect();
    let sqrt_pi: Vec<SVector<f64, 20>> = models.iter().map(|(_, sqrt_pi)| *sqrt_pi).collect();

    let mut grad_branch_lengths = vec![0.0; branch_lengths.len()];
    let result = phylo_grad::calculate_gradients_columns(
        parents,
        columns,
        branch_lengths,
        &S,
        &sqrt_pi,
        Some(std::slice::from_mut(&mut grad_branch_lengths)),
    );
    drop(S);

    // With MeanRate scale = num_sites / sum_s rate_s, so the log likelihood depends on every rate_s with
    // d log_likelihood / d rate_s = -scale^2 / num_sites * sum_s <grad_S_s, S_s> (S before scaling)
    let grad_rate = match scaling {
        RateScaling::Fixed(_) => 0.0,
        RateScaling::MeanRate => {
            let d_scale: f64 = models.par_iter().zip(result.grad_s.par_iter()).map(|((S, _), g)| g.upper_triangle().component_mul(S).sum()).sum();
            -scale * scale / num_sites as f64 * d_scale
        }
    };

    let (grad_log_pi, grad_Mu): (Vec<SVector<f64, 20>>, Vec<SMatrix<f64, 20, 20>>) = models
        .par_iter()
        .zip(log_pi.par_iter())
        .zip(result.grad_s.par_iter().zip(result.grad_sqrt_pi.par_iter()))
        .enumerate()
        .map(|(site, (((S, sqrt_pi), log_pi), (grad_S, grad_sqrt_pi)))| {
            // phylo_grad only reads the upper triangle of S. rate = sum_{i != j} S_ij sqrt_pi_i sqrt_pi_j
            let grad_S = grad_S.upper_triangle() * (scale * site_rate(site)) + sqrt_pi * sqrt_pi.transpose() * grad_rate;
            let grad_sqrt_pi = grad_sqrt_pi + S * sqrt_pi * (2.0 * grad_rate);
            let (grad_Mu, grad_log_pi) = rate_matrix_backward(log_pi, S, sqrt_pi, Mu, &grad_S, &grad_sqrt_pi);
            (grad_log_pi, grad_Mu)
        })
        .unzip();

    BatchGradients {
        log_likelihood: result.log_likelihood.iter().sum(),
        grad_Mu: grad_Mu.into_par_iter().reduce(SMatrix::<f64, 20, 20>::zeros, |a, b| a + b),
        grad_log_pi,
        grad_branch_lengths,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{helmert, optimization::Mu};

    pub(crate) const NUM_SITES: usize = 7;
    pub(crate) const NUM_LEAVES: usize = 5;
    /// ((0,1),(2,3)) plus a fifth leaf at the root
    pub(crate) const PARENTS: [i32; 8] = [5, 5, 6, 6, 7, 7, 7, -1];

    /// NUM_SITES columns, some with gaps
    pub(crate) fn alignment() -> Vec<u8> {
        (0..NUM_SITES * NUM_LEAVES).map(|i| ((i * 7 + i / 5) % 21) as u8).collect()
    }

    pub(crate) fn test_Mu() -> SMatrix<f64, 20, 20> {
        Mu(&crate::prior::tests::test_prior().log_exchangeabilities)
    }

    pub(crate) fn branch_lengths() -> Vec<f64> {
        (0..8).map(|i| 0.05 + 0.07 * i as f64).collect()
    }

    pub(crate) fn log_pi() -> Vec<SVector<f64, 20>> {
        (0..NUM_SITES)
            .map(|site| {
                let x = SVector::from_fn(|k, _| 0.4 * (((site * 19 + k) as f64) * 1.3).sin());
                helmert::coordinates_to_log_freq(&x)
            })
            .collect()
    }

    fn run(scaling: RateScaling, site_rates: Option<&[f64]>, Mu: &SMatrix<f64, 20, 20>, log_pi: &[SVector<f64, 20>], branch_lengths: &[f64]) -> BatchGradients {
        let alignment = alignment();
        let columns: Vec<&[u8]> = alignment.chunks_exact(NUM_LEAVES).collect();
        log_likelihood_and_gradients(&PARENTS, &columns, Mu, log_pi, branch_lengths, scaling, site_rates)
    }

    /// The gradients w.r.t. the off-diagonal Mu, log_pi and the branch lengths against central differences
    fn check_gradients(scaling: RateScaling, site_rates: Option<&[f64]>) {
        let (Mu, log_pi, branch_lengths) = (test_Mu(), log_pi(), branch_lengths());
        let result = run(scaling, site_rates, &Mu, &log_pi, &branch_lengths);
        let h = 1e-6;
        let check = |numeric: f64, analytic: f64, name: &str| {
            assert!((numeric - analytic).abs() < 1e-5 * (1.0 + analytic.abs()), "{}: {} vs {}", name, numeric, analytic);
        };
        for (i, j) in [(0, 1), (4, 17), (3, 19)] {
            let (mut plus, mut minus) = (Mu, Mu);
            plus[(i, j)] += h;
            minus[(i, j)] -= h;
            let fd = (run(scaling, site_rates, &plus, &log_pi, &branch_lengths).log_likelihood
                - run(scaling, site_rates, &minus, &log_pi, &branch_lengths).log_likelihood)
                / (2.0 * h);
            check(fd, result.grad_Mu[(i, j)], &format!("Mu {} {}", i, j));
        }
        for (site, a) in [(0, 0), (3, 7), (6, 19)] {
            let (mut plus, mut minus) = (log_pi.clone(), log_pi.clone());
            plus[site][a] += h;
            minus[site][a] -= h;
            let fd = (run(scaling, site_rates, &Mu, &plus, &branch_lengths).log_likelihood
                - run(scaling, site_rates, &Mu, &minus, &branch_lengths).log_likelihood)
                / (2.0 * h);
            check(fd, result.grad_log_pi[site][a], &format!("log_pi {} {}", site, a));
        }
        for k in 0..7 {
            let (mut plus, mut minus) = (branch_lengths.clone(), branch_lengths.clone());
            plus[k] += h;
            minus[k] -= h;
            let fd = (run(scaling, site_rates, &Mu, &log_pi, &plus).log_likelihood
                - run(scaling, site_rates, &Mu, &log_pi, &minus).log_likelihood)
                / (2.0 * h);
            check(fd, result.grad_branch_lengths[k], &format!("branch length {}", k));
        }
    }

    #[test]
    fn mean_rate_gradients_match_finite_differences() {
        check_gradients(RateScaling::MeanRate, None);
    }

    #[test]
    fn fixed_scaling_gradients_match_finite_differences() {
        let rates: Vec<f64> = (0..NUM_SITES).map(|site| 0.5 + 0.3 * site as f64).collect();
        check_gradients(RateScaling::Fixed(1.3), Some(&rates));
    }

    #[test]
    fn site_rates_scale_the_rate_matrices() {
        let (Mu, log_pi, branch_lengths) = (test_Mu(), log_pi(), branch_lengths());
        let run = |scale: f64, rates: Option<&[f64]>| run(RateScaling::Fixed(scale), rates, &Mu, &log_pi, &branch_lengths);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * (1.0 + b.abs());

        // The same rate at every site is the same as a larger rate scaling
        let equal = run(1.3, Some(&[0.7; NUM_SITES]));
        let scaled = run(1.3 * 0.7, None);
        assert!(close(equal.log_likelihood, scaled.log_likelihood));
        assert!((equal.grad_Mu - scaled.grad_Mu).abs().max() <= 1e-10 * (1.0 + scaled.grad_Mu.abs().max()));
        for (a, b) in equal.grad_branch_lengths.iter().zip(&scaled.grad_branch_lengths) {
            assert!(close(*a, *b), "{} vs {}", a, b);
        }

        // Different rates: the gradient of every site is the one with that site's rate as the scaling
        let rates: Vec<f64> = (0..NUM_SITES).map(|site| 0.5 + 0.3 * site as f64).collect();
        let with_rates = run(1.3, Some(&rates));
        for site in 0..NUM_SITES {
            let alone = run(1.3 * rates[site], None);
            assert!(
                (with_rates.grad_log_pi[site] - alone.grad_log_pi[site]).abs().max() <= 1e-10 * (1.0 + alone.grad_log_pi[site].abs().max()),
                "site {}",
                site
            );
        }
    }
}
