//! The parametrization of Mu and the light PMSF initialization of the MAP estimate (see gen_normal.rs).
//!
//! Mu is parametrized by the log exchangeabilities log S_ij of the entries (i, j), i > j, in row-major order,
//! with the mutation equilibrium fixed to data::M_EQUI.

use phylo_grad::nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::{batch, data::M_EQUI, lbfgs::maximize_finite};

/// Number of log exchangeabilities
pub const NUM_EXCHANGEABILITIES: usize = 190;

/// The entries (i, j), i > j, of the log exchangeabilities in row-major order
fn lower_indices() -> impl Iterator<Item = (usize, usize)> {
    (1..20).flat_map(|i| (0..i).map(move |j| (i, j)))
}

/// Equilibrium pi of the mutation process (M_EQUI normalized), the unnormalized rate matrix Q_ij = S_ij sqrt(pi_j / pi_i)
/// (i != j, S symmetric) and its total rate c = sum_{i != j} pi_i Q_ij
fn Mu_parts(log_S: &[f64]) -> (SVector<f64, 20>, SMatrix<f64, 20, 20>, f64) {
    assert_eq!(log_S.len(), NUM_EXCHANGEABILITIES);
    let pi = SVector::from(M_EQUI) / M_EQUI.iter().sum::<f64>();
    let sqrt_pi = pi.map(f64::sqrt);
    let mut Q = SMatrix::<f64, 20, 20>::zeros();
    for ((i, j), log_S) in lower_indices().zip(log_S) {
        let S = log_S.exp();
        Q[(i, j)] = S * sqrt_pi[j] / sqrt_pi[i];
        Q[(j, i)] = S * sqrt_pi[i] / sqrt_pi[j];
    }
    let c = (0..20).map(|i| pi[i] * Q.row(i).sum()).sum();
    (pi, Q, c)
}

/// The mutation rate matrix Q / c with total rate 1 and zero diagonal
pub fn Mu(log_S: &[f64]) -> SMatrix<f64, 20, 20> {
    let (_, Q, c) = Mu_parts(log_S);
    Q / c
}

/// Gradient w.r.t. the log exchangeabilities given the gradient grad_Mu w.r.t. the off-diagonal entries of Mu
pub fn Mu_backward(log_S: &[f64], grad_Mu: &SMatrix<f64, 20, 20>) -> Vec<f64> {
    let (pi, Q, c) = Mu_parts(log_S);
    let mut grad_c = 0.0;
    for i in 0..20 {
        for j in 0..20 {
            if i != j {
                grad_c -= grad_Mu[(i, j)] * Q[(i, j)] / (c * c);
            }
        }
    }
    // Q_ij enters Mu_ij = Q_ij / c directly and c with weight pi_i
    let grad_Q = |i: usize, j: usize| grad_Mu[(i, j)] / c + pi[i] * grad_c;
    // Q_ij and Q_ji are proportional to S_ij = exp(log_S_ij)
    lower_indices().map(|(i, j)| grad_Q(i, j) * Q[(i, j)] + grad_Q(j, i) * Q[(j, i)]).collect()
}

/// Site frequencies of the light PMSF: the posterior mean over the profile mixture (categories with weights)
/// under Mu, given the tree. The categories are not normalized to a mean substitution rate.
pub fn light_pmsf(
    parents: &[i32],
    columns: &[&[u8]],
    Mu: &SMatrix<f64, 20, 20>,
    categories: &[[f64; 20]],
    weights: &[f64],
    branch_lengths: &[f64],
) -> Vec<SVector<f64, 20>> {
    // log likelihood [category][site] plus the log weight of the category
    let log_joint: Vec<Vec<f64>> = categories
        .iter()
        .zip(weights)
        .map(|(category, weight)| {
            let (S, sqrt_pi) = crate::model::rate_matrix(Mu, &SVector::from(*category).map(f64::ln));
            let result = phylo_grad::calculate_gradients_columns(
                parents,
                columns,
                branch_lengths,
                &[S],
                &[sqrt_pi],
                None::<&mut [Vec<f64>]>,
            );
            result.log_likelihood.iter().map(|ll| ll + weight.ln()).collect()
        })
        .collect();

    (0..columns.len())
        .into_par_iter()
        .map(|site| {
            let max = log_joint.iter().map(|l| l[site]).fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            let mut freq = SVector::<f64, 20>::zeros();
            for (log_joint, category) in log_joint.iter().zip(categories) {
                let posterior = (log_joint[site] - max).exp();
                sum += posterior;
                freq += SVector::from(*category) * posterior;
            }
            freq / sum
        })
        .collect()
}

/// Branch lengths maximizing the log likelihood with Mu and the site frequencies fixed
fn optimize_branch_lengths(
    parents: &[i32],
    columns: &[&[u8]],
    Mu: &SMatrix<f64, 20, 20>,
    log_pi: &[SVector<f64, 20>],
    branch_lengths: &[f64],
    convergence: Convergence,
    site_weights: Option<&[f64]>,
) -> Vec<f64> {
    let mut log_branch_lengths: Vec<f64> = branch_lengths.iter().map(|b| b.ln()).collect();
    maximize_finite(
        |x| {
            let branch_lengths: Vec<f64> = x.iter().map(|l| l.exp()).collect();
            let result = batch::log_likelihood_and_gradients(
                parents,
                columns,
                Mu,
                log_pi,
                &branch_lengths,
                None,
                site_weights,
            );
            let gradient = branch_lengths.iter().zip(&result.grad_branch_lengths).map(|(b, g)| g * b).collect();
            Some((result.log_likelihood, gradient))
        },
        &mut log_branch_lengths,
        convergence.max_iterations,
        convergence.min_rel_improvement,
    );
    log_branch_lengths.iter().map(|l| l.exp()).collect()
}

/// Stopping criteria of the L-BFGS optimizations: at most max_iterations, or until an iteration improves
/// the objective by less than min_rel_improvement relative to its value
#[derive(Debug, Clone, Copy)]
pub struct Convergence {
    pub max_iterations: usize,
    pub min_rel_improvement: f64,
}

impl Convergence {
    /// Criteria of the MAP estimate and of the branch lengths of the light PMSF initialization
    pub const MAP: Convergence = Convergence {
        max_iterations: 500,
        min_rel_improvement: 1e-7,
    };
}

/// The two step light PMSF initialization with the Mu of the log exchangeabilities log_S: light PMSF frequencies on
/// the input tree, the branch lengths optimized for them, light PMSF frequencies on these branch lengths. Returns
/// the site frequencies and the branch lengths.
pub fn light_pmsf_init(
    parents: &[i32],
    columns: &[&[u8]],
    log_S: &[f64],
    branch_lengths: &[f64],
    convergence: Convergence,
    site_weights: Option<&[f64]>,
) -> (Vec<SVector<f64, 20>>, Vec<f64>) {
    let Mu = Mu(log_S);
    let site_freq = light_pmsf(parents, columns, &Mu, crate::data::UDM256, crate::data::UDM256_WEIGHTS, branch_lengths);
    let log_pi: Vec<SVector<f64, 20>> = site_freq.iter().map(|f| f.map(f64::ln)).collect();
    let branch_lengths = optimize_branch_lengths(parents, columns, &Mu, &log_pi, branch_lengths, convergence, site_weights);
    let site_freq = light_pmsf(parents, columns, &Mu, crate::data::UDM256, crate::data::UDM256_WEIGHTS, &branch_lengths);
    (site_freq, branch_lengths)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Log exchangeabilities away from those of the test prior
    fn perturbed_log_S() -> Vec<f64> {
        crate::prior::tests::test_log_exchangeabilities()
            .iter()
            .zip(lower_indices())
            .map(|(l, (i, j))| l + 0.3 * ((i * 20 + j) as f64 * 0.73).sin())
            .collect()
    }

    /// Central differences of f
    fn finite_differences(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
        (0..x.len())
            .map(|k| {
                let (mut plus, mut minus) = (x.to_vec(), x.to_vec());
                plus[k] += h;
                minus[k] -= h;
                (f(&plus) - f(&minus)) / (2.0 * h)
            })
            .collect()
    }

    fn max_difference(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    }

    fn max_abs(a: &[f64]) -> f64 {
        a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
    }

    #[test]
    fn Mu_has_rate_one_and_the_fixed_equilibrium() {
        let Mu = Mu(&perturbed_log_S());
        let pi = SVector::from(M_EQUI) / M_EQUI.iter().sum::<f64>();
        let rate: f64 = (0..20).map(|i| pi[i] * Mu.row(i).sum()).sum();
        assert!((rate - 1.0).abs() < 1e-13);
        assert_eq!(Mu.diagonal(), SVector::<f64, 20>::zeros());
        // Reversible with the equilibrium M_EQUI: pi_i Mu_ij = pi_j Mu_ji
        for (i, j) in lower_indices() {
            assert!((pi[i] * Mu[(i, j)] - pi[j] * Mu[(j, i)]).abs() < 1e-15);
        }
    }

    #[test]
    fn Mu_backward_matches_finite_differences() {
        let log_S = perturbed_log_S();
        let G = SMatrix::<f64, 20, 20>::from_fn(|i, j| ((i * 20 + j) as f64 * 0.37).sin());
        let fd = finite_differences(|log_S| Mu(log_S).component_mul(&G).sum(), &log_S, 1e-6);
        let gradient = Mu_backward(&log_S, &G);
        assert!(max_difference(&fd, &gradient) < 1e-8 * max_abs(&gradient));
    }
}
