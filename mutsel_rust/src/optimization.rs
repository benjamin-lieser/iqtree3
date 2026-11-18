//! The MAP estimate of Mu, the site frequencies and the branch lengths which the Monte Carlo EM starts from
//! (see hmc::fit_with_em), initialized with the light PMSF procedure, and the parametrization of Mu.
//!
//! Mu is parametrized by the log exchangeabilities log S_ij of the entries (i, j), i > j, in row-major order,
//! with the mutation equilibrium fixed to data::M_EQUI.

use phylo_grad::nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::{
    Verbosity, batch,
    data::M_EQUI,
    helmert::{self, NUM_COORDINATES},
    lbfgs::maximize_finite,
    prior::Prior,
};

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

/// Branch lengths maximizing the log likelihood with Mu and the site frequencies fixed (the rate matrices
/// normalized to a mean substitution rate of 1)
fn optimize_branch_lengths(
    parents: &[i32],
    columns: &[&[u8]],
    Mu: &SMatrix<f64, 20, 20>,
    log_pi: &[SVector<f64, 20>],
    branch_lengths: &[f64],
    convergence: Convergence,
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
                batch::RateScaling::MeanRate,
                None,
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
    /// Criteria of the MAP estimate and the branch lengths of the light PMSF initialization
    pub const MAP: Convergence = Convergence {
        max_iterations: 500,
        min_rel_improvement: 1e-7,
    };
}

/// Log posterior of the MAP estimate (log likelihood with the rate matrices normalized to a mean substitution rate
/// of 1 plus the log priors of the exchangeabilities and the site frequencies) and its gradient at
/// x = [log exchangeabilities, Helmert coordinates of every site, log branch lengths]
fn map_objective(parents: &[i32], columns: &[&[u8]], prior: &Prior, x: &[f64]) -> (f64, Vec<f64>) {
    let num_coordinates = columns.len() * NUM_COORDINATES;
    let log_S = &x[..NUM_EXCHANGEABILITIES];
    let coordinates = &x[NUM_EXCHANGEABILITIES..NUM_EXCHANGEABILITIES + num_coordinates];
    let branch_lengths: Vec<f64> = x[NUM_EXCHANGEABILITIES + num_coordinates..].iter().map(|l| l.exp()).collect();
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
        None,
    );

    let (log_prior_S, grad_log_prior_S) = prior.evaluate_exchangeabilities(log_S);
    let mut gradient: Vec<f64> =
        Mu_backward(log_S, &result.grad_Mu).iter().zip(&grad_log_prior_S).map(|(g, p)| g + p).collect();

    let (log_prior_pi, grad_coordinates): (Vec<f64>, Vec<SVector<f64, NUM_COORDINATES>>) = coordinates
        .par_chunks_exact(NUM_COORDINATES)
        .zip(result.grad_log_pi.par_iter())
        .map(|(c, grad_log_pi)| {
            let (log_prior, grad_log_prior) = prior.site_frequencies.evaluate(&SVector::from_column_slice(c));
            (log_prior, basis * grad_log_pi + grad_log_prior)
        })
        .unzip();
    gradient.extend(grad_coordinates.iter().flat_map(|g| g.iter().copied()));
    gradient.extend(branch_lengths.iter().zip(&result.grad_branch_lengths).map(|(b, g)| g * b));

    (result.log_likelihood + log_prior_S + log_prior_pi.iter().sum::<f64>(), gradient)
}

pub struct MapEstimate {
    pub log_S: Vec<f64>,
    pub branch_lengths: Vec<f64>,
    pub coordinates: Vec<SVector<f64, NUM_COORDINATES>>,
}

/// Joint MAP estimate of Mu, the Helmert coordinates of every site and the branch lengths, with the rate matrices
/// normalized to a mean substitution rate of 1 over the sites. Initialized with the prior Mu and the two step
/// light PMSF procedure: light PMSF frequencies on the input tree, the branch lengths optimized for them, light
/// PMSF frequencies on these branch lengths.
pub fn map_estimate(
    parents: &[i32],
    columns: &[&[u8]],
    branch_lengths: &[f64],
    prior: &Prior,
    convergence: Convergence,
    verbosity: Verbosity,
) -> MapEstimate {
    let num_sites = columns.len();
    let prior_Mu = Mu(&prior.log_exchangeabilities);

    let site_freq = light_pmsf(parents, columns, &prior_Mu, crate::data::UDM256, crate::data::UDM256_WEIGHTS, branch_lengths);
    let log_pi: Vec<SVector<f64, 20>> = site_freq.iter().map(|f| f.map(f64::ln)).collect();
    let branch_lengths = optimize_branch_lengths(parents, columns, &prior_Mu, &log_pi, branch_lengths, convergence);
    let site_freq = light_pmsf(parents, columns, &prior_Mu, crate::data::UDM256, crate::data::UDM256_WEIGHTS, &branch_lengths);

    // x = [log exchangeabilities, Helmert coordinates of every site, log branch lengths]
    let num_coordinates = num_sites * NUM_COORDINATES;
    let mut x = prior.log_exchangeabilities.clone();
    x.extend(site_freq.iter().flat_map(|f| helmert::log_freq_to_coordinates(&f.map(f64::ln)).iter().copied().collect::<Vec<_>>()));
    x.extend(branch_lengths.iter().map(|b| b.ln()));

    let start = std::time::Instant::now();
    let num_evaluations = maximize_finite(
        |x| Some(map_objective(parents, columns, prior, x)),
        &mut x,
        convergence.max_iterations,
        convergence.min_rel_improvement,
    );
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL MAP estimate: {} evaluations ({:.1}s)",
            num_evaluations,
            start.elapsed().as_secs_f64()
        );
    }

    MapEstimate {
        log_S: x[..NUM_EXCHANGEABILITIES].to_vec(),
        branch_lengths: x[NUM_EXCHANGEABILITIES + num_coordinates..].iter().map(|l| l.exp()).collect(),
        coordinates: x[NUM_EXCHANGEABILITIES..NUM_EXCHANGEABILITIES + num_coordinates]
            .chunks_exact(NUM_COORDINATES)
            .map(SVector::from_column_slice)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Log exchangeabilities away from those of the test prior
    fn perturbed_log_S() -> Vec<f64> {
        crate::prior::tests::test_prior()
            .log_exchangeabilities
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

    #[test]
    fn map_objective_gradient_matches_finite_differences() {
        use crate::batch::tests::*;
        let alignment = alignment();
        let columns: Vec<&[u8]> = alignment.chunks_exact(NUM_LEAVES).collect();
        let prior = crate::prior::tests::test_prior();
        let mut x = perturbed_log_S();
        x.extend(log_pi().iter().flat_map(|lp| helmert::log_freq_to_coordinates(lp).iter().copied().collect::<Vec<_>>()));
        x.extend(branch_lengths().iter().map(|b| b.ln()));
        let (_, gradient) = map_objective(&PARENTS, &columns, &prior, &x);
        let fd = finite_differences(|x| map_objective(&PARENTS, &columns, &prior, x).0, &x, 1e-5);
        assert!(max_difference(&fd, &gradient) < 1e-5 * max_abs(&gradient), "{}", max_difference(&fd, &gradient));
    }
}
