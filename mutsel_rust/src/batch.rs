//! Log likelihood of all sites and its gradients w.r.t. Mu, the log site frequencies, the log site rates and the
//! branch lengths, with the rate matrices normalized to a mean substitution rate of 1 over the sites.
//!
//! The per-site model computations (S, sqrt_pi, the rate normalization and the backpropagation) run in
//! parallel over the sites, phylo_grad parallelizes the pruning itself. All sums over the sites are
//! deterministic (deterministic_sum), so the results do not depend on the number of threads or their scheduling.

use phylo_grad::nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::model::{rate_matrix, rate_matrix_backward, substitution_rate};

/// Sites per chunk of deterministic_sum
const SUM_CHUNK: usize = 256;

/// Sum over the sites that does not depend on the threads: fixed chunks summed sequentially, in parallel, and the
/// chunk sums added in order (rayon's sum and reduce split the work depending on the scheduling, so their
/// floating-point results can differ between runs).
pub fn deterministic_sum<T, F>(num_sites: usize, zero: T, term: F) -> T
where
    T: Send + Sync + Clone + std::ops::Add<Output = T>,
    F: Fn(usize) -> T + Sync,
{
    let chunk_sums: Vec<T> = (0..num_sites.div_ceil(SUM_CHUNK))
        .into_par_iter()
        .map(|chunk| {
            (chunk * SUM_CHUNK..((chunk + 1) * SUM_CHUNK).min(num_sites)).fold(zero.clone(), |acc, site| acc + term(site))
        })
        .collect();
    chunk_sums.into_iter().fold(zero, |acc, x| acc + x)
}

pub struct BatchGradients {
    pub log_likelihood: f64,
    /// Gradient w.r.t. the upper triangle of Mu
    pub grad_Mu: SMatrix<f64, 20, 20>,
    /// Gradient w.r.t. the unnormalized log frequencies of every site
    pub grad_log_pi: Vec<SVector<f64, 20>>,
    pub grad_branch_lengths: Vec<f64>,
    /// Gradient w.r.t. the log site rates (0 without site rates)
    pub grad_log_site_rates: Vec<f64>,
}

/// Sum of the site weights (the number of sites without weights)
pub fn total_weight(num_sites: usize, site_weights: Option<&[f64]>) -> f64 {
    site_weights.map_or(num_sites as f64, |w| deterministic_sum(w.len(), 0.0, |site| w[site]))
}

/// Summed log likelihood of all sites and its gradients. parents is topologically sorted (see lib.rs), columns
/// [num_sites] holds the leaf states of every site, log_pi [num_sites] are the log frequencies up to a constant
/// per site, branch_lengths in the node order of parents. site_rates [num_sites] (None for 1) multiply the rate
/// matrices of the sites, and all are scaled to a mean substitution rate of 1: the rate matrices are
/// r_s S_s W / sum_t w_t r_t rate_t, rate_t the substitution rate of S_t. site_weights [num_sites] multiply the log
/// likelihood of every site (a pattern standing for that many identical alignment columns) and its weight in the
/// mean rate, None for 1.
pub fn log_likelihood_and_gradients(
    parents: &[i32],
    columns: &[&[u8]],
    Mu: &SMatrix<f64, 20, 20>,
    log_pi: &[SVector<f64, 20>],
    branch_lengths: &[f64],
    site_rates: Option<&[f64]>,
    site_weights: Option<&[f64]>,
) -> BatchGradients {
    let num_sites = log_pi.len();
    assert_eq!(columns.len(), num_sites);
    if let Some(weights) = site_weights {
        assert_eq!(weights.len(), num_sites);
    }
    let weight = |site: usize| site_weights.map_or(1.0, |w| w[site]);
    let total = total_weight(num_sites, site_weights);
    if let Some(rates) = site_rates {
        assert_eq!(rates.len(), num_sites);
    }
    let site_rate = |site: usize| site_rates.map_or(1.0, |rates| rates[site]);
    // S and sqrt_pi of every site before the scaling
    let models: Vec<(SMatrix<f64, 20, 20>, SVector<f64, 20>)> = log_pi.par_iter().map(|lp| rate_matrix(Mu, lp)).collect();
    let scale = total
        / deterministic_sum(num_sites, 0.0, |site| weight(site) * site_rate(site) * substitution_rate(&models[site].0, &models[site].1));
    let S: Vec<SMatrix<f64, 20, 20>> =
        models.par_iter().enumerate().map(|(site, (S, _))| S * (scale * site_rate(site))).collect();
    let sqrt_pi: Vec<SVector<f64, 20>> = models.iter().map(|(_, sqrt_pi)| *sqrt_pi).collect();

    // The branch length gradients per site, summed here in a fixed order (phylo_grad's own sum over the sites
    // depends on the scheduling of the threads)
    let num_nodes = branch_lengths.len();
    let mut grad_per_site = vec![0.0; num_sites * num_nodes];
    let mut grad_slices: Vec<&mut [f64]> = grad_per_site.chunks_mut(num_nodes).collect();
    let result = phylo_grad::calculate_gradients_columns(parents, columns, branch_lengths, &S, &sqrt_pi, Some(&mut grad_slices[..]));
    drop(grad_slices);
    drop(S);
    let chunk_sums: Vec<Vec<f64>> = grad_per_site
        .par_chunks(SUM_CHUNK * num_nodes)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let mut sum = vec![0.0; num_nodes];
            for (grad, site) in chunk.chunks_exact(num_nodes).zip(chunk_index * SUM_CHUNK..) {
                let w = weight(site);
                sum.iter_mut().zip(grad).for_each(|(s, g)| *s += w * g);
            }
            sum
        })
        .collect();
    let mut grad_branch_lengths = vec![0.0; num_nodes];
    for sum in &chunk_sums {
        grad_branch_lengths.iter_mut().zip(sum).for_each(|(s, g)| *s += g);
    }

    // G_s = <grad_S_s, S_s> with S before the scaling, the derivative of the log likelihood of site s w.r.t. its total
    // factor scale * r_s
    let G: Vec<f64> = (0..num_sites)
        .into_par_iter()
        .map(|site| result.grad_s[site].upper_triangle().component_mul(&models[site].0).sum())
        .collect();
    // scale = W / sum_s w_s r_s rate_s (W = sum_s w_s), so the log likelihood depends on every rate_s with
    // d log_likelihood / d rate_s = -scale^2 w_s r_s / W * sum_t w_t r_t G_t; grad_rate is that without the factor
    // w_s r_s
    let grad_rate = -scale * scale / total * deterministic_sum(num_sites, 0.0, |site| weight(site) * site_rate(site) * G[site]);
    // d log_likelihood / d log r_s = w_s r_s (scale G_s + grad_rate rate_s)
    let grad_log_site_rates: Vec<f64> = match site_rates {
        None => vec![0.0; num_sites],
        Some(_) => (0..num_sites)
            .into_par_iter()
            .map(|site| {
                let rate = substitution_rate(&models[site].0, &models[site].1);
                weight(site) * site_rate(site) * (scale * G[site] + grad_rate * rate)
            })
            .collect(),
    };

    let (grad_log_pi, grad_Mu): (Vec<SVector<f64, 20>>, Vec<SMatrix<f64, 20, 20>>) = models
        .par_iter()
        .zip(log_pi.par_iter())
        .zip(result.grad_s.par_iter().zip(result.grad_sqrt_pi.par_iter()))
        .enumerate()
        .map(|(site, (((S, sqrt_pi), log_pi), (grad_S, grad_sqrt_pi)))| {
            // phylo_grad only reads the upper triangle of S. rate = sum_{i != j} S_ij sqrt_pi_i sqrt_pi_j
            let (w, grad_rate) = (weight(site), weight(site) * site_rate(site) * grad_rate);
            let grad_S = grad_S.upper_triangle() * (w * scale * site_rate(site)) + sqrt_pi * sqrt_pi.transpose() * grad_rate;
            let grad_sqrt_pi = grad_sqrt_pi * w + S * sqrt_pi * (2.0 * grad_rate);
            let (grad_Mu, grad_log_pi) = rate_matrix_backward(log_pi, S, sqrt_pi, Mu, &grad_S, &grad_sqrt_pi);
            (grad_log_pi, grad_Mu)
        })
        .unzip();

    BatchGradients {
        log_likelihood: result.log_likelihood.iter().enumerate().map(|(site, ll)| weight(site) * ll).sum(),
        grad_Mu: deterministic_sum(num_sites, SMatrix::<f64, 20, 20>::zeros(), |site| grad_Mu[site]),
        grad_log_pi,
        grad_branch_lengths,
        grad_log_site_rates,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{optimization::Mu, pca};

    pub(crate) const NUM_SITES: usize = 7;
    pub(crate) const NUM_LEAVES: usize = 5;
    /// ((0,1),(2,3)) plus a fifth leaf at the root
    pub(crate) const PARENTS: [i32; 8] = [5, 5, 6, 6, 7, 7, 7, -1];

    /// NUM_SITES columns, some with gaps
    pub(crate) fn alignment() -> Vec<u8> {
        (0..NUM_SITES * NUM_LEAVES).map(|i| ((i * 7 + i / 5) % 21) as u8).collect()
    }

    pub(crate) fn test_Mu() -> SMatrix<f64, 20, 20> {
        Mu(&crate::prior::tests::test_log_exchangeabilities())
    }

    pub(crate) fn branch_lengths() -> Vec<f64> {
        (0..8).map(|i| 0.05 + 0.07 * i as f64).collect()
    }

    pub(crate) fn log_pi() -> Vec<SVector<f64, 20>> {
        (0..NUM_SITES)
            .map(|site| {
                let x = SVector::from_fn(|k, _| 0.4 * (((site * 19 + k) as f64) * 1.3).sin());
                pca::coordinates_to_log_freq(&x)
            })
            .collect()
    }

    fn run(site_rates: Option<&[f64]>, Mu: &SMatrix<f64, 20, 20>, log_pi: &[SVector<f64, 20>], branch_lengths: &[f64]) -> BatchGradients {
        let alignment = alignment();
        let columns: Vec<&[u8]> = alignment.chunks_exact(NUM_LEAVES).collect();
        log_likelihood_and_gradients(&PARENTS, &columns, Mu, log_pi, branch_lengths, site_rates, None)
    }

    /// The gradients w.r.t. the off-diagonal Mu, log_pi and the branch lengths against central differences
    fn check_gradients(site_rates: Option<&[f64]>) {
        let (Mu, log_pi, branch_lengths) = (test_Mu(), log_pi(), branch_lengths());
        let result = run(site_rates, &Mu, &log_pi, &branch_lengths);
        let h = 1e-6;
        let check = |numeric: f64, analytic: f64, name: &str| {
            assert!((numeric - analytic).abs() < 1e-5 * (1.0 + analytic.abs()), "{}: {} vs {}", name, numeric, analytic);
        };
        for (i, j) in [(0, 1), (4, 17), (3, 19)] {
            let (mut plus, mut minus) = (Mu, Mu);
            plus[(i, j)] += h;
            minus[(i, j)] -= h;
            let fd = (run(site_rates, &plus, &log_pi, &branch_lengths).log_likelihood
                - run(site_rates, &minus, &log_pi, &branch_lengths).log_likelihood)
                / (2.0 * h);
            check(fd, result.grad_Mu[(i, j)], &format!("Mu {} {}", i, j));
        }
        for (site, a) in [(0, 0), (3, 7), (6, 19)] {
            let (mut plus, mut minus) = (log_pi.clone(), log_pi.clone());
            plus[site][a] += h;
            minus[site][a] -= h;
            let fd = (run(site_rates, &Mu, &plus, &branch_lengths).log_likelihood
                - run(site_rates, &Mu, &minus, &branch_lengths).log_likelihood)
                / (2.0 * h);
            check(fd, result.grad_log_pi[site][a], &format!("log_pi {} {}", site, a));
        }
        for k in 0..7 {
            let (mut plus, mut minus) = (branch_lengths.clone(), branch_lengths.clone());
            plus[k] += h;
            minus[k] -= h;
            let fd = (run(site_rates, &Mu, &log_pi, &plus).log_likelihood
                - run(site_rates, &Mu, &log_pi, &minus).log_likelihood)
                / (2.0 * h);
            check(fd, result.grad_branch_lengths[k], &format!("branch length {}", k));
        }
        if let Some(rates) = site_rates {
            for site in [0, 2, 6] {
                let (mut plus, mut minus) = (rates.to_vec(), rates.to_vec());
                plus[site] *= h.exp();
                minus[site] *= (-h).exp();
                let fd = (run(Some(&plus), &Mu, &log_pi, &branch_lengths).log_likelihood
                    - run(Some(&minus), &Mu, &log_pi, &branch_lengths).log_likelihood)
                    / (2.0 * h);
                check(fd, result.grad_log_site_rates[site], &format!("log site rate {}", site));
            }
        }
    }

    #[test]
    fn gradients_match_finite_differences() {
        check_gradients(None);
    }

    #[test]
    fn gradients_with_site_rates_match_finite_differences() {
        let rates: Vec<f64> = (0..NUM_SITES).map(|site| 0.5 + 0.3 * site as f64).collect();
        check_gradients(Some(&rates));
    }

    /// The rate matrices are normalized, so the same rate at every site changes nothing
    #[test]
    fn equal_site_rates_change_nothing() {
        let (Mu, log_pi, branch_lengths) = (test_Mu(), log_pi(), branch_lengths());
        let equal = run(Some(&[0.7; NUM_SITES]), &Mu, &log_pi, &branch_lengths);
        let plain = run(None, &Mu, &log_pi, &branch_lengths);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * (1.0 + b.abs());
        assert!(close(equal.log_likelihood, plain.log_likelihood));
        assert!((equal.grad_Mu - plain.grad_Mu).abs().max() <= 1e-10 * (1.0 + plain.grad_Mu.abs().max()));
        for (a, b) in equal.grad_branch_lengths.iter().zip(&plain.grad_branch_lengths) {
            assert!(close(*a, *b), "{} vs {}", a, b);
        }
        // and the gradient w.r.t. the log site rates sums to 0
        assert!(equal.grad_log_site_rates.iter().sum::<f64>().abs() < 1e-8 * equal.grad_log_site_rates.iter().map(|g| g.abs()).sum::<f64>());
    }

    /// Bitwise the same results with 1, 3 and 8 threads (several chunks of deterministic_sum)
    #[test]
    fn results_do_not_depend_on_the_threads() {
        let num_sites = 3 * SUM_CHUNK + 17;
        let alignment: Vec<u8> = (0..num_sites * NUM_LEAVES).map(|i| ((i * 7 + i / 5 + i / 37) % 21) as u8).collect();
        let columns: Vec<&[u8]> = alignment.chunks_exact(NUM_LEAVES).collect();
        let log_pi: Vec<SVector<f64, 20>> = (0..num_sites)
            .map(|site| pca::coordinates_to_log_freq(&SVector::from_fn(|k, _| 0.4 * (((site * 19 + k) as f64) * 1.3).sin())))
            .collect();
        let rates: Vec<f64> = (0..num_sites).map(|site| 0.5 + (site % 11) as f64 * 0.1).collect();
        let (Mu, branch_lengths) = (test_Mu(), branch_lengths());
        let run = |threads: usize, rates: Option<&[f64]>| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| log_likelihood_and_gradients(&PARENTS, &columns, &Mu, &log_pi, &branch_lengths, rates, None))
        };
        for rates in [None, Some(&rates[..])] {
            let reference = run(1, rates);
            for threads in [3, 8] {
                let result = run(threads, rates);
                assert_eq!(result.log_likelihood.to_bits(), reference.log_likelihood.to_bits());
                assert_eq!(result.grad_Mu, reference.grad_Mu);
                assert_eq!(result.grad_log_pi, reference.grad_log_pi);
                assert_eq!(result.grad_branch_lengths, reference.grad_branch_lengths);
                assert_eq!(result.grad_log_site_rates, reference.grad_log_site_rates);
            }
        }
    }

    /// A pattern with weight c is the same as c identical columns (with and without site rates)
    #[test]
    fn site_weights_count_identical_columns() {
        let alignment = alignment();
        let columns: Vec<&[u8]> = alignment.chunks_exact(NUM_LEAVES).collect();
        let (Mu, log_pi, branch_lengths) = (test_Mu(), log_pi(), branch_lengths());
        let counts = [1usize, 3, 1, 2, 1, 1, 4];
        let rates: Vec<f64> = (0..NUM_SITES).map(|site| 0.5 + 0.3 * site as f64).collect();
        let expand = |n: usize| -> Vec<usize> { (0..n).flat_map(|site| std::iter::repeat(site).take(counts[site])).collect() };
        let index = expand(NUM_SITES);
        let full_columns: Vec<&[u8]> = index.iter().map(|&site| columns[site]).collect();
        let full_log_pi: Vec<SVector<f64, 20>> = index.iter().map(|&site| log_pi[site]).collect();
        let full_rates: Vec<f64> = index.iter().map(|&site| rates[site]).collect();
        let weights: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + b.abs());
        for with_rates in [false, true] {
            let weighted = log_likelihood_and_gradients(
                &PARENTS, &columns, &Mu, &log_pi, &branch_lengths, with_rates.then_some(&rates[..]), Some(&weights),
            );
            let full = log_likelihood_and_gradients(
                &PARENTS, &full_columns, &Mu, &full_log_pi, &branch_lengths, with_rates.then_some(&full_rates[..]), None,
            );
            assert!(close(weighted.log_likelihood, full.log_likelihood), "{} vs {}", weighted.log_likelihood, full.log_likelihood);
            assert!((weighted.grad_Mu - full.grad_Mu).abs().max() <= 1e-9 * (1.0 + full.grad_Mu.abs().max()));
            for (a, b) in weighted.grad_branch_lengths.iter().zip(&full.grad_branch_lengths) {
                assert!(close(*a, *b), "{} vs {}", a, b);
            }
            // the gradient of a pattern is the sum over its columns
            for site in 0..NUM_SITES {
                let sum: SVector<f64, 20> = index.iter().zip(&full.grad_log_pi).filter(|(s, _)| **s == site).map(|(_, g)| *g).sum();
                assert!((weighted.grad_log_pi[site] - sum).abs().max() <= 1e-9 * (1.0 + sum.abs().max()), "site {}", site);
                if with_rates {
                    let sum: f64 = index.iter().zip(&full.grad_log_site_rates).filter(|(s, _)| **s == site).map(|(_, g)| g).sum();
                    assert!(close(weighted.grad_log_site_rates[site], sum), "site {}", site);
                }
            }
        }
    }
}
