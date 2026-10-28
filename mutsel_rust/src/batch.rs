//! Log likelihood of all sites and its gradients w.r.t. Mu, the log site frequencies and the branch lengths.
//!
//! Same model as model::calc_rate_matrix followed by the FelsensteinWithEdgeOp, but the per-site model
//! computations (S, sqrt_pi, the rate normalization and the backpropagation) run in parallel over the
//! sites instead of as candle tensor operations, which are single threaded. phylo_grad parallelizes the
//! pruning itself. The results enter candle graphs through linear_surrogate.

use std::sync::Mutex;

use candle_core::Tensor;
use phylo_grad::{
    FelsensteinTree,
    nalgebra::{SMatrix, SVector},
};
use rayon::prelude::*;

use crate::hmc::{dx_log_fixation, fixation};

/// Factor the rate matrices S of all sites are multiplied with
#[derive(Debug, Clone, Copy)]
pub enum RateScaling {
    Fixed(f64),
    /// 1 / the mean substitution rate over the sites (optimization::ModelParameters)
    MeanRate,
}

pub struct BatchGradients {
    pub log_likelihood: f64,
    pub grad_Mu: SMatrix<f64, 20, 20>,
    /// Gradient w.r.t. the unnormalized log frequencies of every site
    pub grad_log_pi: Vec<SVector<f64, 20>>,
    pub grad_branch_lengths: Vec<f64>,
}

struct SiteModel {
    pi: SVector<f64, 20>,
    sqrt_pi: SVector<f64, 20>,
    fitness: SVector<f64, 20>,
    /// S_ij = sqrt(pi_i / pi_j) * Mu_ij * g(F_j - F_i) before the rate scaling, both triangles
    S: SMatrix<f64, 20, 20>,
    /// Substitution rate sum_{i != j} S_ij sqrt_pi_i sqrt_pi_j
    rate: f64,
}

fn site_model(Mu: &SMatrix<f64, 20, 20>, log_mutation_equilibrium: &SVector<f64, 20>, log_pi: &SVector<f64, 20>) -> SiteModel {
    let max_log_pi = log_pi.max();
    let unnormalized_pi = log_pi.map(|l| (l - max_log_pi).exp());
    let pi = unnormalized_pi / unnormalized_pi.sum();
    let sqrt_pi = pi.map(f64::sqrt);
    let fitness = log_pi - log_mutation_equilibrium;
    let mut S = SMatrix::<f64, 20, 20>::zeros();
    let mut rate = 0.0;
    for i in 0..20 {
        for j in 0..20 {
            if i != j {
                S[(i, j)] = Mu[(i, j)] * (0.5 * (log_pi[i] - log_pi[j])).exp() * fixation(fitness[j] - fitness[i]);
                rate += S[(i, j)] * sqrt_pi[i] * sqrt_pi[j];
            }
        }
    }
    SiteModel { pi, sqrt_pi, fitness, S, rate }
}

/// Summed log likelihood of all sites bound to the tree and its gradients. log_pi [num_sites] are the log
/// frequencies up to a constant per site, branch_lengths in the node order of the tree's parents.
pub fn log_likelihood_and_gradients(
    tree: &Mutex<FelsensteinTree<20>>,
    Mu: &SMatrix<f64, 20, 20>,
    log_pi: &[SVector<f64, 20>],
    branch_lengths: &[f64],
    scaling: RateScaling,
) -> BatchGradients {
    let num_sites = log_pi.len();
    let log_mutation_equilibrium = Mu.diagonal().map(f64::ln);
    let models: Vec<SiteModel> = log_pi
        .par_iter()
        .map(|lp| site_model(Mu, &log_mutation_equilibrium, lp))
        .collect();
    let scale = match scaling {
        RateScaling::Fixed(scale) => scale,
        RateScaling::MeanRate => num_sites as f64 / models.par_iter().map(|m| m.rate).sum::<f64>(),
    };
    let S: Vec<SMatrix<f64, 20, 20>> = models.par_iter().map(|m| m.S * scale).collect();
    let sqrt_pi: Vec<SVector<f64, 20>> = models.iter().map(|m| m.sqrt_pi).collect();

    let result = {
        let mut tree = tree.lock().unwrap();
        assert_eq!(tree.num_sites(), num_sites);
        tree.calculate_gradients_with_branch_lengths_sum(&S, &sqrt_pi, branch_lengths)
    };
    drop(S);

    // With MeanRate scale = num_sites / sum_s rate_s, so the log likelihood depends on every rate_s with
    // d log_likelihood / d rate_s = -scale^2 / num_sites * sum_s sum_{i<j} grad_S_ij S_ij (before scaling)
    let grad_rate = match scaling {
        RateScaling::Fixed(_) => 0.0,
        RateScaling::MeanRate => {
            let d_scale: f64 = models
                .par_iter()
                .zip(result.grad_s.par_iter())
                .map(|(m, g)| {
                    let mut sum = 0.0;
                    for i in 0..20 {
                        for j in (i + 1)..20 {
                            sum += g[(i, j)] * m.S[(i, j)];
                        }
                    }
                    sum
                })
                .sum();
            -scale * scale / num_sites as f64 * d_scale
        }
    };

    let (grad_log_pi, grad_Mu): (Vec<SVector<f64, 20>>, Vec<SMatrix<f64, 20, 20>>) = models
        .par_iter()
        .zip(result.grad_s.par_iter())
        .zip(result.grad_sqrt_pi.par_iter())
        .map(|((m, grad_S), grad_sqrt_pi)| {
            // Gradient w.r.t. the sqrt_pi given to phylo_grad and through the site's substitution rate
            let mut grad_sqrt_pi = *grad_sqrt_pi;
            let mut grad_log_pi = SVector::<f64, 20>::zeros();
            let mut grad_fitness = SVector::<f64, 20>::zeros();
            let mut grad_Mu = SMatrix::<f64, 20, 20>::zeros();
            for i in 0..20 {
                for j in 0..20 {
                    if i == j {
                        continue;
                    }
                    let mut grad = grad_rate * m.sqrt_pi[i] * m.sqrt_pi[j];
                    if i < j {
                        grad += scale * grad_S[(i, j)];
                    }
                    grad_sqrt_pi[i] += grad_rate * m.S[(i, j)] * m.sqrt_pi[j];
                    grad_sqrt_pi[j] += grad_rate * m.S[(i, j)] * m.sqrt_pi[i];
                    let w = grad * m.S[(i, j)];
                    if w == 0.0 {
                        continue;
                    }
                    let dl = w * dx_log_fixation(m.fitness[j] - m.fitness[i]);
                    grad_log_pi[i] += 0.5 * w;
                    grad_log_pi[j] -= 0.5 * w;
                    grad_fitness[j] += dl;
                    grad_fitness[i] -= dl;
                    grad_Mu[(i, j)] += w / Mu[(i, j)];
                }
            }
            // fitness = log_pi - log diag(Mu)
            grad_log_pi += grad_fitness;
            for k in 0..20 {
                grad_Mu[(k, k)] -= grad_fitness[k] / Mu[(k, k)];
            }
            // d sqrt_pi_a / d log_pi_b = sqrt_pi_a * (delta_ab - pi_b) / 2
            let weighted = grad_sqrt_pi.component_mul(&m.sqrt_pi);
            grad_log_pi += (weighted - m.pi * weighted.sum()) * 0.5;
            (grad_log_pi, grad_Mu)
        })
        .unzip();

    BatchGradients {
        log_likelihood: result.log_likelihood.iter().sum(),
        grad_Mu: grad_Mu.into_par_iter().reduce(SMatrix::<f64, 20, 20>::zeros, |a, b| a + b),
        grad_log_pi,
        grad_branch_lengths: result.grad_tree,
    }
}

/// Scalar tensor with the value `value` and the gradient `gradient` w.r.t. `tensor` for every pair, for
/// gradients computed outside of candle: value + sum <tensor, gradient> - the same sum detached
pub fn linear_surrogate(value: f64, terms: &[(&Tensor, Tensor)]) -> Result<Tensor, candle_core::Error> {
    let mut sum = crate::utils::tensor_full(0.0, &[]);
    for (tensor, gradient) in terms {
        sum = (sum + tensor.mul(gradient)?.sum_all()?)?;
    }
    (&sum - sum.detach())? + value
}

pub fn smatrix_from_tensor(t: &Tensor) -> Result<SMatrix<f64, 20, 20>, candle_core::Error> {
    Ok(SMatrix::<f64, 20, 20>::from_row_slice(&t.flatten_all()?.to_vec1::<f64>()?))
}

pub fn smatrix_to_tensor(m: &SMatrix<f64, 20, 20>) -> Result<Tensor, candle_core::Error> {
    Tensor::from_vec(m.transpose().as_slice().to_vec(), &[20, 20], &candle_core::Device::Cpu)
}

/// [n, 20] tensor to one vector per row
pub fn rows_from_tensor(t: &Tensor) -> Result<Vec<SVector<f64, 20>>, candle_core::Error> {
    let values = t.flatten_all()?.to_vec1::<f64>()?;
    Ok(values.chunks_exact(20).map(SVector::<f64, 20>::from_column_slice).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_core::Var;

    use super::*;
    use crate::{
        felsenstein::FelsensteinOp,
        model::calc_rate_matrix,
        optimization::{ModelParameters, Mu},
        pca::PCA,
        utils::tensor_full,
    };

    const NUM_SITES: usize = 7;
    const NUM_LEAVES: usize = 5;

    /// ((0,1),(2,3)) plus a fifth leaf at the root, NUM_SITES columns with a gap
    fn test_op() -> FelsensteinOp {
        let parents = [5, 5, 6, 6, 7, 7, 7, -1];
        let branch_lengths = [0.1, 0.3, 0.2, 0.05, 0.4, 0.15, 0.25, 0.0];
        let alignment: Vec<u8> = (0..NUM_SITES * NUM_LEAVES).map(|i| ((i * 7 + i / 5) % 21) as u8).collect();
        let tree = crate::create_felsenstein_tree(&parents, &branch_lengths, &alignment, NUM_SITES, NUM_LEAVES);
        FelsensteinOp::new(Arc::new(Mutex::new(tree)))
    }

    fn log_R() -> Var {
        let init_R = crate::data::load_lower_R_with_equi(crate::data::M_TXT);
        Var::from_tensor(&init_R.log().unwrap()).unwrap()
    }

    fn log_branch_lengths() -> Var {
        let values: Vec<f64> = (0..8).map(|i| (0.05 + 0.07 * i as f64).ln()).collect();
        Var::from_slice(&values, &[8], &candle_core::Device::Cpu).unwrap()
    }

    fn pca_coordinates() -> Var {
        let pca = PCA::new(19);
        let mean = pca.mean.to_vec1::<f64>().unwrap();
        let values: Vec<f64> = (0..NUM_SITES * 19)
            .map(|i| mean[i % 19] + 0.4 * ((i as f64) * 1.3).sin())
            .collect();
        Var::from_slice(&values, &[NUM_SITES, 19], &candle_core::Device::Cpu).unwrap()
    }

    fn assert_close(a: &Tensor, b: &Tensor, name: &str) {
        let a = a.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!((x - y).abs() <= 1e-10 * (1.0 + scale), "{} entry {}: {} vs {}", name, i, x, y);
        }
    }

    #[test]
    fn map_likelihood_matches_candle() {
        let model = ModelParameters {
            felsenstein_op: test_op().into_with_edge_op(),
            log_R: log_R(),
            pca_coordinates: pca_coordinates(),
            log_branch_lengths: log_branch_lengths(),
            init_log_branch_lengths: log_branch_lengths().as_tensor().copy().unwrap(),
            reg_para: crate::MutselParams { pi_reg: 0.32, Mu_reg: 9.67, branch_length_reg: 0.0 },
            init_log_R: log_R().as_tensor().copy().unwrap(),
            pca_data: PCA::new(19),
        };
        let reference = model.likelihood_candle();
        let parallel = model.likelihood_parallel().unwrap();
        assert_close(&parallel, &reference, "log likelihood");
        let reference_grads = reference.backward().unwrap();
        let parallel_grads = parallel.backward().unwrap();
        for (v, name) in [(&model.log_R, "log_R"), (&model.pca_coordinates, "pca_coordinates"), (&model.log_branch_lengths, "log_branch_lengths")] {
            assert_close(
                parallel_grads.get(v.as_tensor()).unwrap(),
                reference_grads.get(v.as_tensor()).unwrap(),
                name,
            );
        }
    }

    #[test]
    fn fixed_scaling_matches_candle() {
        let op = test_op().into_with_edge_op();
        let log_R = log_R();
        let log_branch_lengths = log_branch_lengths();
        let log_pi = Var::from_tensor(&PCA::new(19).pca_coordinates_to_log_freq(pca_coordinates().as_tensor())).unwrap();

        let (S, sqrt_pi) = calc_rate_matrix(&Mu(&log_R), log_pi.as_tensor(), &tensor_full(1.3, &[]));
        let reference = S.apply_op3(&sqrt_pi, &log_branch_lengths.exp().unwrap(), op.clone()).unwrap();
        let reference_grads = reference.backward().unwrap();

        let Mu_tensor = Mu(&log_R);
        let branch_lengths = log_branch_lengths.exp().unwrap();
        let result = log_likelihood_and_gradients(
            op.tree(),
            &smatrix_from_tensor(&Mu_tensor).unwrap(),
            &rows_from_tensor(log_pi.as_tensor()).unwrap(),
            &branch_lengths.to_vec1::<f64>().unwrap(),
            RateScaling::Fixed(1.3),
        );
        let grad_log_pi: Vec<f64> = result.grad_log_pi.iter().flat_map(|g| g.iter().copied()).collect();
        assert_close(
            &Tensor::from_vec(grad_log_pi, &[NUM_SITES, 20], &candle_core::Device::Cpu).unwrap(),
            reference_grads.get(log_pi.as_tensor()).unwrap(),
            "log_pi",
        );
        let parallel = linear_surrogate(
            result.log_likelihood,
            &[
                (&Mu_tensor, smatrix_to_tensor(&result.grad_Mu).unwrap()),
                (&branch_lengths, Tensor::from_vec(result.grad_branch_lengths, &[8], &candle_core::Device::Cpu).unwrap()),
            ],
        )
        .unwrap();
        assert_close(&parallel, &reference, "log likelihood");
        let parallel_grads = parallel.backward().unwrap();
        for (v, name) in [(&log_R, "log_R"), (&log_branch_lengths, "log_branch_lengths")] {
            assert_close(
                parallel_grads.get(v.as_tensor()).unwrap(),
                reference_grads.get(v.as_tensor()).unwrap(),
                name,
            );
        }
    }
}
