use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use candle_core::{Tensor, Var};
use candle_nn::{Optimizer, ops::softmax};
use phylo_grad::FelsensteinTree;

use phylo_grad::nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::{
    batch, MutselParams, Verbosity, felsenstein::{self, FelsensteinOp, FelsensteinWithEdgeOp}, model::{self, calc_rate_matrix}, pca::PCA, utils::tensor_full,
};

pub trait Optimizable {
    fn variables(&self) -> Vec<Var>;
    fn variables_names(&self) -> Vec<String>;
    fn model_name(&self) -> String;
    fn likelihood(&self) -> Tensor;
    fn penalty(&self) -> Tensor;
    fn print_state(&self) {}
}

fn calc_likelihood(
    mu: &Tensor,
    log_pi: &Tensor,
    log_branch_lengths: &Tensor,
    felsenstein_op: FelsensteinWithEdgeOp,
) -> Tensor {
    let (S, sqrt_pi) = calc_rate_matrix(mu, log_pi, &tensor_full(1.0, &[]));

    let branch_lengths = log_branch_lengths.exp().unwrap();

    let average_rate = model::substitution_rates_tensor(&S, &sqrt_pi)
        .mean_all()
        .unwrap();
    let S = S.broadcast_div(&average_rate).unwrap();

    S.apply_op3(&sqrt_pi, &branch_lengths, felsenstein_op)
        .unwrap()
        .sum_all()
        .unwrap()
}

pub struct BranchParameters {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    pub log_branch_lengths: Var,
    pub reg_para: MutselParams,
    pub Mu: Tensor,
    pub log_pi: Tensor,
}

impl Optimizable for BranchParameters {
    fn variables(&self) -> Vec<Var> {
        vec![self.log_branch_lengths.clone()]
    }
    fn variables_names(&self) -> Vec<String> {
        vec!["log_branch_lengths".to_string()]
    }
    fn model_name(&self) -> String {
        "BranchParameters".to_string()
    }

    fn likelihood(&self) -> Tensor {
        calc_likelihood(
            &self.Mu,
            &self.log_pi,
            &self.log_branch_lengths,
            self.felsenstein_op.clone(),
        )
    }

    fn penalty(&self) -> Tensor {
        let branch_penalty = (&self.log_branch_lengths.exp().unwrap()).sum_all().unwrap();
        (branch_penalty * self.reg_para.branch_length_reg).unwrap()
    }

    fn print_state(&self) {}
}

pub struct ModelParameters {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    pub log_R: Var,
    /// Defines the log pi per site
    pub pca_coordinates: Var,
    pub log_branch_lengths: Var,
    pub init_log_branch_lengths: Tensor,
    pub reg_para: MutselParams,
    pub init_log_R: Tensor,
    pub pca_data: PCA,
}

impl ModelParameters {
    pub fn log_pi(&self) -> Tensor {
        self.pca_data
            .pca_coordinates_to_log_freq(&self.pca_coordinates)
    }

    pub fn calc_rate_matrix(&self) -> (Tensor, Tensor) {
        let (S, sqrt_pi) =
            calc_rate_matrix(&Mu(&self.log_R), &self.log_pi(), &tensor_full(1.0, &[]));
        let average_rate = model::substitution_rates_tensor(&S, &sqrt_pi)
            .mean_all()
            .unwrap();
        let S = S.broadcast_div(&average_rate).unwrap();
        (S, sqrt_pi)
    }

    /// Same as likelihood_candle, with the per-site computations parallelized over the sites
    pub fn likelihood_parallel(&self) -> Result<Tensor, candle_core::Error> {
        let Mu = Mu(&self.log_R);
        let pca_coordinates = self.pca_coordinates.as_tensor();
        let branch_lengths = self.log_branch_lengths.exp()?;
        let components = batch_components(&self.pca_data)?;
        let coordinates = pca_coordinates.flatten_all()?.to_vec1::<f64>()?;
        let num_components = self.pca_data.num_components;
        let log_pi: Vec<SVector<f64, 20>> = coordinates
            .par_chunks_exact(num_components)
            .map(|x| components.transpose() * SVector::<f64, 19>::from_column_slice(x))
            .collect();

        let result = batch::log_likelihood_and_gradients(
            self.felsenstein_op.tree(),
            &batch::smatrix_from_tensor(&Mu)?,
            &log_pi,
            &branch_lengths.to_vec1::<f64>()?,
            batch::RateScaling::MeanRate,
            None,
        );
        let grad_coordinates: Vec<f64> = result
            .grad_log_pi
            .par_iter()
            .flat_map_iter(|g| (components * g).iter().copied().collect::<Vec<_>>())
            .collect();
        let num_branches = result.grad_branch_lengths.len();
        batch::linear_surrogate(
            result.log_likelihood,
            &[
                (&Mu, batch::smatrix_to_tensor(&result.grad_Mu)?),
                (pca_coordinates, Tensor::from_vec(grad_coordinates, pca_coordinates.shape(), &candle_core::Device::Cpu)?),
                (&branch_lengths, Tensor::from_vec(result.grad_branch_lengths, &[num_branches], &candle_core::Device::Cpu)?),
            ],
        )
    }

    /// Log likelihood as a candle graph, single threaded outside of phylo_grad
    pub fn likelihood_candle(&self) -> Tensor {
        let (S, sqrt_pi) = self.calc_rate_matrix();
        let branch_lengths = self.log_branch_lengths.exp().unwrap();
        S.apply_op3(&sqrt_pi, &branch_lengths, self.felsenstein_op.clone())
            .unwrap()
            .sum_all()
            .unwrap()
    }

    pub fn save_npz(&self, path: &Path) {
        let Mu = Mu(&self.log_R.as_detached_tensor());
        let pi = softmax(&self.log_pi().detach(), 1).unwrap();

        Tensor::write_npz(
            &[
                ("Mu", &Mu),
                ("pi", &pi),
                ("init_log_R", &self.init_log_R),
                ("branch_lengths", &self.log_branch_lengths.exp().unwrap()),
            ],
            path,
        )
        .unwrap();
    }
}

impl Optimizable for ModelParameters {
    fn variables(&self) -> Vec<Var> {
        vec![
            self.log_R.clone(),
            self.pca_coordinates.clone(),
            self.log_branch_lengths.clone(),
        ]
    }

    fn variables_names(&self) -> Vec<String> {
        vec![
            "log_R".to_string(),
            "pca_coordinates".to_string(),
            "log_branch_lengths".to_string(),
        ]
    }

    fn model_name(&self) -> String {
        "ModelParameters".to_string()
    }

    /// Evaluated with batch::log_likelihood_and_gradients, parallel over the sites
    fn likelihood(&self) -> Tensor {
        self.likelihood_parallel().unwrap()
    }

    fn penalty(&self) -> Tensor {
        let pi_penalty = self
            .pca_data
            .penalty_on_pca_coordinates(&self.pca_coordinates, self.reg_para.pi_reg);

        let Mu_penalty = Mu_penalty(&self.log_R, &self.init_log_R, self.reg_para.Mu_reg);

        let branch_penalty = (&self.log_branch_lengths.exp().unwrap()).sum_all().unwrap();
        let branch_penalty = (branch_penalty * self.reg_para.branch_length_reg).unwrap();

        (pi_penalty + Mu_penalty + branch_penalty).unwrap()
    }

    fn print_state(&self) {
    }
}

/// PCA components [19, 20] (all 19 coordinates are optimized in the MAP estimate)
fn batch_components(pca: &PCA) -> Result<SMatrix<f64, 19, 20>, candle_core::Error> {
    assert_eq!(pca.num_components, 19);
    Ok(SMatrix::<f64, 19, 20>::from_row_slice(&pca.components.flatten_all()?.to_vec1::<f64>()?))
}

/// Squared difference of the log off-diagonal Mu entries to the initial ones, times Mu_reg
pub fn Mu_penalty(log_R: &Tensor, init_log_R: &Tensor, Mu_reg: f64) -> Tensor {
    fn log_Mu(log_R: &Tensor) -> Tensor {
        let Mu = Mu(log_R);

        // One out the diagonal, since we only want to penalize the off-diagonal elements
        let Mu = (&Mu
            - (&Mu - 1.0).unwrap()
                * Tensor::eye(20, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
        .unwrap();

        return Mu.log().unwrap();
    }

    let Mu = log_Mu(log_R)
        .sub(&log_Mu(init_log_R))
        .unwrap()
        .powf(2.0)
        .unwrap()
        .sum_all()
        .unwrap();
    (Mu * Mu_reg).unwrap()
}

pub fn optimize(
    model: &impl Optimizable,
    min_iterations: usize,
    max_iterations: usize,
    min_rel_improvement: f64,
    no_improve_patience: usize,
    verbosity: Verbosity,
    _prefix: &str,
) {
    let variables = model.variables();
    // let variable_names = model.variables_names();
    // let mut trajectory_tensors: Vec<Vec<Tensor>> = variables
    //     .iter()
    //     .map(|variable| vec![variable.as_tensor().copy().unwrap()])
    //     .collect();
    let mut opt = candle_nn::optim::AdamW::new_lr(variables, 0.05).unwrap();
    let parameter = candle_nn::optim::ParamsAdamW {
        lr: 0.03,
        weight_decay: 0.0,
        ..Default::default()
    };
    opt.set_params(parameter);

    let variables = model.variables();

    let mut best_opt = f64::INFINITY;
    let mut best_params: Vec<Tensor> = variables
        .iter()
        .map(|variable| variable.as_tensor().copy().unwrap())
        .collect();
    let mut no_improve_count = 0;

    for iteration in 0.. {
        let neg_likelihood = model.likelihood().neg().unwrap();
        let penalty = model.penalty();
        let opt_fn = (&neg_likelihood + &penalty).unwrap();

        let current_opt = opt_fn.to_scalar::<f64>().unwrap();

        if current_opt < best_opt {
            best_params = variables
                .iter()
                .map(|variable| variable.as_tensor().copy().unwrap())
                .collect();
        }

        if verbosity.should_print(Verbosity::Med) {
            println!(
                "Iteration {}: neg Loglikelihood {:.3}, Optfn {:.3}",
                iteration,
                neg_likelihood.to_scalar::<f64>().unwrap(),
                current_opt
            );
            model.print_state();
        }

        let grads = opt_fn.backward().unwrap();
        opt.step(&grads).unwrap();

        // for (traj, new) in &mut trajectory_tensors.iter_mut().zip(variables.iter()) {
        //     traj.push(new.as_tensor().copy().unwrap());
        // }
        if iteration > min_iterations.saturating_sub(no_improve_patience) {
            let rel_improvement = (best_opt - current_opt) / best_opt.abs().max(1e-12);
            if rel_improvement > min_rel_improvement {
                no_improve_count = 0;
            } else {
                no_improve_count += 1;
            }
            if no_improve_count >= no_improve_patience {
                println!(
                    "Stopping at iteration {} (relative loss improvement {:.2e} <= {:.2e} for {} consecutive iterations)",
                    iteration, rel_improvement, min_rel_improvement, no_improve_patience
                );
                break;
            }
        }

        if iteration > max_iterations {
            println!("Reached maximum iterations ({})", max_iterations);
            break;
        }

        best_opt = current_opt.min(best_opt);
    }

    // Obtain parameters from best iteration
    for (variable, value) in variables.iter().zip(best_params.iter()) {
        variable.set(value).unwrap();
    }

    // Combine trajectory tensors into a single tensor for each variable
    // let trajectory_tensors: Vec<Tensor> = trajectory_tensors
    //     .into_iter()
    //     .map(|tensors| Tensor::stack(&tensors, 0).unwrap())
    //     .collect();
    // let filename = format!("{}.traj_{}.npz", _prefix, model.model_name());
    // Tensor::write_npz(
    //     &variable_names
    //         .iter()
    //         .zip(trajectory_tensors.iter())
    //         .collect::<Vec<_>>(),
    //     Path::new(&filename),
    // )
    // .unwrap();
}

pub fn optimize_branch_lengths(
    felsenstein_op: FelsensteinWithEdgeOp,
    log_pi: &Tensor,
    Mu: &Tensor,
    log_branch_lengths: &Tensor,
    verbosity: Verbosity,
    prefix: &str,
    mutsel_params: super::MutselParams
) -> Tensor {
    let model = BranchParameters {
        felsenstein_op,
        log_branch_lengths: Var::from_tensor(log_branch_lengths).unwrap(),
        reg_para: mutsel_params,
        Mu: Mu.clone(),
        log_pi: log_pi.clone(),
    };

    optimize(&model, 10, 200, 1e-6, 5, verbosity, prefix);

    model.log_branch_lengths.as_tensor().copy().unwrap()
}

pub fn two_step_light_pmsf(
    felsenstein_op: FelsensteinOp,
    categories: &[[f64; 20]],
    weights: &[f64],
    log_branch_lengths: &Tensor,
    mutsel_params: super::MutselParams,
    verbosity: Verbosity,
    prefix: &str,
) -> (Tensor, Tensor) {
    let step1_site_freq = light_pmsf(
        felsenstein_op.into_with_edge_op(),
        categories,
        weights,
        log_branch_lengths,
    );

    let Mu = loadMu();

    let log_pi = step1_site_freq.log().unwrap();

    let log_branch_lengths = optimize_branch_lengths(
        felsenstein_op.into_with_edge_op(),
        &log_pi,
        &Mu,
        log_branch_lengths,
        verbosity,
        prefix,
        mutsel_params,
    );

    let final_site_freq = light_pmsf(
        felsenstein_op.into_with_edge_op(),
        categories,
        weights,
        &log_branch_lengths,
    );

    (final_site_freq, log_branch_lengths)
}

pub fn light_pmsf(
    felsenstein_op: FelsensteinWithEdgeOp,
    categories: &[[f64; 20]],
    weights: &[f64],
    log_branch_lengths: &Tensor,
) -> Tensor {
    let mut likelihoods = vec![];

    let Mu = loadMu();

    for category in categories.iter() {
        let category_tensor =
            Tensor::from_vec(category.to_vec(), &[20], &candle_core::Device::Cpu).unwrap();
        let log_pi = category_tensor.log().unwrap().unsqueeze(0).unwrap();

        let (S, sqrt_pi) = model::calc_rate_matrix(&Mu, &log_pi, &tensor_full(1.0, &[]));

        let likelihood = S
            .apply_op3(
                &sqrt_pi,
                &log_branch_lengths.exp().unwrap(),
                felsenstein_op.into_fwd_op(),
            )
            .unwrap();
        likelihoods.push(likelihood);
    }

    let likelihoods = Tensor::stack(&likelihoods, 0).unwrap();
    let weights_tensor =
        Tensor::from_slice(&weights, &[weights.len()], &candle_core::Device::Cpu).unwrap();
    let log_weights_tensor = weights_tensor.log().unwrap().unsqueeze(1).unwrap();

    let weighted_likelihoods = (likelihoods.broadcast_add(&log_weights_tensor)).unwrap();

    let posteriors = candle_nn::ops::softmax(&weighted_likelihoods, 0).unwrap();

    let category_tensor = Tensor::from_vec(
        categories.iter().flatten().copied().collect(),
        &[categories.len(), 20],
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let site_freq = posteriors.t().unwrap().matmul(&category_tensor).unwrap();

    site_freq
}

/// Calculates the mutation rate matrix from the log_R variable. This is a lower triangular matrix with the exchanabilities and the equilibirum in the diagonal. Both as in log space.
/// Returns Mut and the equilibrium frequencies in the diagonal. (not log space)
pub fn Mu(log_parameter: &Tensor) -> Tensor {
    let parameter = log_parameter.exp().unwrap();
    let diagonal = (&parameter
        * Tensor::eye(20, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
    .unwrap();
    let off_diagonal = (parameter - &diagonal).unwrap();

    // Mutation equilibrium should stay fixed to neutral
    let diagonal = diagonal.detach();

    let pi = diagonal
        .broadcast_div(&diagonal.sum_all().unwrap())
        .unwrap();
    let pi_vec = pi.sum(1).unwrap();
    let sqrt_pi = pi_vec.sqrt().unwrap();
    let sqrt_pi_inv = sqrt_pi.recip().unwrap();

    let S = (&off_diagonal + off_diagonal.t().unwrap()).unwrap();

    // Q = diag(sqrt_pi_inv) * S * diag(sqrt_pi)
    let Q = sqrt_pi_inv
        .unsqueeze(1)
        .unwrap()
        .broadcast_mul(&S)
        .unwrap()
        .broadcast_mul(&sqrt_pi.unsqueeze(0).unwrap())
        .unwrap();
    let Q = (&Q
        - &Q * Tensor::eye(20, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
    .unwrap();

    let row_sum = Q.sum(1).unwrap();
    let sum = row_sum.dot(&pi_vec).unwrap();
    let Q = Q.broadcast_div(&sum).unwrap();

    // The scaling of Q does not matter, since we always scale it so the subsitution rate over all rate averages to 1.0.

    // diagonal is zero here, but they are not used anyway
    (Q + pi).unwrap()
}

// Used with the MutSel model.
fn loadMu() -> Tensor {
    let R_lower = crate::data::load_lower_R_with_equi(crate::data::M_TXT);
    let log_R = R_lower.log().unwrap();
    Mu(&log_R)
}

/// Convergence criteria of optimize
#[derive(Debug, Clone, Copy)]
pub struct Convergence {
    pub min_iterations: usize,
    pub max_iterations: usize,
    pub min_rel_improvement: f64,
    pub no_improve_patience: usize,
}

impl Convergence {
    /// Criteria of the MAP estimate in optimize_internal
    pub const MAP: Convergence = Convergence {
        min_iterations: 100,
        max_iterations: 500,
        min_rel_improvement: 1e-6,
        no_improve_patience: 5,
    };
}

/// log of the lower triangular R with the equilibrium in the diagonal (-inf above it), from prior_R_file or
/// the default M_TXT. Mu is penalized relative to it.
pub fn initial_log_R(prior_R_file: Option<&Path>) -> Result<Tensor, candle_core::Error> {
    let init_R = if let Some(prior_R_file) = prior_R_file {
        let file_content = std::fs::read_to_string(prior_R_file)?;
        crate::data::load_lower_R_with_equi(&file_content)
    } else {
        crate::data::load_lower_R_with_equi(crate::data::M_TXT)
    };
    init_R.log()
}

/// log_R with Mu(log_R) = Mu up to the normalization, the inverse of Mu. The upper triangle is taken
/// from template (an output of initial_log_R).
pub fn log_R_from_Mu(Mu: &Tensor, template: &Tensor) -> Result<Tensor, candle_core::Error> {
    let Mu = Mu.to_vec2::<f64>()?;
    let mut log_R = template.to_vec2::<f64>()?;
    let pi: Vec<f64> = (0..20).map(|i| Mu[i][i]).collect();
    for i in 0..20 {
        log_R[i][i] = pi[i].ln();
        // Q_ij = S_ij sqrt(pi_j / pi_i) / normalization, R holds S_ij in the lower triangle
        for j in 0..i {
            log_R[i][j] = (Mu[i][j] * (pi[i] / pi[j]).sqrt()).ln();
        }
    }
    Tensor::from_vec(log_R.concat(), &[20, 20], &candle_core::Device::Cpu)
}

/// Joint MAP estimate of Mu, the pca coordinates of every site and the branch lengths,
/// initialized with the two step light PMSF procedure.
pub fn map_estimate(
    op: FelsensteinOp,
    distances: &[f64],
    mutsel_params: super::MutselParams,
    prior_R_file: Option<&Path>,
    convergence: Convergence,
    verbosity: Verbosity,
    out_prefix: &str,
) -> Result<ModelParameters, candle_core::Error> {
    let init_log_R = initial_log_R(prior_R_file)?;

    let log_branch_lengths =
        Tensor::from_slice(distances, &[distances.len()], &candle_core::Device::Cpu)?.log()?;

    // Do our lightweight PMSF procedure for initialization:
    let (site_freq, log_branch_lengths) = two_step_light_pmsf(
        op.clone(),
        crate::data::UDM256,
        crate::data::UDM256_WEIGHTS,
        &log_branch_lengths,
        mutsel_params,
        verbosity,
        out_prefix,
    );

    // Variable which gets optimized
    let log_R = Var::from_tensor(&init_log_R)?;
    let init_log_pi = site_freq.log()?;

    let pca = PCA::new(19);

    let pca_coordinates = pca.log_freq_to_pca_coordinates(&init_log_pi);

    let model = ModelParameters {
        felsenstein_op: op.into_with_edge_op(),
        log_R,
        pca_coordinates: Var::from_tensor(&pca_coordinates).unwrap(),
        log_branch_lengths: Var::from_tensor(&log_branch_lengths).unwrap(),
        init_log_branch_lengths: log_branch_lengths.detach().copy().unwrap(),
        reg_para: mutsel_params,
        init_log_R,
        pca_data: pca,
    };

    optimize(
        &model,
        convergence.min_iterations,
        convergence.max_iterations,
        convergence.min_rel_improvement,
        convergence.no_improve_patience,
        verbosity,
        out_prefix,
    );

    Ok(model)
}

/// Returns the optimal S, sqrt_pi and rate parameters.
/// S: [L, 20, 20], sqrt_pi: [L, 20]
/// rate parameters is either alpha shape: [1] or [2 * num_cat] for free rate model (first num_cat weights, second num_cat are rates)
pub fn optimize_internal(
    felsenstein: FelsensteinTree<20>,
    distances: &[f64],
    mutsel_params: super::MutselParams,
    prior_R_file: Option<&Path>,
    verbosity: Verbosity,
    out_prefix: &str,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let op = felsenstein::FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));

    let model = map_estimate(
        op,
        distances,
        mutsel_params,
        prior_R_file,
        Convergence::MAP,
        verbosity,
        out_prefix,
    )?;

    let (S, sqrt_pi) = model.calc_rate_matrix();

    if verbosity.should_print(Verbosity::Med) {
        model.save_npz(Path::new(&format!("{}.mutsel.npz", out_prefix)));
    }

    Ok((S, sqrt_pi))
}
