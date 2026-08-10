use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;

use candle_core::DType::U32;
use candle_core::Device;
use candle_core::Tensor;
use candle_core::Var;
use linfa::Dataset;
use linfa::traits::Predict;
use ndarray::Array2;
use linfa_clustering::Dbscan;
use linfa::traits::{Fit, Transformer};
use linfa_reduction::Pca;
use phylo_grad::FelsensteinTree;

use crate::SubstitutionModel;
use crate::Verbosity;
use crate::felsenstein;
use crate::felsenstein::FelsensteinOp;
use crate::felsenstein::FelsensteinWithEdgeFwdOp;
use crate::model;
use crate::optimization::Optimizable;
use crate::utils::tensor_full;

use crate::MutselParams;
use crate::felsenstein::FelsensteinWithEdgeOp;
use crate::optimization::Mu;


fn cluster_log_pi(
    log_pi: &Tensor,
    min_cluster_size: usize,
    n_components: usize, // e.g. 8-10, tune to your data
) -> Tensor {
    let num_sites = log_pi.dim(0).unwrap();
    let num_features = log_pi.dim(1).unwrap();
    let data = log_pi.flatten_all().unwrap().to_vec1::<f64>().unwrap();
    let data = Array2::from_shape_vec((num_sites, num_features), data).unwrap();

    // Wrap as a linfa Dataset (required for Pca's Fit trait)
    let dataset = Dataset::from(data);

    // Fit PCA and project down to n_components dimensions
    let pca = Pca::params(n_components).fit(&dataset).unwrap();
    let reduced = pca.predict(dataset); // Dataset<f64, ()>, now with n_components columns
    let reduced_records = reduced.records().to_owned(); // Array2<f64>

    // Run DBSCAN on the reduced data
    let clusters = Dbscan::params(min_cluster_size)
        .transform(&reduced_records)
        .unwrap();

    // Make labels contiguous
    let mut unique_ids: Vec<usize> = clusters.iter().flatten().copied().collect::<HashSet<_>>().into_iter().collect();
    unique_ids.sort();
    let num_clusters = unique_ids.len();
    let mut id_map = std::collections::HashMap::new();
    for (new_id, old_id) in unique_ids.iter().enumerate() {
        id_map.insert(*old_id, new_id);
    }
    let labels: Vec<usize> = clusters
        .iter()
        .map(|c| *id_map.get(c).unwrap())
        .collect();
    }

    Tensor::from_slice(&labels, &[labels.len()], &Device::Cpu).unwrap()
}


pub struct GlobalScalingPiMuParameters {
    pub felsenstein_op: FelsensteinOp,
    pub log_global_scaling: Var,
    pub mu: Mu,
    pub log_pi: Var,
    pub log_pi_init: Tensor,
    pub pi_reg: f64,
    pub Mu_reg: f64,
}

impl Optimizable for GlobalScalingPiMuParameters {
    fn variables(&self) -> Vec<Var> {
        vec![
            self.log_global_scaling.clone(),
            self.mu.variable(),
            self.log_pi.clone(),
        ]
    }
    fn likelihood(&self) -> Tensor {
        let global_scaling = self.log_global_scaling.exp().unwrap();
        let Mu = self.mu.mu();

        let (S, sqrt_pi) = model::calc_rate_matrix(
            &Mu,
            &self.log_pi,
            &global_scaling,
            SubstitutionModel::MutSel,
        );

        S.apply_op2(&sqrt_pi, self.felsenstein_op.clone())
            .unwrap()
            .sum_all()
            .unwrap()
    }
    fn penalty(&self) -> Tensor {
        let log_pi_mean = self.log_pi.mean_keepdim(1).unwrap();
        let pi_penalty = self
            .log_pi
            .broadcast_sub(&log_pi_mean)
            .unwrap()
            .powf(2.0)
            .unwrap()
            .sum_all()
            .unwrap()
            * self.pi_reg;
        let Mu_penalty = (self.mu.penalty() * self.Mu_reg).unwrap();
        (pi_penalty + Mu_penalty).unwrap()
    }
}

pub struct CATParameters {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    pub log_branch_lengths: Var,
    pub log_pi: Var,
    pub mu: Mu,
    pub hyperparameters: MutselParams,
    pub center_centers: Option<Tensor>,
    pub clustering: Tensor,
}

impl CATParameters {
    pub fn new(
        felsenstein_op: FelsensteinWithEdgeOp,
        log_branch_lengths: &Tensor,
        mu: &Mu,
        log_pi: &Tensor,
        hyperparameters: MutselParams,
        clustering: &Tensor,
        center_centers: Option<Tensor>,
    ) -> Self {
        Self {
            felsenstein_op,
            log_branch_lengths: Var::from_tensor(log_branch_lengths).unwrap(),
            mu: mu.clone(),
            log_pi: Var::from_tensor(log_pi).unwrap(),
            hyperparameters,
            center_centers,
            clustering: clustering.clone(),
        }
    }
    pub fn calc_rate_matrix(&self) -> (Tensor, Tensor) {
        let Mu = self.mu.mu();

        model::calc_rate_matrix(
            &Mu,
            &self.log_pi.as_detached_tensor(),
            &tensor_full(1.0, &[]),
            SubstitutionModel::MutSel,
        )
    }

    pub fn cluster_mean_log_pi(&self) -> Tensor {
        let num_clusters = self
            .clustering
            .max_all()
            .unwrap()
            .to_scalar::<u32>()
            .unwrap() as usize
            + 1;

        let mut means = Vec::with_capacity(num_clusters);
        for k in 0..num_clusters {
            let mask = self.clustering.eq(k as u32).unwrap().to_dtype(U32).unwrap();
            let count = mask.sum_all().unwrap().to_scalar::<u32>().unwrap();
            if count == 0 {
                means.push(tensor_full(0.0, &[20]));
                continue;
            }
            let sum = self
                .log_pi
                .broadcast_mul(
                    &mask
                        .to_dtype(candle_core::DType::F64)
                        .unwrap()
                        .unsqueeze(1)
                        .unwrap(),
                )
                .unwrap()
                .sum(0)
                .unwrap();
            means.push((sum / count as f64).unwrap());
        }

        Tensor::stack(&means, 0).unwrap()
    }
}

impl Optimizable for CATParameters {
    fn variables(&self) -> Vec<Var> {
        vec![
            self.log_branch_lengths.clone(),
            self.mu.variable(),
            self.log_pi.clone(),
        ]
    }

    fn likelihood(&self) -> Tensor {
        let branch_lengths = self.log_branch_lengths.exp().unwrap();
        let Mu = self.mu.mu();

        let (S, sqrt_pi) = model::calc_rate_matrix(
            &Mu,
            &self.log_pi,
            &tensor_full(1.0, &[]),
            SubstitutionModel::MutSel,
        );

        let log_likelihoods = S
            .apply_op3(&sqrt_pi, &branch_lengths, self.felsenstein_op.clone())
            .unwrap();
        log_likelihoods.sum_all().unwrap()
    }

    fn penalty(&self) -> Tensor {
        let log_pi_centers = self.cluster_mean_log_pi();

        let per_site_centers = log_pi_centers.index_select(&self.clustering, 0).unwrap();

        let pi_penalty = (&per_site_centers - self.log_pi.as_tensor())
            .unwrap()
            .powf(2.0)
            .unwrap()
            .sum_all()
            .unwrap();
        let pi_penalty = (pi_penalty * self.hyperparameters.pi_reg).unwrap();

        let Mu_penalty = (self.mu.penalty() * self.hyperparameters.Mu_reg).unwrap();

        let center_penalty = if let Some(center_centers) = self.center_centers.as_ref() {
            (center_centers - &log_pi_centers)
                .unwrap()
                .powf(2.0)
                .unwrap()
                .sum_all()
                .unwrap()
        } else {
            tensor_full(0.0, &[])
        };
        let center_penalty = (center_penalty * self.hyperparameters.branch_reg).unwrap();

        println!(
            "Penalty: pi_penalty = {}, Mu_penalty = {}, center_penalty = {}",
            pi_penalty.to_scalar::<f64>().unwrap(),
            Mu_penalty.to_scalar::<f64>().unwrap(),
            center_penalty.to_scalar::<f64>().unwrap()
        );

        (pi_penalty + Mu_penalty + center_penalty).unwrap()
    }
}

pub fn mixture_posteriors(
    felsenstein_op: FelsensteinWithEdgeFwdOp,
    log_branch_lengths: &Tensor,
    log_categories: &Tensor,
    log_weights: &Tensor,
    Mu: &Mu,
) -> Tensor {
    let mut likelihoods = vec![];

    let Mu = Mu.mu();

    let branch_lengths = log_branch_lengths.exp().unwrap();

    for category in 0..log_categories.dim(0).unwrap() {
        let category_tensor = log_categories.get(category).unwrap();
        let log_pi = category_tensor.unsqueeze(0).unwrap();

        let (S, sqrt_pi) = model::calc_rate_matrix(
            &Mu,
            &log_pi,
            &tensor_full(1.0, &[]),
            SubstitutionModel::MutSel,
        );

        let log_likelihoods = S
            .apply_op3(&sqrt_pi, &branch_lengths, felsenstein_op.clone())
            .unwrap();
        likelihoods.push(log_likelihoods);
    }

    let likelihoods = Tensor::stack(&likelihoods, 0).unwrap();

    let log_weights_tensor = log_weights.unsqueeze(1).unwrap();

    let weighted_likelihoods = (likelihoods.broadcast_add(&log_weights_tensor)).unwrap();

    let posteriors = candle_nn::ops::softmax(&weighted_likelihoods, 0).unwrap();

    posteriors
}

pub fn cat_mutsel(
    felsenstein: FelsensteinTree<20>,
    distances: &[f64],
    hyperparameters: MutselParams,
    verbosity: Verbosity,
) -> (Tensor, Tensor) {
    let op = felsenstein::FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));

    let log_branch_lengths =
        Tensor::from_slice(distances, &[distances.len()], &candle_core::Device::Cpu)
            .unwrap()
            .log()
            .unwrap();

    let (init_pi, log_global_scaling) = super::optimization::two_step_light_pmsf(
        op.clone(),
        crate::data::C60,
        &crate::data::C60_WEIGHTS,
        &log_branch_lengths,
        &hyperparameters,
        verbosity,
    );
    let init_log_pi = init_pi.log().unwrap();


    let model = GlobalScalingPiMuParameters {
        felsenstein_op: op.clone(),
        log_global_scaling: Var::from_tensor(&tensor_full(log_global_scaling, &[])).unwrap(),
        mu: Mu::new(),
        log_pi: Var::from_tensor(&init_log_pi).unwrap(),
        log_pi_init: init_log_pi.clone(),
        pi_reg: hyperparameters.branch_reg, // Will use a better name later
        Mu_reg: hyperparameters.Mu_reg,
    };

    crate::optimization::optimize(&model, 10, 1000, 1e-3, 5, verbosity);

    let cluster_assignments = cluster_log_pi(&model.log_pi.as_detached_tensor(), 30);

    let model = CATParameters::new(
        op.into_with_edge_op(),
        &(log_branch_lengths.broadcast_add(&model.log_global_scaling)).unwrap(),
        &model.mu,
        &model.log_pi,
        hyperparameters,
        &cluster_assignments,
        None
    );

    crate::optimization::optimize(&model, 100, 1000, 1e-5, 5, verbosity);

    model.calc_rate_matrix()
}
