use std::sync::Arc;
use std::sync::Mutex;

use candle_core::Tensor;
use candle_core::Var;
use phylo_grad::FelsensteinTree;

use crate::SubstitutionModel;
use crate::Verbosity;
use crate::felsenstein;
use crate::felsenstein::FelsensteinWithEdgeFwdOp;
use crate::model;
use crate::optimization::Optimizable;
use crate::utils::tensor_full;

use crate::MutselParams;
use crate::felsenstein::FelsensteinWithEdgeOp;
use crate::optimization::Mu;

pub struct CATParameters {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    pub log_branch_lengths: Var,
    pub log_pi: Var,
    pub mu: Mu,
    pub hyperparameters: MutselParams,
    pub center_centers: Tensor,
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
        center_centers: &Tensor,
    ) -> Self {
        Self {
            felsenstein_op,
            log_branch_lengths: Var::from_tensor(log_branch_lengths).unwrap(),
            mu: mu.clone(),
            log_pi: Var::from_tensor(log_pi).unwrap(),
            hyperparameters,
            center_centers: center_centers.detach().copy().unwrap(),
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
        let num_clusters = self.center_centers.dim(0).unwrap();

        let mut means = Vec::with_capacity(num_clusters);
        for k in 0..num_clusters {
            let mask = self
                .clustering
                .eq(k as u32)
                .unwrap();
            let count = mask.sum_all().unwrap().to_scalar::<u32>().unwrap();
            if count == 0 {
                means.push(tensor_full(0.0, &[20]));
                continue;
            }
            let sum = self
                .log_pi
                .broadcast_mul(&mask.unsqueeze(1).unwrap())
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

        let center_penalty = (&self.center_centers - &log_pi_centers)
            .unwrap()
            .powf(2.0)
            .unwrap()
            .sum_all()
            .unwrap();
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

    let orig_categories = crate::data::C60;
    let orig_categories_weights = crate::data::C60_WEIGHTS;

    let orig_log_categories = Tensor::from_slice(
        &orig_categories.concat(),
        &[orig_categories.len(), 20],
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .log()
    .unwrap();

    let orig_categories_log_weights = Tensor::from_slice(
        &orig_categories_weights,
        &[orig_categories_weights.len()],
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .log()
    .unwrap();

    let posteriors = mixture_posteriors(
        op.into_with_edge_fwd_op(),
        &log_branch_lengths,
        &orig_log_categories,
        &orig_categories_log_weights,
        &Mu::new(),
    );

    let cluster_assignments = posteriors.argmax(0).unwrap();

    let mut model = CATParameters::new(
        op.into_with_edge_op(),
        &log_branch_lengths,
        &Mu::new(),
        &orig_log_categories
            .index_select(&cluster_assignments, 0)
            .unwrap(),
        hyperparameters,
        &cluster_assignments,
        &orig_log_categories,
    );

    for _epoch in 0..20 {
        crate::optimization::optimize(&model, 10, 1000, 1e-5, 5, verbosity);

        // Assign new cluster centers based on the current log_pi estimates
        let log_pi_centers = model.cluster_mean_log_pi();
        let euclidian_distances = model
            .log_pi
            .unsqueeze(1)
            .unwrap()
            .broadcast_sub(&log_pi_centers.unsqueeze(0).unwrap())
            .unwrap()
            .powf(2.0)
            .unwrap()
            .sum(2)
            .unwrap();

        let new_assignment = euclidian_distances.argmin(1).unwrap();

        // Print the number of sites that changed cluster assignments
        let changed = new_assignment
            .ne(&model.clustering)
            .unwrap()
            .to_vec1::<u8>()
            .unwrap();
        let num_changed = changed.iter().map(|&x| x as usize).sum::<usize>();

        println!(
            "Epoch {}: {} sites changed cluster assignments",
            _epoch + 1,
            num_changed
        );

        if num_changed == 0 {
            println!("No sites changed cluster assignments. Stopping optimization.");
            break;
        }

        // Print assignment summary
        let mut assignment_summary = vec![0; log_pi_centers.dim(0).unwrap()];
        for &assignment in new_assignment.to_vec1::<u32>().unwrap().iter() {
            assignment_summary[assignment as usize] += 1;
        }
        println!("Cluster assignment summary: {:?}", assignment_summary);

        model.clustering = new_assignment;
    }

    model.calc_rate_matrix()
}
