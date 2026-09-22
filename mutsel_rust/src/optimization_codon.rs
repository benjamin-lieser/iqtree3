//! Codon-level counterpart of `optimization.rs`'s `ModelParameters`/
//! `optimize_internal`. The fitness/selection machinery (the per-site
//! amino-acid PCA-parametrized log-frequency landscape, and its
//! regularization penalty) is reused completely unchanged from the amino
//! acid model -- only the mutation process (now a 4-state nucleotide GTR
//! expanded through the genetic code, see `codon.rs`) and the likelihood
//! evaluation (now over 61 codon states) are new.

use std::path::Path;

use candle_core::{Tensor, Var};
use candle_nn::ops::softmax;

use crate::{
    MutselParams, Verbosity,
    codon::{self, N_CODON},
    felsenstein_codon::FelsensteinWithEdgeOp,
    optimization::{Optimizable, optimize},
    pca::PCA,
    utils::tensor_full,
};

fn calc_likelihood_codon(
    mu: &Tensor,
    log_pi: &Tensor,
    log_branch_lengths: &Tensor,
    log_site_rate: &Tensor,
    felsenstein_op: FelsensteinWithEdgeOp,
) -> Tensor {
    let (S, sqrt_pi) = codon::calc_rate_matrix_codon(mu, log_pi, &tensor_full(1.0, &[]));

    let S = S
        .broadcast_mul(
            &log_site_rate
                .exp()
                .unwrap()
                .unsqueeze(1)
                .unwrap()
                .unsqueeze(2)
                .unwrap(),
        )
        .unwrap();

    let branch_lengths = log_branch_lengths.exp().unwrap();

    let average_rate = codon::substitution_rates_tensor_codon(&S, &sqrt_pi)
        .mean_all()
        .unwrap();
    let S = S.broadcast_div(&average_rate).unwrap();

    S.apply_op3(&sqrt_pi, &branch_lengths, felsenstein_op)
        .unwrap()
        .sum_all()
        .unwrap()
}

/// Codon-level counterpart of `optimization::BranchParameters`: optimizes
/// branch lengths (and per-site rate) against a *fixed* codon mutation
/// matrix and per-site codon target frequency, using the real codon
/// Felsenstein tree -- so the branch-length scale (substitutions per
/// codon site, including synonymous substitutions) matches the model
/// that will actually be fit, rather than an amino-acid-level scale.
pub struct BranchParametersCodon {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    pub log_branch_lengths: Var,
    pub log_site_rate: Var,
    pub reg_para: MutselParams,
    pub Mu: Tensor,
    pub log_pi: Tensor,
}

impl Optimizable for BranchParametersCodon {
    fn variables(&self) -> Vec<Var> {
        vec![self.log_branch_lengths.clone(), self.log_site_rate.clone()]
    }
    fn variables_names(&self) -> Vec<String> {
        vec!["log_branch_lengths".to_string(), "log_site_rate".to_string()]
    }
    fn model_name(&self) -> String {
        "BranchParametersCodon".to_string()
    }

    fn likelihood(&self) -> Tensor {
        calc_likelihood_codon(
            &self.Mu,
            &self.log_pi,
            &self.log_branch_lengths,
            &self.log_site_rate,
            self.felsenstein_op.clone(),
        )
    }

    fn penalty(&self) -> Tensor {
        let rate_penalty = (&self.log_site_rate.powf(2.0).unwrap()).sum_all().unwrap();
        let rate_penalty = (rate_penalty * self.reg_para.site_rate_reg).unwrap();

        let branch_penalty = (&self.log_branch_lengths.exp().unwrap()).sum_all().unwrap();
        let branch_penalty = (branch_penalty * self.reg_para.branch_length_reg).unwrap();

        (rate_penalty + branch_penalty).unwrap()
    }

    fn print_state(&self) {}
}

/// Codon-level counterpart of `optimization::optimize_branch_lengths`.
pub fn optimize_branch_lengths_codon(
    felsenstein_op: FelsensteinWithEdgeOp,
    log_pi: &Tensor,
    Mu: &Tensor,
    log_branch_lengths: &Tensor,
    verbosity: Verbosity,
    prefix: &str,
    mutsel_params: MutselParams,
) -> (Tensor, Tensor) {
    let L = log_pi.dim(0).unwrap();

    let model = BranchParametersCodon {
        felsenstein_op,
        log_branch_lengths: Var::from_tensor(log_branch_lengths).unwrap(),
        log_site_rate: Var::from_tensor(&tensor_full(0.0, &[L])).unwrap(),
        reg_para: mutsel_params,
        Mu: Mu.clone(),
        log_pi: log_pi.clone(),
    };

    optimize(&model, 10, 200, 1e-6, 5, verbosity, prefix);

    (
        model.log_branch_lengths.as_tensor().copy().unwrap(),
        model.log_site_rate.as_tensor().copy().unwrap(),
    )
}

/// Codon-level counterpart of `optimization::light_pmsf`. The UDM256
/// category profiles describe per-site amino-acid *target* frequencies --
/// the fitness/selection side of the mutation-selection model -- which is
/// meaningful independently of whatever the mutation process is. So this
/// evaluates each category's likelihood with the real codon Felsenstein
/// tree and a codon mutation matrix (broadcasting each 20-dim amino-acid
/// category profile to 61 codons through the genetic code), while the
/// posterior-weighted combination of categories still happens at the
/// amino-acid level -- the returned site frequencies are `[L, 20]`, exactly
/// like the amino-acid `light_pmsf`.
pub fn light_pmsf_codon(
    felsenstein_op: FelsensteinWithEdgeOp,
    categories: &[[f64; 20]],
    weights: &[f64],
    log_branch_lengths: &Tensor,
    codon_mu: &Tensor,
    codon_aa: &[u8; N_CODON],
) -> Tensor {
    let mut likelihoods = vec![];

    for category in categories.iter() {
        let category_tensor =
            Tensor::from_vec(category.to_vec(), &[20], &candle_core::Device::Cpu).unwrap();
        let log_pi_aa = category_tensor.log().unwrap().unsqueeze(0).unwrap();
        let log_pi_codon = codon::broadcast_aa_fitness_to_codon(&log_pi_aa, codon_aa);

        let (S, sqrt_pi) = codon::calc_rate_matrix_codon(codon_mu, &log_pi_codon, &tensor_full(1.0, &[]));

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
        Tensor::from_slice(weights, &[weights.len()], &candle_core::Device::Cpu).unwrap();
    let log_weights_tensor = weights_tensor.log().unwrap().unsqueeze(1).unwrap();

    let weighted_likelihoods = (likelihoods.broadcast_add(&log_weights_tensor)).unwrap();

    let posteriors = candle_nn::ops::softmax(&weighted_likelihoods, 0).unwrap();

    let category_tensor = Tensor::from_vec(
        categories.iter().flatten().copied().collect(),
        &[categories.len(), 20],
        &candle_core::Device::Cpu,
    )
    .unwrap();

    posteriors.t().unwrap().matmul(&category_tensor).unwrap()
}

pub struct ModelParametersCodon {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    /// The shared 4-state nucleotide mutation process: 10 log-parameters
    /// (6 symmetric exchangeabilities + 4 frequency-like factors), see
    /// `codon::Mu4`.
    pub log_R_nt: Var,
    /// Defines the log amino-acid fitness per site -- identical in meaning
    /// to the amino-acid model's `pca_coordinates`.
    pub pca_coordinates: Var,
    pub log_branch_lengths: Var,
    pub log_site_rate: Var,
    pub init_log_branch_lengths: Tensor,
    pub reg_para: MutselParams,
    pub init_log_R_nt: Tensor,
    pub pca_data: PCA,
    pub codon_nt: [[u8; 3]; N_CODON],
    pub codon_aa: [u8; N_CODON],
}

impl ModelParametersCodon {
    /// Per-site amino-acid log-frequency (fitness) -- identical call as the
    /// amino-acid model's `ModelParameters::log_pi`.
    pub fn log_pi_aa(&self) -> Tensor {
        self.pca_data.pca_coordinates_to_log_freq(&self.pca_coordinates)
    }

    pub fn calc_rate_matrix(&self) -> (Tensor, Tensor) {
        let nt_mu = codon::Mu4(&self.log_R_nt);
        let codon_mu = codon::expand_nt_to_codon_mu(&nt_mu, &self.codon_nt);
        let log_pi_codon = codon::broadcast_aa_fitness_to_codon(&self.log_pi_aa(), &self.codon_aa);

        let (S, sqrt_pi) = codon::calc_rate_matrix_codon(&codon_mu, &log_pi_codon, &tensor_full(1.0, &[]));
        let S = S
            .broadcast_mul(
                &self
                    .log_site_rate
                    .exp()
                    .unwrap()
                    .unsqueeze(1)
                    .unwrap()
                    .unsqueeze(2)
                    .unwrap(),
            )
            .unwrap();
        let average_rate = codon::substitution_rates_tensor_codon(&S, &sqrt_pi)
            .mean_all()
            .unwrap();
        let S = S.broadcast_div(&average_rate).unwrap();
        (S, sqrt_pi)
    }

    pub fn save_npz(&self, path: &Path) {
        let nt_mu = codon::Mu4(&self.log_R_nt.as_detached_tensor());
        let pi_aa = softmax(&self.log_pi_aa().detach(), 1).unwrap();

        Tensor::write_npz(
            &[
                ("Mu_nt", &nt_mu),
                ("pi_aa", &pi_aa),
                ("init_log_R_nt", &self.init_log_R_nt),
                ("branch_lengths", &self.log_branch_lengths.exp().unwrap()),
            ],
            path,
        )
        .unwrap();
    }
}

impl Optimizable for ModelParametersCodon {
    fn variables(&self) -> Vec<Var> {
        vec![
            self.log_R_nt.clone(),
            self.pca_coordinates.clone(),
            self.log_branch_lengths.clone(),
            self.log_site_rate.clone(),
        ]
    }

    fn variables_names(&self) -> Vec<String> {
        vec![
            "log_R_nt".to_string(),
            "pca_coordinates".to_string(),
            "log_branch_lengths".to_string(),
            "log_site_rate".to_string(),
        ]
    }

    fn model_name(&self) -> String {
        "ModelParametersCodon".to_string()
    }

    fn likelihood(&self) -> Tensor {
        let (S, sqrt_pi) = self.calc_rate_matrix();
        let branch_lengths = self.log_branch_lengths.exp().unwrap();
        S.apply_op3(&sqrt_pi, &branch_lengths, self.felsenstein_op.clone())
            .unwrap()
            .sum_all()
            .unwrap()
    }

    fn penalty(&self) -> Tensor {
        // Same fitness penalty as the amino-acid model, unchanged.
        let pi_penalty = self
            .pca_data
            .penalty_on_pca_coordinates(&self.pca_coordinates);
        let pi_penalty = (pi_penalty * self.reg_para.pi_reg).unwrap();

        // Mu4's diagonal is a fixed constant (1.0, independent of the
        // log-parameters), so it contributes nothing to this diff -- no
        // need to mask it out separately.
        let mu_diff = codon::Mu4(&self.log_R_nt)
            .log()
            .unwrap()
            .sub(&codon::Mu4(&self.init_log_R_nt).log().unwrap())
            .unwrap()
            .powf(2.0)
            .unwrap()
            .sum_all()
            .unwrap();
        let mu_penalty = (mu_diff * self.reg_para.Mu_reg).unwrap();

        let rate_penalty = (&self.log_site_rate.powf(2.0).unwrap()).sum_all().unwrap();
        let rate_penalty = (rate_penalty * self.reg_para.site_rate_reg).unwrap();

        let branch_penalty = (&self.log_branch_lengths.exp().unwrap()).sum_all().unwrap();
        let branch_penalty = (branch_penalty * self.reg_para.branch_length_reg).unwrap();

        (pi_penalty + mu_penalty + rate_penalty + branch_penalty).unwrap()
    }

    fn print_state(&self) {}
}

/// Returns the optimal S, sqrt_pi for the codon model.
/// S: [L, 61, 61], sqrt_pi: [L, 61]
///
/// The whole initialization pipeline (light-PMSF site-frequency estimation
/// and branch-length pre-optimization) runs entirely on the real codon
/// Felsenstein tree: the UDM256 category profiles describe per-site
/// amino-acid *target* frequencies -- the fitness/selection side of the
/// model -- which is independent of the mutation process, so they're
/// broadcast to codons through the genetic code and evaluated with the
/// real codon mutation process (see `light_pmsf_codon`), rather than with
/// an amino-acid-level proxy model.
pub fn optimize_internal_codon(
    felsenstein_codon: phylo_grad::FelsensteinTree<N_CODON>,
    distances: &[f64],
    mutsel_params: MutselParams,
    codon_nt: [[u8; 3]; N_CODON],
    codon_aa: [u8; N_CODON],
    verbosity: Verbosity,
    out_prefix: &str,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    use crate::felsenstein_codon::FelsensteinOp as CodonFelsensteinOp;
    use std::sync::{Arc, Mutex};

    let codon_op = CodonFelsensteinOp::new(Arc::new(Mutex::new(felsenstein_codon)));

    let log_branch_lengths = Tensor::from_slice(distances, &[distances.len()], &candle_core::Device::Cpu)?.log()?;

    // No codon-level empirical prior exists yet; start from a uniform
    // (JC69-like) 4-state GTR. Used as the fixed mutation process for the
    // light-PMSF site-frequency estimation and branch-length
    // pre-optimization steps below.
    let init_log_r_nt = Tensor::zeros(&[10], candle_core::DType::F64, &candle_core::Device::Cpu)?;
    let init_codon_mu = codon::expand_nt_to_codon_mu(&codon::Mu4(&init_log_r_nt), &codon_nt);

    // Step 1: light-PMSF site-frequency estimate, using the initial
    // (guide-tree) branch lengths.
    let site_freq = light_pmsf_codon(
        codon_op.into_with_edge_op(),
        crate::data::UDM256,
        crate::data::UDM256_WEIGHTS,
        &log_branch_lengths,
        &init_codon_mu,
        &codon_aa,
    );

    // Step 2: branch-length (and site-rate) pre-optimization, using the
    // step-1 frequencies broadcast to codons.
    let log_pi_codon_init = codon::broadcast_aa_fitness_to_codon(&site_freq.log()?, &codon_aa);
    let (log_branch_lengths, log_site_rate) = optimize_branch_lengths_codon(
        codon_op.into_with_edge_op(),
        &log_pi_codon_init,
        &init_codon_mu,
        &log_branch_lengths,
        verbosity,
        out_prefix,
        mutsel_params,
    );

    // Step 3: re-estimate site frequencies at the now-optimized branch
    // lengths (mirrors the amino-acid model's two-step light-PMSF
    // refinement).
    let site_freq = light_pmsf_codon(
        codon_op.into_with_edge_op(),
        crate::data::UDM256,
        crate::data::UDM256_WEIGHTS,
        &log_branch_lengths,
        &init_codon_mu,
        &codon_aa,
    );

    // Step 4: full codon-level optimization.
    let log_r_nt = Var::from_tensor(&init_log_r_nt)?;
    let init_log_r_nt = log_r_nt.detach().copy()?;

    let init_log_pi = site_freq.log()?;

    let pca = PCA::new(19);
    let pca_coordinates = pca.log_freq_to_pca_coordinates(&init_log_pi);

    let model = ModelParametersCodon {
        felsenstein_op: codon_op.into_with_edge_op(),
        log_R_nt: log_r_nt,
        pca_coordinates: Var::from_tensor(&pca_coordinates)?,
        log_branch_lengths: Var::from_tensor(&log_branch_lengths)?,
        log_site_rate: Var::from_tensor(&log_site_rate)?,
        init_log_branch_lengths: log_branch_lengths.detach().copy()?,
        reg_para: mutsel_params,
        init_log_R_nt: init_log_r_nt,
        pca_data: pca,
        codon_nt,
        codon_aa,
    };

    optimize(&model, 100, 500, 1e-6, 5, verbosity, out_prefix);

    let (S, sqrt_pi) = model.calc_rate_matrix();

    if verbosity.should_print(Verbosity::Med) {
        model.save_npz(Path::new(&format!("{}.mutselcodon.npz", out_prefix)));
    }

    Ok((S, sqrt_pi))
}
