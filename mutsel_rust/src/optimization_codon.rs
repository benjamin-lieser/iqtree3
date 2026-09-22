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
    optimization::{Optimizable, optimize, two_step_light_pmsf},
    pca::PCA,
    utils::tensor_full,
};

pub struct ModelParametersCodon {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    /// The shared 4-state nucleotide GTR mutation process, in the same
    /// lower-triangular log-parametrization as `optimization::Mu`/`Mu4`.
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

        fn log_mu4(log_r_nt: &Tensor) -> Tensor {
            let mu = codon::Mu4(log_r_nt);

            // One out the diagonal, since we only want to penalize the off-diagonal elements
            let mu = (&mu
                - (&mu - 1.0).unwrap()
                    * Tensor::eye(4, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
            .unwrap();

            mu.log().unwrap()
        }

        let mu_diff = log_mu4(&self.log_R_nt)
            .sub(&log_mu4(&self.init_log_R_nt))
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
/// `felsenstein_aa` should be an amino-acid (20-state) view of the same
/// codon alignment (each leaf codon translated to its encoded amino acid
/// through `codon_aa`) -- it's used, completely unmodified, to run the same
/// light-PMSF initialization phase the amino-acid model uses, so the
/// codon model's fitness landscape starts from the same place the amino
/// acid model's would. `felsenstein_codon` is the real 61-state tree used
/// for the final optimization.
pub fn optimize_internal_codon(
    felsenstein_aa: phylo_grad::FelsensteinTree<20>,
    felsenstein_codon: phylo_grad::FelsensteinTree<N_CODON>,
    distances: &[f64],
    mutsel_params: MutselParams,
    codon_nt: [[u8; 3]; N_CODON],
    codon_aa: [u8; N_CODON],
    verbosity: Verbosity,
    out_prefix: &str,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    use crate::felsenstein::FelsensteinOp as AaFelsensteinOp;
    use crate::felsenstein_codon::FelsensteinOp as CodonFelsensteinOp;
    use std::sync::{Arc, Mutex};

    let aa_op = AaFelsensteinOp::new(Arc::new(Mutex::new(felsenstein_aa)));

    let log_branch_lengths = Tensor::from_slice(distances, &[distances.len()], &candle_core::Device::Cpu)?.log()?;

    // Phase 1: amino-acid-level light-PMSF initialization, fully reused
    // (unmodified) from the amino-acid model.
    let (site_freq, log_branch_lengths, log_site_rate) = two_step_light_pmsf(
        aa_op,
        crate::data::UDM256,
        crate::data::UDM256_WEIGHTS,
        &log_branch_lengths,
        mutsel_params,
        verbosity,
        out_prefix,
    );

    // Phase 2: codon-level optimization.
    let codon_op = CodonFelsensteinOp::new(Arc::new(Mutex::new(felsenstein_codon)));

    // No codon-level empirical prior exists yet; start from a uniform
    // (JC69-like) 4-state GTR.
    let log_r_nt = Var::from_tensor(&Tensor::zeros(
        &[4, 4],
        candle_core::DType::F64,
        &candle_core::Device::Cpu,
    )?)?;
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
