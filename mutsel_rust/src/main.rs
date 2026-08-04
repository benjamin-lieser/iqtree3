#![allow(non_snake_case)]

use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use candle_core::{Tensor, Var};
use mutsel_rust::optimization;
use mutsel_rust::{felsenstein::FelsensteinWithEdgeOp, io::process_newick_alignment, optimization::{Mu, Optimizable, RateParameters}};

fn mixture_model_ll(
    felsenstein_op: FelsensteinWithEdgeOp,
    categories: &[[f64; 20]],
    weights: &[f64],
    rate_model: &mutsel_rust::optimization::RateParameters,
    log_branch_lengths: &Tensor,
    R: &Tensor,
    substitution_model : mutsel_rust::SubstitutionModel,
    L : usize
) -> Tensor {
    let mut likelihoods = vec![];

    for category in categories.iter() {
        let category_tensor =
            Tensor::from_vec(category.to_vec(), &[20], &candle_core::Device::Cpu).unwrap();
        let log_pi = category_tensor.log().unwrap().unsqueeze(0).unwrap().broadcast_as(&[L, 20]).unwrap();

        likelihoods.push(rate_model.likelihood_per_site(
            felsenstein_op.clone(),
            R,
            &log_pi,
            substitution_model,
            true,
            log_branch_lengths,
        ));
    }

    let likelihoods = Tensor::stack(&likelihoods, 0).unwrap();

    let weights_tensor =
        Tensor::from_slice(&weights, &[weights.len()], &candle_core::Device::Cpu).unwrap();
    let log_weights_tensor = weights_tensor.log().unwrap().unsqueeze(1).unwrap();

    let weighted_likelihoods = (likelihoods.broadcast_add(&log_weights_tensor)).unwrap();

    let log_likelihoods = weighted_likelihoods.log_sum_exp(0).unwrap();
    log_likelihoods
}

struct MixtureModel {
    felsenstein_op: FelsensteinWithEdgeOp,
    log_R : Var,
    alpha: RateParameters,
    log_branch_lengths: Var,
    substitution_model: mutsel_rust::SubstitutionModel,
    L : usize
}

impl Optimizable for MixtureModel {
    fn variables(&self) -> Vec<candle_core::Var> {
        let mut vars = Vec::new();
        vars.push(self.log_R.clone());
        vars.extend(self.alpha.variables());
        vars.push(self.log_branch_lengths.clone());
        vars
    }

    fn likelihood(&self) -> Tensor {
        let R = match self.substitution_model {
            mutsel_rust::SubstitutionModel::MutSel => {
                Mu(&self.log_R)
            }
            _ => (self.log_R.exp().unwrap() + self.log_R.exp().unwrap().transpose(0, 1).unwrap()).unwrap(),
        };
        

        let likelihood = mixture_model_ll(
            self.felsenstein_op.clone(),
            &mutsel_rust::data::C20,
            &mutsel_rust::data::C20_WEIGHTS,
            &self.alpha,
            &self.log_branch_lengths,
            &R,
            self.substitution_model,
            self.L
        );

        likelihood.sum_all().unwrap()
    }

    fn penalty(&self) -> Tensor {
        Tensor::zeros(&[], candle_core::DType::F64, &candle_core::Device::Cpu).unwrap()
    }

    fn print_state(&self) {
        println!("Current state:");
        println!("alpha: {:?}", self.alpha.variables()[0]);
    }
}


fn run(args: &[String]) {
    let newick = args[0].clone();
    let fasta = args[1].clone();
    let model = args[2].clone();
    let num_categories = args[3].parse::<usize>().unwrap();

    let substitution_model = match model.to_lowercase().as_str() {
        "mutselapprox" => mutsel_rust::SubstitutionModel::MutSelApprox,
        "pmsfnorm" => mutsel_rust::SubstitutionModel::PMSFNormalize,
        "pmsfnonorm" => mutsel_rust::SubstitutionModel::PMSFNoNormalize,
        "mutsel" => mutsel_rust::SubstitutionModel::MutSel,
        _ => panic!("Unknown substitution model"),
    };

    let sequences = mutsel_rust::io::read_alignment(Path::new(&fasta));

    let L = sequences.iter().next().unwrap().1.len();

    let (felsenstein, branch_lenghts) = process_newick_alignment(&String::from_utf8(std::fs::read(&newick).unwrap()).unwrap(), &sequences);

    let N = branch_lenghts.len();

    let branch_lenghts = Tensor::from_vec(branch_lenghts, &[N], &candle_core::Device::Cpu).unwrap();
    let log_branch_lenghts = branch_lenghts.log().unwrap();

    let log_R = match substitution_model {
        mutsel_rust::SubstitutionModel::MutSel => {
            let R = mutsel_rust::data::load_lower_R_with_equi(mutsel_rust::data::CODON2_TXT);
            let log_R = R.log().unwrap();
            log_R
        }
        _ => {
            let R = mutsel_rust::data::load_lower_R(mutsel_rust::data::CODON2_TXT);
            let log_R = R.log().unwrap();
            log_R
        }
    };
    
    let model = MixtureModel {
        felsenstein_op: FelsensteinWithEdgeOp::new(Arc::new(Mutex::new(felsenstein))),
        log_R: Var::from_tensor(&log_R).unwrap(),
        alpha: RateParameters::gamma(num_categories, 0.5),
        log_branch_lengths: Var::from_tensor(&log_branch_lenghts).unwrap(),
        substitution_model,
        L
    };

    optimization::optimize(&model, 100, 1000, 1e-6, 5, mutsel_rust::Verbosity::Debug);

}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    run(&args[1..].to_vec());
}
