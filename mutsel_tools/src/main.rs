use std::sync::{Arc, Mutex};

use candle_core::{DType::F64, Tensor, Var};

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    let alignment = mutsel_tools::read_alignment(std::path::Path::new(&args[1]));
    let tree = std::fs::read_to_string(&args[2]).expect("Could not read tree file");
    let (felsenstein, distances) = mutsel_tools::process_newick_alignment(&tree, &alignment);

    let distances = Tensor::from_slice(&distances, &[distances.len()], &candle_core::Device::Cpu).unwrap();
    let log_branch_lengths = distances.log().unwrap();
    
    let model = Tensor::read_npz(std::path::Path::new(&args[3])).expect("Could not read model file");

    let mut Mu = Tensor::zeros(&[0], candle_core::DType::F64, &candle_core::Device::Cpu).unwrap();
    let mut pi = Tensor::zeros(&[0], candle_core::DType::F64, &candle_core::Device::Cpu).unwrap();
    for (name, tensor) in model.into_iter() {
        if name == "Mu" {
            Mu = tensor;
        } else if name == "pi" {
            pi = tensor;
        }
    }

    dbg!("Mu: {:?}", &Mu);
    dbg!("pi: {:?}", &pi);

    let felsenstein_op = mutsel_rust::felsenstein::FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));

    let model = mutsel_rust::optimization::BranchParameters {
        felsenstein_op: felsenstein_op.into_with_edge_op(),
        Mu: Mu,
        log_branch_length: Var::from_tensor(&log_branch_lengths).unwrap(),
        log_pi: pi.log().unwrap(),
    };

    mutsel_rust::optimization::optimize(&model, 10, 10, 1e-6, 5, mutsel_rust::Verbosity::Med, &"debug");

    model.log_branch_length.set(&Tensor::zeros(&[distances.dim(0).unwrap()], F64, &candle_core::Device::Cpu).unwrap()).unwrap();

    mutsel_rust::optimization::optimize(&model, 100, 1000, 1e-7, 5, mutsel_rust::Verbosity::Med, &"debug");

    println!("Final log branch lengths: {:?}", model.log_branch_length.to_vec1::<f64>().unwrap());
}
