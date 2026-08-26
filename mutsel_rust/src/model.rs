use std::ops::Deref;

use candle_core::{CpuStorage, CustomOp1, Storage, Tensor};

fn g_scalar(x: f64) -> f64 {
    if x.abs() < 1e-6 {
        return 1.0 + x / 2.0 + x * x / 12.0;
    } else {
        return x / (1.0 - (-x).exp());
    }
}

fn dx_g_scalar(x: f64) -> f64 {
    if x.abs() < 1e-6 {
        return 0.5 + x / 6.0 - x * x * x / 180.0;
    } else {
        let exp_x = x.exp();
        return exp_x * (-1.0 + exp_x - x) / (exp_x - 1.0).powi(2);
    }
}

#[derive(Debug, Clone)]
pub struct GOp {}

impl CustomOp1 for GOp {
    fn name(&self) -> &'static str {
        "g_op"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let data = match storage {
            CpuStorage::F64(data) => data,
            _ => panic!("Expected f64 storage."),
        };

        let output = candle_core::cpu_backend::unary_map(&data, layout, g_scalar);

        Ok((CpuStorage::F64(output), layout.shape().clone()))
    }

    fn bwd(
        &self,
        _arg: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (storage, layout) = _arg.storage_and_layout();

        let data = match storage.deref() {
            Storage::Cpu(CpuStorage::F64(data)) => data,
            _ => panic!("Expected f64 storage."),
        };

        let grads = candle_core::cpu_backend::unary_map(&data, &layout, dx_g_scalar);

        let grad_tensor = Tensor::from_vec(grads, _arg.shape(), &candle_core::Device::Cpu)?;

        Ok(Some(grad_tensor.mul(_grad_res)?))
    }
}

/// We assume a mutation process Mu is the rate matrix of the mutation process with its equlibirium in the diagonal.
/// It should be the output of the Mu function.
pub fn calc_rate_matrix(
    Mu: &Tensor,
    log_pi: &Tensor,
    global_scaling: &Tensor,
) -> (Tensor, Tensor) {
    let L = log_pi.dims()[0];

    let equi_mutation = (Mu * Tensor::eye(20, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap()).unwrap();
    let equi_mutation = equi_mutation.sum(1).unwrap();
    let log_equi_mutation = equi_mutation.log().unwrap();

    // This is a correction because of the equilibirum of the mutation process.
    let fitness = log_pi.broadcast_sub(&log_equi_mutation.unsqueeze(0).unwrap()).unwrap();

    let log_pi_diff = fitness.unsqueeze(1).unwrap().broadcast_sub(&fitness.unsqueeze(2).unwrap()).unwrap();

    let fixation = log_pi_diff.apply_op1(GOp {}).unwrap();

    let R = Mu.unsqueeze(0).unwrap().broadcast_as(&[L, 20, 20]).unwrap();

    let Q = R.mul(&fixation).unwrap();

    let pi = candle_nn::ops::softmax(&log_pi, 1).unwrap();
    let pi_sqrt = pi.sqrt().unwrap();

    let sqrt_pi_Q = pi_sqrt.unsqueeze(2).unwrap().broadcast_mul(&Q).unwrap();
    let S = sqrt_pi_Q.broadcast_div(&pi_sqrt.unsqueeze(1).unwrap()).unwrap();

    let S = S.broadcast_mul(&global_scaling).unwrap();
    (S, pi_sqrt)
}

// From S, sqrt_pi to R, pi
pub fn phylograd2iqtree_parametrization(
    S: &Tensor,
    sqrt_pi: &Tensor,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let S_sqrt_pi_inv = S.broadcast_div(&sqrt_pi.unsqueeze(1)?)?;
    let R = S_sqrt_pi_inv.broadcast_div(&sqrt_pi.unsqueeze(2)?)?;

    let pi = sqrt_pi.mul(sqrt_pi)?;

    Ok((R, pi))
}

pub fn substitution_rates(S: &Tensor, sqrt_pi: &Tensor) -> Vec<f64> {
    // Q_ij = S_ij * sqrt_pi_j / sqrt_pi_i

    let S = S.to_vec3::<f64>().unwrap();
    let sqrt_pi = sqrt_pi.to_vec2::<f64>().unwrap();

    assert_eq!(S.len(), sqrt_pi.len(), "Batch dimensions do not match.");

    let mut rates = Vec::with_capacity(S.len());
    for site in 0..S.len() {
        let mut total_rate = 0.0;
        for i in 0..20 {
            for j in 0..20 {
                if i != j {
                    total_rate += S[site][i][j] * sqrt_pi[site][j] * sqrt_pi[site][i];
                }
            }
        }
        rates.push(total_rate);
    }
    rates
}

// Tests
#[cfg(test)]
mod tests {
    use crate::{optimization::Mu, utils::tensor_full, model::calc_rate_matrix};

use super::*;
    #[test]
    fn test_mutsel() {
        let L = 2;
        let log_pi = Tensor::rand(-1.0, 1.0, &[L, 20], &candle_core::Device::Cpu).unwrap();

        let R = Tensor::rand(0.1, 2.0, &[20, 20], &candle_core::Device::Cpu).unwrap();

        let R = Mu(&R);

        let (S, sqrt_pi) = calc_rate_matrix(&R, &log_pi, &tensor_full(1.0, &[]));

        println!("S: {}", S);
        println!("sqrt_pi: {}", (&sqrt_pi * &sqrt_pi).unwrap());

        println!("S - S.T: {}", (&S - &S.transpose(1, 2).unwrap()).unwrap());

        assert!((&S - &S.transpose(1, 2).unwrap()).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f64>().unwrap() < 1e-12);
    }
}