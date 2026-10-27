use std::ops::Deref;

use candle_core::{CpuStorage, CustomOp1, Storage, Tensor};
use phylo_grad::nalgebra::{SMatrix, SVector};

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
    } else if x > 0.0 {
        // Same as below, but with exp(-x) so it does not overflow for large x
        let exp_neg_x = (-x).exp();
        return (1.0 - exp_neg_x * (1.0 + x)) / (1.0 - exp_neg_x).powi(2);
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

pub type Matrix20 = SMatrix<f64, 20, 20>;
pub type Vector20 = SVector<f64, 20>;

/// Mutation-selection model for a single site.
/// Mu is the symmetric mutation exchangeability matrix with the (unnormalized) mutation equilibrium on the diagonal.
/// log_pi are the unnormalized log equilibrium frequencies of the site.
/// Returns the symmetric S = diag(sqrt_pi) Q diag(1/sqrt_pi) and sqrt_pi.
/// The diagonal of S is set to 0, phylo_grad recomputes it from the off diagonal.
pub fn mutsel(Mu: &Matrix20, log_pi: &Vector20) -> (Matrix20, Vector20) {
    // Correct the fitness for the equilibrium of the mutation process
    let fitness = log_pi - Mu.diagonal().map(f64::ln);

    let sqrt_pi = softmax(log_pi).map(f64::sqrt);

    // S_ij = sqrt_pi_i * Q_ij / sqrt_pi_j, the ratio is computed in log space so tiny pi do not divide by 0
    let S = Matrix20::from_fn(|i, j| {
        if i == j {
            0.0
        } else {
            let Q_ij = Mu[(i, j)] * g_scalar(fitness[j] - fitness[i]);
            Q_ij * (0.5 * (log_pi[i] - log_pi[j])).exp()
        }
    });

    (S, sqrt_pi)
}

/// Gradient with respect to log_pi of a function of the outputs of `mutsel`, given its gradients grad_S and grad_sqrt_pi.
/// Mu is held fixed. Like phylo_grad, only the upper triangle of grad_S is used.
pub fn mutsel_backward_log_pi(
    Mu: &Matrix20,
    log_pi: &Vector20,
    grad_S: &Matrix20,
    grad_sqrt_pi: &Vector20,
) -> Vector20 {
    let fitness = log_pi - Mu.diagonal().map(f64::ln);
    let pi = softmax(log_pi);

    // d sqrt_pi_k / d log_pi_m = sqrt_pi_k * (delta_km - pi_m) / 2
    let weighted = grad_sqrt_pi.component_mul(&pi.map(f64::sqrt));
    let mut grad = 0.5 * (weighted - pi * weighted.sum());

    // S_ij = Mu_ij * exp((log_pi_i - log_pi_j) / 2) * g(fitness_j - fitness_i)
    for i in 0..20 {
        for j in (i + 1)..20 {
            let x = fitness[j] - fitness[i];
            let scale = Mu[(i, j)] * (0.5 * (log_pi[i] - log_pi[j])).exp();
            let dS_dlog_pi_i = scale * (0.5 * g_scalar(x) - dx_g_scalar(x));
            grad[i] += grad_S[(i, j)] * dS_dlog_pi_i;
            grad[j] -= grad_S[(i, j)] * dS_dlog_pi_i;
        }
    }

    grad
}

fn softmax(x: &Vector20) -> Vector20 {
    let max = x.max();
    let exp = x.map(|x_i| (x_i - max).exp());
    exp / exp.sum()
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

/// Returns a tensor of shape [L] with the total substitution rate per site,
/// keeping the computation differentiable w.r.t. S and sqrt_pi.
/// Assumes S is symmetric.
pub fn substitution_rates_tensor(S: &Tensor, sqrt_pi: &Tensor) -> Tensor {
    // Q_ij = S_ij * sqrt_pi_j * sqrt_pi_i (i != j)

    let pi_outer = sqrt_pi.unsqueeze(2).unwrap().broadcast_mul(&sqrt_pi.unsqueeze(1).unwrap()).unwrap();
    let weighted = S.mul(&pi_outer).unwrap();

    let total = weighted.sum(2).unwrap().sum(1).unwrap();

    let eye = Tensor::eye(20, candle_core::DType::F64, S.device()).unwrap();
    let diag = weighted.broadcast_mul(&eye).unwrap().sum(2).unwrap().sum(1).unwrap();

    total.sub(&diag).unwrap()
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

    #[test]
    fn test_mutsel_matches_candle() {
        let L = 3;
        let log_pi = Tensor::rand(-1.0, 1.0, &[L, 20], &candle_core::Device::Cpu).unwrap();
        let R = Tensor::rand(0.1, 2.0, &[20, 20], &candle_core::Device::Cpu).unwrap();
        let R = Mu(&R);

        let (S_ref, sqrt_pi_ref) = calc_rate_matrix(&R, &log_pi, &tensor_full(1.0, &[]));
        let S_ref = S_ref.to_vec3::<f64>().unwrap();
        let sqrt_pi_ref = sqrt_pi_ref.to_vec2::<f64>().unwrap();

        let R = R.to_vec2::<f64>().unwrap();
        let R = Matrix20::from_fn(|i, j| R[i][j]);
        let log_pi = log_pi.to_vec2::<f64>().unwrap();

        for site in 0..L {
            let (S, sqrt_pi) = mutsel(&R, &Vector20::from_column_slice(&log_pi[site]));

            assert!((S - S.transpose()).abs().max() < 1e-12);
            for i in 0..20 {
                assert!((sqrt_pi[i] - sqrt_pi_ref[site][i]).abs() < 1e-12);
                for j in 0..20 {
                    if i != j {
                        assert!((S[(i, j)] - S_ref[site][i][j]).abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn test_mutsel_backward_finite_differences() {
        let log_Mu = Tensor::rand(-1.0, 1.0, &[20, 20], &candle_core::Device::Cpu).unwrap();
        let Mu_vec = Mu(&log_Mu).to_vec2::<f64>().unwrap();
        let Mu = Matrix20::from_fn(|i, j| Mu_vec[i][j]);

        let log_pi = Vector20::from_fn(|i, _| ((i * 7) % 11) as f64 * 0.3 - 1.5);
        // Includes x = 0 for g, and a lower triangle that has to be ignored
        let grad_S = Matrix20::from_fn(|i, j| ((i * 3 + j * 5) % 7) as f64 - 3.0);
        let grad_sqrt_pi = Vector20::from_fn(|i, _| ((i * 5) % 9) as f64 - 4.0);

        let f = |log_pi: &Vector20| {
            let (S, sqrt_pi) = mutsel(&Mu, log_pi);
            S.upper_triangle().component_mul(&grad_S).sum() + sqrt_pi.dot(&grad_sqrt_pi)
        };

        let grad = mutsel_backward_log_pi(&Mu, &log_pi, &grad_S, &grad_sqrt_pi);

        let h = 1e-6;
        for k in 0..20 {
            let mut plus = log_pi;
            plus[k] += h;
            let mut minus = log_pi;
            minus[k] -= h;
            let fd = (f(&plus) - f(&minus)) / (2.0 * h);
            assert!((fd - grad[k]).abs() < 1e-6 * (1.0 + fd.abs()), "{k}: {fd} vs {}", grad[k]);
        }
    }

    #[test]
    fn test_dx_g_scalar_large_values() {
        assert!((dx_g_scalar(1000.0) - 1.0).abs() < 1e-12);
        assert!(dx_g_scalar(-1000.0).abs() < 1e-12);
        let h = 1e-6;
        for x in [-30.0, -2.0, -1e-3, 1e-3, 2.0, 30.0] {
            let fd = (g_scalar(x + h) - g_scalar(x - h)) / (2.0 * h);
            assert!((fd - dx_g_scalar(x)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_substitution_rates_tensor_matches_vec() {
        let L = 2;
        let log_pi = Tensor::rand(-1.0, 1.0, &[L, 20], &candle_core::Device::Cpu).unwrap();
        let R = Tensor::rand(0.1, 2.0, &[20, 20], &candle_core::Device::Cpu).unwrap();
        let R = Mu(&R);

        let (S, sqrt_pi) = calc_rate_matrix(&R, &log_pi, &tensor_full(1.0, &[]));

        let rates_vec = substitution_rates(&S, &sqrt_pi);
        let rates_tensor = substitution_rates_tensor(&S, &sqrt_pi).to_vec1::<f64>().unwrap();

        assert_eq!(rates_vec.len(), rates_tensor.len());
        for (a, b) in rates_vec.iter().zip(rates_tensor.iter()) {
            assert!((a - b).abs() < 1e-10, "rates differ: {} vs {}", a, b);
        }
    }
}