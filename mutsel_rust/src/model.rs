use phylo_grad::nalgebra::{SMatrix, SVector};

use crate::data::LOG_M_EQUI;

/// Fixation factor g(x) = x / (1 - exp(-x))
pub(crate) fn g_scalar(x: f64) -> f64 {
    if x.abs() < 1e-6 {
        1.0 + x / 2.0 + x * x / 12.0
    } else {
        x / (-(-x).exp_m1())
    }
}

/// d/dx log g(x) = 1/x - 1/(exp(x) - 1)
pub(crate) fn dx_log_g_scalar(x: f64) -> f64 {
    // The closed form loses about eps / |x| to cancellation, so small arguments use the
    // Bernoulli series 1/2 - sum_k B_2k x^(2k-1) / (2k)!, whose truncation error is below 1e-20 for |x| < 0.1.
    if x.abs() < 0.1 {
        let x2 = x * x;
        0.5 - x * (1.0 / 12.0
            - x2 * (1.0 / 720.0
                - x2 * (1.0 / 30240.0 - x2 * (1.0 / 1209600.0 - x2 / 47900160.0))))
    } else {
        1.0 / x - 1.0 / x.exp_m1()
    }
}

/// S and sqrt_pi of a site with the mutation rate matrix Mu (zero diagonal, reversible with the equilibrium
/// data::M_EQUI) and the log frequencies log_pi (up to a constant): S_ij = sqrt(pi_i / pi_j) Mu_ij g(F_j - F_i),
/// F = log_pi - log M_EQUI. S is symmetric with zero diagonal, since g(x) / g(-x) = exp(x), so only the upper
/// triangle of Mu is read and the lower triangle of S is a copy of the upper one.
pub fn rate_matrix(Mu: &SMatrix<f64, 20, 20>, log_pi: &SVector<f64, 20>) -> (SMatrix<f64, 20, 20>, SVector<f64, 20>) {
    // pi = softmax(log_pi)
    let max_log_pi = log_pi.max();
    let unnormalized_pi = log_pi.map(|l| (l - max_log_pi).exp());
    let sqrt_pi = (unnormalized_pi / unnormalized_pi.sum()).map(f64::sqrt);

    let fitness = log_pi - LOG_M_EQUI;
    let mut S = SMatrix::<f64, 20, 20>::zeros();
    for i in 0..20 {
        for j in (i + 1)..20 {
            S[(i, j)] = Mu[(i, j)] * (0.5 * (log_pi[i] - log_pi[j])).exp() * g_scalar(fitness[j] - fitness[i]);
            S[(j, i)] = S[(i, j)];
        }
    }
    (S, sqrt_pi)
}

/// Backward pass of rate_matrix: the gradients w.r.t. Mu (upper triangle) and log_pi, given the gradients w.r.t. S
/// (the two triangles are added, as they are the same entries) and sqrt_pi
pub fn rate_matrix_backward(
    log_pi: &SVector<f64, 20>,
    S: &SMatrix<f64, 20, 20>,
    sqrt_pi: &SVector<f64, 20>,
    Mu: &SMatrix<f64, 20, 20>,
    grad_S: &SMatrix<f64, 20, 20>,
    grad_sqrt_pi: &SVector<f64, 20>,
) -> (SMatrix<f64, 20, 20>, SVector<f64, 20>) {
    let fitness = log_pi - LOG_M_EQUI;
    let mut grad_Mu = SMatrix::<f64, 20, 20>::zeros();
    let mut grad_log_pi = SVector::<f64, 20>::zeros();
    for i in 0..20 {
        for j in (i + 1)..20 {
            // w = dL / d log S_ij, every factor of S_ij enters multiplicatively
            let w = (grad_S[(i, j)] + grad_S[(j, i)]) * S[(i, j)];
            grad_Mu[(i, j)] = w / Mu[(i, j)];
            // d log S_ij / d log_pi_i = 1/2 - (log g)'(F_j - F_i), the opposite for log_pi_j
            let d = w * (0.5 - dx_log_g_scalar(fitness[j] - fitness[i]));
            grad_log_pi[i] += d;
            grad_log_pi[j] -= d;
        }
    }
    // d sqrt_pi_a / d log_pi_b = sqrt_pi_a * (delta_ab - pi_b) / 2
    let pi = sqrt_pi.component_mul(sqrt_pi);
    let weighted = grad_sqrt_pi.component_mul(sqrt_pi);
    grad_log_pi += (weighted - pi * weighted.sum()) * 0.5;
    (grad_Mu, grad_log_pi)
}

/// From S, sqrt_pi to the exchangeabilities R_ij = S_ij / (sqrt_pi_i sqrt_pi_j) and pi of IQ-TREE
pub fn phylograd2iqtree_parametrization(
    S: &SMatrix<f64, 20, 20>,
    sqrt_pi: &SVector<f64, 20>,
) -> (SMatrix<f64, 20, 20>, SVector<f64, 20>) {
    let R = SMatrix::<f64, 20, 20>::from_fn(|i, j| S[(i, j)] / (sqrt_pi[i] * sqrt_pi[j]));
    (R, sqrt_pi.component_mul(sqrt_pi))
}

/// Total substitution rate sum_{i != j} pi_i Q_ij = sum_{i != j} S_ij sqrt_pi_i sqrt_pi_j of a symmetric S
pub fn substitution_rate(S: &SMatrix<f64, 20, 20>, sqrt_pi: &SVector<f64, 20>) -> f64 {
    let mut rate = 0.0;
    for i in 0..20 {
        for j in (i + 1)..20 {
            rate += S[(i, j)] * sqrt_pi[i] * sqrt_pi[j];
        }
    }
    2.0 * rate
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_Mu() -> SMatrix<f64, 20, 20> {
        crate::optimization::Mu(&crate::prior::tests::test_prior().log_exchangeabilities)
    }

    fn test_log_pi() -> SVector<f64, 20> {
        SVector::from_fn(|i, _| ((7 * i) % 9) as f64 * 0.4 - 1.5)
    }

    #[test]
    fn lower_triangle_is_the_rate_matrix_formula() {
        let (Mu, log_pi) = (test_Mu(), test_log_pi());
        let (S, sqrt_pi) = rate_matrix(&Mu, &log_pi);
        let fitness = log_pi - LOG_M_EQUI;
        for i in 0..20 {
            for j in 0..i {
                let S_ij = sqrt_pi[i] / sqrt_pi[j] * Mu[(i, j)] * g_scalar(fitness[j] - fitness[i]);
                assert!((S[(i, j)] - S_ij).abs() < 1e-12 * S_ij, "{} {}: {} vs {}", i, j, S[(i, j)], S_ij);
            }
        }
        assert!((sqrt_pi.norm_squared() - 1.0).abs() < 1e-14);
        let (R, pi) = phylograd2iqtree_parametrization(&S, &sqrt_pi);
        // Q_ij = R_ij pi_j has the total rate of S
        let rate: f64 = (0..20).flat_map(|i| (0..20).map(move |j| (i, j))).filter(|(i, j)| i != j).map(|(i, j)| pi[i] * R[(i, j)] * pi[j]).sum();
        assert!((rate - substitution_rate(&S, &sqrt_pi)).abs() < 1e-12 * rate);
    }

    #[test]
    fn rate_matrix_backward_matches_finite_differences() {
        let (Mu, log_pi) = (test_Mu(), test_log_pi());
        let grad_S = SMatrix::<f64, 20, 20>::from_fn(|i, j| ((i * 20 + j) % 7) as f64 - 3.0);
        let grad_sqrt_pi = SVector::<f64, 20>::from_fn(|i, _| ((2 * i) % 9) as f64 - 4.0);
        let loss = |Mu: &SMatrix<f64, 20, 20>, log_pi: &SVector<f64, 20>| {
            let (S, sqrt_pi) = rate_matrix(Mu, log_pi);
            S.component_mul(&grad_S).sum() + sqrt_pi.dot(&grad_sqrt_pi)
        };
        let (S, sqrt_pi) = rate_matrix(&Mu, &log_pi);
        let (grad_Mu, grad_log_pi) = rate_matrix_backward(&log_pi, &S, &sqrt_pi, &Mu, &grad_S, &grad_sqrt_pi);

        let h = 1e-6;
        let check = |numeric: f64, analytic: f64| {
            assert!((numeric - analytic).abs() < 1e-6 * (1.0 + analytic.abs()), "{} vs {}", numeric, analytic);
        };
        for k in 0..20 {
            let (mut plus, mut minus) = (log_pi, log_pi);
            plus[k] += h;
            minus[k] -= h;
            check((loss(&Mu, &plus) - loss(&Mu, &minus)) / (2.0 * h), grad_log_pi[k]);
        }
        for (i, j) in [(0, 1), (1, 0), (4, 17), (19, 3)] {
            let (mut plus, mut minus) = (Mu, Mu);
            plus[(i, j)] += h;
            minus[(i, j)] -= h;
            check((loss(&plus, &log_pi) - loss(&minus, &log_pi)) / (2.0 * h), grad_Mu[(i, j)]);
        }
    }
}
