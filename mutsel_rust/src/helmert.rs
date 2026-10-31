//! Helmert coordinates of the site frequencies: an orthonormal basis of the sum-zero log frequencies, so that
//! log_pi = basis^T * x up to a constant and x = basis * log_pi.

use phylo_grad::nalgebra::{SMatrix, SVector};

/// Number of coordinates of the site frequencies
pub const NUM_COORDINATES: usize = 19;

/// Helmert basis [19, 20]: row k is (1, ..., 1, -(k + 1), 0, ..., 0) / sqrt((k + 1)(k + 2)) with k + 1 ones
pub fn basis() -> SMatrix<f64, NUM_COORDINATES, 20> {
    SMatrix::from_fn(|k, i| {
        let norm = (((k + 1) * (k + 2)) as f64).sqrt();
        if i <= k {
            1.0 / norm
        } else if i == k + 1 {
            -((k + 1) as f64) / norm
        } else {
            0.0
        }
    })
}

pub fn log_freq_to_coordinates(log_freq: &SVector<f64, 20>) -> SVector<f64, NUM_COORDINATES> {
    basis() * log_freq
}

/// Log frequencies up to a constant
pub fn coordinates_to_log_freq(x: &SVector<f64, NUM_COORDINATES>) -> SVector<f64, 20> {
    basis().transpose() * x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_is_orthonormal_and_sums_to_zero() {
        let basis = basis();
        assert!((basis * basis.transpose() - SMatrix::<f64, NUM_COORDINATES, NUM_COORDINATES>::identity()).abs().max() < 1e-14);
        assert!((basis * SVector::<f64, 20>::from_element(1.0)).abs().max() < 1e-14);
    }

    #[test]
    fn coordinates_roundtrip_up_to_a_constant() {
        let log_freq = SVector::<f64, 20>::from_fn(|i, _| ((i * 7) % 5) as f64 * 0.3 - 1.0);
        let difference = coordinates_to_log_freq(&log_freq_to_coordinates(&log_freq)) - log_freq;
        assert!(difference.add_scalar(-difference.mean()).abs().max() < 1e-12);
    }
}
