//! PCA coordinates of the site frequencies: c = P log_pi and log_pi = P^T c up to a constant, P the first 19 PCA
//! components of the log site frequencies (data::PCA_COMPONENTS, orthonormal rows orthogonal to the constant
//! direction, the 20th component), with the parameters of the generalized normal distribution of every coordinate.

use std::sync::LazyLock;

use phylo_grad::nalgebra::{SMatrix, SVector};

use crate::data;

/// Number of coordinates of the site frequencies
pub const NUM_COORDINATES: usize = 19;

pub struct Pca {
    /// P [19, 20]
    pub components: SMatrix<f64, NUM_COORDINATES, 20>,
    pub mean: SVector<f64, NUM_COORDINATES>,
    pub alpha: SVector<f64, NUM_COORDINATES>,
    pub beta: SVector<f64, NUM_COORDINATES>,
}

fn parse_values(text: &str) -> Vec<f64> {
    text.split_whitespace().map(|s| s.parse::<f64>().unwrap()).collect()
}

pub static PCA: LazyLock<Pca> = LazyLock::new(|| {
    let components = SMatrix::<f64, 20, 20>::from_row_slice(&parse_values(data::PCA_COMPONENTS));
    Pca {
        components: components.fixed_rows::<NUM_COORDINATES>(0).into_owned(),
        mean: SVector::from_vec(parse_values(data::PCA_MEAN)),
        alpha: SVector::from_vec(parse_values(data::PCA_ALPHA)),
        beta: SVector::from_vec(parse_values(data::PCA_BETA)),
    }
});

pub fn log_freq_to_coordinates(log_freq: &SVector<f64, 20>) -> SVector<f64, NUM_COORDINATES> {
    PCA.components * log_freq
}

/// Log frequencies up to a constant (summing to 0)
pub fn coordinates_to_log_freq(c: &SVector<f64, NUM_COORDINATES>) -> SVector<f64, 20> {
    PCA.components.transpose() * c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_are_orthonormal_and_sum_to_zero() {
        let P = PCA.components;
        assert!((P * P.transpose() - SMatrix::<f64, NUM_COORDINATES, NUM_COORDINATES>::identity()).abs().max() < 1e-12);
        assert!((P * SVector::<f64, 20>::from_element(1.0)).abs().max() < 1e-12);
        assert!(PCA.beta.min() > 1.0);
    }

    #[test]
    fn coordinates_roundtrip_up_to_a_constant() {
        let log_freq = SVector::<f64, 20>::from_fn(|i, _| ((i * 7) % 5) as f64 * 0.3 - 1.0);
        let difference = coordinates_to_log_freq(&log_freq_to_coordinates(&log_freq)) - log_freq;
        assert!(difference.add_scalar(-difference.mean()).abs().max() < 1e-12);
    }
}
