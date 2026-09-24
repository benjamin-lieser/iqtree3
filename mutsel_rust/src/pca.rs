use candle_core::Tensor;

pub struct PCA {
    /// The components are per row
    pub components: Tensor,
    pub alpha: Tensor,
    pub beta: Tensor,
    pub mean: Tensor,
    pub num_components: usize,
}

impl PCA {
    pub fn new(num_components: usize) -> PCA {
        let components: Vec<f64> = super::data::PCA_COMPONENTS
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        let mean_values = super::data::PCA_MEAN
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        let alpha = super::data::PCA_ALPHA
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        let beta = super::data::PCA_BETA
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        let components =
            Tensor::from_vec(components, &[20, 20], &candle_core::Device::Cpu).unwrap();
        let mean = Tensor::from_vec(mean_values, &[19], &candle_core::Device::Cpu).unwrap();
        let alpha = Tensor::from_vec(alpha, &[19], &candle_core::Device::Cpu).unwrap();
        let beta = Tensor::from_vec(beta, &[19], &candle_core::Device::Cpu).unwrap();

        PCA {
            components: components.narrow(0, 0, 19).unwrap(),
            alpha,
            beta,
            mean,
            num_components,
        }
    }

    pub fn log_freq_to_pca_coordinates(&self, data: &Tensor) -> Tensor {
        let pca_coordinates = data
            .matmul(&self.components.transpose(0, 1).unwrap())
            .unwrap();
        pca_coordinates.narrow(1, 0, self.num_components).unwrap()
    }

    pub fn pca_coordinates_to_log_freq(&self, pca_coordinates: &Tensor) -> Tensor {
        let pad_means = self
            .mean
            .narrow(0, self.num_components, 19 - self.num_components)
            .unwrap();

        let full_pca_coordinates = if self.num_components == 19 {
            pca_coordinates.clone()
        } else {
            Tensor::cat(&[pca_coordinates, &pad_means.unsqueeze(0).unwrap()], 1).unwrap()
        };
        let log_freq = full_pca_coordinates.matmul(&self.components).unwrap();
        log_freq
    }

    pub fn penalty_on_pca_coordinates(&self, pca_coordinates: &Tensor) -> Tensor {
        let means = self.mean.narrow(0, 0, self.num_components).unwrap();
        let alpha = self.alpha.narrow(0, 0, self.num_components).unwrap();
        let beta = self.beta.narrow(0, 0, self.num_components).unwrap();

        let centered_pca_coordinates = pca_coordinates
            .broadcast_sub(&means.unsqueeze(0).unwrap())
            .unwrap();
        let penalty = centered_pca_coordinates
            .abs()
            .unwrap()
            .broadcast_div(&alpha.unsqueeze(0).unwrap())
            .unwrap()
            .broadcast_pow(&beta.unsqueeze(0).unwrap())
            .unwrap()
            .sum_all()
            .unwrap();
        penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
    }

    #[test]
    fn projection_returns_requested_number_of_components() {
        let pca = PCA::new(4);
        let data = Tensor::from_vec(
            (0..40).map(|x| x as f64 * 0.01).collect(),
            &[2, 20],
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let coords = pca.log_freq_to_pca_coordinates(&data);

        assert_eq!(coords.dims(), &[2, 4]);
    }

    #[test]
    fn projection_returns_requested_number_of_components_19() {
        let pca = PCA::new(19);
        let data = Tensor::from_vec(
            (0..40).map(|x| x as f64 * 0.01).collect(),
            &[2, 20],
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let coords = pca.log_freq_to_pca_coordinates(&data);

        assert_eq!(coords.dims(), &[2, 19]);
    }
    #[test]
    fn inverse_fills_missing_components_with_means() {
        let pca = PCA::new(3);
        let coords =
            Tensor::from_vec(vec![0.2, -0.3, 0.5], &[1, 3], &candle_core::Device::Cpu).unwrap();

        let reconstructed = pca.pca_coordinates_to_log_freq(&coords);

        let expected_tail = pca.mean.narrow(0, 3, 16).unwrap().unsqueeze(0).unwrap();
        let full_coords = Tensor::cat(&[&coords, &expected_tail], 1).unwrap();
        let expected = full_coords.matmul(&pca.components).unwrap();

        assert_eq!(reconstructed.dims(), &[1, 20]);
        assert!(max_abs_diff(&reconstructed, &expected) < 1e-12);
    }

    #[test]
    fn penalty_is_zero_at_component_means() {
        let pca = PCA::new(5);
        let mean_coords = pca.mean.narrow(0, 0, 5).unwrap().unsqueeze(0).unwrap();

        let penalty = pca.penalty_on_pca_coordinates(&mean_coords);

        assert!(penalty.to_scalar::<f64>().unwrap().abs() < 1e-12);
    }
}
