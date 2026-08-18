use candle_core::Tensor;


/// Read PCA components and singular values from the data module and return them as Tensors
/// The components are per row
pub fn read_pca_components() -> (Tensor, Tensor, Tensor) {
    let components: Vec<f64> = super::data::PCA_COMPONENTS
        .split_whitespace()
        .map(|s| s.parse::<f64>().unwrap())
        .collect();

    let singular_values = super::data::PCA_SINGULAR_VALUES
        .split_whitespace()
        .map(|s| s.parse::<f64>().unwrap())
        .collect();

    let mean_values = super::data::PCA_MEAN
        .split_whitespace()
        .map(|s| s.parse::<f64>().unwrap())
        .collect();

    let (components, singular_values, mean_values )= (
        Tensor::from_vec(components, &[20, 20], &candle_core::Device::Cpu).unwrap().narrow(0, 0, 19).unwrap(),
        Tensor::from_vec(singular_values, &[20], &candle_core::Device::Cpu).unwrap(),
        Tensor::from_vec(mean_values, &[20], &candle_core::Device::Cpu).unwrap(),
    );

    // We store the mean in PCA coordinates, so we need to convert it from log-frequencies to PCA coordinates
    let mean_values = log_freq_to_pca_coordinates(&components, &mean_values.unsqueeze(0).unwrap());

    (components, singular_values, mean_values)
}

pub fn log_freq_to_pca_coordinates(components: &Tensor, data: &Tensor) -> Tensor {
    let pca_coordinates = data.matmul(&components.transpose(0, 1).unwrap()).unwrap();
    pca_coordinates
}

pub fn pca_coordinates_to_log_freq(components: &Tensor, pca_coordinates: &Tensor) -> Tensor {
    let log_freq = pca_coordinates.matmul(&components).unwrap();
    log_freq
}

/// pca_mean needs to be in pca coordinates!
pub fn penalty_on_pca_coordinates(_singular_values: &Tensor, pca_coordinates: &Tensor, pca_mean: &Tensor) -> Tensor {
    // We ignore the last coordiante, because it has 0 singular value and is not penalized

    let alpha = super::data::PCA_ALPHA
        .split_whitespace()
        .map(|s| s.parse::<f64>().unwrap())
        .collect::<Vec<f64>>();
    let beta = super::data::PCA_BETA
        .split_whitespace()
        .map(|s| s.parse::<f64>().unwrap())
        .collect::<Vec<f64>>();

    let alpha = Tensor::from_vec(alpha, &[19], &candle_core::Device::Cpu).unwrap();
    let beta = Tensor::from_vec(beta, &[19], &candle_core::Device::Cpu).unwrap();


    let centered_pca_coordinates = pca_coordinates.broadcast_sub(&pca_mean).unwrap();
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
