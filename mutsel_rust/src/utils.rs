use candle_core::Tensor;

pub fn tensor_full(value: f64, dims: &[usize]) -> Tensor {
    Tensor::full(value, dims, &candle_core::Device::Cpu).unwrap()
}
