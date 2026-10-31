use phylo_grad::nalgebra::DVector;

/// Maximizes f with L-BFGS (memory 6) and a backtracking line search, starting at init, until the gradient norm
/// is below gradient_tolerance or an iteration improves f by less than value_tolerance relative to |f|. eval
/// returns f and its gradient, None where f is not finite. Returns init if f is not finite there; the result is
/// never worse than init. init has to be finite.
pub fn lbfgs<F: FnMut(&DVector<f64>) -> Option<(f64, DVector<f64>)>>(
    mut eval: F,
    init: DVector<f64>,
    max_iterations: usize,
    gradient_tolerance: f64,
    value_tolerance: f64,
) -> DVector<f64> {
    let mut x = init;
    let Some((mut value, mut gradient)) = eval(&x) else {
        return x;
    };
    // (s, y) pairs of the minimization of -f
    let mut history: std::collections::VecDeque<(DVector<f64>, DVector<f64>)> = std::collections::VecDeque::new();
    for _ in 0..max_iterations {
        if gradient.norm() < gradient_tolerance {
            break;
        }
        // Two-loop recursion for the ascent direction
        let mut q = gradient.clone();
        let mut coefficients = Vec::with_capacity(history.len());
        for (s, y) in history.iter().rev() {
            let rho = 1.0 / y.dot(s);
            let alpha = rho * s.dot(&q);
            q -= y * alpha;
            coefficients.push((alpha, rho));
        }
        let gamma = match history.back() {
            Some((s, y)) => s.dot(y) / y.dot(y),
            None => 1.0f64.min(1.0 / gradient.norm()),
        };
        let mut direction = q * gamma;
        for ((s, y), (alpha, rho)) in history.iter().zip(coefficients.iter().rev()) {
            let beta = rho * y.dot(&direction);
            direction += s * (alpha - beta);
        }
        if gradient.dot(&direction) <= 0.0 {
            history.clear();
            direction = &gradient * 1.0f64.min(1.0 / gradient.norm());
        }

        let slope = gradient.dot(&direction);
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..30 {
            let candidate = &x + &direction * step;
            if let Some((candidate_value, candidate_gradient)) = eval(&candidate) {
                if candidate_value >= value + 1e-4 * step * slope {
                    accepted = Some((candidate, candidate_value, candidate_gradient));
                    break;
                }
            }
            step *= 0.5;
        }
        let Some((next, next_value, next_gradient)) = accepted else {
            break;
        };
        let improvement = next_value - value;
        let s = &next - &x;
        let y = &gradient - &next_gradient;
        if s.dot(&y) > 1e-12 {
            history.push_back((s, y));
            if history.len() > 6 {
                history.pop_front();
            }
        }
        x = next;
        value = next_value;
        gradient = next_gradient;
        if improvement <= value_tolerance * value.abs() {
            break;
        }
    }
    x
}

/// lbfgs over the finite entries of x, the others (the log of a zero, e.g. of the root branch length) stay fixed.
/// eval gets and returns the full vector and gradient. Returns the number of evaluations.
pub fn maximize_finite<F: FnMut(&[f64]) -> Option<(f64, Vec<f64>)>>(
    mut eval: F,
    x: &mut [f64],
    max_iterations: usize,
    value_tolerance: f64,
) -> usize {
    let free: Vec<usize> = (0..x.len()).filter(|&k| x[k].is_finite()).collect();
    let mut full = x.to_vec();
    let mut num_evaluations = 0;
    let optimum = lbfgs(
        |free_values| {
            num_evaluations += 1;
            for (&k, &value) in free.iter().zip(free_values.iter()) {
                full[k] = value;
            }
            let (value, gradient) = eval(&full)?;
            value.is_finite().then(|| (value, DVector::from_iterator(free.len(), free.iter().map(|&k| gradient[k]))))
        },
        DVector::from_iterator(free.len(), free.iter().map(|&k| x[k])),
        max_iterations,
        0.0,
        value_tolerance,
    );
    for (&k, &value) in free.iter().zip(optimum.iter()) {
        x[k] = value;
    }
    num_evaluations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximize_finite_keeps_infinite_entries_and_finds_the_optimum() {
        // f = -sum_k (x_k - k)^2 over the finite entries
        let mut x = vec![0.0, f64::NEG_INFINITY, 5.0, -3.0];
        let eval = |x: &[f64]| {
            let mut value = 0.0;
            let mut gradient = vec![0.0; x.len()];
            for k in (0..x.len()).filter(|&k| x[k].is_finite()) {
                value -= (x[k] - k as f64).powi(2);
                gradient[k] = -2.0 * (x[k] - k as f64);
            }
            Some((value, gradient))
        };
        maximize_finite(eval, &mut x, 100, 0.0);
        assert_eq!(x[1], f64::NEG_INFINITY);
        for k in [0, 2, 3] {
            assert!((x[k] - k as f64).abs() < 1e-8, "{}: {}", k, x[k]);
        }
    }
}
