//! # ternary-activation
//!
//! Ternary activation functions for neural networks operating in ℤ₃ = {-1, 0, 1}.
//!
//! Each activation maps continuous or ternary input to ternary output. These
//! functions are the ℤ₃ analogues of classical activation functions, quantized
//! to the nearest trit.

/// A trit in ℤ₃ = {-1, 0, 1}.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit {
    NegOne = -1,
    Zero = 0,
    One = 1,
}

impl Trit {
    pub fn to_i8(self) -> i8 {
        self as i8
    }

    pub fn from_i8(v: i8) -> Self {
        match v.cmp(&0) {
            std::cmp::Ordering::Less => Trit::NegOne,
            std::cmp::Ordering::Equal => Trit::Zero,
            std::cmp::Ordering::Greater => Trit::One,
        }
    }
}

/// Ternary sign function: identity on {-1, 0, 1}.
///
/// Maps continuous input to the nearest trit:
///   - x < 0 → -1
///   - x == 0 → 0
///   - x > 0 → 1
///
/// For already-ternary input, this is the identity function.
pub fn ternary_sign(x: f64) -> Trit {
    if x < 0.0 {
        Trit::NegOne
    } else if x == 0.0 {
        Trit::Zero
    } else {
        Trit::One
    }
}

/// Ternary ReLU: negative → 0, zero and positive pass through.
///
/// Maps:
///   - x ≤ 0 → 0
///   - x > 0 → 1
pub fn ternary_relu(x: f64) -> Trit {
    if x <= 0.0 {
        Trit::Zero
    } else {
        Trit::One
    }
}

/// Ternary sigmoid: step function mapping to nearest trit.
///
/// Approximates sigmoid as a three-level step:
///   - x < -threshold → -1
///   - |x| ≤ threshold → 0
///   - x > threshold → 1
///
/// Default threshold is 0.5.
pub fn ternary_sigmoid(x: f64, threshold: f64) -> Trit {
    if x < -threshold {
        Trit::NegOne
    } else if x > threshold {
        Trit::One
    } else {
        Trit::Zero
    }
}

/// Ternary tanh: scale continuous tanh to {-1, 0, 1}.
///
/// Applies standard tanh then quantizes:
///   - tanh(x) < -1/3 → -1
///   - |tanh(x)| ≤ 1/3 → 0
///   - tanh(x) > 1/3 → 1
pub fn ternary_tanh(x: f64) -> Trit {
    let t = x.tanh();
    if t < -1.0 / 3.0 {
        Trit::NegOne
    } else if t > 1.0 / 3.0 {
        Trit::One
    } else {
        Trit::Zero
    }
}

/// Ternary softmax: argmax-style, returns one-hot in ternary.
///
/// Given a slice of logits, returns a vector of trits where the maximum
/// position is 1, the minimum is -1, and all others are 0.
///
/// For a single value, maps to sign. For multiple values, does hard argmax.
pub fn ternary_softmax(logits: &[f64]) -> Vec<Trit> {
    if logits.is_empty() {
        return vec![];
    }
    if logits.len() == 1 {
        return vec![ternary_sign(logits[0])];
    }

    let mut result = vec![Trit::Zero; logits.len()];

    // Find max and min indices
    let mut max_idx = 0;
    let mut min_idx = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[max_idx] {
            max_idx = i;
        }
        if v < logits[min_idx] {
            min_idx = i;
        }
    }

    if max_idx != min_idx {
        result[max_idx] = Trit::One;
        result[min_idx] = Trit::NegOne;
    } else {
        // All equal: assign One to the first element
        result[max_idx] = Trit::One;
    }

    result
}

/// Ternary GELU approximation.
///
/// Approximates GELU(x) = x · Φ(x) where Φ is the standard normal CDF.
/// Quantizes to ℤ₃:
///   - GELU(x) < -0.5 → -1
///   - |GELU(x)| ≤ 0.5 → 0
///   - GELU(x) > 0.5 → 1
///
/// Uses the tanh approximation: GELU(x) ≈ x · 0.5 · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
pub fn ternary_gelu(x: f64) -> Trit {
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    let gelu = x * 0.5 * (1.0 + inner.tanh());

    if gelu < -0.5 {
        Trit::NegOne
    } else if gelu > 0.5 {
        Trit::One
    } else {
        Trit::Zero
    }
}

/// Swish ternary: ternary approximation of Swish(x) = x · sigmoid(x).
///
/// Quantizes swish to nearest trit:
///   - swish(x) < -0.5 → -1
///   - |swish(x)| ≤ 0.5 → 0
///   - swish(x) > 0.5 → 1
pub fn swish_ternary(x: f64) -> Trit {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    let swish = x * sigmoid;

    if swish < -0.5 {
        Trit::NegOne
    } else if swish > 0.5 {
        Trit::One
    } else {
        Trit::Zero
    }
}

/// Leaky ternary ReLU: like ternary ReLU but negative values map to -1
/// instead of 0 (scaled by a leak factor).
///
///   - x < -leak_threshold → -1
///   - |x| ≤ leak_threshold → 0
///   - x > 0 → 1
pub fn leaky_ternary_relu(x: f64, leak_threshold: f64) -> Trit {
    if x < -leak_threshold {
        Trit::NegOne
    } else if x > 0.0 {
        Trit::One
    } else {
        Trit::Zero
    }
}

/// Apply a ternary activation to a slice of continuous inputs.
pub fn apply_activation(inputs: &[f64], func: &dyn Fn(f64) -> Trit) -> Vec<Trit> {
    inputs.iter().map(|&x| func(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Sign function tests ===

    #[test]
    fn test_sign_identity_on_trits() {
        // Sign is identity for already-ternary inputs
        assert_eq!(ternary_sign(-1.0), Trit::NegOne);
        assert_eq!(ternary_sign(0.0), Trit::Zero);
        assert_eq!(ternary_sign(1.0), Trit::One);
    }

    #[test]
    fn test_sign_maps_continuous() {
        assert_eq!(ternary_sign(-0.5), Trit::NegOne);
        assert_eq!(ternary_sign(0.001), Trit::One);
        assert_eq!(ternary_sign(-100.0), Trit::NegOne);
        assert_eq!(ternary_sign(100.0), Trit::One);
    }

    // === ReLU tests ===

    #[test]
    fn test_relu_zeroing() {
        // Negative and zero → 0
        assert_eq!(ternary_relu(-1.0), Trit::Zero);
        assert_eq!(ternary_relu(-0.001), Trit::Zero);
        assert_eq!(ternary_relu(0.0), Trit::Zero);
        // Positive → 1
        assert_eq!(ternary_relu(0.001), Trit::One);
        assert_eq!(ternary_relu(1.0), Trit::One);
        assert_eq!(ternary_relu(100.0), Trit::One);
    }

    // === Sigmoid tests ===

    #[test]
    fn test_sigmoid_stepping() {
        // Default threshold = 0.5
        assert_eq!(ternary_sigmoid(-1.0, 0.5), Trit::NegOne);
        assert_eq!(ternary_sigmoid(-0.5, 0.5), Trit::Zero); // boundary
        assert_eq!(ternary_sigmoid(0.0, 0.5), Trit::Zero);
        assert_eq!(ternary_sigmoid(0.5, 0.5), Trit::Zero); // boundary
        assert_eq!(ternary_sigmoid(1.0, 0.5), Trit::One);
    }

    #[test]
    fn test_sigmoid_custom_threshold() {
        assert_eq!(ternary_sigmoid(-0.2, 0.1), Trit::NegOne);
        assert_eq!(ternary_sigmoid(0.0, 0.1), Trit::Zero);
        assert_eq!(ternary_sigmoid(0.2, 0.1), Trit::One);
    }

    // === Tanh tests ===

    #[test]
    fn test_tanh_scaling() {
        // Large negative → -1
        assert_eq!(ternary_tanh(-10.0), Trit::NegOne);
        // Near zero → 0
        assert_eq!(ternary_tanh(0.0), Trit::Zero);
        // Large positive → 1
        assert_eq!(ternary_tanh(10.0), Trit::One);
    }

    #[test]
    fn test_tanh_symmetry() {
        let a = ternary_tanh(2.0);
        let b = ternary_tanh(-2.0);
        assert_eq!(a, Trit::One);
        assert_eq!(b, Trit::NegOne);
    }

    // === Softmax tests ===

    #[test]
    fn test_softmax_argmax_behavior() {
        let logits = vec![0.1, 0.5, 0.3];
        let result = ternary_softmax(&logits);
        assert_eq!(result[1], Trit::One);   // max at index 1
        assert_eq!(result[0], Trit::NegOne); // min at index 0
    }

    #[test]
    fn test_softmax_single_element() {
        let result = ternary_softmax(&[2.5]);
        assert_eq!(result[0], Trit::One);

        let result = ternary_softmax(&[-1.0]);
        assert_eq!(result[0], Trit::NegOne);

        let result = ternary_softmax(&[0.0]);
        assert_eq!(result[0], Trit::Zero);
    }

    #[test]
    fn test_softmax_empty() {
        let result = ternary_softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_min_max_different() {
        let logits = vec![-5.0, 0.0, 5.0];
        let result = ternary_softmax(&logits);
        assert_eq!(result[0], Trit::NegOne); // min
        assert_eq!(result[1], Trit::Zero);   // middle
        assert_eq!(result[2], Trit::One);    // max
    }

    #[test]
    fn test_softmax_all_equal() {
        let logits = vec![1.0, 1.0, 1.0];
        let result = ternary_softmax(&logits);
        // All equal: max_idx == min_idx, so just One at that index
        assert_eq!(result[0], Trit::One);
    }

    // === GELU tests ===

    #[test]
    fn test_gelu_bounds() {
        // GELU for large negative → GELU ≈ 0 → Zero (GELU squashes negatives toward 0)
        assert_eq!(ternary_gelu(-10.0), Trit::Zero);
        // GELU near zero → small value → 0
        assert_eq!(ternary_gelu(0.0), Trit::Zero);
        // GELU for large positive → should be 1 (GELU(5) ≈ 5.0)
        assert_eq!(ternary_gelu(5.0), Trit::One);
    }

    #[test]
    fn test_gelu_approximation() {
        // GELU(1) ≈ 0.841 → 1
        assert_eq!(ternary_gelu(1.0), Trit::One);
        // GELU(-1) ≈ -0.159 → within [-0.5, 0.5] → Zero
        assert_eq!(ternary_gelu(-1.0), Trit::Zero);
        // GELU(-4) ≈ -0.0001 → still essentially 0 → Zero
        assert_eq!(ternary_gelu(-4.0), Trit::Zero);
    }

    // === Swish tests ===

    #[test]
    fn test_swish_ternary() {
        // Swish(0) = 0 → 0
        assert_eq!(swish_ternary(0.0), Trit::Zero);
        // Swish for large positive → 1
        assert_eq!(swish_ternary(10.0), Trit::One);
        // Swish for large negative → sigmoid(-x) ≈ 0, so swish ≈ 0 → 0
        // Actually swish(-10) = -10 * sigmoid(-10) ≈ -10 * 0 ≈ 0 → 0
        assert_eq!(swish_ternary(-10.0), Trit::Zero);
        // Swish(-1) = -1 * sigmoid(-1) = -1 * 0.269 = -0.269 → 0
        assert_eq!(swish_ternary(-1.0), Trit::Zero);
    }

    // === Leaky ReLU tests ===

    #[test]
    fn test_leaky_relu() {
        assert_eq!(leaky_ternary_relu(1.0, 0.1), Trit::One);
        assert_eq!(leaky_ternary_relu(0.0, 0.1), Trit::Zero);
        assert_eq!(leaky_ternary_relu(-0.05, 0.1), Trit::Zero);
        assert_eq!(leaky_ternary_relu(-1.0, 0.1), Trit::NegOne);
    }

    // === Cross-function tests: all handle {-1, 0, 1} input correctly ===

    #[test]
    fn test_all_activations_on_trit_inputs() {
        let trit_values = [-1.0_f64, 0.0, 1.0];

        for &v in &trit_values {
            let sign_result = ternary_sign(v);
            assert!(matches!(sign_result, Trit::NegOne | Trit::Zero | Trit::One));

            let relu_result = ternary_relu(v);
            assert!(matches!(relu_result, Trit::NegOne | Trit::Zero | Trit::One));

            let sigmoid_result = ternary_sigmoid(v, 0.5);
            assert!(matches!(sigmoid_result, Trit::NegOne | Trit::Zero | Trit::One));

            let tanh_result = ternary_tanh(v);
            assert!(matches!(tanh_result, Trit::NegOne | Trit::Zero | Trit::One));

            let gelu_result = ternary_gelu(v);
            assert!(matches!(gelu_result, Trit::NegOne | Trit::Zero | Trit::One));

            let swish_result = swish_ternary(v);
            assert!(matches!(swish_result, Trit::NegOne | Trit::Zero | Trit::One));

            let leaky_result = leaky_ternary_relu(v, 0.5);
            assert!(matches!(leaky_result, Trit::NegOne | Trit::Zero | Trit::One));
        }
    }

    #[test]
    fn test_apply_activation_batch() {
        let inputs = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let outputs = apply_activation(&inputs, &ternary_sign);
        assert_eq!(outputs.len(), 5);
        assert_eq!(outputs[0], Trit::NegOne);
        assert_eq!(outputs[2], Trit::Zero);
        assert_eq!(outputs[4], Trit::One);
    }
}
