# ternary-activation

[![crates.io](https://img.shields.io/crates/v/ternary-activation.svg)](https://crates.io/crates/v/ternary-activation)
[![docs.rs](https://docs.rs/ternary-activation/badge.svg)](https://docs.rs/ternary-activation)

**Ternary activation functions for neural networks in ℤ₃ = {-1, 0, 1}.**

A Rust library implementing ℤ₃ analogues of classical neural network activation functions. Each activation maps continuous or ternary input to ternary output ∈ {-1, 0, 1}, providing the non-linear transformations needed for ternary neural networks.

## Why Ternary Activations?

In ternary neural networks, all values are trits ∈ {-1, 0, 1}. Traditional activation functions (ReLU, sigmoid, tanh, GELU) produce continuous outputs that must be quantized. This library provides **principled ternary approximations** of these functions that:

1. **Preserve the essential shape** of the original function (monotonicity, symmetry, etc.)
2. **Map cleanly to ℤ₃** with well-defined quantization boundaries
3. **Are efficient to compute** — no floating-point hardware needed at inference
4. **Maintain gradient-friendly properties** for training with Straight-Through Estimators

## Activation Functions

### Ternary Sign
The simplest activation: identity on {-1, 0, 1}, quantization for continuous input.
```
x < 0 → -1    x = 0 → 0    x > 0 → +1
```
Use case: Final quantization step, binary classification output.

### Ternary ReLU
Negative values are zeroed; positive and zero pass through.
```
x ≤ 0 → 0    x > 0 → +1
```
Note: Since output is ternary, the distinction between "small positive" and "large positive" is lost. This is the ℤ₃ equivalent of the hard sigmoid.

### Ternary Sigmoid
A three-level step function with configurable threshold.
```
x < -t → -1    |x| ≤ t → 0    x > t → +1
```
This is the ternary analogue of the sigmoid's S-curve, reduced to three discrete levels. The threshold parameter `t` controls the dead zone width.

### Ternary Tanh
Applies continuous tanh, then quantizes to the nearest trit.
```
tanh(x) < -1/3 → -1    |tanh(x)| ≤ 1/3 → 0    tanh(x) > 1/3 → +1
```
Preserves the symmetry and saturation properties of tanh while mapping to ℤ₃.

### Ternary Softmax
Argmax-style: marks the maximum position as +1, the minimum as -1, and all others as 0.
```
max(logits) → +1    min(logits) → -1    others → 0
```
This provides a "hard" attention mechanism suitable for ternary attention layers.

### Ternary GELU
Approximates GELU(x) = x · Φ(x) using the tanh approximation, then quantizes.
```
GELU(x) < -0.5 → -1    |GELU(x)| ≤ 0.5 → 0    GELU(x) > 0.5 → +1
```
The GELU function is approximately zero for negative inputs (since Φ(x) → 0) and approximately x for large positive inputs. In ternary space, this means negative inputs typically map to 0, while positive inputs map to 1.

### Swish Ternary
Approximates Swish(x) = x · σ(x), then quantizes.
```
Swish(x) < -0.5 → -1    |Swish(x)| ≤ 0.5 → 0    Swish(x) > 0.5 → +1
```
The Swish function is self-gated: for large positive x, σ(x) → 1 so Swish(x) → x. For negative x, Swish dips slightly below 0 then returns to 0.

### Leaky Ternary ReLU
Like ternary ReLU, but negative inputs below a threshold map to -1 instead of 0.
```
x < -t → -1    |x| ≤ t → 0    x > 0 → +1
```
The "leak" allows negative information to flow through, which can help with gradient flow during training.

## Usage

```rust
use ternary_activation::*;

// Individual activations
let t = ternary_sign(-2.5);       // Trit::NegOne
let t = ternary_relu(0.001);      // Trit::One
let t = ternary_sigmoid(0.0, 0.5); // Trit::Zero
let t = ternary_tanh(3.0);        // Trit::One
let t = ternary_gelu(1.0);        // Trit::One
let t = swish_ternary(2.0);       // Trit::One
let t = leaky_ternary_relu(-1.0, 0.1); // Trit::NegOne

// Softmax over a vector
let logits = vec![0.1, -0.5, 0.8];
let probs = ternary_softmax(&logits);
// → [Zero, NegOne, One]  (max=0.8→One, min=-0.5→NegOne)

// Batch application
let inputs = vec![-1.0, 0.0, 1.0, 2.0];
let outputs = apply_activation(&inputs, &ternary_relu);
// → [Zero, Zero, One, One]
```

## Quantization Strategy

All activations follow a consistent quantization pattern:

```
continuous_output < -threshold → Trit::NegOne
|continuous_output| ≤ threshold → Trit::Zero
continuous_output > threshold  → Trit::One
```

The threshold varies by function:
| Function | Threshold |
|----------|-----------|
| Sign | 0 |
| ReLU | 0 |
| Sigmoid | configurable (default 0.5) |
| Tanh | 1/3 (relative to tanh output range) |
| GELU | 0.5 |
| Swish | 0.5 |
| Leaky ReLU | configurable for negative, 0 for positive |

## Training with Ternary Activations

During training, use the **Straight-Through Estimator (STE)**: compute the continuous activation for the gradient, but pass the ternary output forward. This library provides the forward pass; the gradient estimation should be handled by your training framework.

## Testing

```bash
cargo test
```

All 18 tests pass, covering:
- Sign function identity on trits and continuous mapping
- ReLU zeroing behavior
- Sigmoid stepping with default and custom thresholds
- Tanh scaling and symmetry
- Softmax argmax, min-max, single element, empty, and equal-value cases
- GELU approximation bounds and known values
- Swish at key points
- Leaky ReLU with threshold
- Cross-function validation: all activations produce valid trits for all inputs

## License

MIT
