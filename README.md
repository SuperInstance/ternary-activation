# ternary-activation

**Non-linearities born ternary — ReLU, sigmoid, tanh, GELU, and softmax, mapped to {-1, 0, +1}.**

[![crates.io](https://img.shields.io/crates/v/ternary-activation.svg)](https://crates.io/crates/ternary-activation)
[![docs.rs](https://docs.rs/ternary-activation/badge.svg)](https://docs.rs/ternary-activation)

## Why This Exists

Activation functions are what make neural networks nonlinear. Without them, stacking layers is just one big linear transformation. ReLU kills negatives. Sigmoid squashes to (0, 1). GELU gates by magnitude. Tanh saturates symmetrically.

In a ternary network, all values are {-1, 0, +1}. You can't apply continuous ReLU and get a continuous output — you'd leave ternary space. You need activation functions that *quantize* their outputs while preserving the essential shape of the original: monotonicity, symmetry, saturation behavior.

This crate provides principled ternary approximations of seven classic activation functions, plus batch application utilities.

## The Key Insight

Every activation in this crate follows one pattern: compute the continuous version, then quantize to the nearest trit via threshold rounding:

```
f(x) < -threshold → -1
|f(x)| ≤ threshold →  0
f(x) >  threshold → +1
```

The threshold varies by function. For sign, it's 0. For tanh, it's 1/3 (relative to tanh's output range). For GELU and Swish, it's 0.5. This isn't arbitrary — each threshold is chosen so the ternary approximation preserves the critical inflection points of the continuous function.

The result: activation functions that behave like their continuous counterparts *at decision boundaries* while living entirely in Z₃.

## Quick Start

```toml
[dependencies]
ternary-activation = "0.1"
```

```rust
use ternary_activation::*;

// Individual activations — continuous input → Trit output
let a = ternary_sign(-2.5);        // NegOne
let b = ternary_relu(0.001);       // One
let c = ternary_sigmoid(0.0, 0.5); // Zero (in the dead zone)
let d = ternary_tanh(3.0);         // One (saturated positive)
let e = ternary_gelu(1.0);         // One
let f = swish_ternary(2.0);        // One
let g = leaky_ternary_relu(-1.0, 0.1); // NegOne (leaks negative signal)

// Ternary softmax — attention / classification
let logits = vec![0.1, -0.5, 0.8];
let attn = ternary_softmax(&logits);
// → [Zero, NegOne, One]  (max=0.8→One, min=-0.5→NegOne)

// Batch application
let inputs = vec![-1.0, 0.0, 1.0, 2.0];
let outputs = apply_activation(&inputs, &ternary_relu);
// → [Zero, Zero, One, One]
```

## The Activation Functions

### Ternary Sign — The Identity

```
x < 0 → -1    x = 0 → 0    x > 0 → +1
```

Identity on ternary input. Quantization for continuous input. The simplest possible nonlinearity.

**Use when:** Final output quantization, binary decisions.

### Ternary ReLU — The Killer of Negatives

```
x ≤ 0 → 0    x > 0 → +1
```

ReLU kills negative values. Ternary ReLU does the same: everything non-positive becomes zero, everything positive becomes +1. Since the output is ternary, there's no distinction between "slightly positive" and "very positive" — it's all +1.

**Use when:** Standard hidden layer activation, like ReLU in float networks.

### Ternary Sigmoid — The Three-Level Step

```
x < -t → -1    |x| ≤ t → 0    x > t → +1
```

A step function with a configurable dead zone. Threshold `t` controls the width of the "neutral" region. This is the ternary analog of sigmoid's S-curve, reduced to three discrete levels.

**Use when:** Gated architectures, where you want a clear "off / uncertain / on" decision.

### Ternary Tanh — Symmetric Saturation

```
tanh(x) < -1/3 → -1    |tanh(x)| ≤ 1/3 → 0    tanh(x) > 1/3 → +1
```

Applies continuous tanh, then quantizes at ±1/3 boundaries. Preserves tanh's symmetry (f(-x) = -f(x)) and saturation behavior.

**Use when:** Symmetric activations, recurrent networks, anywhere tanh is traditional.

### Ternary Softmax — Hard Attention

```
max(logits) → +1    min(logits) → -1    others → 0
```

Not a probability distribution — it's a hard decision. The maximum gets +1, the minimum gets -1, everything else is 0. This is ternary attention: focus on the strongest signal, anti-focus on the weakest.

**Use when:** Ternary attention mechanisms, hard classification, sparse selection.

### Ternary GELU — The Gaussian Gate

```
GELU(x) < -0.5 → -1    |GELU(x)| ≤ 0.5 → 0    GELU(x) > 0.5 → +1
```

GELU gates by magnitude: small values are suppressed, large values pass through. In ternary space, this means negative inputs → 0 (suppressed), large positive inputs → +1. Uses the tanh approximation for the CDF.

**Use when:** Transformer architectures where GELU is the standard. Drop-in ternary replacement.

### Swish Ternary — Self-Gated

```
Swish(x) < -0.5 → -1    |Swish(x)| ≤ 0.5 → 0    Swish(x) > 0.5 → +1
```

Swish(x) = x × σ(x). Self-gated: the sigmoid acts as a gate on the input itself. For large positive x, Swish → x. For negative x, Swish dips below zero then returns to 0.

**Use when:** When Swish outperforms ReLU in your architecture and you want the ternary equivalent.

### Leaky Ternary ReLU — Negative Information Flows

```
x < -t → -1    |x| ≤ t → 0    x > 0 → +1
```

Like ternary ReLU, but negative inputs below `-t` map to -1 instead of 0. The "leak" allows negative information to flow through the network, improving gradient flow during training.

**Use when:** Gradient flow through negative activations matters. Ternary networks with deep architectures.

## Quantization Thresholds

| Function | Threshold | Rationale |
|----------|-----------|-----------|
| Sign | 0 | Zero is the natural boundary |
| ReLU | 0 | Positive/non-positive split |
| Sigmoid | configurable | Task-dependent dead zone |
| Tanh | 1/3 | Relative to tanh output range [-1, 1] |
| GELU | 0.5 | Half the ternary step size |
| Swish | 0.5 | Half the ternary step size |
| Leaky ReLU | configurable | Task-dependent negative threshold |

## API Reference

### Core Type

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Trit { NegOne = -1, Zero = 0, One = 1 }

impl Trit {
    fn to_i8(self) -> i8;
    fn from_i8(v: i8) -> Self;
}
```

### Activation Functions

```rust
fn ternary_sign(x: f64) -> Trit;
fn ternary_relu(x: f64) -> Trit;
fn ternary_sigmoid(x: f64, threshold: f64) -> Trit;
fn ternary_tanh(x: f64) -> Trit;
fn ternary_softmax(logits: &[f64]) -> Vec<Trit>;
fn ternary_gelu(x: f64) -> Trit;
fn swish_ternary(x: f64) -> Trit;
fn leaky_ternary_relu(x: f64, leak_threshold: f64) -> Trit;
```

### Batch Application

```rust
fn apply_activation(inputs: &[f64], func: &dyn Fn(f64) -> Trit) -> Vec<Trit>;
```

## Training: The Straight-Through Estimator

Ternary activations are not differentiable (they're step functions). During training, use the **Straight-Through Estimator (STE)**: pass the ternary value forward, but use the continuous activation for the gradient.

```rust
// Pseudocode for STE during training:
let continuous = x.tanh();
let ternary = ternary_tanh(x);
// Forward: use ternary
// Backward: gradient flows through continuous as if ternary == continuous
```

This crate provides the forward pass. Your training framework handles the STE gradient estimation.

## Real-World Example: Ternary Transformer Block

```rust
// Attention scores → ternary softmax → hard attention
let attention_scores = vec![-0.5, 0.3, 1.2, -0.1];
let attention_weights = ternary_softmax(&attention_scores);
// [NegOne, Zero, One, Zero] — focus on position 2, ignore position 0

// After attention, apply GELU to FFN output
let ffn_output = vec![-0.3, 0.8, 1.5, -2.0];
let activated = apply_activation(&ffn_output, &ternary_gelu);
// [Zero, One, One, Zero] — GELU suppresses negatives and small values
```

## Performance Characteristics

All activations are O(1) per element (constant-time comparisons and lookups). The tanh-based activations (ternary_tanh, ternary_gelu, swish_ternary) involve one floating-point operation each — still O(1), but slightly more expensive than pure comparison-based activations (sign, relu, sigmoid).

Ternary softmax is O(n) for n logits (find max and min in one pass).

Memory: Trit is an enum (1 byte). A Vec<Trit> for 1M activations is 1 MB.

## Ecosystem Connections

Activations are the glue between layers:

- [`ternary-matmul`](https://github.com/SuperInstance/ternary-matmul) — outputs fed to these activations
- [`ternary-conv`](https://github.com/SuperInstance/ternary-conv) — convolution outputs fed to these activations
- [`ternary-norm`](https://github.com/SuperInstance/ternary-norm) — normalization typically precedes activation
- [`ternary-loss`](https://github.com/SuperInstance/ternary-loss) — losses computed on activated outputs
- [`ternary-optimizer`](https://github.com/SuperInstance/ternary-optimizer) — STE gradients flow through these

## Open Questions

- **Learnable thresholds**: The sigmoid and leaky ReLU thresholds are fixed. Could they be learned during training, like PReLU learns its leak slope?
- **Ternary SiLU / Mish**: Other modern activations haven't been ternarized yet. The pattern is straightforward (continuous → quantize), but the thresholds need empirical validation.
- **Multi-threshold activations**: Instead of one threshold, use two: a narrow "strong activation" band and a wider "weak activation" band. Would require more than three output levels.

## Testing

```bash
cargo test
```

18 tests covering: sign identity on trits and continuous mapping, ReLU zeroing behavior, sigmoid stepping with default and custom thresholds, tanh scaling and symmetry, softmax argmax/min-max/single element/empty/equal values, GELU approximation bounds and known values, Swish at key points, leaky ReLU with threshold, cross-function validation (all activations produce valid trits for all inputs).

## License

MIT
