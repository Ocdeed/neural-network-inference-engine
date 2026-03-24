# Chapter 3: Activation Functions

## 3.1 Why Activation Functions Exist

Activation functions are the **non-linear** element of neural networks. They transform the output of a linear operation (matrix multiplication + bias) in a non-linear way, enabling the network to learn complex patterns.

Without activation functions, stacking multiple layers provides **no benefit** over a single layer - the entire network collapses into one linear transformation.

---

## 3.2 The Linear Collapse Problem

### The Math

Consider a 2-layer network:

```
Layer 1: h = W1 * x + b1
Layer 2: y = W2 * h + b2
```

Substituting:
```
y = W2 * (W1 * x + b1) + b2
  = (W2 * W1) * x + (W2 * b1 + b2)
  = W_combined * x + b_combined
```

This is **identical to a single linear layer**! We gained nothing by adding depth.

### With Activation Functions

```
Layer 1: h = σ(W1 * x + b1)    ← non-linear!
Layer 2: y = σ(W2 * h + b2)    ← non-linear!
```

Now we have genuine depth. The non-linear σ breaks the chain, allowing the network to approximate any complex function (Universal Approximation Theorem).

---

## 3.3 ReLU — The Workhorse

### Definition
```
f(x) = max(0, x)
```

### Graph
```
      │
    3 │     ╱
      │    ╱
    0 ──┴────────
      │       3
      │
```

### Why It Dominates

1. **Computationally efficient** - just a comparison, no exponentials
2. **Non-saturating** - gradient stays at 1 for positive inputs (no vanishing!)
3. **Sparse** - outputs are zero for half the input space, creating efficient representations
4. **Works well in practice** - default choice for most modern architectures

### Real Example: ReLU in Our MNIST Network

After matrix multiplication in Layer 1 (784→256), before activation:

```
Pre-activation z (first 10 values out of 256):
  z = [ 3.45, -1.23,  0.87, -0.05,  5.12, -2.31,  0.00,  1.98, -0.45,  2.67]

After ReLU:  a = max(0, z)
  a = [ 3.45,  0.00,  0.87,  0.00,  5.12,  0.00,  0.00,  1.98,  0.00,  2.67]
           ↑        ↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑
         positive  negative positive zero    positive negative zero   positive positive

Sparsity: 4 out of 10 = 40% of neurons are zero!
(Real networks often see 50-70% zeros with ReLU)
```

This sparsity is actually BENEFICIAL:
- Fewer non-zero values = less computation in next layer
- Acts as automatic feature selection
- Makes the network more efficient

### Code Implementation

Here's how we implement ReLU in C:

```c
void activation_relu(Matrix* input, Matrix* output) {
    for (int i = 0; i < input->rows * input->cols; i++) {
        // This is ONE instruction: max(0, x)
        output->data[i] = input->data[i] > 0 ? input->data[i] : 0;
    }
}

// Compare to Sigmoid (much slower!):
//   output->data[i] = 1.0f / (1.0f + expf(-input->data[i]));
// Requires: division, negation, exponential (expensive!)
```

### The Dying ReLU Problem

**Problem**: If a neuron gets negative input once during training, its gradient becomes 0. It can never recover!

**Visualization**:
```
Input: -5 → ReLU(0) = 0 → gradient = 0 → weights never update → neuron dies forever!
```

**Real example of a dying neuron:**

```
During training, a neuron's weights become:
  weights = [-0.5, -0.3, -0.8, 0.1, -0.2]  (mostly negative!)
  
For ANY input, the weighted sum is negative:
  z = input · weights + bias
    = 0.1×(-0.5) + 0.2×(-0.3) + ... 
    = -0.05 - 0.06 - ... 
    = -0.82 (negative!)
    
ReLU(z) = max(0, -0.82) = 0

Gradient through ReLU = 0 (for negative input)
Weight update = gradient × input = 0 × input = 0

The neuron is stuck at 0 forever! It's "dead".
```

**Solution**: Leaky ReLU allows small gradient even for negative values:

```
LeakyReLU(x) = x if x > 0
             = 0.01x if x <= 0  (small slope!)

For z = -0.82:
  LeakyReLU(-0.82) = 0.01 × (-0.82) = -0.0082
  
Gradient = 0.01 (not zero!)
Weights can still update!
```

---

## 3.4 Sigmoid and Tanh — The Classics

### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
```

Range: (0, 1)

Graph:
```
      │   ╭────
    1 ┤  ╱
      │ ╱
  0.5 ┤╱
      │
    0 ┼───────────
      │
```

**Historical importance**: Used in early neural networks (perceptrons)
**Problem**: Vanishing gradients - at extremes, gradient ≈ 0
**Modern use**: Output layer for binary classification (probability)

### Tanh (Hyperbolic Tangent)
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

Range: (-1, 1)

Graph:
```
      │  ╭──
    1 ┤ ╱
      │╱
    0 ┼───────
      │╲
   -1 ┤ ╲__
      │
```

**Advantage over sigmoid**: Zero-centered! Helps with gradient flow
**Still has saturation** at extreme values
**Modern use**: LSTMs, RNNs, some CNNs

### Comparison

| Property | Sigmoid | Tanh | ReLU |
|----------|---------|------|------|
| Range | (0,1) | (-1,1) | [0,∞) |
| Zero-centered | No | Yes | No |
| Dying neurons | Yes | Yes | Yes |
| Computation | exp() | exp() | max() |

---

## 3.5 Softmax — Turning Scores Into Probabilities

### Definition
```
softmax(x)[i] = e^(x[i]) / Σ e^(x[j])
```

### The Mathematics

1. **Exponential**: Amplifies differences - small advantages become large
2. **Normalization**: Divide by sum to get probabilities

### Example
```
Input:     [2.0, 1.0, 0.1]
Exp:       [7.39, 2.72, 1.11]
Sum:       11.22
Output:    [0.66, 0.24, 0.10]
```

The highest input (2.0) gets 66% despite being only 2x the second!
Exponential magnifies small differences.

### Real Neural Network Example

After matrix multiplication in our final layer, we get raw scores (logits):

```
Pre-softmax (logits):  [-2.3,  5.1,  0.8, -1.2,  0.3, -0.5,  0.1,  3.2, -0.8,  1.1]
                       digit digit digit digit digit digit digit digit digit digit
                         0     1     2     3     4     5     6     7     8     9

After softmax (probabilities):
                       [ 0.01,  0.84,  0.03,  0.00,  0.02,  0.00,  0.01,  0.04,  0.00,  0.04]
                        digit digit digit digit digit digit digit digit digit digit
                          0     1     2     3     4     5     6     7     8     9

The network is 84% confident this is the digit "1"!
```

Notice how softmax dramatically amplifies the difference between scores. The second score (5.1) is only about 2× higher than the first score at index 2 (0.8), but after softmax, it gets 84% of the probability vs just 3%!

### Why Use Softmax?

- **Multi-class classification**: Output is a probability distribution
- **Interpretable**: Can say "87% confident it's class 3"
- **Differentiable**: Gradients flow through it

---

## 3.6 Function Pointers in C — A Powerful Pattern

### What Are Function Pointers?

In C, functions have addresses too! You can store a function's address in a pointer and call it later.

```c
typedef void (*ActivationFn)(Matrix*, Matrix*);
```

This creates a type `ActivationFn` that is:
- A pointer to a function
- That returns void
- Takes two Matrix* parameters

### Why This Matters for Neural Networks

```c
ActivationFn activations[] = {
    activation_relu,
    activation_sigmoid,
    activation_tanh_act
};

for (int i = 0; i < 3; i++) {
    activations[i](&input, &output);  // Call each one!
}
```

**Benefits**:
1. **Configurable networks**: Choose activation at runtime
2. **Clean architecture**: Pass activations as parameters to layer functions
3. **Flexibility**: Swap ReLU for Sigmoid without changing core code
4. **Arrays of functions**: Iterate over different activations

### Real-World Use

Many ML frameworks use this pattern:
- PyTorch: `torch.nn.ReLU()`, `torch.nn.Sigmoid()` are objects
- TensorFlow: `tf.keras.layers.Activation('relu')` takes string
- Our C engine: Function pointers give similar flexibility!

---

## 3.7 Numerical Stability — Why Small Details Matter

### The Softmax Overflow Problem

```
Input: [1000, 1001, 999]
exp(1000) = 10^434  ← OVERFLOW!
```

IEEE 754 double precision max ≈ 10^308

### The Solution: Subtract Max

```
softmax(x)[i] = e^(x[i]) / Σ e^(x[j])
              = e^(x[i]-max) / Σ e^(x[j]-max)  ← divide top and bottom by e^max
```

Now:
```
Input:     [1000, 1001, 999]
Subtract max (1001):
            [-1, 0, -2]
Exp:       [0.37, 1.00, 0.14]
Sum:       1.51
Output:    [0.24, 0.67, 0.09]
```

All exponentiated values ≤ e^0 = 1!

### This Is Why Engineering Matters

A tiny detail like subtracting max before exponentiating:
- Prevents crashes on extreme inputs
- Makes the code robust to any input range
- Is the difference between "works in theory" and "works in production"

---

## What Just Happened

- **Implemented 6 activation functions**: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, plus derivative
- **Learned why non-linearity matters**: Without activation functions, stacked layers collapse to one linear transform
- **Understood dying ReLU**: Neurons stuck at zero can't recover - Leaky ReLU fixes this
- **Mastered numerical stability**: Subtracting max before exp() prevents overflow in softmax
- **Used function pointers**: Created ActivationFn typedef for flexible activation selection
- **Built test suite**: Verified all activations with edge cases like large values
- **Traced real values**: See how ReLU creates sparsity in actual layer outputs

### Run the Activation Tests

```bash
cd nn_engine
make test_activations
```

You should see output like:

```
Testing ReLU:
  Input:  [3.45, -1.23, 0.0]
  Output: [3.45, 0.0, 0.0]
  ✓ PASSED

Testing Softmax (with stability fix):
  Input:  [1000, 1001, 999]
  Output: [0.24, 0.67, 0.09]
  ✓ PASSED (no overflow!)

Testing Sigmoid:
  Input:  [0.0]
  Output: [0.5]
  ✓ PASSED
```