# Chapter 4: Forward Pass

## 4.1 What Is a Forward Pass?

The **forward pass** (also called **inference**) is the process of feeding input data through a neural network to get output predictions.

```
Input → Layer1 → Layer2 → ... → LayerN → Output
```

For each layer:
1. Take input from previous layer (or original data)
2. Compute linear transform: z = Wx + b
3. Apply activation: a = σ(z)
4. Pass output to next layer

This is called "forward" because data flows from input to output, in the forward direction. (The opposite is "backward" which is used in training).

### When Does Forward Pass Happen?

1. **Inference/Prediction**: When using a trained model to make predictions
2. **Training**: In each training iteration, forward pass is done BEFORE backpropagation

---

## 4.2 The Neuron — Biological Inspiration vs Math Reality

### Biological Neuron ( inspiration)
```
       Dendrites          Axon
          ↓                 ↓
    [Receive signals] → [Process] → [Send output]
```

### Mathematical Neuron (what we actually implement)

```
Input: x₁, x₂, x₃
         ↓    ↓    ↓
    ┌─────────────────────┐
    │  w₁   w₂   w₃       │  ← weights
    │  (learnable params) │
    │   Σ wᵢxᵢ + b        │  ← weighted sum + bias
    │        ↓            │
    │       σ()           │  ← activation function
    │        ↓            │
    └─────────────────────┘
         ↓
    Output: a
```

### Single Neuron Computation

```
a = σ(w₁x₁ + w₂x₂ + w₃x₃ + b)
```

This is exactly what our DenseLayer does! A dense layer is just a bunch of neurons stacked together.

---

## 4.3 Weights and Biases — The Learnable Parameters

### Weights (W)

The weight matrix stores the strength of connection between inputs and outputs.

- Shape: [output_size × input_size]
- Each row j contains the weights for output neuron j
- Weight w[j,i] connects input i to output j

### Biases (b)

The bias vector shifts the activation function. 

**Why is bias necessary?**

Without bias, every neuron passes through the origin:
```
output = σ(w·x + 0)
```

This means:
- If input is 0, output is always 0 (or 0.5 for sigmoid)
- The decision boundary is forced through the origin

With bias:
```
output = σ(w·x + b)
```

The bias allows the boundary to shift! Without it, neural networks would be severely limited.

**Visual intuition:**
- Without bias: You're at the origin (0,0), can only draw lines through it
- With bias: You're free to position the line anywhere

---

## 4.4 Shape Math — How Matrices Must Align

For a dense layer with input_size = m, output_size = n:

```
Input:      [batch × m]
Weights:    [n × m]
Biases:     [1 × n]
Output:     [batch × n]
```

### Step-by-Step

1. **Matrix multiply**: input [1×m] × weights^T [m×n] = [1×n]
2. **Add bias**: [1×n] + [1×n] = [1×n]
3. **Activate**: apply σ element-wise

### Why Weights Need Transpose

Weights are stored as [output × input], but we need [input × output] to multiply correctly!

- Weights stored: [256 × 784] (output neurons × input features)
- Need to multiply by its transpose to align dimensions
- Mathematically: input × W^T = [1 × 784] × [784 × 256] = [1 × 256]

---

## 4.5 The Dense Layer Implementation

Our implementation stores:

```c
typedef struct {
    Matrix* weights;        // [output_size × input_size]
    Matrix* biases;         // [1 × output_size]
    Matrix* output;         // Cached output [batch × output_size]
    Matrix* pre_activation; // z before activation
    ActivationFn activation; // ReLU, Sigmoid, etc.
    int input_size;
    int output_size;
} DenseLayer;
```

### Forward Function

```c
Matrix* dense_layer_forward(DenseLayer* layer, Matrix* input) {
    // 1. Compute weighted sum
    for (each output neuron j):
        sum = Σ(input[i] × weights[j,i]) + bias[j]
        pre_activation[j] = sum
    
    // 2. Apply activation
    if (layer->activation != NULL):
        activation(pre_activation, output)
    
    return output;
}
```

---

## 4.6 Trace-Through: Following One Input Through the Layer

Let's trace a single input through a layer:

**Setup:**
- Input: [1.0, 2.0, 3.0] (3 values)
- Weights: [[1,2,3], [4,5,6]] (2 output neurons)
- Biases: [0.1, 0.2]
- Activation: ReLU

**Step 1: Compute weighted sum**

Output neuron 0:
```
z₀ = 1×1 + 2×2 + 3×3 + 0.1 = 1 + 4 + 9 + 0.1 = 14.1
```

Output neuron 1:
```
z₁ = 1×4 + 2×5 + 3×6 + 0.2 = 4 + 10 + 18 + 0.2 = 32.2
```

**Step 2: Apply ReLU**

```
a₀ = max(0, 14.1) = 14.1  (positive, unchanged)
a₁ = max(0, 32.2) = 32.2  (positive, unchanged)
```

**Result:** [14.1, 32.2]

If z had been negative, ReLU would convert it to 0!

---

## 4.7 Memory Management in Layers

Layers allocate several matrices that must be freed to prevent memory leaks:

### Allocations per Layer

- `weights`: [output_size × input_size] floats
- `biases`: [1 × output_size] floats
- `output`: [batch × output_size] floats
- `pre_activation`: [batch × output_size] floats

### Cleanup Function

```c
void dense_layer_free(DenseLayer* layer) {
    free(layer->weights);    // Must free each matrix
    free(layer->biases);
    free(layer->output);
    free(layer->pre_activation);
    free(layer);             // And the layer struct itself
}
```

### Output Caching

We cache the output in `layer->output` because:
- In inference, you might need the output multiple times
- Recomputing would require another matrix multiplication (expensive!)

This is a classic **time-memory tradeoff**:
- Time: recompute if not cached
- Memory: store cached result

---

## What Just Happened

- **Created DenseLayer**: A fully connected layer with weights, biases, activation, and caching
- **Implemented forward pass**: Computes z = Wx + b, then applies activation
- **Learned shape math**: Input [1×m] × W^T [m×n] = [1×n]
- **Understood bias**: Without bias, decision boundaries are forced through origin
- **Implemented output caching**: Store output to avoid recomputation
- **All tests pass**: Creation, forward with/without activation, edge cases verified