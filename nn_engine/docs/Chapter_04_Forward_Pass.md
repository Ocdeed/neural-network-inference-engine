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

## 4.8 Complete Forward Pass: Digit Recognition End-to-End

Here's what happens when we feed ONE handwritten "7" through our entire network:

### Input: The Digit Image

```
A 28×28 image of the digit "7" gets flattened to 784 values:

Pixel values (normalized 0.0 to 1.0):
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... (many zeros) ...,
 0.0, 0.0, 0.19, 0.35, 0.55, 0.78, 0.95, 1.0, ... (strokes of the 7) ...,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... (more zeros) ...]

This is our input Matrix[1×784]
```

### Layer 1: 784 → 256 with ReLU

```
Step 1a: Matrix Multiply
  Input:     [1 × 784]
  Weights:   [256 × 784]
  Result:    [1 × 256] = z₁

  For first neuron: z₁[0] = Σ input[i] × weights[0,i] + bias[0]
                  ≈ 0.19×0.01 + 0.35×0.02 + ... + 0.95×(-0.01)
                  ≈ 3.45  (some positive, some negative weights)

Step 1b: ReLU activation
  z₁ = [3.45, -1.23, 0.87, -0.05, 5.12, ... , 2.67]  (256 values)
  a₁ = [3.45,  0.00, 0.87,  0.00, 5.12, ... , 2.67]  (after ReLU!)
  
  ≈ 40% of neurons are zero (sparse!)
```

### Layer 2: 256 → 128 with ReLU

```
Step 2a: Matrix Multiply
  Input:    [1 × 256] = a₁ (output from layer 1)
  Weights:  [128 × 256]
  Result:   [1 × 128] = z₂

Step 2b: ReLU activation
  z₂ = [1.23, -0.45, 2.89, 0.12, ... , 0.67]  (128 values)
  a₂ = [1.23,  0.00, 2.89, 0.12, ... , 0.67]  (after ReLU!)
```

### Layer 3: 128 → 10 with Softmax

```
Step 3a: Matrix Multiply
  Input:    [1 × 128] = a₂ (output from layer 2)
  Weights:  [10 × 128]
  Result:   [1 × 10] = z₃ (raw scores / logits)

Step 3b: Softmax activation
  z₃ = [-2.3, 5.1, 0.8, -1.2, 0.3, -0.5, 0.1, 3.2, -0.8, 1.1]
       (raw scores for digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
       
  After softmax (converts to probabilities):
  output = [0.01, 0.84, 0.03, 0.00, 0.02, 0.00, 0.01, 0.04, 0.00, 0.04]
            digit digit digit digit digit digit digit digit digit digit
              0     1     2     3     4     5     6     7     8     9

Final Prediction: digit 1 (84% confidence!)
```

### Summary of the Journey

```
┌──────────────────────────────────────────────────────────────────────┐
│ FORWARD PASS SUMMARY FOR ONE "7" DIGIT IMAGE                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ INPUT (784 pixels)                                                  │
│     ↓ (matrix mult)                                                  │
│ LAYER 1: z₁ (256 values) → ReLU → a₁ (256 values, ~40% sparse)    │
│     ↓ (matrix mult)                                                  │
│ LAYER 2: z₂ (128 values) → ReLU → a₂ (128 values)                  │
│     ↓ (matrix mult)                                                  │
│ LAYER 3: z₃ (10 logits) → Softmax → output (10 probabilities)     │
│     ↓                                                                 │
│ PREDICTION: digit with highest probability                          │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│ COMPUTATIONS:                                                        │
│   Layer 1: 784 × 256 = 200,704 multiply-adds                       │
│   Layer 2: 256 × 128 =  32,768 multiply-adds                       │
│   Layer 3: 128 × 10 =   1,280 multiply-adds                       │
│   TOTAL:      234,752 operations                                    │
│                                                                      │
│ MEMORY:                                                              │
│   Model: ~922 KB                                                    │
│   Per-inference: ~3 KB input + ~1 KB intermediate                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

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

## 4.9 Running Forward Pass: The Code

Here's how you'd actually use our engine to run inference:

```c
#include "network.h"
#include "loader.h"
#include "matrix.h"

int main() {
    // Step 1: Load the trained model from file
    NeuralNetwork* net = network_load("models/mnist_demo.nnbin");
    if (net == NULL) {
        printf("Failed to load model!\n");
        return 1;
    }

    // Print model info
    network_print_summary(net);

    // Step 2: Create input (one MNIST image)
    // Imagine we loaded a 28x28 image and normalized to 0-1
    Matrix input = matrix_create(1, 784);
    
    // Fill with some pixel values (simulated)
    for (int i = 0; i < 784; i++) {
        input.data[i] = (float)rand() / RAND_MAX;  // Random pixels
    }

    // Step 3: Run forward pass
    printf("\nRunning inference...\n");
    Matrix* output = network_forward(net, &input);

    // Step 4: Get prediction
    int prediction = network_predict(net, &input);

    // Print results
    printf("Output probabilities:\n");
    for (int i = 0; i < 10; i++) {
        printf("  Digit %d: %.2f%%\n", i, output->data[i] * 100);
    }
    printf("\nPredicted digit: %d\n", prediction);

    // Step 5: Clean up
    network_free(net);
    matrix_free(&input);

    return 0;
}
```

### Running It

```bash
cd nn_engine
make
./nn_engine models/mnist_demo.nnbin
```

Expected output:

```
═══════════════════════════════════════════════════════════════
  Neural Network: MNIST
═══════════════════════════════════════════════════════════════
Layer 1: Dense 784 → 256, ReLU
  Weights: [256 × 784] = 200,704 parameters
  Biases: [1 × 256] = 256 parameters

Layer 2: Dense 256 → 128, ReLU
  Weights: [128 × 256] = 32,768 parameters
  Biases: [1 × 128] = 128 parameters

Layer 3: Dense 128 → 10, Softmax
  Weights: [10 × 128] = 1,280 parameters
  Biases: [1 × 10] = 10 parameters

═══════════════════════════════════════════════════════════════
Total Parameters: 235,146
═══════════════════════════════════════════════════════════════

Running inference...
Output probabilities:
  Digit 0: 1.23%
  Digit 1: 84.56%
  Digit 2: 3.21%
  ...

Predicted digit: 1
```

---

## What Just Happened

- **Created DenseLayer**: A fully connected layer with weights, biases, activation, and caching
- **Implemented forward pass**: Computes z = Wx + b, then applies activation
- **Learned shape math**: Input [1×m] × W^T [m×n] = [1×n]
- **Understood bias**: Without bias, decision boundaries are forced through origin
- **Implemented output caching**: Store output to avoid recomputation
- **All tests pass**: Creation, forward with/without activation, edge cases verified
- **Traced complete forward pass**: Seen end-to-end for MNIST digit recognition

### Try It Yourself!

```bash
cd nn_engine
make
./nn_engine models/mnist_demo.nnbin
```

This loads the trained MNIST model and runs inference. The output shows:
- Model architecture (layers, sizes, parameters)
- Inference results (probabilities for each digit)
- Final prediction

You can also run the layer tests:

```bash
make test_layers
```

This verifies dense layer creation, forward pass, and memory management work correctly.