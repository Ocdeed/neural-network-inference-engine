# Chapter 0: Project Overview

## Neural Network Inference Engine

## What Is a Neural Network Inference Engine?

A **neural network inference engine** is a software system that takes a *pre-trained* neural network model and runs it on *new input data* to produce predictions. This is fundamentally different from **training**, where the network learns from data.

### Inference vs Training

| Aspect | Training | Inference |
|--------|----------|-----------|
| Goal | Adjust weights to minimize error | Use fixed weights to make predictions |
| Operations | Forward pass + backward pass (gradients) | Forward pass only |
| Computational cost | Very high (hours/days/weeks) | Lower (milliseconds) |
| Memory | Large (stores gradients, optimizer state) | Smaller (weights only) |
| Frequency | Once (or occasionally) | Many times (production) |

Think of it like a recipe:
- **Training** = developing and testing a recipe (takes lots of experimentation)
- **Inference** = actually cooking using that recipe (quick, can be done many times)

---

## Why C for This Project?

C is uniquely suited for building a high-performance neural network inference engine for several reasons:

### 1. **Memory Control**
C gives you direct control over memory allocation. Neural networks process large matrices, and controlling memory layout (row-major vs column-major, contiguous vs fragmented) directly impacts performance.

### 2. **Predictable Performance**
No garbage collection pauses, no JIT compilation overhead. Every operation's timing is deterministic. This is critical for real-time systems.

### 3. **Hardware-Near Programming**
You can directly use CPU features like:
- SIMD (Single Instruction Multiple Data) instructions
- Cache-aware memory access patterns
- Memory prefetching

### 4. **Portability**
C code compiles cleanly on any platform with a C compiler. No runtime dependencies.

### 5. **Foundation for Optimization**
Most deep learning frameworks (TensorFlow, PyTorch, ONNX Runtime) have C/C++ cores for performance - we're doing the same thing, just from scratch!

---

## A Real Example: MNIST Handwritten Digit Recognition

Let's look at a concrete example of what our inference engine does. MNIST is the "Hello World" of neural networks - recognizing handwritten digits (0-9).

### The Input: A Handwritten Digit

An MNIST image is a 28×28 grayscale image:
```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│     ████████████████                                        │
│     ██          ████                                        │
│     ██  ████    ████    ████████                          │
│     ██  ████    ████    ████████                          │
│     ██  ████    ████    ████████                          │
│     ██          ████    ████████                          │
│     ██  ████    ████    ████████                          │
│     ██  ████    ████    ████████                          │
│     ██          ████    ████████                          │
│     ████████████████    ████████                          │
│                         ██                                  │
│                                                            │
│     (28 pixels × 28 pixels = 784 input values)             │
└────────────────────────────────────────────────────────────┘
```

When we feed this to our network, each pixel becomes a number between 0.0 (black) and 1.0 (white):
```
Input array (first 20 values):
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19, 0.35, ...]
```

### The Output: Probability Distribution

After running through all layers, we get 10 probabilities (one for each digit 0-9):

```
Output after softmax:
Index:  0      1      2      3      4      5      6      7      8      9
      ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
Prob: │ 0.01 │ 0.01 │ 0.02 │ 0.03 │ 0.01 │ 0.01 │ 0.01 │ 0.88 │ 0.01 │ 0.01 │
      └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
       digit  digit  digit  digit  digit  digit  digit  digit  digit  digit
         0      1      2      3      4      5      6      7      8      9
      
The network is 88% confident this is the digit "7"!
```

### How Many Operations?

For ONE image through our 784→256→128→10 network:
- **Layer 1 (784→256)**: 784 × 256 = 200,704 multiplications + 256 biases
- **Layer 2 (256→128)**: 256 × 128 = 32,768 multiplications + 128 biases  
- **Layer 3 (128→10)**: 128 × 10 = 1,280 multiplications + 10 biases
- **Total**: ~235,000 multiply-add operations per image!

At 60,000 images per second, that's **14 billion operations per second**!

---

## What Is SIMD?

**SIMD** stands for **Single Instruction Multiple Data**. It's a type of parallel processing where the same operation is applied to multiple data points simultaneously.

### Example: Adding Two Arrays

Without SIMD (scalar):
```c
// Process 4 elements one at a time
for (int i = 0; i < 4; i++) {
    result[i] = a[i] + b[i];  // 4 separate additions
}
```

With SIMD (vector):
```c
// Process 4 elements in one instruction
__m128 va = _mm_loadu_ps(a);
__m128 vb = _mm_loadu_ps(b);
__m128 vr = _mm_add_ps(va, vb);  // 1 instruction adds ALL 4!
_mm_storeu_ps(result, vr);
```

### Why It Matters

Modern CPUs can perform 8, 16, or 32 operations in parallel with SIMD:
- **SSE**: 128-bit = 4 floats at once
- **AVX**: 256-bit = 8 floats at once
- **AVX-512**: 512-bit = 16 floats at once

That's potentially a **16x speedup** for matrix operations! We'll implement this in Phase 5.

---

## Glossary of Terms

### Tensor
A multi-dimensional array of numbers. 
- Scalar = 0D tensor (single number)
- Vector = 1D tensor 
- Matrix = 2D tensor
- 3D/4D tensors common in neural networks (e.g., images with color channels)

### Weight
A trainable parameter in a neural network layer. Stored in matrices, these values determine how the network transforms input to output. Weights are learned during training.

### Bias
A trainable parameter added to the weighted sum before activation. Each neuron has its own bias. Formula: `output = activation(dot(input, weights) + bias)`

### Layer
A fundamental building block of neural networks. Layers transform their input:
- **Dense/Linear**: Weighted sum + bias
- **Convolution**: Local pattern detection
- **Recurrent**: Sequential processing

### Activation Function
A non-linear function applied after linear transformation. Without activations, neural networks would just be linear regression!
- **ReLU**: max(0, x) - simple, fast, effective
- **Sigmoid**: 1/(1+e^-x) - squashes to 0-1
- **Tanh**: (e^x - e^-x)/(e^x + e^-x) - squashes to -1 to 1
- **Softmax**: converts logits to probabilities (sum = 1)

### Inference
The process of running a trained model on new data to make predictions. Also called "forward pass."

### Forward Pass
The sequence of computations from input to output through all layers. Data flows "forward" through the network - no backpropagation involved.

---

## What Our Engine Will Do

Here's a visual diagram of the inference process:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NEURAL NETWORK INFERENCE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT DATA                    NEURAL NETWORK MODEL                 │
│  ┌─────────┐                  ┌─────────────────────┐              │
│  │ [0.234] │                  │   LAYER 1 (Dense)   │              │
│  │ [0.891] │ ───────────────▶│   weights: 784x256  │              │
│  │ [0.012] │                  │   biases: 256        │              │
│  │  ...    │                  └──────────┬──────────┘              │
│  │ [0.567] │                             │                         │
│  │ 784 vals│                             ▼                         │
│  └─────────┘                  ┌─────────────────────┐              │
│                              │  ReLU Activation    │              │
│                              └──────────┬──────────┘              │
│                                         │                         │
│                                         ▼                         │
│                              ┌─────────────────────┐              │
│                              │   LAYER 2 (Dense)   │              │
│                              │   weights: 256x128  │              │
│                              │   biases: 128        │              │
│                              └──────────┬──────────┘              │
│                                         │                         │
│                                         ▼                         │
│                              ┌─────────────────────┐              │
│                              │  ReLU Activation    │              │
│                              └──────────┬──────────┘              │
│                                         │                         │
│                                         ▼                         │
│                              ┌─────────────────────┐              │
│                              │  LAYER 3 (Dense)   │              │
│                              │   weights: 128x10  │              │
│                              │   biases: 10        │              │
│                              └──────────┬──────────┘              │
│                                         │                         │
│                                         ▼                         │
│                              ┌─────────────────────┐              │
│                              │    Softmax          │              │
│                              │  (probabilities)    │              │
│                              └──────────┬──────────┘              │
│                                         │                         │
│                                         ▼                         │
│                              ┌─────────────────────┐              │
│                              │     OUTPUT          │              │
│                              │  [0.02, 0.95, ...] │              │
│                              │   Predicted class: 1│              │
│                              └─────────────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### The Flow:
1. **Load model** from file (weights, biases, architecture)
2. **Load input** (e.g., image pixels, text embeddings)
3. **Forward pass**: Input → Layer1 → ReLU → Layer2 → ReLU → Layer3 → Softmax
4. **Output**: Predicted class probabilities

### What This Looks Like in Practice

When you run our engine on an MNIST image, here's the actual output:

```
═══════════════════════════════════════════════════════════════
  Neural Network: MNIST_Digit_Classifier
═══════════════════════════════════════════════════════════════
Layer 1: Dense 784 → 256, ReLU
  Weights: [256 × 784] = 200,704 parameters
  Biases:  [1 × 256]   =     256 parameters
  Memory:  200,960 floats × 4 bytes = ~784 KB
           
Layer 2: Dense 256 → 128, ReLU
  Weights: [128 × 256] = 32,768 parameters  
  Biases:  [1 × 128]   =     128 parameters
  Memory:  32,896 floats × 4 bytes = ~128 KB

Layer 3: Dense 128 → 10, Softmax
  Weights: [10 × 128]  =  1,280 parameters
  Biases:  [1 × 10]    =      10 parameters
  Memory:   1,290 floats × 4 bytes = ~5 KB

═══════════════════════════════════════════════════════════════
Total Parameters: 109,450
Total Memory: ~917 KB (~1 MB)
═══════════════════════════════════════════════════════════════

Running inference on sample image...
Input:  784 values (28×28 image)
Layer 1 output (after ReLU):  256 values
Layer 2 output (after ReLU):  128 values  
Layer 3 output (after softmax): 10 values

Prediction: 7 (confidence: 88.3%)
```

---

## Common C Pitfalls to Avoid

As we build this engine, watch out for these common beginner mistakes:

### 1. Forgetting to Free Memory
```c
// WRONG - memory leak!
Tensor* create_tensor(size_t size) {
    Tensor* t = malloc(sizeof(Tensor));
    t->data = malloc(size * sizeof(float));
    return t;  // Caller must free, but who?
}

// RIGHT - clear ownership
void tensor_free(Tensor* t) {
    if (t) {
        free(t->data);
        free(t);
    }
}
```

### 2. Not Checking malloc Return
```c
// WRONG - crashes on out-of-memory
float* data = malloc(size * sizeof(float));

// RIGHT - graceful handling
float* data = malloc(size * sizeof(float));
if (!data) {
    fprintf(stderr, "Failed to allocate %zu bytes\n", size);
    return NULL;
}
```

### 3. Buffer Overflows
```c
// WRONG - writes beyond array bounds!
float arr[10];
for (int i = 0; i < 20; i++) arr[i] = i;

// RIGHT - bounds checking
for (int i = 0; i < 10; i++) arr[i] = i;
```

### 4. Not Using `const` for Read-Only Data
```c
// WRONG - doesn't communicate intent
float dot_product(float* a, float* b, int n);

// RIGHT - compiler enforces read-only
float dot_product(const float* a, const float* b, size_t n);
```

We'll use `-Wall -Wextra` compiler flags to catch many of these automatically!

---

## Next Steps

Now that we understand the foundation:
- **Phase 1**: Build tensor data structures (the backbone of everything)
- **Phase 2**: Implement dense layers (matrix multiplication)
- **Phase 3**: Add activation functions (the non-linearity)
- **Phase 4**: Load models from disk
- **Phase 5**: Optimize with SIMD

Let's build something amazing! 🚀