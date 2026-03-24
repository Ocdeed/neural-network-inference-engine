# Chapter 5: Network Architecture

## 5.1 Stacking Layers — The Deep in Deep Learning

The "deep" in deep learning refers to having **many layers** stacked together. Each layer transforms the representation of data, and stacking them creates progressively more abstract features.

### Why More Layers?

- **1 layer (linear)**: Can only learn straight lines/decision boundaries
- **2-3 layers**: Can learn non-linear patterns
- **10+ layers**: Can learn incredibly complex patterns (images, speech, text)

### Layer Composition

```
Input → Layer1 → Layer2 → ... → LayerN → Output
         ↓         ↓                ↓
       [W1,b1]   [W2,b2]          [Wn,bn]
        +σ()      +σ()             +σ()
```

Each layer does: `output = activation(W × input + b)`

---

## 5.2 Dynamic Arrays in C — realloc Explained

### The Problem

C arrays have fixed size. If you need to add more elements than the array can hold, you must:
1. Allocate a larger array
2. Copy existing data to new array
3. Free old array
4. Update pointer

This is what `realloc()` does!

### realloc Behavior

```c
ptr = realloc(ptr, new_size);
```

Can do one of three things:
1. **Extend in place**: If room to grow, return same pointer
2. **Move to new memory**: If need to move, allocate new block, copy data, FREE OLD, return new
3. **Fail**: Return NULL (old memory still valid!)

### The Danger

```c
// WRONG!
DenseLayer** old = net->layers;
net->layers = realloc(net->layers, new_size);
// old might now point to freed memory!

// RIGHT: use temp variable
DenseLayer** new_layers = realloc(net->layers, new_size);
if (new_layers != NULL) {
    net->layers = new_layers;  // Only update on success!
}
```

---

## 5.3 Chaining Layers — Shape Compatibility Rules

For layers to chain, output shape of layer i must match input shape of layer i+1:

```
Layer 1: input 784 → output 128
Layer 2: input 128 → output 64  
Layer 3: input 64 → output 10

✓ Shapes: 784→128→64→10 (valid)
✗ Shapes: 784→256→64→10 (invalid! Layer 2 output 256 != Layer 3 input 64)
```

Our code validates this in `network_add_layer()`:

```c
if (prev->output_size != layer->input_size) {
    fprintf(stderr, "ERROR: Shape mismatch!...\n");
    return;
}
```

---

## 5.4 The Full Forward Pass — End to End

For our 3-layer network (784→128→64→10):

```
Step 1: Layer 1 (784→128, ReLU)
  Input:    [1 × 784]       (1 image, 784 pixels)
  Weights:  [128 × 784]    (128 neurons, each connected to 784 inputs)
  z1:       [1 × 128] = input × W^T + b
  a1:       [1 × 128] = ReLU(z1)

Step 2: Layer 2 (128→64, ReLU)
  Input:    [1 × 128]       (output from Layer 1)
  Weights:  [64 × 128]      (64 neurons)
  z2:       [1 × 64] = a1 × W^T + b
  a2:       [1 × 64] = ReLU(z2)

Step 3: Layer 3 (64→10, Softmax)
  Input:    [1 × 64]        (output from Layer 2)
  Weights:  [10 × 64]       (10 output neurons = 10 digits)
  z3:       [1 × 10] = a2 × W^T + b
  Output:   [1 × 10] = Softmax(z3)  ← probabilities!
```

---

## 5.5 Parameter Counting — How Big Is Our Model?

For each Dense layer:
- **Weights**: output_size × input_size
- **Biases**: output_size (one per neuron)

### Our Network (784→256→128→10)

| Layer | Weights | Biases | Total | Memory |
|-------|---------|--------|-------|--------|
| 1 | 784 × 256 = 200,704 | 256 | 200,960 | 784 KB |
| 2 | 256 × 128 = 32,768 | 128 | 32,896 | 128 KB |
| 3 | 128 × 10 = 1,280 | 10 | 1,290 | 5 KB |
| **Total** | | | **235,146** | **~917 KB** |

### Why Memory Matters

```
917 KB is tiny! 

Comparison:
- Our MNIST model:        ~1 MB
- BERT-base (NLP):        ~400 MB  
- GPT-3 (LLM):            ~175,000 MB (175 GB!)

Our model is 189,000× smaller than GPT-3!

This is why MNIST inference can run on:
- Arduino boards (2 KB RAM!)
- Raspberry Pi
- Mobile phones
- Web browsers (via WebAssembly)
```

### Where Each Layer's Memory Goes

```
Layer 1: 784 → 256
  weights: 200,704 floats × 4 bytes = 802,816 bytes = 784 KB
  biases:     256 floats × 4 bytes =   1,024 bytes =   1 KB
  output:     256 floats × 4 bytes =   1,024 bytes =   1 KB
  pre_act:    256 floats × 4 bytes =   1,024 bytes =   1 KB
  ───────────────────────────────────────────────
  Total per forward pass: 788 KB

Layer 2: 256 → 128
  Total per forward pass: 130 KB

Layer 3: 128 → 10  
  Total per forward pass: 5 KB

Total memory for inference: ~923 KB
```

---

## 5.6 Classification — argmax and Final Prediction

### The Problem

The network outputs 10 probabilities (one per digit):
```
[0.09, 0.10, 0.09, 0.11, 0.10, 0.09, 0.09, 0.11, 0.10, 0.11]
  0      1      2      3      4      5      6      7      8      9
```

Which digit is it? The one with highest probability!

### argmax

```c
int network_predict(NeuralNetwork* net, Matrix* input) {
    Matrix* output = network_forward(net, input);
    
    int best_idx = 0;
    float best_val = output->data[0];
    
    for (int i = 1; i < output->cols; i++) {
        if (output->data[i] > best_val) {
            best_val = output->data[i];
            best_idx = i;
        }
    }
    
    return best_idx;
}
```

In our run: predicted digit = 3 (highest probability at index 3)

---

## 5.7 Real Example: MNIST Digit Classifier

### What is MNIST?

- **M**odified **N**ational **I**nstitute of **S**tandards and **T**echnology
- 70,000 handwritten digits (0-9)
- 60,000 training, 10,000 test
- 28×28 grayscale images = 784 pixels

### Our Architecture

```
Input Layer:     784 neurons (flattened 28×28 image)
                  ↓
Hidden Layer 1:  256 neurons + ReLU   ← (we use 256, not 128!)
                  ↓
Hidden Layer 2: 128 neurons + ReLU  
                  ↓
Output Layer:   10 neurons + Softmax (probabilities for digits 0-9)
```

### Why This Architecture?

- **784 input**: Each pixel is a feature
- **256 first hidden**: Compress 784→256, capture main patterns
- **128 second hidden**: Further compress, more abstraction
- **10 output**: One per digit (0-9)
- **ReLU in hidden**: Fast, effective, prevents vanishing gradients
- **Softmax in output**: Converts to valid probabilities (sum=1)

### Building the Network in Code

Here's how we actually create this network in C:

```c
// Create the network
NeuralNetwork* net = network_create("MNIST_Digit_Classifier");

// Layer 1: 784 inputs -> 256 outputs, ReLU activation
DenseLayer* layer1 = dense_layer_create(784, 256, activation_relu);
network_add_layer(net, layer1);

// Layer 2: 256 inputs -> 128 outputs, ReLU activation  
DenseLayer* layer2 = dense_layer_create(256, 128, activation_relu);
network_add_layer(net, layer2);

// Layer 3: 128 inputs -> 10 outputs, Softmax for classification
DenseLayer* layer3 = dense_layer_create(128, 10, activation_softmax);
network_add_layer(net, layer3);

// Print the summary
network_print_summary(net);
```

The network validates shape compatibility automatically:
- Layer 1 output (256) = Layer 2 input (256) ✓
- Layer 2 output (128) = Layer 3 input (128) ✓

If shapes don't match, `network_add_layer()` prints an error and doesn't add the layer!

---

## What Just Happened

- **Created NeuralNetwork container**: Dynamic array of layers that grows with realloc
- **Implemented network_forward**: Chains all layers - output of one becomes input to next
- **Added network_predict**: Returns argmax (index of highest probability)
- **Built demo network**: 784→256→128→10 for MNIST digit classification
- **Ran full inference**: Forward pass works end-to-end, softmax outputs sum to 1.0
- **Learned parameter counting**: 235K parameters total (~917 KB), verified in code
- **Created documentation**: Chapter 05 explains stacking, dynamic arrays, shape math

### Build and Run

```bash
cd nn_engine
make
./nn_engine models/mnist_demo.nnbin
```

You should see the full network architecture printed with all layer details!