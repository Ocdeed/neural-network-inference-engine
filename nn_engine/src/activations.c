/**
 * @file activations.c
 * @brief Implementation of activation functions
 * 
 * ACTIVATION FUNCTIONS EXPLAINED:
 * ===============================
 * 
 * Neural networks without activation functions are just LINEAR TRANSFORMS.
 * A stack of linear layers collapses into ONE linear layer:
 * 
 *   W2 * (W1 * x) = (W2 * W1) * x = W_combined * x
 * 
 * Non-linear activations break this chain, allowing the network to learn
 * complex, non-linear patterns like:
 *   - Image classification
 *   - Natural language processing  
 *   - Complex function approximation
 * 
 * Each function is applied ELEMENT-WISE to the input matrix.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "activations.h"
#include "matrix.h"

/**
 * ReLU: Rectified Linear Unit
 * f(x) = max(0, x)
 * 
 * The workhorse of modern deep learning!
 * - Extremely fast (just a comparison and branch)
 * - Gradient is 1 for positive inputs (no vanishing gradient!)
 * - Creates sparse, efficient representations
 * 
 * BUT: "Dying ReLU" - neurons stuck at 0 never recover
 * If a neuron gets negative during training, its gradient is 0 forever!
 */
void activation_relu(Matrix* input, Matrix* output)
{
    if (input == NULL || output == NULL) {
        return;
    }
    
    if (input->rows != output->rows || input->cols != output->cols) {
        fprintf(stderr, "ERROR: Dimension mismatch in ReLU\n");
        return;
    }
    
    int total = input->rows * input->cols;
    
    for (int i = 0; i < total; i++) {
        float x = input->data[i];
        /* Branchless would be faster: output = (x > 0) ? x : 0 */
        output->data[i] = (x > 0.0f) ? x : 0.0f;
    }
}

/**
 * ReLU derivative: f'(x) = 1 if x > 0, else 0
 * 
 * Used during backpropagation to compute gradients.
 * We won't use this for inference, but it's essential for training.
 */
void activation_relu_derivative(Matrix* input, Matrix* output)
{
    if (input == NULL || output == NULL) {
        return;
    }
    
    int total = input->rows * input->cols;
    
    for (int i = 0; i < total; i++) {
        output->data[i] = (input->data[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

/**
 * Leaky ReLU: fixes dying ReLU problem
 * f(x) = x if x > 0, else alpha * x
 * 
 * With alpha = 0.01 (common default):
 * - For x > 0: gradient = 1 (same as ReLU)
 * - For x < 0: gradient = 0.01 (small but non-zero!)
 * 
 * The small negative slope allows gradients to flow even when input is negative.
 * Neurons won't "die" completely!
 */
void activation_leaky_relu(Matrix* input, Matrix* output, float alpha)
{
    if (input == NULL || output == NULL) {
        return;
    }
    
    int total = input->rows * input->cols;
    
    for (int i = 0; i < total; i++) {
        float x = input->data[i];
        output->data[i] = (x > 0.0f) ? x : alpha * x;
    }
}

/**
 * Sigmoid: classic squashing function
 * f(x) = 1 / (1 + e^(-x))
 * 
 * Maps any real number to (0, 1) - perfect for probabilities!
 * 
 * PROBLEM: Vanishing gradients
 * - At x >> 0: f(x) ≈ 1, gradient ≈ 0
 * - At x << 0: f(x) ≈ 0, gradient ≈ 0
 * - Only in middle range (-4 to 4) is gradient significant
 * 
 * This caused problems in early deep networks (5+ layers).
 * Modern networks use ReLU instead.
 * 
 * Still used: output layer for binary classification!
 */
void activation_sigmoid(Matrix* input, Matrix* output)
{
    if (input == NULL || output == NULL) {
        return;
    }
    
    int total = input->rows * input->cols;
    
    for (int i = 0; i < total; i++) {
        float x = input->data[i];
        
        /* Clamp to prevent overflow in exp() */
        if (x < -50.0f) {
            output->data[i] = 0.0f;
        } else if (x > 50.0f) {
            output->data[i] = 1.0f;
        } else {
            output->data[i] = 1.0f / (1.0f + expf(-x));
        }
    }
}

/**
 * Tanh: Hyperbolic Tangent
 * f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * 
 * Range: (-1, 1) - zero-centered!
 * 
 * Advantage over sigmoid: outputs can be negative
 * This helps with gradient flow (zero-centered = symmetric)
 * 
 * Still has saturation problem at extremes.
 * 
 * Often used in: LSTMs, RNNs, some CNN architectures
 */
void activation_tanh_act(Matrix* input, Matrix* output)
{
    if (input == NULL || output == NULL) {
        return;
    }
    
    int total = input->rows * input->cols;
    
    for (int i = 0; i < total; i++) {
        float x = input->data[i];
        
        /* Use tanhf for efficiency */
        output->data[i] = tanhf(x);
    }
}

/**
 * Softmax: converts logits to probabilities
 * 
 * Given input vector x of length n:
 *   output[i] = e^(x[i]) / Σ e^(x[j])
 * 
 * Result: all outputs are positive and sum to 1!
 * 
 * MATHEMATICAL INTUITION:
 * - Exponential amplifies differences: small advantages become large
 * - Normalization makes them sum to 1 (probability distribution)
 * 
 * Example:
 *   Input:  [2.0, 1.0, 0.1]
 *   Exp:    [7.39, 2.72, 1.11]
 *   Sum:    11.22
 *   Output: [0.66, 0.24, 0.10]
 * 
 * The "2.0" input got 66% probability despite only being 2x the second one!
 * This is because exp(2)/exp(1) = e^1 ≈ 2.72x multiplier
 * 
 * CRITICAL: NUMERICAL STABILITY!
 * 
 * Problem: e^1000 overflows! Even e^100 is too big.
 * 
 * Solution: Subtract max before exponentiating:
 *   e^(x[i]) / Σ e^(x[j])
 *   = e^(x[i] - max) / Σ e^(x[j] - max)  ← divide top and bottom by e^max
 * 
 * This works because:
 *   e^(a - max) / Σ e^(b - max) = e^a / e^max  /  (Σ e^b / e^max) = e^a / Σ e^b
 * 
 * Now all exponentiated values are ≤ e^0 = 1!
 */
void activation_softmax(Matrix* input, Matrix* output)
{
    if (input == NULL || output == NULL) {
        return;
    }
    
    /* Softmax is typically used on vectors (1D), but we'll handle any shape */
    int total = input->rows * input->cols;
    
    /* Find maximum for numerical stability */
    float max_val = input->data[0];
    for (int i = 1; i < total; i++) {
        if (input->data[i] > max_val) {
            max_val = input->data[i];
        }
    }
    
    /* Compute sum of exp(x - max) */
    float sum = 0.0f;
    for (int i = 0; i < total; i++) {
        sum += expf(input->data[i] - max_val);
    }
    
    /* Compute softmax */
    for (int i = 0; i < total; i++) {
        output->data[i] = expf(input->data[i] - max_val) / sum;
    }
}