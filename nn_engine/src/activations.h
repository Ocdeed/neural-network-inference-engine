/**
 * @file activations.h
 * @brief Activation functions for neural networks
 * 
 * Activation functions introduce NON-LINEARITY into neural networks.
 * Without them, stacking layers would just be one giant linear transform!
 * 
 * THE BIG PICTURE:
 * 
 *   Layer 1: output = W1 * input + b1  (linear)
 *   Layer 2: output = W2 * (output) + b2  (linear)
 *   
 *   Combined: output = W2 * (W1 * input + b1) + b2
 *            = (W2 * W1) * input + (W2 * b1 + b2)
 *            = W_combined * input + b_combined
 *            
 *   It's still just ONE linear layer! We haven't gained anything.
 *   
 *   WITH activations:
 *   Layer 1: h1 = σ(W1 * input + b1)  (non-linear!)
 *   Layer 2: output = σ(W2 * h1 + b2)
 *   
 *   Now we have a real deep network that can learn complex patterns!
 */

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "matrix.h"

/**
 * @brief Function pointer type for activation functions
 * 
 * This allows us to:
 * - Store different activations in arrays
 * - Pass activations as parameters to layer functions
 * - Swap activations at runtime
 * 
 * Example usage:
 *   ActivationFn activations[] = {activation_relu, activation_sigmoid};
 *   for (int i = 0; i < 2; i++) activations[i](&input, &output);
 */
typedef void (*ActivationFn)(Matrix*, Matrix*);

/**
 * @brief ReLU (Rectified Linear Unit)
 * 
 * f(x) = max(0, x)
 * 
 * The most popular activation function!
 * - Fast to compute (just a comparison)
 * - Doesn't saturate for positive values
 * - Creates sparse representations (many zeros)
 * 
 * Graph:
 *      │
 *    3 │     ╱
 *      │    ╱
 *    0 ──┴────────
 *      │       3
 *      │
 * 
 * Problem: "Dying ReLU" - if a neuron gets negative once, it dies forever.
 * Solution: Leaky ReLU (see below)
 */
void activation_relu(Matrix* input, Matrix* output);

/**
 * @brief Derivative of ReLU (for backpropagation understanding)
 * 
 * f'(x) = 1 if x > 0, else 0
 * 
 * This tells us how to adjust weights during training.
 * We won't use it for inference, but it's good to understand!
 */
void activation_relu_derivative(Matrix* input, Matrix* output);

/**
 * @brief Leaky ReLU — fixes the "dying ReLU" problem
 * 
 * f(x) = x if x > 0, else α * x
 * 
 * Common α value: 0.01
 * 
 * Graph (with α=0.1):
 *      │
 *    3 │   ╱
 *      │  ╱
 *    0 ┼─┴───────
 *      │╱
 * -0.3 │
 * 
 * Now negative inputs still have a small gradient, so neurons don't die!
 */
void activation_leaky_relu(Matrix* input, Matrix* output, float alpha);

/**
 * @brief Sigmoid — classic activation
 * 
 * f(x) = 1 / (1 + e^(-x))
 * 
 * Output range: (0, 1) — useful for probabilities!
 * 
 * Problem: "vanishing gradients" - sigmoid saturates at 0 and 1,
 *          making gradients nearly zero. This slows down training.
 * 
 * Graph:
 *      │   ╭────
 *    1 ┤  ╱
 *      │ ╱
 *  0.5 ┤╱
 *      │
 *    0 ┼───────────
 *      │
 * 
 * Common when: output needs to be probability (0-1)
 * Less common now: in hidden layers (ReLU is better)
 */
void activation_sigmoid(Matrix* input, Matrix* output);

/**
 * @brief Hyperbolic Tangent (tanh)
 * 
 * f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * 
 * Output range: (-1, 1) — zero-centered!
 * 
 * Like sigmoid but steeper and zero-centered.
 * Still has vanishing gradient problem for extreme values.
 * 
 * Graph:
 *      │  ╭──
 *    1 ┤ ╱
 *      │╱
 *    0 ┼───────
 *      │╲
 *   -1 ┤ ╲__
 *      │
 */
void activation_tanh_act(Matrix* input, Matrix* output);

/**
 * @brief Softmax — for multi-class classification
 * 
 * Given vector x, output[i] = e^(x[i]) / Σ e^(x[j])
 * 
 * Converts raw "logits" to probabilities that sum to 1!
 * 
 * Example:
 *   input:  [2.0, 1.0, 0.1]
 *   exp:    [7.39, 2.72, 1.11]
 *   sum:    11.22
 *   output: [0.66, 0.24, 0.10]  ← probabilities!
 * 
 * IMPORTANT: Numerical stability!
 *   Subtract max before exp to prevent overflow:
 *   exp(x - max(x))
 * 
 * Used in: Output layer for classification (especially multi-class)
 */
void activation_softmax(Matrix* input, Matrix* output);

#endif /* ACTIVATIONS_H */