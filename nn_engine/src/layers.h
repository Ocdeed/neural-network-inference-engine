/**
 * @file layers.h
 * @brief Neural network layer implementations
 * 
 * A "layer" is the fundamental building block of a neural network.
 * Each layer:
 *   1. Takes input (from previous layer or original data)
 *   2. Applies linear transform: z = Wx + b
 *   3. Applies non-linear activation: a = σ(z)
 *   4. Passes output to next layer
 * 
 * The "forward pass" is just calling layer->forward() on each layer in sequence!
 */

#ifndef LAYERS_H
#define LAYERS_H

#include "matrix.h"
#include "activations.h"

/**
 * @brief Dense (Fully Connected) Layer
 * 
 * A dense layer connects EVERY input to EVERY output.
 * It's the most common layer type in neural networks.
 * 
 * Forward computation:
 *   1. Linear: z = input * weights^T + bias  (matrix multiply + add)
 *   2. Non-linear: output = activation(z)   (ReLU, Sigmoid, etc.)
 * 
 * Weight matrix shape: [output_size, input_size]
 *   - Each ROW is the weights for ONE output neuron
 *   - weights[0] = weights for output neuron 0
 * 
 * Input shape: [batch_size, input_size]  (we use batch_size=1 for inference)
 * Output shape: [batch_size, output_size]
 * 
 * Example for MNIST digit classification:
 *   Input: 784 pixels (28x28 image)
 *   Hidden: 256 neurons
 *   Output: 10 classes (digits 0-9)
 *   
 *   Layer 1: 784 → 256 with ReLU
 *   Layer 2: 256 → 10 with Softmax
 */
typedef struct {
    Matrix* weights;        /**< Weight matrix [output_size x input_size] */
    Matrix* biases;         /**< Bias vector [1 x output_size] */
    Matrix* output;         /**< Cached output [batch_size x output_size] */
    Matrix* pre_activation; /**< z = input * W^T + b before activation */
    ActivationFn activation; /**< Activation function (ReLU, Sigmoid, etc) */
    int input_size;         /**< Number of input features */
    int output_size;        /**< Number of output neurons */
    char name[64];          /**< Layer name for debugging */
} DenseLayer;

/**
 * @brief Create a new dense layer
 * @param input_size Number of input features
 * @param output_size Number of output neurons
 * @param activation Activation function to use
 * @return Newly created DenseLayer with Xavier-initialized weights
 * 
 * Weights are initialized with Xavier/He initialization to prevent
 * vanishing/exploding gradients during training (or to ensure inference
 * with pretrained weights works correctly).
 * 
 * Biases are initialized to zero.
 */
DenseLayer* dense_layer_create(int input_size, int output_size, ActivationFn activation);

/**
 * @brief Free all memory associated with a dense layer
 * @param layer Pointer to layer to free
 */
void dense_layer_free(DenseLayer* layer);

/**
 * @brief Forward pass through the dense layer
 * @param layer The dense layer
 * @param input Input matrix [batch_size x input_size]
 * @return Output matrix [batch_size x output_size]
 * 
 * Computation:
 *   1. z = input * weights^T + bias   (linear transform)
 *   2. output = activation(z)         (non-linear)
 * 
 * The output is CACHED in layer->output for potential reuse.
 * 
 * WHY TRANSPOSE WEIGHTS?
 * ====================
 * 
 * Input: [batch x input_size]  e.g., [1 x 784]
 * Weights: [output_size x input_size]  e.g., [256 x 784]
 * 
 * To multiply correctly, we need:
 *   [1 x 784] * [784 x 256] = [1 x 256]
 *                    ↑
 *                This is weights^T!
 * 
 * The weight matrix is stored as [output x input], but we need
 * to multiply by its transpose [input x output] during forward pass.
 */
Matrix* dense_layer_forward(DenseLayer* layer, Matrix* input);

/**
 * @brief Print layer information
 * @param layer The dense layer
 * 
 * Shows: name, input size, output size, activation function name
 */
void dense_layer_print_info(DenseLayer* layer);

/**
 * @brief Set layer weights from pretrained model
 * @param layer The dense layer
 * @param weights Weight matrix (will be copied)
 * @param biases Bias vector (will be copied)
 * 
 * Used when loading a trained model from file.
 */
void dense_layer_set_weights(DenseLayer* layer, Matrix* weights, Matrix* biases);

#endif /* LAYERS_H */