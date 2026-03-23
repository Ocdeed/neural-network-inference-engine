/**
 * @file network.h
 * @brief Neural Network container - holds multiple layers
 * 
 * A NeuralNetwork is just a collection of DenseLayers.
 * The "forward pass" through the network is simply calling
 * forward() on each layer in sequence.
 * 
 * This is the "deep" in deep learning - multiple stacked layers!
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "layers.h"
#include "matrix.h"

/**
 * @brief Neural Network - a container for multiple layers
 * 
 * The network owns its layers (will free them on destroy).
 * Layers are stored in a dynamic array that grows as needed.
 */
typedef struct {
    DenseLayer** layers;  /**< Array of layer pointers */
    int num_layers;       /**< Number of layers currently added */
    int capacity;         /**< Allocated capacity (for realloc) */
    char name[128];       /**< Network name for identification */
} NeuralNetwork;

/**
 * @brief Create a new neural network
 * @param name Optional name (e.g., "MNIST_Classifier")
 * @return Newly allocated NeuralNetwork
 */
NeuralNetwork* network_create(const char* name);

/**
 * @brief Free all memory associated with network
 * @param net Network to free
 * 
 * Frees:
 * - All layers (and their weights, biases, caches)
 * - The layers array
 * - The network struct itself
 */
void network_free(NeuralNetwork* net);

/**
 * @brief Add a layer to the network
 * @param net Network to add to
 * @param layer Layer to add (network takes ownership)
 * 
 * Uses dynamic array with realloc for growth:
 * - Start with capacity 4
 * - When full, double capacity
 * - realloc preserves old data AND can move memory!
 * 
 * IMPORTANT: After realloc, the old pointer may be invalid!
 * Always use the returned pointer.
 */
void network_add_layer(NeuralNetwork* net, DenseLayer* layer);

/**
 * @brief Forward pass through entire network
 * @param net The neural network
 * @param input Input matrix [batch_size x first_layer_input_size]
 * @return Output matrix from final layer
 * 
 * This chains all layers:
 *   output = input
 *   for each layer:
 *       output = layer->forward(output)
 *   return output
 */
Matrix* network_forward(NeuralNetwork* net, Matrix* input);

/**
 * @brief Get prediction (class index) from network
 * @param net The neural network
 * @param input Input matrix
 * @return Index of class with highest probability (argmax)
 * 
 * Assumes final layer outputs probabilities via softmax
 * or similar function, and the highest value is the prediction.
 */
int network_predict(NeuralNetwork* net, Matrix* input);

/**
 * @brief Print network architecture summary
 * @param net The neural network
 * 
 * Similar to Keras model.summary() - shows:
 * - Layer type and output shape
 * - Number of parameters per layer
 * - Total parameters
 */
void network_print_summary(NeuralNetwork* net);

/**
 * @brief Count total parameters in network
 * @param net The neural network
 * @return Total number of trainable parameters
 * 
 * Parameter = weights + biases
 * For layer: weights = output_size × input_size, biases = output_size
 */
long network_count_parameters(NeuralNetwork* net);

#endif /* NETWORK_H */