/**
 * @file layers.c
 * @brief Implementation of neural network layers
 * 
 * LAYERS EXPLAINED:
 * ================
 * 
 * A neural network layer transforms input to output:
 *   input → [linear: Wx + b] → [activation: σ()] → output
 * 
 * The "forward pass" through the entire network is just calling
 * forward() on each layer in sequence.
 * 
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║                    FORWARD PASS FLOW                          ║
 * ╠═══════════════════════════════════════════════════════════════╣
 * ║                                                               ║
 * ║  Input     Layer 1      Layer 2      Layer 3      Output      ║
 * ║  Data    ┌────────┐  ┌────────┐  ┌────────┐                ║
 * ║  ──────▶│Wx+b, σ  │─▶│Wx+b, σ  │─▶│Wx+b, σ  │──────▶ Class  ║
 * ║          └────────┘  └────────┘  └────────┘                ║
 * ║            ↓           ↓           ↓                        ║
 * ║         z1 = W1*x   z2 = W2*a1   z3 = W3*a2               ║
 * ║         a1 = σ(z1)  a2 = σ(z2)  a3 = σ(z3)                ║
 * ║                                                               ║
 * ╚═══════════════════════════════════════════════════════════════╝
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layers.h"
#include "matrix.h"
#include "activations.h"

/**
 * Create a new dense (fully connected) layer
 * 
 * Allocates:
 * - weights matrix [output_size x input_size] - Xavier initialized
 * - biases vector [1 x output_size] - zeros
 * - output cache [1 x output_size]
 * - pre_activation cache [1 x output_size]
 */
DenseLayer* dense_layer_create(int input_size, int output_size, ActivationFn activation)
{
    if (input_size <= 0 || output_size <= 0) {
        fprintf(stderr, "ERROR: Invalid layer dimensions %d x %d\n", input_size, output_size);
        return NULL;
    }
    
    /* Allocate the layer struct */
    DenseLayer* layer = (DenseLayer*)malloc(sizeof(DenseLayer));
    if (layer == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate DenseLayer\n");
        return NULL;
    }
    
    /* Initialize to NULL for safe cleanup */
    layer->weights = NULL;
    layer->biases = NULL;
    layer->output = NULL;
    layer->pre_activation = NULL;
    
    /* Store dimensions */
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;
    
    /* Create name */
    snprintf(layer->name, 64, "Dense_%dx%d", input_size, output_size);
    
    /* Allocate weights: [output_size x input_size] */
    /* Each row is the weights for one output neuron */
    Matrix weights = matrix_create(output_size, input_size);
    layer->weights = (Matrix*)malloc(sizeof(Matrix));
    *layer->weights = weights;
    matrix_random(layer->weights);  /* Xavier initialization */
    
    /* Allocate biases: [1 x output_size] */
    Matrix biases = matrix_create(1, output_size);
    layer->biases = (Matrix*)malloc(sizeof(Matrix));
    *layer->biases = biases;
    matrix_zero(layer->biases);  /* Initialize to zero */
    
    /* Allocate output cache: [1 x output_size] */
    Matrix output = matrix_create(1, output_size);
    layer->output = (Matrix*)malloc(sizeof(Matrix));
    *layer->output = output;
    
    /* Allocate pre-activation cache: [1 x output_size] */
    Matrix pre = matrix_create(1, output_size);
    layer->pre_activation = (Matrix*)malloc(sizeof(Matrix));
    *layer->pre_activation = pre;
    
    return layer;
}

/**
 * Free all memory associated with a layer
 * 
 * CRITICAL: Must free ALL allocated memory to prevent leaks!
 *   - weights matrix
 *   - biases matrix
 *   - output cache
 *   - pre_activation cache
 *   - layer struct itself
 */
void dense_layer_free(DenseLayer* layer)
{
    if (layer == NULL) {
        return;
    }
    
    /* Free all matrices */
    if (layer->weights != NULL) {
        matrix_free(layer->weights);
        free(layer->weights);
        layer->weights = NULL;
    }
    
    if (layer->biases != NULL) {
        matrix_free(layer->biases);
        free(layer->biases);
        layer->biases = NULL;
    }
    
    if (layer->output != NULL) {
        matrix_free(layer->output);
        free(layer->output);
        layer->output = NULL;
    }
    
    if (layer->pre_activation != NULL) {
        matrix_free(layer->pre_activation);
        free(layer->pre_activation);
        layer->pre_activation = NULL;
    }
    
    /* Free the layer struct */
    free(layer);
}

/**
 * FORWARD PASS — The Core Computation!
 * ===================================
 * 
 * This is where the magic happens. For each input, we compute:
 * 
 *   1. Linear transform: z = x * W^T + b
 *      - x is [1 x input_size]
 *      - W is [output_size x input_size]
 *      - W^T is [input_size x output_size]
 *      - x * W^T is [1 x output_size]
 *      - Add bias: [1 x output_size] + [1 x output_size] = [1 x output_size]
 * 
 *   2. Apply activation: a = σ(z)
 *      - Apply element-wise to get [1 x output_size]
 * 
 * WHY CACHE OUTPUT?
 * =================
 * In inference, we might need the output multiple times (for debugging,
 * for different inputs, etc.). Recomputing is expensive (matrix multiply!).
 * Caching saves computation at the cost of some memory.
 * 
 * In training (not implemented here), we'd also need to cache pre_activation
 * for computing gradients during backpropagation.
 */
Matrix* dense_layer_forward(DenseLayer* layer, Matrix* input)
{
    if (layer == NULL || input == NULL) {
        fprintf(stderr, "ERROR: NULL input to dense_layer_forward\n");
        return NULL;
    }
    
    /* Validate input dimensions */
    if (input->cols != layer->input_size) {
        fprintf(stderr, "ERROR: Input size mismatch. Expected %d, got %d\n",
                layer->input_size, input->cols);
        return NULL;
    }
    
    /*
     * Step 1: Linear transform
     * 
     * input:        [1 x input_size]
     * weights^T:    [input_size x output_size]  ← TRANSPOSE!
     * result:       [1 x output_size]
     * 
     * Then add biases: [1 x output_size] + [1 x output_size] = [1 x output_size]
     */
    
    /* 
     * Compute output for each neuron:
     * output[j] = sum over i of input[i] * weights[j,i] + bias[j]
     * 
     * This is essentially doing input * weights^T (the transpose is implicit
     * because we access weights row-by-row)
     */
    for (int j = 0; j < layer->output_size; j++) {
        float sum = 0.0f;
        for (int i = 0; i < layer->input_size; i++) {
            sum += input->data[i] * layer->weights->data[j * layer->weights->stride + i];
        }
        layer->pre_activation->data[j] = sum + layer->biases->data[j];
    }
    
    /*
     * Step 2: Apply activation function
     * 
     * If activation is NULL (shouldn't happen), skip it
     */
    if (layer->activation != NULL) {
        /* Copy pre_activation to output, then apply activation in-place */
        matrix_copy(layer->pre_activation, layer->output);
        layer->activation(layer->output, layer->output);
    } else {
        /* No activation - just copy pre_activation to output */
        matrix_copy(layer->pre_activation, layer->output);
    }
    
    return layer->output;
}

/**
 * Print layer information
 */
void dense_layer_print_info(DenseLayer* layer)
{
    if (layer == NULL) {
        printf("Layer: (null)\n");
        return;
    }
    
    const char* act_name = "Linear";
    if (layer->activation == activation_relu) {
        act_name = "ReLU";
    } else if (layer->activation == activation_sigmoid) {
        act_name = "Sigmoid";
    } else if (layer->activation == activation_tanh_act) {
        act_name = "Tanh";
    } else if (layer->activation == activation_softmax) {
        act_name = "Softmax";
    }
    
    printf("Layer: %s\n", layer->name);
    printf("  Input size:  %d\n", layer->input_size);
    printf("  Output size: %d\n", layer->output_size);
    printf("  Activation: %s\n", act_name);
    printf("  Weights shape: [%d x %d]\n", layer->output_size, layer->input_size);
    printf("  Biases shape: [1 x %d]\n", layer->output_size);
}

/**
 * Set pretrained weights
 */
void dense_layer_set_weights(DenseLayer* layer, Matrix* weights, Matrix* biases)
{
    if (layer == NULL || weights == NULL || biases == NULL) {
        return;
    }
    
    /* Validate dimensions */
    if (weights->rows != layer->output_size || weights->cols != layer->input_size) {
        fprintf(stderr, "ERROR: Weight dimensions don't match layer\n");
        return;
    }
    
    if (biases->cols != layer->output_size || biases->rows != 1) {
        fprintf(stderr, "ERROR: Bias dimensions don't match layer\n");
        return;
    }
    
    /* Copy weights */
    matrix_copy(weights, layer->weights);
    
    /* Copy biases */
    matrix_copy(biases, layer->biases);
}