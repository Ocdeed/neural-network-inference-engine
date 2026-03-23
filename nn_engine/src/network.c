/**
 * @file network.c
 * @brief Implementation of NeuralNetwork container
 * 
 * NEURAL NETWORKS EXPLAINED:
 * =========================
 * 
 * A neural network is simply a sequence of layers. The "deep" in "deep learning"
 * refers to having many layers stacked together.
 * 
 * WHY STACK LAYERS?
 * 
 * Without depth (just 1 layer = linear model, can't learn complex patterns)
 * With 2-3 layers (can learn simple non-linear patterns)
 * With 10+ layers (can learn incredibly complex patterns - images, text, etc.)
 * 
 * The magic is that each layer transforms the representation of the data,
 * and stacking them creates progressively more abstract representations.
 * 
 * Example:
 *   Layer 1: learns edges from raw pixels
 *   Layer 2: learns shapes from edges
 *   Layer 3: learns digits from shapes
 * 
 * ╔═══════════════════════════════════════════════════════════════════╗
 * ║              3-LAYER NETWORK FORWARD PASS                          ║
 * ╠═══════════════════════════════════════════════════════════════════╣
 * ║                                                                   ║
 * ║  Input        Layer 1         Layer 2         Layer 3        Out ║
 * ║  (784)     784→128,ReLU    128→64,ReLU     64→10,Softmax    (10) ║
 * ║   ↓            ↓              ↓               ↓             ↓    ║
 * ║  [x]  →  W1*x+b1  →  ReLU  →  W2*x+b2  →  ReLU  →  W3*x+b3 → Softmax
 * ║         [1×128]      [1×128]  [1×64]      [1×64]   [1×10]    [1×10]
 * ║                                                                   ║
 * ║  Parameters:                                                        ║
 * ║    Layer 1: 784×128 weights + 128 biases = 100,480                ║
 * ║    Layer 2: 128×64 weights + 64 biases = 8,320                    ║
 * ║    Layer 3: 64×10 weights + 10 biases = 650                       ║
 * ║    Total: 109,450 parameters                                       ║
 * ║                                                                   ║
 * ╚═══════════════════════════════════════════════════════════════════╝
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "network.h"
#include "layers.h"
#include "matrix.h"

/**
 * Create a new neural network
 */
NeuralNetwork* network_create(const char* name)
{
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (net == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate NeuralNetwork\n");
        return NULL;
    }
    
    /* Initialize fields */
    net->num_layers = 0;
    net->capacity = 4;  /* Start with capacity for 4 layers */
    net->layers = (DenseLayer**)malloc(net->capacity * sizeof(DenseLayer*));
    
    if (net->layers == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate layers array\n");
        free(net);
        return NULL;
    }
    
    /* Set name (default if NULL or empty) */
    if (name != NULL && name[0] != '\0') {
        snprintf(net->name, 128, "%s", name);
    } else {
        snprintf(net->name, 128, "UnnamedNetwork");
    }
    
    return net;
}

/**
 * Free all memory in the network
 * 
 * CRITICAL: Must free all layers AND the layers array!
 */
void network_free(NeuralNetwork* net)
{
    if (net == NULL) {
        return;
    }
    
    /* Free each layer */
    for (int i = 0; i < net->num_layers; i++) {
        if (net->layers[i] != NULL) {
            dense_layer_free(net->layers[i]);
            net->layers[i] = NULL;
        }
    }
    
    /* Free the layers array */
    if (net->layers != NULL) {
        free(net->layers);
        net->layers = NULL;
    }
    
    /* Free the network struct */
    free(net);
}

/**
 * Add a layer to the network
 * 
 * REALLOC EXPLAINED:
 * =================
 * 
 * realloc(ptr, new_size) does one of three things:
 * 
 * 1. If there's room to grow in place: extend the existing block, return same ptr
 * 2. If need to move: allocate new block, copy data, FREE OLD BLOCK, return NEW ptr
 * 3. If new_size is 0: free old block, return NULL
 * 
 * THE DANGER:
 * Old pointers become INVALID after realloc! If you saved 'layers' in a variable
 * before realloc, that variable now points to freed memory (dangling pointer)!
 * 
 * Correct pattern:
 *   DenseLayer** new_layers = realloc(net->layers, new_size);
 *   if (new_layers == NULL) { error! }
 *   net->layers = new_layers;  // Only update AFTER realloc succeeds
 */
void network_add_layer(NeuralNetwork* net, DenseLayer* layer)
{
    if (net == NULL || layer == NULL) {
        fprintf(stderr, "ERROR: NULL network or layer in network_add_layer\n");
        return;
    }
    
    /* Check shape compatibility with previous layer */
    if (net->num_layers > 0) {
        DenseLayer* prev = net->layers[net->num_layers - 1];
        if (prev->output_size != layer->input_size) {
            fprintf(stderr, "ERROR: Shape mismatch! Layer %d outputs %d, next layer expects %d\n",
                    net->num_layers - 1, prev->output_size, layer->input_size);
            return;
        }
    }
    
    /* Grow array if needed */
    if (net->num_layers >= net->capacity) {
        /* Double the capacity */
        int new_capacity = net->capacity * 2;
        
        /* IMPORTANT: Use temp pointer! */
        DenseLayer** new_layers = (DenseLayer**)realloc(net->layers, 
                                                       new_capacity * sizeof(DenseLayer*));
        
        if (new_layers == NULL) {
            fprintf(stderr, "ERROR: Failed to realloc layers array\n");
            return;
        }
        
        /* Only update after successful realloc! */
        net->layers = new_layers;
        net->capacity = new_capacity;
    }
    
    /* Add the layer */
    net->layers[net->num_layers] = layer;
    net->num_layers++;
}

/**
 * Forward pass through entire network
 * 
 * Simply chains all layers - output of one becomes input to next!
 */
Matrix* network_forward(NeuralNetwork* net, Matrix* input)
{
    if (net == NULL || input == NULL) {
        fprintf(stderr, "ERROR: NULL network or input in network_forward\n");
        return NULL;
    }
    
    if (net->num_layers == 0) {
        fprintf(stderr, "ERROR: No layers in network\n");
        return NULL;
    }
    
    /* Validate first layer input size */
    if (input->cols != net->layers[0]->input_size) {
        fprintf(stderr, "ERROR: Input size %d doesn't match first layer %d\n",
                input->cols, net->layers[0]->input_size);
        return NULL;
    }
    
    /* Chain through all layers */
    Matrix* current = input;
    
    for (int i = 0; i < net->num_layers; i++) {
        current = dense_layer_forward(net->layers[i], current);
        
        if (current == NULL) {
            fprintf(stderr, "ERROR: Forward pass failed at layer %d\n", i);
            return NULL;
        }
    }
    
    return current;  /* Return final output */
}

/**
 * Get prediction (argmax)
 * 
 * ARGMAX EXPLAINED:
 * ===============
 * 
 * For classification, the network outputs probabilities (via softmax).
 * Each of the 10 output values represents P(class=i | input).
 * 
 * To get the final prediction, we pick the index with highest probability:
 *   prediction = argmax(output) = argmax([p0, p1, ..., p9]) = index of max
 * 
 * Example:
 *   output = [0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.90, 0.01, 0.01]
 *   argmax = 7 (because 0.90 is the largest!)
 */
int network_predict(NeuralNetwork* net, Matrix* input)
{
    if (net == NULL || input == NULL) {
        return -1;
    }
    
    /* Forward pass to get probabilities */
    Matrix* output = network_forward(net, input);
    if (output == NULL) {
        return -1;
    }
    
    /* Find index of maximum value (argmax) */
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

/**
 * Print network architecture summary
 * 
 * Similar to Keras model.summary()!
 */
void network_print_summary(NeuralNetwork* net)
{
    if (net == NULL) {
        printf("Network: (null)\n");
        return;
    }
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  Neural Network: %s\n", net->name);
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("%-4s %-20s %-15s %15s\n", "#", "Layer", "Output Shape", "Parameters");
    printf("─────────────────────────────────────────────────────────────────\n");
    
    long total_params = 0;
    
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer* layer = net->layers[i];
        
        /* Get activation name */
        const char* act_name = "Linear";
        if (layer->activation == activation_relu) act_name = "ReLU";
        else if (layer->activation == activation_sigmoid) act_name = "Sigmoid";
        else if (layer->activation == activation_tanh_act) act_name = "Tanh";
        else if (layer->activation == activation_softmax) act_name = "Softmax";
        
        /* Count parameters: weights + biases */
        long layer_params = (long)layer->weights->rows * layer->weights->cols + 
                          (long)layer->biases->cols;
        total_params += layer_params;
        
        /* Print layer info */
        printf("%-4d %-20s %-15s %15ld\n", 
               i + 1,
               act_name,
               "None",  /* We'd need more info to show output shape */
               layer_params);
        
        printf("     └─ Dense: %d → %d\n", layer->input_size, layer->output_size);
    }
    
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("%-25s %15ld\n", "Total Parameters:", total_params);
    printf("═══════════════════════════════════════════════════════════════\n");
}

/**
 * Count total parameters
 */
long network_count_parameters(NeuralNetwork* net)
{
    if (net == NULL) {
        return 0;
    }
    
    long total = 0;
    
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer* layer = net->layers[i];
        
        /* Weights */
        total += (long)layer->weights->rows * layer->weights->cols;
        /* Biases */
        total += (long)layer->biases->cols;
    }
    
    return total;
}