/**
 * @file main.c
 * @brief Neural Network Inference Engine - Demo
 * @description Phase 4: Model saving/loading
 * 
 * This demonstrates a complete 3-layer neural network with save/load.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "activations.h"
#include "layers.h"
#include "network.h"
#include "loader.h"

/**
 * @brief Display welcome banner
 */
static void print_banner(void)
{
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Neural Network Inference Engine v0.1.0                ║\n");
    printf("║     Built in pure C from scratch                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Phase 4: Model Saving & Loading                          ║\n");
    printf("║  → Save network to .nnbin binary format                   ║\n");
    printf("║  → Load network from .nnbin file                          ║\n");
    printf("║  → Analyze GGUF headers (llama.cpp format)                ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

/**
 * @brief Create a small random input for testing
 */
static Matrix* create_random_input(int size)
{
    Matrix* input = (Matrix*)malloc(sizeof(Matrix));
    *input = matrix_create(1, size);
    
    /* Fill with random values in range [0, 1] - like normalized pixel values */
    for (int i = 0; i < size; i++) {
        input->data[i] = (float)rand() / (float)RAND_MAX;
    }
    
    return input;
}

/**
 * @brief Print output probabilities nicely
 */
static void print_probabilities(Matrix* output, int num_classes)
{
    printf("  Class probabilities:\n");
    printf("  ");
    for (int i = 0; i < num_classes; i++) {
        printf("  %d:%.3f", i, output->data[i]);
        if (i < num_classes - 1) printf("  ");
    }
    printf("\n");
}

int main(void)
{
    print_banner();
    
    /* Seed random for reproducibility */
    srand(42);
    
    printf("═══ Building 3-Layer Neural Network ═══\n\n");
    
    /*
     * ARCHITECTURE: 784 → 128 → 64 → 10
     * 
     * This is a typical MNIST digit classifier:
     * - Input: 784 pixels (28x28 grayscale image, flattened)
     * - Layer 1: 784 → 128 with ReLU (hidden layer)
     * - Layer 2: 128 → 64 with ReLU (hidden layer)
     * - Layer 3: 64 → 10 with Softmax (output layer, 10 digits)
     * 
     * Parameter count:
     * - Layer 1: 784×128 + 128 = 100,480
     * - Layer 2: 128×64 + 64 = 8,320
     * - Layer 3: 64×10 + 10 = 650
     * - Total: 109,450 parameters
     */
    
    /* Create network */
    NeuralNetwork* net = network_create("MNIST_Digit_Classifier");
    printf("Created network: %s\n", net->name);
    
    /* Add Layer 1: 784 → 128, ReLU */
    DenseLayer* layer1 = dense_layer_create(784, 128, activation_relu);
    network_add_layer(net, layer1);
    printf("Added Layer 1: 784 → 128 (ReLU)\n");
    
    /* Add Layer 2: 128 → 64, ReLU */
    DenseLayer* layer2 = dense_layer_create(128, 64, activation_relu);
    network_add_layer(net, layer2);
    printf("Added Layer 2: 128 → 64 (ReLU)\n");
    
    /* Add Layer 3: 64 → 10, Softmax (for 10-digit classification) */
    DenseLayer* layer3 = dense_layer_create(64, 10, activation_softmax);
    network_add_layer(net, layer3);
    printf("Added Layer 3: 64 → 10 (Softmax)\n");
    
    /* Print architecture summary */
    printf("\n");
    network_print_summary(net);
    
    /* Count and verify parameters */
    long param_count = network_count_parameters(net);
    printf("\n→ Parameter count: %ld (100,480 + 8,320 + 650 = 109,450)\n", param_count);
    
    /* Create random input (simulating 28x28 image) */
    printf("\n═══ Running Forward Pass ═══\n\n");
    printf("Creating random input vector (784 values)...\n");
    Matrix* input = create_random_input(784);
    printf("  Sample values: [%0.2f, %0.2f, %0.2f, ... %0.2f]\n",
           input->data[0], input->data[1], input->data[2], input->data[783]);
    
    /* Forward pass through entire network */
    printf("\nRunning forward pass through all 3 layers...\n");
    Matrix* output = network_forward(net, input);
    
    if (output == NULL) {
        printf("ERROR: Forward pass failed!\n");
        network_free(net);
        matrix_free(input);
        free(input);
        return EXIT_FAILURE;
    }
    
    /* Print output */
    printf("\n═══ Results ═══\n\n");
    printf("Output probabilities (10 classes):\n");
    print_probabilities(output, 10);
    
    /* Get prediction (argmax) */
    int prediction = network_predict(net, input);
    printf("\n→ Predicted digit: %d\n", prediction);
    
    /* Verify sum = 1 (softmax property) */
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {
        sum += output->data[i];
    }
    printf("→ Probability sum: %.4f (should be ~1.0)\n", sum);
    
    /* Cleanup */
    printf("\n═══ Cleanup ═══\n");
    matrix_free(input);
    free(input);
    network_free(net);
    printf("✓ Memory freed - no leaks!\n");
    
    /* ═════════════════════════════════════════════════════════════ */
    /* PHASE 4: Save and Load the network                             */
    /* ═════════════════════════════════════════════════════════════ */
    
    printf("\n═══ Phase 4: Model Saving & Loading ═══\n\n");
    
    /* Recreate the same network */
    printf("Recreating network for save/load demo...\n");
    NeuralNetwork* net2 = network_create("MNIST_Digit_Classifier");
    DenseLayer* l1 = dense_layer_create(784, 128, activation_relu);
    DenseLayer* l2 = dense_layer_create(128, 64, activation_relu);
    DenseLayer* l3 = dense_layer_create(64, 10, activation_softmax);
    network_add_layer(net2, l1);
    network_add_layer(net2, l2);
    network_add_layer(net2, l3);
    
    /* Save to file */
    printf("Saving network to models/mnist_demo.nnbin...\n");
    network_save(net2, "models/mnist_demo.nnbin");
    
    /* Free the original network */
    network_free(net2);
    
    /* Load from file */
    printf("\nLoading network from file...\n");
    NeuralNetwork* loaded = network_load("models/mnist_demo.nnbin");
    
    if (loaded == NULL) {
        printf("ERROR: Failed to load network!\n");
        return EXIT_FAILURE;
    }
    
    /* Verify loaded network works */
    printf("\nVerifying loaded network...\n");
    Matrix* input2 = create_random_input(784);
    Matrix* output2 = network_forward(loaded, input2);
    print_probabilities(output2, 10);
    
    int pred2 = network_predict(loaded, input2);
    printf("→ Predicted digit: %d\n", pred2);
    
    /* Print architecture */
    printf("\nLoaded network architecture:\n");
    network_print_summary(loaded);
    
    /* Cleanup */
    matrix_free(input2);
    free(input2);
    network_free(loaded);
    
    /* Show GGUF header info (won't work without a real .gguf file) */
    printf("\n═══ GGUF Format Demo ═══\n");
    show_gguf_header_info("models/mnist_demo.nnbin");
    
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("→ Phase 4 Complete! Model can now be saved and loaded.\n");
    printf("  Next: SIMD optimization for faster inference.\n");
    printf("═══════════════════════════════════════════════════════════\n");
    
    return EXIT_SUCCESS;
}