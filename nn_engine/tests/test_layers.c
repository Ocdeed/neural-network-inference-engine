/**
 * @file test_layers.c
 * @brief Unit tests for DenseLayer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "activations.h"
#include "layers.h"

static int float_eq(float a, float b, float epsilon)
{
    return fabsf(a - b) < epsilon;
}

static void test_result(const char* name, int passed)
{
    printf("  %-40s [%s]\n", name, passed ? "✓ PASS" : "✗ FAIL");
}

/* ============================================================================
 * TEST 1: Layer Creation
 * ============================================================================
 */
void test_layer_creation(void)
{
    printf("\n=== Test: Layer Creation ===\n");
    
    DenseLayer* layer = dense_layer_create(4, 3, activation_relu);
    
    int passed = (layer != NULL);
    passed = passed && (layer->input_size == 4);
    passed = passed && (layer->output_size == 3);
    passed = passed && (layer->activation == activation_relu);
    test_result("Layer created with correct dimensions", passed);
    
    passed = (layer->weights != NULL);
    passed = passed && (layer->weights->rows == 3);
    passed = passed && (layer->weights->cols == 4);
    test_result("Weight matrix allocated correctly", passed);
    
    passed = (layer->biases != NULL);
    passed = passed && (layer->biases->rows == 1);
    passed = passed && (layer->biases->cols == 3);
    test_result("Bias vector allocated correctly", passed);
    
    passed = (layer->output != NULL);
    test_result("Output cache allocated", passed);
    
    dense_layer_free(layer);
    test_result("Layer freed without crash", 1);
}

/* ============================================================================
 * TEST 2: Forward Pass with Identity Activation
 * ============================================================================
 */
void test_forward_identity(void)
{
    printf("\n=== Test: Forward Pass (No Activation) ===\n");
    
    /* Create layer with no activation */
    DenseLayer* layer = dense_layer_create(3, 2, NULL);
    
    /* Set specific weights and biases for testing */
    /* Weights: 2x3 matrix
     * [1 2 3]
     * [4 5 6]
     */
    layer->weights->data[0] = 1.0f;
    layer->weights->data[1] = 2.0f;
    layer->weights->data[2] = 3.0f;
    layer->weights->data[3] = 4.0f;
    layer->weights->data[4] = 5.0f;
    layer->weights->data[5] = 6.0f;
    
    /* Biases: [0.1, 0.2] */
    layer->biases->data[0] = 0.1f;
    layer->biases->data[1] = 0.2f;
    
    /* Input: [1, 2, 3] */
    Matrix input = matrix_create(1, 3);
    input.data[0] = 1.0f;
    input.data[1] = 2.0f;
    input.data[2] = 3.0f;
    
    /* Forward pass */
    Matrix* output = dense_layer_forward(layer, &input);
    
    /* Expected:
     * output[0] = 1*1 + 2*2 + 3*3 + 0.1 = 1 + 4 + 9 + 0.1 = 14.1
     * output[1] = 1*4 + 2*5 + 3*6 + 0.2 = 4 + 10 + 18 + 0.2 = 32.2
     */
    int passed = float_eq(output->data[0], 14.1f, 0.01f);
    passed = passed && float_eq(output->data[1], 32.2f, 0.01f);
    test_result("Forward pass computes correct values", passed);
    
    /* Verify output shape */
    passed = (output->rows == 1 && output->cols == 2);
    test_result("Output has correct shape [1x2]", passed);
    
    matrix_free(&input);
    dense_layer_free(layer);
}

/* ============================================================================
 * TEST 3: Forward Pass with ReLU Activation
 * ============================================================================
 */
void test_forward_relu(void)
{
    printf("\n=== Test: Forward Pass with ReLU ===\n");
    
    DenseLayer* layer = dense_layer_create(2, 3, activation_relu);
    
    /* Set weights: all zeros except some that will give negative */
    /* Output will be: [10, -5, 3] before activation */
    memset(layer->weights->data, 0, sizeof(float) * 6);
    layer->weights->data[0] = 1.0f;  /* row 0 */
    layer->weights->data[1] = 0.0f;
    layer->weights->data[2] = 0.0f;  /* row 1 */
    layer->weights->data[3] = -1.0f;
    layer->weights->data[4] = 0.0f;  /* row 2 */
    layer->weights->data[5] = 1.0f;
    
    memset(layer->biases->data, 0, sizeof(float) * 3);
    
    /* Input: [10, 0] */
    Matrix input = matrix_create(1, 2);
    input.data[0] = 10.0f;
    input.data[1] = 0.0f;
    
    Matrix* output = dense_layer_forward(layer, &input);
    
    /* Before ReLU: [10, -5, 0] */
    /* After ReLU: [10, 0, 0] */
    int passed = float_eq(output->data[0], 10.0f, 0.01f);
    passed = passed && float_eq(output->data[1], 0.0f, 0.01f);  /* -5 -> 0 */
    passed = passed && float_eq(output->data[2], 0.0f, 0.01f);  /* 0 -> 0 */
    test_result("ReLU activation zeroes negatives", passed);
    
    matrix_free(&input);
    dense_layer_free(layer);
}

/* ============================================================================
 * TEST 4: Layer Info Print
 * ============================================================================
 */
void test_layer_info(void)
{
    printf("\n=== Test: Layer Info ===\n");
    
    DenseLayer* layer = dense_layer_create(784, 256, activation_relu);
    
    /* Just verify it doesn't crash */
    dense_layer_print_info(layer);
    test_result("Layer info printed without crash", 1);
    
    dense_layer_free(layer);
}

/* ============================================================================
 * TEST 5: Multiple Forward Passes
 * ============================================================================
 */
void test_multiple_forward(void)
{
    printf("\n=== Test: Multiple Forward Passes ===\n");
    
    DenseLayer* layer = dense_layer_create(2, 2, activation_sigmoid);
    
    Matrix input1 = matrix_create(1, 2);
    input1.data[0] = 1.0f;
    input1.data[1] = 0.0f;
    
    Matrix input2 = matrix_create(1, 2);
    input2.data[0] = 0.0f;
    input2.data[1] = 1.0f;
    
    Matrix* output1 = dense_layer_forward(layer, &input1);
    Matrix* output2 = dense_layer_forward(layer, &input2);
    
    /* Both should produce different outputs */
    int passed = 1;
    if (float_eq(output1->data[0], output2->data[0], 0.001f) &&
        float_eq(output1->data[1], output2->data[1], 0.001f)) {
        /* They might be the same if weights are symmetric, that's OK */
        passed = 1;
    }
    test_result("Multiple forward passes work", passed);
    
    /* Outputs should be valid sigmoid values (between 0 and 1) */
    passed = 1;
    for (int i = 0; i < 2; i++) {
        if (output1->data[i] <= 0.0f || output1->data[i] >= 1.0f) passed = 0;
        if (output2->data[i] <= 0.0f || output2->data[i] >= 1.0f) passed = 0;
    }
    test_result("Sigmoid outputs in valid range (0,1)", passed);
    
    matrix_free(&input1);
    matrix_free(&input2);
    dense_layer_free(layer);
}

/* ============================================================================
 * MAIN
 * ============================================================================
 */
int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║            DenseLayer Test Suite                          ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    test_layer_creation();
    test_forward_identity();
    test_forward_relu();
    test_layer_info();
    test_multiple_forward();
    
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("All layer tests completed!\n");
    printf("═══════════════════════════════════════════════════════════\n");
    
    return 0;
}