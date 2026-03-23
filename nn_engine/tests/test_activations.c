/**
 * @file test_activations.c
 * @brief Unit tests for activation functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "activations.h"

static int float_eq(float a, float b, float epsilon)
{
    return fabsf(a - b) < epsilon;
}

static void test_result(const char* name, int passed)
{
    printf("  %-40s [%s]\n", name, passed ? "✓ PASS" : "✗ FAIL");
}

/* ============================================================================
 * TEST 1: ReLU
 * ============================================================================
 */
void test_relu(void)
{
    printf("\n=== Test: ReLU Activation ===\n");
    
    Matrix input = matrix_create(1, 5);
    float in_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    memcpy(input.data, in_data, 5 * sizeof(float));
    
    Matrix output = matrix_create(1, 5);
    activation_relu(&input, &output);
    
    float expected[] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
    int passed = 1;
    for (int i = 0; i < 5; i++) {
        if (!float_eq(output.data[i], expected[i], 0.001f)) {
            printf("  Index %d: got %f, expected %f\n", i, output.data[i], expected[i]);
            passed = 0;
        }
    }
    test_result("ReLU: negatives become zero", passed);
    
    /* Test that positive values unchanged */
    passed = float_eq(output.data[4], 2.0f, 0.001f);
    test_result("ReLU: positives unchanged", passed);
    
    matrix_free(&input);
    matrix_free(&output);
}

/* ============================================================================
 * TEST 2: Leaky ReLU
 * ============================================================================
 */
void test_leaky_relu(void)
{
    printf("\n=== Test: Leaky ReLU Activation ===\n");
    
    Matrix input = matrix_create(1, 4);
    float in_data[] = {-2.0f, -1.0f, 1.0f, 2.0f};
    memcpy(input.data, in_data, 4 * sizeof(float));
    
    Matrix output = matrix_create(1, 4);
    float alpha = 0.1f;
    activation_leaky_relu(&input, &output, alpha);
    
    /* With alpha = 0.1: */
    /* -2.0 -> -0.2, -1.0 -> -0.1, 1.0 -> 1.0, 2.0 -> 2.0 */
    int passed = float_eq(output.data[0], -0.2f, 0.001f);
    passed = passed && float_eq(output.data[1], -0.1f, 0.001f);
    passed = passed && float_eq(output.data[2], 1.0f, 0.001f);
    passed = passed && float_eq(output.data[3], 2.0f, 0.001f);
    test_result("Leaky ReLU: small negative slope", passed);
    
    matrix_free(&input);
    matrix_free(&output);
}

/* ============================================================================
 * TEST 3: Sigmoid
 * ============================================================================
 */
void test_sigmoid(void)
{
    printf("\n=== Test: Sigmoid Activation ===\n");
    
    Matrix input = matrix_create(1, 5);
    float in_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    memcpy(input.data, in_data, 5 * sizeof(float));
    
    Matrix output = matrix_create(1, 5);
    activation_sigmoid(&input, &output);
    
    /* Sigmoid values (approximate): */
    /* sigmoid(-2) = 0.119, sigmoid(-1) = 0.269, sigmoid(0) = 0.5 */
    /* sigmoid(1) = 0.731, sigmoid(2) = 0.881 */
    int passed = float_eq(output.data[0], 0.119f, 0.01f);
    passed = passed && float_eq(output.data[2], 0.5f, 0.01f);
    passed = passed && float_eq(output.data[4], 0.881f, 0.01f);
    test_result("Sigmoid: squashes to (0,1)", passed);
    
    /* Verify all values in (0,1) */
    passed = 1;
    for (int i = 0; i < 5; i++) {
        if (output.data[i] <= 0.0f || output.data[i] >= 1.0f) {
            passed = 0;
        }
    }
    test_result("Sigmoid: all values in (0,1)", passed);
    
    matrix_free(&input);
    matrix_free(&output);
}

/* ============================================================================
 * TEST 4: Tanh
 * ============================================================================
 */
void test_tanh(void)
{
    printf("\n=== Test: Tanh Activation ===\n");
    
    Matrix input = matrix_create(1, 5);
    float in_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    memcpy(input.data, in_data, 5 * sizeof(float));
    
    Matrix output = matrix_create(1, 5);
    activation_tanh_act(&input, &output);
    
    /* tanh values: */
    /* tanh(-2) = -0.96, tanh(-1) = -0.76, tanh(0) = 0 */
    /* tanh(1) = 0.76, tanh(2) = 0.96 */
    int passed = float_eq(output.data[0], -0.96f, 0.01f);
    passed = passed && float_eq(output.data[2], 0.0f, 0.01f);
    passed = passed && float_eq(output.data[4], 0.96f, 0.01f);
    test_result("Tanh: squashes to (-1,1)", passed);
    
    /* Verify zero-centered (symmetric) */
    passed = float_eq(output.data[0], -output.data[4], 0.001f);
    passed = passed && float_eq(output.data[1], -output.data[3], 0.001f);
    test_result("Tanh: zero-centered", passed);
    
    matrix_free(&input);
    matrix_free(&output);
}

/* ============================================================================
 * TEST 5: Softmax
 * ============================================================================
 */
void test_softmax(void)
{
    printf("\n=== Test: Softmax Activation ===\n");
    
    Matrix input = matrix_create(1, 3);
    float in_data[] = {2.0f, 1.0f, 0.1f};
    memcpy(input.data, in_data, 3 * sizeof(float));
    
    Matrix output = matrix_create(1, 3);
    activation_softmax(&input, &output);
    
    /* Expected: [0.66, 0.24, 0.10] approximately */
    int passed = float_eq(output.data[0], 0.66f, 0.02f);
    passed = passed && float_eq(output.data[1], 0.24f, 0.02f);
    passed = passed && float_eq(output.data[2], 0.10f, 0.02f);
    test_result("Softmax: converts to probabilities", passed);
    
    /* Verify sum = 1 */
    float sum = output.data[0] + output.data[1] + output.data[2];
    passed = float_eq(sum, 1.0f, 0.001f);
    test_result("Softmax: probabilities sum to 1", passed);
    
    /* Test numerical stability with large values */
    Matrix input2 = matrix_create(1, 3);
    float large_data[] = {1000.0f, 1001.0f, 999.0f};
    memcpy(input2.data, large_data, 3 * sizeof(float));
    
    Matrix output2 = matrix_create(1, 3);
    activation_softmax(&input2, &output2);
    
    /* Without stability fix, this would overflow!
     * With fix: exp(0), exp(1), exp(-1) = 1, 2.72, 0.37
     * Sum = 4.09, results = [0.24, 0.67, 0.09] */
    passed = float_eq(output2.data[0], 0.24f, 0.02f);
    passed = passed && float_eq(output2.data[1], 0.67f, 0.02f);
    passed = passed && float_eq(output2.data[2], 0.09f, 0.02f);
    test_result("Softmax: numerical stability", passed);
    
    matrix_free(&input);
    matrix_free(&output);
    matrix_free(&input2);
    matrix_free(&output2);
}

/* ============================================================================
 * TEST 6: Function Pointer Demo
 * ============================================================================
 */
void test_function_pointer(void)
{
    printf("\n=== Test: Function Pointer Pattern ===\n");
    
    /* Demonstrate using activation functions via pointers */
    ActivationFn funcs[] = {activation_relu, activation_sigmoid, activation_tanh_act};
    const char* names[] = {"ReLU", "Sigmoid", "Tanh"};
    
    Matrix input = matrix_create(1, 3);
    input.data[0] = -1.0f;
    input.data[1] = 0.0f;
    input.data[2] = 1.0f;
    
    int passed = 1;
    for (int i = 0; i < 3; i++) {
        Matrix output = matrix_create(1, 3);
        funcs[i](&input, &output);
        
        /* Just verify it runs without crashing */
        /* Values checked in individual tests */
        passed = passed && (output.data[0] != 0.0f || i == 0); /* ReLU zero, others not */
        
        matrix_free(&output);
    }
    test_result("Function pointer array works", passed);
    
    matrix_free(&input);
}

/* ============================================================================
 * MAIN
 * ============================================================================
 */
int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║       Activation Functions Test Suite                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    test_relu();
    test_leaky_relu();
    test_sigmoid();
    test_tanh();
    test_softmax();
    test_function_pointer();
    
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("All activation tests completed!\n");
    printf("═══════════════════════════════════════════════════════════\n");
    
    return 0;
}