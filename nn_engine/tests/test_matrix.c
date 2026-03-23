/**
 * @file test_matrix.c
 * @brief Unit tests for Matrix operations
 * 
 * A solid test suite verifies our code works correctly.
 * We'll test:
 * - Basic operations with known inputs
 * - Edge cases (single element, empty-looking, non-square)
 * - Properties (commutativity, associativity where applicable)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix.h"

/* Test helper: check if two floats are approximately equal */
static int float_eq(float a, float b, float epsilon)
{
    return fabsf(a - b) < epsilon;
}

/* Test helper: check if two matrices are approximately equal */
static int matrix_eq(const Matrix* A, const Matrix* B, float epsilon)
{
    if (A->rows != B->rows || A->cols != B->cols) {
        return 0;
    }
    
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            float a = A->data[i * A->stride + j];
            float b = B->data[i * B->stride + j];
            if (!float_eq(a, b, epsilon)) {
                printf("  Mismatch at [%d,%d]: got %f, expected %f\n", i, j, a, b);
                return 0;
            }
        }
    }
    return 1;
}

/* Print test result */
static void test_result(const char* name, int passed)
{
    printf("  %-40s [%s]\n", name, passed ? "✓ PASS" : "✗ FAIL");
}

/* ============================================================================
 * TEST 1: Matrix Multiply
 * ============================================================================
 */
void test_multiply(void)
{
    printf("\n=== Test: Matrix Multiplication ===\n");
    
    /* 
     * A = [1 2]     B = [5 6]
     *     [3 4]         [7 8]
     * 
     * Expected: C = A*B = [1*5+2*7  1*6+2*8] = [19 22]
     *                            [3*5+4*7  3*6+4*8]   [43 50]
     */
    Matrix A = matrix_create(2, 2);
    A.data[0] = 1.0f; A.data[1] = 2.0f;
    A.data[2] = 3.0f; A.data[3] = 4.0f;
    
    Matrix B = matrix_create(2, 2);
    B.data[0] = 5.0f; B.data[1] = 6.0f;
    B.data[2] = 7.0f; B.data[3] = 8.0f;
    
    Matrix C = matrix_create(2, 2);
    matrix_multiply(&A, &B, &C);
    
    float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    int passed = 1;
    for (int i = 0; i < 4; i++) {
        if (!float_eq(C.data[i], expected[i], 0.001f)) {
            passed = 0;
            printf("  Element %d: got %f, expected %f\n", i, C.data[i], expected[i]);
        }
    }
    
    test_result("2x2 * 2x2 = 2x2", passed);
    
    /* Test non-square: (2x3) * (3x2) = (2x2) */
    Matrix A2 = matrix_create(2, 3);
    float a2_data[] = {1, 2, 3, 4, 5, 6};
    memcpy(A2.data, a2_data, 6 * sizeof(float));
    
    Matrix B2 = matrix_create(3, 2);
    float b2_data[] = {7, 8, 9, 10, 11, 12};
    memcpy(B2.data, b2_data, 6 * sizeof(float));
    
    Matrix C2 = matrix_create(2, 2);
    matrix_multiply(&A2, &B2, &C2);
    
    /* 
     * C2[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
     * C2[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
     * C2[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
     * C2[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
     */
    float expected2[4] = {58.0f, 64.0f, 139.0f, 154.0f};
    passed = 1;
    for (int i = 0; i < 4; i++) {
        if (!float_eq(C2.data[i], expected2[i], 0.001f)) {
            passed = 0;
        }
    }
    test_result("(2x3) * (3x2)", passed);
    
    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&C);
    matrix_free(&A2);
    matrix_free(&B2);
    matrix_free(&C2);
}

/* ============================================================================
 * TEST 2: Matrix Add
 * ============================================================================
 */
void test_add(void)
{
    printf("\n=== Test: Matrix Addition ===\n");
    
    Matrix A = matrix_create(2, 2);
    A.data[0] = 1.0f; A.data[1] = 2.0f;
    A.data[2] = 3.0f; A.data[3] = 4.0f;
    
    Matrix B = matrix_create(2, 2);
    B.data[0] = 5.0f; B.data[1] = 6.0f;
    B.data[2] = 7.0f; B.data[3] = 8.0f;
    
    Matrix C = matrix_create(2, 2);
    matrix_add(&A, &B, &C);
    
    float expected[4] = {6.0f, 8.0f, 10.0f, 12.0f};
    int passed = 1;
    for (int i = 0; i < 4; i++) {
        if (!float_eq(C.data[i], expected[i], 0.001f)) {
            passed = 0;
        }
    }
    test_result("Element-wise addition", passed);
    
    /* Test that addition is commutative: A + B = B + A */
    Matrix D = matrix_create(2, 2);
    matrix_add(&B, &A, &D);
    
    passed = matrix_eq(&C, &D, 0.001f);
    test_result("Addition is commutative (A+B = B+A)", passed);
    
    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&C);
    matrix_free(&D);
}

/* ============================================================================
 * TEST 3: Scalar Multiply
 * ============================================================================
 */
void test_scalar_multiply(void)
{
    printf("\n=== Test: Scalar Multiplication ===\n");
    
    Matrix A = matrix_create(2, 2);
    A.data[0] = 1.0f; A.data[1] = 2.0f;
    A.data[2] = 3.0f; A.data[3] = 4.0f;
    
    Matrix result = matrix_create(2, 2);
    matrix_scalar_multiply(&A, 2.5f, &result);
    
    float expected[4] = {2.5f, 5.0f, 7.5f, 10.0f};
    int passed = 1;
    for (int i = 0; i < 4; i++) {
        if (!float_eq(result.data[i], expected[i], 0.001f)) {
            passed = 0;
        }
    }
    test_result("Scalar * matrix", passed);
    
    /* Test zero scalar */
    matrix_scalar_multiply(&A, 0.0f, &result);
    passed = 1;
    for (int i = 0; i < 4; i++) {
        if (!float_eq(result.data[i], 0.0f, 0.001f)) {
            passed = 0;
        }
    }
    test_result("Zero scalar gives zero matrix", passed);
    
    matrix_free(&A);
    matrix_free(&result);
}

/* ============================================================================
 * TEST 4: Transpose
 * ============================================================================
 */
void test_transpose(void)
{
    printf("\n=== Test: Transpose ===\n");
    
    Matrix A = matrix_create(2, 3);
    float a_data[] = {1, 2, 3, 4, 5, 6};
    memcpy(A.data, a_data, 6 * sizeof(float));
    
    /* 
     * A = [1 2 3]
     *     [4 5 6]
     * 
     * A^T = [1 4]
     *        [2 5]
     *        [3 6]
     */
    Matrix At = matrix_create(3, 2);
    matrix_transpose(&A, &At);
    
    int passed = 1;
    if (At.rows != 3 || At.cols != 2) passed = 0;
    if (passed && !float_eq(At.data[0], 1.0f, 0.001f)) passed = 0;
    if (passed && !float_eq(At.data[1], 4.0f, 0.001f)) passed = 0;
    if (passed && !float_eq(At.data[2], 2.0f, 0.001f)) passed = 0;
    if (passed && !float_eq(At.data[3], 5.0f, 0.001f)) passed = 0;
    if (passed && !float_eq(At.data[4], 3.0f, 0.001f)) passed = 0;
    if (passed && !float_eq(At.data[5], 6.0f, 0.001f)) passed = 0;
    
    test_result("2x3 transpose -> 3x2", passed);
    
    /* Test that (A^T)^T = A */
    Matrix At2 = matrix_create(2, 3);
    matrix_transpose(&At, &At2);
    
    passed = matrix_eq(&A, &At2, 0.001f);
    test_result("(A^T)^T = A", passed);
    
    /* Test square matrix */
    Matrix S = matrix_create(3, 3);
    float s_data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  /* Identity */
    memcpy(S.data, s_data, 9 * sizeof(float));
    
    Matrix St = matrix_create(3, 3);
    matrix_transpose(&S, &St);
    
    passed = matrix_eq(&S, &St, 0.001f);
    test_result("Identity transpose = itself", passed);
    
    matrix_free(&A);
    matrix_free(&At);
    matrix_free(&At2);
    matrix_free(&S);
    matrix_free(&St);
}

/* ============================================================================
 * TEST 5: Dot Product
 * ============================================================================
 */
void test_dot_product(void)
{
    printf("\n=== Test: Dot Product ===\n");
    
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    
    float result = matrix_dot_product(a, b, 3);
    /* 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32 */
    int passed = float_eq(result, 32.0f, 0.001f);
    test_result("dot([1,2,3], [4,5,6]) = 32", passed);
    
    /* Test with zeros */
    float zeros[] = {0.0f, 0.0f, 0.0f};
    result = matrix_dot_product(a, zeros, 3);
    passed = float_eq(result, 0.0f, 0.001f);
    test_result("dot(any, zeros) = 0", passed);
    
    /* Test unit vectors */
    float u[] = {1.0f, 0.0f, 0.0f};
    float v[] = {0.0f, 1.0f, 0.0f};
    result = matrix_dot_product(u, v, 3);
    passed = float_eq(result, 0.0f, 0.001f);
    test_result("dot(orthogonal) = 0", passed);
    
    /* Test with itself = squared magnitude */
    result = matrix_dot_product(a, a, 3);
    /* 1 + 4 + 9 = 14 */
    passed = float_eq(result, 14.0f, 0.001f);
    test_result("dot(a, a) = ||a||^2", passed);
}

/* ============================================================================
 * TEST 6: Copy
 * ============================================================================
 */
void test_copy(void)
{
    printf("\n=== Test: Copy ===\n");
    
    Matrix src = matrix_create(3, 4);
    for (int i = 0; i < 12; i++) {
        src.data[i] = (float)(i + 1);
    }
    
    Matrix dst = matrix_create(3, 4);
    matrix_copy(&src, &dst);
    
    int passed = matrix_eq(&src, &dst, 0.001f);
    test_result("Copy preserves all values", passed);
    
    /* Modify dst, verify src unchanged */
    dst.data[0] = 999.0f;
    passed = (src.data[0] == 1.0f);
    test_result("Copy creates independent copy", passed);
    
    matrix_free(&src);
    matrix_free(&dst);
}

/* ============================================================================
 * TEST 7: Edge Cases
 * ============================================================================
 */
void test_edge_cases(void)
{
    printf("\n=== Test: Edge Cases ===\n");
    
    /* Single element matrix */
    Matrix single = matrix_create(1, 1);
    single.data[0] = 42.0f;
    
    Matrix result = matrix_create(1, 1);
    matrix_scalar_multiply(&single, 2.0f, &result);
    
    int passed = float_eq(result.data[0], 84.0f, 0.001f);
    test_result("Single element multiply", passed);
    
    /* Non-square rectangular */
    Matrix rect = matrix_create(1, 5);
    float rect_data[] = {1, 2, 3, 4, 5};
    memcpy(rect.data, rect_data, 5 * sizeof(float));
    
    Matrix rect_t = matrix_create(5, 1);
    matrix_transpose(&rect, &rect_t);
    
    passed = (rect_t.rows == 5 && rect_t.cols == 1);
    test_result("1x5 transpose -> 5x1", passed);
    
    /* Verify values after transpose */
    passed = 1;
    for (int i = 0; i < 5; i++) {
        if (!float_eq(rect_t.data[i], rect_data[i], 0.001f)) {
            passed = 0;
        }
    }
    test_result("1x5 values preserved in transpose", passed);
    
    matrix_free(&single);
    matrix_free(&result);
    matrix_free(&rect);
    matrix_free(&rect_t);
}

/* ============================================================================
 * MAIN
 * ============================================================================
 */
int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║         Matrix Operations Test Suite                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    test_multiply();
    test_add();
    test_scalar_multiply();
    test_transpose();
    test_dot_product();
    test_copy();
    test_edge_cases();
    
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("All tests completed!\n");
    printf("═══════════════════════════════════════════════════════════\n");
    
    return 0;
}