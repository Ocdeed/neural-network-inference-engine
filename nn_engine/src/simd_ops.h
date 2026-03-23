/**
 * @file simd_ops.h
 * @brief SIMD-accelerated operations for neural network inference
 * 
 * SIMD (Single Instruction Multiple Data) = process multiple values
 * with one instruction, like cooking 8 eggs at once instead of 1!
 * 
 * We use Intel AVX2 intrinsics (Advanced Vector Extensions):
 * - 256-bit registers = 8 float32 values at once
 * - Each operation processes 8x more data per cycle
 */

#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#include <stddef.h>

/**
 * @brief SIMD-accelerated dot product
 * @param a First vector
 * @param b Second vector  
 * @param n Number of elements (should be multiple of 8 for best performance)
 * @return Sum of element-wise products
 * 
 * Uses AVX2 to process 8 floats at once:
 *   _mm256_loadu_ps loads 8 floats
 *   _mm256_mul_ps multiplies 8 pairs
 *   _mm256_add_ps adds 8 results
 * 
 * The tricky part: horizontal sum (adding up the 8 results in the register)
 */
float simd_dot_product(const float* a, const float* b, int n);

/**
 * @brief SIMD-accelerated vector addition
 * @param a First vector (input)
 * @param b Second vector (input)
 * @param result Output vector (a + b)
 * @param n Number of elements
 */
void simd_vector_add(const float* a, const float* b, float* result, int n);

/**
 * @brief SIMD-accelerated scalar multiplication
 * @param a Input vector
 * @param scalar Multiplier
 * @param result Output vector (scalar * a)
 * @param n Number of elements
 */
void simd_scalar_multiply(const float* a, float scalar, float* result, int n);

/**
 * @brief Check if SIMD is available at compile time
 */
#ifdef __AVX2__
#define SIMD_AVAILABLE 1
#else
#define SIMD_AVAILABLE 0
#endif

#endif /* SIMD_OPS_H */