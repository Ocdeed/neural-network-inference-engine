/**
 * @file matrix.h
 * @brief Matrix data structure for neural network inference
 * 
 * This is our fundamental data structure. Everything in a neural network
 * is ultimately a matrix - weights, biases, activations, gradients.
 * 
 * Why flat array (float*) instead of float[][]?
 * - Cache locality: contiguous memory is MUCH faster
 * - Flexibility: stride lets us represent sub-matrices without copying
 * - Simplicity: one pointer to manage, not an array of pointers
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>  /* for size_t */

/**
 * @brief Matrix structure using flat array with stride
 * 
 * The 'stride' field is crucial for our memory layout:
 * - Standard matrix: stride == cols (contiguous rows)
 * - Sub-matrix view: stride points to parent matrix's column
 * - Transposed view: stride == 1 (column-major)
 * 
 * This allows us to create "views" into existing data without copying.
 */
typedef struct {
    float* data;   /**< Pointer to heap-allocated array of floats */
    int rows;      /**< Number of rows in this view */
    int cols;      /**< Number of columns in this view */
    int stride;    /**< Distance between consecutive elements in memory */
} Matrix;

/**
 * @brief Create a new matrix with allocated memory
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Newly allocated Matrix (zero-initialized)
 * 
 * Allocation happens on the HEAP (via malloc).
 * Don't forget to call matrix_free() when done!
 */
Matrix matrix_create(int rows, int cols);

/**
 * @brief Free all memory associated with a matrix
 * @param m Pointer to Matrix to free
 * 
 * Frees both the data pointer AND the matrix struct itself.
 * After calling, the matrix is invalid - don't use it.
 * 
 * Common beginner mistake: forgetting to free = MEMORY LEAK!
 */
void matrix_free(Matrix* m);

/**
 * @brief Print matrix to stdout with nice formatting
 * @param m Matrix to print
 * @param name Optional name for display
 */
void matrix_print(const Matrix* m, const char* name);

/**
 * @brief Set all elements to zero
 * @param m Matrix to zero out
 */
void matrix_zero(Matrix* m);

/**
 * @brief Initialize with Xavier/He initialization
 * @param m Matrix to fill with random values
 * 
 * Xavier initialization: random values scaled by sqrt(2.0 / (fan_in + fan_out))
 * This prevents vanishing/exploding gradients by keeping variance stable.
 * 
 * For a layer with n inputs, we scale by sqrt(1.0/n) (Xavier) or sqrt(2.0/n) (He)
 */
void matrix_random(Matrix* m);

/**
 * @brief Matrix multiplication: C = A * B
 * @param A First matrix (m x k)
 * @param B Second matrix (k x n)
 * @param C Result matrix (m x n) - must be pre-allocated
 * 
 * This is the WORKHORSE of neural networks!
 * Each element C[i,j] = sum over k of A[i,k] * B[k,j]
 * 
 * Complexity: O(m * k * n) - three nested loops!
 * 
 * IMPORTANT: A's columns must match B's rows (k dimension)
 */
void matrix_multiply(const Matrix* A, const Matrix* B, Matrix* C);

/**
 * @brief Element-wise matrix addition: C = A + B
 * @param A First matrix
 * @param B Second matrix (same dimensions as A)
 * @param C Result matrix - must be pre-allocated
 * 
 * Used in: Adding biases to weighted sums
 */
void matrix_add(const Matrix* A, const Matrix* B, Matrix* C);

/**
 * @brief Scalar multiplication: result = scalar * A
 * @param A Input matrix
 * @param scalar Float multiplier
 * @param result Output matrix - must be pre-allocated
 * 
 * Used in: Gradient descent updates, weight decay
 */
void matrix_scalar_multiply(const Matrix* A, float scalar, Matrix* result);

/**
 * @brief Matrix transpose: result = A^T
 * @param A Input matrix (m x n)
 * @param result Output matrix (n x m) - must be pre-allocated
 * 
 * Transpose flips rows and columns: A[i,j] → result[j,i]
 * 
 * Used in: Backpropagation, certain weight layouts
 */
void matrix_transpose(const Matrix* A, Matrix* result);

/**
 * @brief Compute dot product of two vectors
 * @param a First vector (length n)
 * @param b Second vector (length n)
 * @param n Length of vectors
 * @return Sum of element-wise products: Σ a[i] * b[i]
 * 
 * This is THE fundamental operation of neural networks!
 * Every matrix multiplication is just a bunch of dot products.
 * 
 * We'll optimize this with SIMD instructions later.
 */
float matrix_dot_product(const float* a, const float* b, int n);

/**
 * @brief Copy matrix src to dst
 * @param src Source matrix
 * @param dst Destination matrix (must have same dimensions)
 * 
 * Used in: Backpropagation (need to save activations)
 */
void matrix_copy(const Matrix* src, Matrix* dst);

/* Forward declaration - actual definition in thread_pool.h */
typedef struct ThreadPool ThreadPool;

/**
 * @brief Threaded matrix multiplication using thread pool
 * @param A First matrix (m x k)
 * @param B Second matrix (k x n)
 * @param C Result matrix (m x n) - must be pre-allocated
 * @param pool Thread pool to use for parallelization
 * 
 * Parallelizes by splitting rows of A across threads.
 * Each thread computes its slice of the output matrix.
 * 
 * For example, with 4 threads on a 512x512 matrix:
 *   Thread 0: rows 0-127
 *   Thread 1: rows 128-255
 *   Thread 2: rows 256-383
 *   Thread 3: rows 384-511
 */
void matrix_multiply_threaded(const Matrix* A, const Matrix* B, Matrix* C, ThreadPool* pool);

#endif /* MATRIX_H */