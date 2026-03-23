/**
 * @file matrix.c
 * @brief Implementation of Matrix operations
 * 
 * TEACHING MOMENT: Memory Management in C
 * =========================================
 * 
 * In C, you have manual memory management - the compiler doesn't do it for you.
 * This is powerful but dangerous. Here's the deal:
 * 
 * THE HEAP (malloc/calloc):
 * - Stored in dynamic memory that persists until you free it
 * - Use for data that needs to outlive the function that creates it
 * - Slower to allocate but flexible in size
 * - Our Matrix data lives here
 * 
 * THE STACK:
 * - Fast allocation, automatic cleanup when function returns
 * - Use for local variables that don't need to persist
 * - Limited size (typically 1-8 MB)
 * - Example: int x = 5; lives on stack
 * 
 * ANALOGY: Think of the heap as a warehouse where you rent storage units.
 * You get a key (pointer) and must remember to return it (free)!
 * The stack is like your desk - fast to use, but limited space.
 * 
 * MEMORY LEAK EXAMPLE:
 * If you call malloc() but never free(), the memory stays allocated forever.
 * Run this program repeatedly and your computer will eventually run out of RAM!
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"
#include "thread_pool.h"
#include "simd_ops.h"

/**
 * Matrix multiplication: C = A * B
 * 
 * Uses SIMD dot product for each element when available.
 * Column-major access pattern for cache efficiency.
 */
void matrix_multiply(const Matrix* A, const Matrix* B, Matrix* C)
{
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "ERROR: NULL matrix in multiply\n");
        return;
    }
    
    if (A->cols != B->rows) {
        fprintf(stderr, "ERROR: Dimension mismatch: A(%dx%d) * B(%dx%d)\n",
                A->rows, A->cols, B->rows, B->cols);
        return;
    }
    
    int m = A->rows;
    int k = A->cols;
    int n = B->cols;
    
    /* Use SIMD-accelerated dot product for each output element */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            /* Get column j from B - this is the tricky part for cache */
            /* For each row i in A, compute dot product with column j in B */
            C->data[i * C->stride + j] = simd_dot_product(
                A->data + i * A->stride,
                B->data + j * B->stride,
                k
            );
        }
    }
}

/**
 * Work item for threaded multiplication
 */
typedef struct {
    const Matrix* A;
    const Matrix* B;
    Matrix* C;
    int row_start;
    int row_end;
} MultiplyTask;

/**
 * Worker function for a thread
 * Computes a slice of the output matrix using SIMD
 */
static void* multiply_worker(void* arg)
{
    MultiplyTask* task = (MultiplyTask*)arg;
    
    const Matrix* A = task->A;
    const Matrix* B = task->B;
    Matrix* C = task->C;
    int row_start = task->row_start;
    int row_end = task->row_end;
    
    int k = A->cols;
    int n = B->cols;
    
    /* Compute rows from row_start to row_end (exclusive) */
    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < n; j++) {
            /* Use SIMD dot product for better performance */
            C->data[i * C->stride + j] = simd_dot_product(
                A->data + i * A->stride,
                B->data + j * B->stride,
                k
            );
        }
    }
    
    return NULL;
}

/**
 * Threaded matrix multiplication using thread pool
 */
void matrix_multiply_threaded(const Matrix* A, const Matrix* B, Matrix* C, ThreadPool* pool)
{
    if (A == NULL || B == NULL || C == NULL || pool == NULL) {
        fprintf(stderr, "ERROR: NULL parameter in threaded multiply\n");
        return;
    }
    
    /* Validate dimensions */
    if (A->cols != B->rows) {
        fprintf(stderr, "ERROR: Dimension mismatch: A(%dx%d) * B(%dx%d)\n",
                A->rows, A->cols, B->rows, B->cols);
        return;
    }
    
    int num_threads = thread_pool_get_num_threads(pool);
    int rows_per_thread = A->rows / num_threads;
    int extra_rows = A->rows % num_threads;
    
    /* Zero the output first */
    matrix_zero(C);
    
    /* Create and submit tasks */
    int current_row = 0;
    for (int t = 0; t < num_threads; t++) {
        MultiplyTask* task = (MultiplyTask*)malloc(sizeof(MultiplyTask));
        task->A = A;
        task->B = B;
        task->C = C;
        task->row_start = current_row;
        current_row += rows_per_thread;
        if (t < extra_rows) {
            current_row++;  /* Distribute extra rows to first threads */
        }
        task->row_end = current_row;
        
        if (task->row_start < task->row_end) {
            thread_pool_submit(pool, multiply_worker, task);
        } else {
            free(task);  /* No work for this thread */
        }
    }
    
    /* Wait for all threads to complete */
    thread_pool_wait(pool);
}

/**
 * Create a new matrix with allocated memory
 */
Matrix matrix_create(int rows, int cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = (float*)calloc(rows * cols, sizeof(float));
    return m;
}

/**
 * Free all memory associated with a matrix
 */
void matrix_free(Matrix* m)
{
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
    }
    if (m) {
        m->rows = 0;
        m->cols = 0;
        m->stride = 0;
    }
}

/**
 * Print matrix to stdout
 */
void matrix_print(const Matrix* m, const char* name)
{
    if (m == NULL || m->data == NULL) {
        printf("Matrix %s: NULL\n", name ? name : "");
        return;
    }
    
    printf("Matrix %s (%d x %d):\n", name ? name : "", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("  [");
        for (int j = 0; j < m->cols; j++) {
            printf("%7.3f", m->data[i * m->stride + j]);
            if (j < m->cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

/**
 * Set all elements to zero
 */
void matrix_zero(Matrix* m)
{
    if (m && m->data) {
        memset(m->data, 0, m->rows * m->cols * sizeof(float));
    }
}

/**
 * Initialize with Xavier/He initialization
 */
void matrix_random(Matrix* m)
{
    if (m == NULL || m->data == NULL) return;
    
    float scale = sqrtf(2.0f / (m->cols));
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
    }
}

/**
 * Element-wise matrix addition: C = A + B
 */
void matrix_add(const Matrix* A, const Matrix* B, Matrix* C)
{
    if (A == NULL || B == NULL || C == NULL) return;
    
    for (int i = 0; i < A->rows * A->cols; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }
}

/**
 * Scalar multiplication: result = scalar * A
 */
void matrix_scalar_multiply(const Matrix* A, float scalar, Matrix* result)
{
    if (A == NULL || result == NULL) return;
    
    for (int i = 0; i < A->rows * A->cols; i++) {
        result->data[i] = scalar * A->data[i];
    }
}

/**
 * Matrix transpose: result = A^T
 */
void matrix_transpose(const Matrix* A, Matrix* result)
{
    if (A == NULL || result == NULL) return;
    
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            result->data[j * result->stride + i] = A->data[i * A->stride + j];
        }
    }
}

/**
 * Compute dot product of two vectors
 */
float matrix_dot_product(const float* a, const float* b, int n)
{
    if (a == NULL || b == NULL) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * Copy matrix src to dst
 */
void matrix_copy(const Matrix* src, Matrix* dst)
{
    if (src == NULL || dst == NULL || src->data == NULL || dst->data == NULL) return;
    
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(float));
}