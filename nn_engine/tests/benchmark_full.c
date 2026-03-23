#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matrix.h"
#include "thread_pool.h"

#define NUM_ITERATIONS 10

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void naive_multiply(const Matrix* A, const Matrix* B, Matrix* C) {
    int m = A->rows;
    int k = A->cols;
    int n = B->cols;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                sum += A->data[i * A->stride + kk] * B->data[kk * B->stride + j];
            }
            C->data[i * C->stride + j] = sum;
        }
    }
}

static double run_naive(int rows, int cols) {
    Matrix A = matrix_create(rows, cols);
    Matrix B = matrix_create(cols, rows);
    Matrix C = matrix_create(rows, rows);
    
    for (int i = 0; i < rows * cols; i++) {
        A.data[i] = (float)(rand() % 100) / 10.0f;
        B.data[i] = (float)(rand() % 100) / 10.0f;
    }
    
    double start = get_time_ms();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        naive_multiply(&A, &B, &C);
    }
    double elapsed = get_time_ms() - start;
    
    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&C);
    
    return elapsed / NUM_ITERATIONS;
}

static double run_threaded(int rows, int cols, int num_threads) {
    Matrix A = matrix_create(rows, cols);
    Matrix B = matrix_create(cols, rows);
    Matrix C = matrix_create(rows, rows);
    
    for (int i = 0; i < rows * cols; i++) {
        A.data[i] = (float)(rand() % 100) / 10.0f;
        B.data[i] = (float)(rand() % 100) / 10.0f;
    }
    
    ThreadPool* pool = thread_pool_create(num_threads);
    
    double start = get_time_ms();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        matrix_multiply_threaded(&A, &B, &C, pool);
    }
    double elapsed = get_time_ms() - start;
    
    thread_pool_destroy(pool);
    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&C);
    
    return elapsed / NUM_ITERATIONS;
}

int main(void) {
    srand(42);
    
    int sizes[] = {128, 256, 512};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║     NEURAL NETWORK INFERENCE ENGINE - FULL BENCHMARK     ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Comparing: Naive C vs SIMD+Threaded (Multithreaded)     ║\n");
    printf("║  Iterations: %d                                          ║\n", NUM_ITERATIONS);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    printf("+----------+---------------+---------------+\n");
    printf("|   Size   |  Naive (ms)   | Threaded (ms) |\n");
    printf("+----------+---------------+---------------+\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int sz = sizes[i];
        double t_naive = run_naive(sz, sz);
        double t_threaded = run_threaded(sz, sz, 4);
        
        printf("| %6dx%2d |    %7.2f    |    %7.2f    |\n", sz, sz, t_naive, t_threaded);
        printf("|          |               | %7.2fx     |\n", t_naive / t_threaded);
        printf("+----------+---------------+---------------+\n");
    }
    
    printf("\nBENCHMARK COMPLETE\n");
    
    return 0;
}
