#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "matrix.h"
#include "thread_pool.h"

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void benchmark_thread_count(Matrix *A, Matrix *B, Matrix *C, int num_threads) {
    ThreadPool *pool = thread_pool_create(num_threads);
    
    double start = get_time_ms();
    matrix_multiply_threaded(A, B, C, pool);
    double elapsed = get_time_ms() - start;
    
    printf("  %d thread(s): %8.2f ms\n", num_threads, elapsed);
    
    thread_pool_destroy(pool);
}

int main(void) {
    const int sizes[] = {128, 256, 512, 1024};
    const int thread_counts[] = {1, 2, 4, 8};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_thread_opts = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    printf("=== Threaded Matrix Multiply Benchmark ===\n\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        printf("Matrix size: %d x %d\n", size, size);
        
        Matrix A = matrix_create(size, size);
        Matrix B = matrix_create(size, size);
        Matrix C = matrix_create(size, size);
        
        for (int i = 0; i < size * size; i++) {
            A.data[i] = (float)(rand() % 100) / 10.0f;
            B.data[i] = (float)(rand() % 100) / 10.0f;
        }
        
        for (int t = 0; t < num_thread_opts; t++) {
            int num_threads = thread_counts[t];
            memset(C.data, 0, C.rows * C.cols * sizeof(float));
            benchmark_thread_count(&A, &B, &C, num_threads);
        }
        
        printf("\n");
        
        matrix_free(&A);
        matrix_free(&B);
        matrix_free(&C);
    }
    
    printf("=== Benchmark Complete ===\n");
    
    return 0;
}
