/**
 * @file simd_ops.c
 * @brief SIMD-accelerated implementations
 * 
 * SIMD EXPLAINED:
 * ==============
 * 
 * Imagine cooking eggs in a pan:
 * - WITHOUT SIMD: Cook 1 egg at a time, flip it, take it out, repeat
 * - WITH SIMD: Put 8 eggs in an 8-egg pan, cook all at once!
 * 
 * SIMD does the same for numbers:
 * - Without: a[0]*b[0] + a[1]*b[1] + ... (one at a time)
 * - With: _mm256_mul_ps processes 8 multiplications in ONE instruction
 * 
 * ╔═══════════════════════════════════════════════════════════╗
 * ║           AVX2 REGISTERS: 256 BITS = 8 FLOATS              ║
 * ╠═══════════════════════════════════════════════════════════╣
 * ║                                                               ║
 * ║  __m256 register:                                           ║
 * ║  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐  ║
 * ║  │ f[0] │ f[1] │ f[2] │ f[3] │ f[4] │ f[5] │ f[6] │ f[7] │  ║
 * ║  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘  ║
 * ║                                                               ║
 * ║  _mm256_mul_ps(a, b):                                        ║
 * ║  ┌─────────┐    ┌─────────┐    ┌─────────┐                  ║
 * ║  │ a0 * b0 │    │ a1 * b1 │    │ a2 * b2 │  ...              ║
 * ║  └─────────┘    └─────────┘    └─────────┘                  ║
 * ║                                                               ║
 * ╚═══════════════════════════════════════════════════════════╝
 * 
 * HORIZONTAL SUM PROBLEM:
 * After _mm256_mul_ps, we have 8 partial sums in one register.
 * We need to ADD them all together to get one result!
 * 
 * Solution: Use _mm256_hadd_ps to add adjacent pairs:
 *   [a0,a1,a2,a3,a4,a5,a6,a7] 
 *   → [a0+a1, a2+a3, a4+a5, a6+a7, a0+a1, a2+a3, a4+a5, a6+a7]
 *   Then do it again with different shuffles to get the final sum.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "simd_ops.h"

/**
 * Runtime CPU detection for AVX2
 * Returns 1 if AVX2 is available, 0 otherwise
 */
static int have_avx2(void)
{
    #ifdef __AVX2__
    /* Check using CPUID - check if OS supports AVX2 */
    int got_ebx, got_ecx, got_edx;
    /* Check CPUID function 1 (ECX bit 5 = AVX) */
    __asm__ __volatile__(
        "cpuid" : "=b"(got_ebx), "=c"(got_ecx), "=d"(got_edx) : "a"(1), "c"(0)
    );
    if (!(got_ecx & (1 << 5))) return 0;
    
    /* Check CPUID function 7 (EBX bit 5 = AVX2) */
    __asm__ __volatile__(
        "cpuid" : "=b"(got_ebx), "=c"(got_ecx), "=d"(got_edx) : "a"(7), "c"(0)
    );
    return (got_ebx & (1 << 5)) ? 1 : 0;
    #else
    return 0;
    #endif
}

/* Global flag set at startup */
static int simd_enabled = 0;

/**
 * Naive (non-SIMD) fallback implementations
 * These work on any CPU, but are slower
 */
static float naive_dot_product(const float* a, const float* b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void naive_vector_add(const float* a, const float* b, float* result, int n)
{
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

static void naive_scalar_multiply(const float* a, float scalar, float* result, int n)
{
    for (int i = 0; i < n; i++) {
        result[i] = a[i] * scalar;
    }
}

/**
 * AVX2 implementation - uses 256-bit registers for 8x speedup
 */
#ifdef __AVX2__
#include <immintrin.h>

/**
 * SIMD Dot Product using AVX2
 * 
 * Process 8 floats at a time, then sum all results together.
 */
float simd_dot_product(const float* a, const float* b, int n)
{
    if (n <= 0) return 0.0f;
    
    __m256 sum_vec = _mm256_setzero_ps();  /* Start with 0s */
    
    int i = 0;
    
    /* Process 8 floats at a time */
    for (; i + 8 <= n; i += 8) {
        /* Load 8 floats from a and b (unaligned, works with any address) */
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        
        /* Multiply all 8 pairs at once! */
        __m256 prod_vec = _mm256_mul_ps(a_vec, b_vec);
        
        /* Add to running sum */
        sum_vec = _mm256_add_ps(sum_vec, prod_vec);
    }
    
    /* Handle remaining elements (less than 8) */
    for (; i < n; i++) {
        sum_vec = _mm256_add_ps(sum_vec, _mm256_set1_ps(a[i] * b[i]));
    }
    
    /*
     * HORIZONTAL SUM: We have 8 partial sums in sum_vec
     * Need to add them all together!
     * 
     * Technique: Add upper 128 bits to lower 128 bits, then hadd within 128-bit
     */
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);  /* Upper 128 bits */
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);     /* Lower 128 bits */
    __m128 sum = _mm_add_ps(sum_low, sum_high);           /* Combine */
    
    /* Final horizontal add within 128 bits */
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    
    return _mm_cvtss_f32(sum);  /* Extract first float */
}

/**
 * SIMD Vector Add
 */
void simd_vector_add(const float* a, const float* b, float* result, int n)
{
    int i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 r_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(result + i, r_vec);
    }
    
    /* Handle remaining */
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

/**
 * SIMD Scalar Multiply
 */
void simd_scalar_multiply(const float* a, float scalar, float* result, int n)
{
    __m256 scalar_vec = _mm256_set1_ps(scalar);  /* Broadcast scalar to all 8 */
    
    int i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 r_vec = _mm256_mul_ps(a_vec, scalar_vec);
        _mm256_storeu_ps(result + i, r_vec);
    }
    
    /* Handle remaining */
    for (; i < n; i++) {
        result[i] = a[i] * scalar;
    }
}

#else /* No AVX2 available - use naive fallback */

float simd_dot_product(const float* a, const float* b, int n)
{
    return naive_dot_product(a, b, n);
}

void simd_vector_add(const float* a, const float* b, float* result, int n)
{
    naive_vector_add(a, b, result, n);
}

void simd_scalar_multiply(const float* a, float scalar, float* result, int n)
{
    naive_scalar_multiply(a, scalar, result, n);
}

#endif /* __AVX2__ */