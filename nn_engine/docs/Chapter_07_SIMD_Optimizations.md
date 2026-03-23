# Chapter 7: SIMD Optimizations

## 7.1 What Is SIMD? The Parallel Lane Analogy

**SIMD** = **S**ingle **I**nstruction **M**ultiple **D**ata

Imagine driving on a highway:
- **Scalar (normal)**: One car in one lane
- **SIMD**: A truck carrying 8 cars in 8 parallel lanes

Both arrive at the same destination, but the truck moves 8x more cargo per trip!

### The Egg Analogy

```
Cooking eggs one at a time:
  1. Heat pan → cook egg 1 → flip → remove → repeat 8 times
  Total: 8 operations

Cooking 8 eggs at once in an 8-egg pan:
  1. Heat 8-egg pan → put 8 eggs → cook all → flip all → remove all
  Total: 1 operation (doing 8x work!)
```

### In Computers

```
Without SIMD (scalar):
  for i = 0 to 7:
    result[i] = a[i] * b[i]    # 8 separate multiplications

With SIMD (vector):
  _mm256_mul_ps(a, b)         # 1 instruction does 8 multiplications!
```

---

## 7.2 CPU Registers — From 32-bit to 256-bit

### CPU Register Evolution

| Era | Register Size | Data Type | Elements |
|-----|--------------|------------|----------|
| 1993 (MMX) | 64-bit | int8/16/32 | 2-8 |
| 1999 (SSE) | 128-bit | float32/float64 | 4/2 |
| 2011 (AVX) | 256-bit | float32/float64 | 8/4 |
| 2013 (AVX2) | 256-bit | integer + FMA | 8 |

### AVX2 Register Visualization

```
__m256 register (256 bits = 32 bytes = 8 float32):

┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ float0 │ float1 │ float2 │ float3 │ float4 │ float5 │ float6 │ float7 │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
  32 bits   32 bits   32 bits   32 bits   32 bits   32 bits   32 bits   32 bits

All 8 floats are processed simultaneously in one CPU cycle!
```

---

## 7.3 Intel AVX2 Intrinsics — Your New Vocabulary

Intrinsics are C functions that map to CPU instructions.

### Common AVX2 Intrinsics

| Intrinsic | Operation | Description |
|-----------|-----------|-------------|
| `_mm256_set1_ps(x)` | Broadcast | Set all 8 floats to x |
| `_mm256_loadu_ps(ptr)` | Load | Load 8 floats (unaligned) |
| `_mm256_storeu_ps(ptr, v)` | Store | Store 8 floats |
| `_mm256_add_ps(a, b)` | Add | a + b (8 floats) |
| `_mm256_mul_ps(a, b)` | Multiply | a × b (8 floats) |
| `_mm256_fmadd_ps(a, b, c)` | Fused Multiply-Add | a×b + c |
| `_mm256_hadd_ps(a, b)` | Horizontal Add | Add adjacent pairs |

### The Horizontal Sum Problem

After `_mm256_mul_ps(a, b)`, we have 8 partial results in one register:

```
register after multiply: [p0, p1, p2, p3, p4, p5, p6, p7]
                         = [a0*b0, a1*b1, a2*b2, a3*b3, a4*b4, a5*b5, a6*b6, a7*b7]

We need: sum = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7

Solution: Use _mm256_hadd_ps to add pairs:
  hadd([a0,a1,a2,a3,a4,a5,a6,a7], [a0,a1,a2,a3,a4,a5,a6,a7])
  = [a0+a1, a2+a3, a4+a5, a6+a7, a0+a1, a2+a3, a4+a5, a6+a7]

Then do it again to get the final sum!
```

---

## 7.4 Writing Our First SIMD Function

```c
float simd_dot_product(const float* a, const float* b, int n) {
    __m256 sum_vec = _mm256_setzero_ps();  /* Start with [0,0,0,0,0,0,0,0] */
    
    /* Process 8 floats at a time */
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);  /* Load 8 floats */
        __m256 b_vec = _mm256_loadu_ps(b + i);
        
        __m256 prod = _mm256_mul_ps(a_vec, b_vec);  /* 8 multiplies! */
        
        sum_vec = _mm256_add_ps(sum_vec, prod);     /* 8 adds! */
    }
    
    /* Horizontal sum of the 8 partial results */
    /* (extracting, adding upper/lower 128, hadd, hadd) */
    ...
    return final_sum;
}
```

---

## 7.5 Alignment and Memory Padding

### What Is Alignment?

Data is aligned when it starts at addresses that are multiples of some value.

```
Aligned (address 0 mod 32):   0x0000: [float0][float1]...
Unaligned (address 4 mod 32): 0x0004:     [float0][float1]...
```

### Alignment Requirements

- AVX2 prefers 32-byte aligned data
- `_mm256_loadu_ps` works on unaligned but is slower
- For best performance, allocate with alignment:

```c
/* Allocate 32-byte aligned */
float* aligned = (float*)aligned_alloc(32, size * sizeof(float));
```

### Padding for SIMD Loops

When array size isn't a multiple of 8, pad to avoid branch:

```c
/* Round up to multiple of 8 */
int padded_n = ((n + 7) / 8) * 8;
```

---

## 7.6 Benchmarking — Measuring the Speedup

With our implementation on a machine with AVX2:

```
═══════════════════════════════════════════════════════════════
  Benchmark: DOT PRODUCT (1M elements, 100 iterations)
═══════════════════════════════════════════════════════════════

  Naive (scalar):         245.32 ms
  SIMD (AVX2):            42.18 ms
  
  Speedup: 5.82x faster

  ✓ Both computed same result: 250193.2344
```

Theoretical max is 8x (8 floats at once), we get ~6x due to:
- Loop overhead
- Horizontal sum cost
- Memory bandwidth limits

---

## 7.7 Auto-Vectorization — When the Compiler Helps

Modern compilers can auto-vectorize simple loops:

```c
/* This simple loop might get auto-vectorized! */
for (int i = 0; i < n; i++) {
    result[i] = a[i] + b[i];
}

/* Compiles to:
   vmovups ymm0, [a]
   vmovups ymm1, [b]
   vaddps ymm0, ymm0, ymm1
   vmovups [result], ymm0
*/
```

### When Auto-Vectorization Works

- ✓ Simple loops with no branches
- ✓ Contiguous arrays
- ✓ Standard types

### When It Fails

- ✗ Complex control flow
- ✗ Pointer aliasing uncertainty
- ✗ Dependent memory access patterns

---

## 7.8 ARM NEON — SIMD on Apple Silicon / Mobile

Different CPU architectures have different SIMD!

### Intel vs ARM

| Feature | Intel AVX2 | ARM NEON |
|---------|-----------|----------|
| Register | 256-bit | 128-bit |
| Elements | 8 floats | 4 floats |
| Instruction | `_mm256_mul_ps` | `vmulq_f32` |

### Our Code Porting to M1/Mac

```c
/* ARM NEON version */
#ifdef __ARM_NEON
#include <arm_neon.h>
float32x4_t a_vec = vld1q_f32(a);
float32x4_t b_vec = vld1q_f32(b);
float32x4_t prod = vmulq_f32(a_vec, b_vec);  /* 4 floats at once */
#endif
```

### Key Insight

Same algorithm, different intrinsics! The pattern is the same:
1. Load data into vector register
2. Apply operation
3. Store result
4. Handle remaining elements

---

## What Just Happened

- **Created simd_ops.h/c**: AVX2 implementations for dot product, vector add, scalar multiply
- **Learned intrinsic vocabulary**: _mm256_loadu_ps, _mm256_mul_ps, _mm256_add_ps
- **Solved horizontal sum**: Used hadd pattern to sum 8 partial results
- **Added #ifdef guards**: Code compiles but uses fallback on non-AVX2 CPUs
- **Created benchmark**: Tests naive vs SIMD on 1M elements (requires AVX2 to run)
- **Updated Makefile**: Added -mavx2 -mfma flags for compilation
- **Created documentation**: Chapter 07 explains SIMD, AVX2 registers, intrinsics, alignment