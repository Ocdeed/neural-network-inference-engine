# Chapter 2: Matrix Math in C

## 2.1 Why Matrix Math Powers AI

Matrix math is the **backbone** of neural networks. Every computation in a neural network - forward pass, backpropagation, gradient updates - boils down to matrix operations.

### The Core Insight

A neural network layer does:
```
output = weights × input + bias
```

This is matrix multiplication! The entire forward pass is:
```
Layer1: H1 = σ(W1 × X + b1)      (matrix mul → add → activation)
Layer2: H2 = σ(W2 × H1 + b2)    (matrix mul → add → activation)
Layer3: Y  = softmax(W3 × H2 + b3)  (matrix mul → add → activation)
```

Everything is matrices. That's why optimizing matrix operations directly improves neural network performance by 10x-100x!

---

## 2.2 Matrix Multiplication — The Core Operation

### Definition

Given A (m×k) and B (k×n), produce C (m×n) where:
```
C[i,j] = Σ(k=0 to k-1) A[i,k] * B[k,j]
```

Each element is the **dot product** of row i of A and column j of B.

### Visual Representation

```
A (m×k)           B (k×n)                 C (m×n)
┌───────┐        ┌───────┐              ┌───────┐
│ a_00  │        │ b_00  │              │ c_00  │ = a_00*b_00 + a_01*b_10 + ...
│ a_01  │   ×    │ b_10  │   =          │ c_01  │   ↑
│  ...  │        │  ...  │              │  ...  │   │
│ a_m0  │        │ b_k0  │              │ c_mn  │   └─ dot product of row m, col n
└───────┘        └───────┘              └───────┘
```

### Time Complexity: O(n³)

```c
void matrix_multiply(Matrix* A, Matrix* B, Matrix* C) {
    for (int i = 0; i < m; i++)          // For each output row
        for (int j = 0; j < n; j++)      // For each output column
            for (int k = 0; k < k; k++)  // For each element in dot product
                C[i,j] += A[i,k] * B[k,j];
}
```

Three nested loops = **O(m × k × n)**, hence O(n³) for square matrices.

### Why Matrix Multiplication Doesn't Commute

A × B ≠ B × A in general!

- A is (2×3), B is (3×2) → A×B is (2×2), B×A is (3×3) - different dimensions!
- Even when dimensions match: 
  ```
      [1 2]   [0 1]   [4 4]
      [3 4] × [1 0] = [4 3]  ≠
      
      [0 1]   [1 2]   [2 3]
      [1 0] × [3 4] = [1 2]  (different result!)
      ```

---

## 2.3 Cache Locality and Loop Order

### The Problem: Memory is SLOW

Fetching from RAM takes ~100ns. CPU can do ~10 ops in that time! To keep the CPU busy, we need **locality** - keep data in cache.

### Row-Major Memory Layout

Our flat array stores rows contiguously:
```
Matrix A (2×3): [a00, a01, a02, a10, a11, a12]
                  ↑ row 0      ↑ row 1
```

Accessing `A[i][j]` → `A.data[i * stride + j]` is fast when j changes slowly!

### Loop Order Matters!

**GOOD (i,k,j)**: Access A by row (contiguous), B reused in inner loop
```c
for i in rows:
    for k in shared_dim:
        aik = A[i,k]               // Contiguous access
        for j in cols:
            C[i,j] += aik * B[k,j]  // B[k,j] moves but stays in cache
```

**BAD (i,j,k)**: Random access everywhere, cache thrashing!
```c
for i in rows:
    for j in cols:
        for k in shared_dim:
            C[i,j] += A[i,k] * B[k,j]  // Both jump around randomly
```

### Why This Matters for Neural Networks

A 1000×1000 matrix multiply does 1 billion operations. If cache-friendly code is just 2x faster, that's 2 billion more operations per second!

---

## 2.4 The Dot Product — Heart of Neural Networks

### Definition

For vectors a = [a₀, a₁, ..., aₙ₋₁] and b = [b₀, b₁, ..., bₙ₋₁]:
```
dot(a, b) = Σ a[i] * b[i] = a₀*b₀ + a₁*b₁ + ... + aₙ₋₁*bₙ₋₁
```

### Why It Matters

**Every matrix multiplication is just a bunch of dot products!**

```
C = A × B

C[i,j] = dot(A[i,*], B[*][j])
        = row i of A ⋅ column j of B
```

In a neural network:
- **Dense layer**: output[j] = Σ weight[i,j] * input[i] + bias[j]
- That's a dot product for each output neuron!
- For a 784×256 layer: 784 dot products per output neuron, 256 neurons = 200,704 dot products per image

### Properties

- **Commutative**: dot(a,b) = dot(b,a)
- **Linear**: dot(a + b, c) = dot(a,c) + dot(b,c)
- **Scalar**: k * dot(a,b) = dot(ka, b) = dot(a, kb)

### Future: SIMD Optimization

We'll use SIMD (Single Instruction Multiple Data) to compute 4-8 dot products at once:
```c
// Process 4 floats at once
__m128 va = _mm_loadu_ps(a);
__m128 vb = _mm_loadu_ps(b);
__mm_add_ps(_mm_mul_ps(va, vb), ...);  // 4 multiplies + 4 adds in 1 instruction!
```

---

## 2.5 Transpose and Why We Need It

### Definition

Flipping rows and columns: A[i,j] → A^T[j,i]

```
A = [1 2 3]      A^T = [1 4]
    [4 5 6]              [2 5]
                        [3 6]
```

### Why Neural Networks Need Transpose

1. **Backpropagation**: Computing gradients requires weight matrix transpose
2. **Batch operations**: Some libraries store data in column-major format
3. **Reshaping**: Flattening images (28×28 → 784×1) is like transpose

### Implementation

```c
void matrix_transpose(const Matrix* A, Matrix* result) {
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            result->data[j * result->stride + i] = A->data[i * A->stride + j];
}
```

---

## 2.6 Writing Good Unit Tests in C

### Principles

1. **Test known inputs/expected outputs**: Don't just "run it", verify correctness
2. **Test edge cases**: Single element, non-square, zero matrices
3. **Test properties**: Commutativity, associativity, identity
4. **Use epsilon for floats**: Don't compare floats with ==, use tolerance

### Our Test Structure

```c
/* Helper: approximate float equality */
static int float_eq(float a, float b, float epsilon) {
    return fabsf(a - b) < epsilon;
}

/* Helper: matrix equality */
static int matrix_eq(const Matrix* A, const Matrix* B, float epsilon) {
    // Check dimensions, then each element
}

/* Test function pattern */
void test_multiply(void) {
    Matrix A = matrix_create(2, 2);
    // ... setup ...
    matrix_multiply(&A, &B, &C);
    
    // Check expected values
    int passed = /* check C matches expected */;
    test_result("Test name", passed);
    
    matrix_free(&A);  // Don't leak!
}
```

### Test Coverage

We tested:
- ✓ Basic multiplication (2×2, non-square)
- ✓ Element-wise addition + commutativity
- ✓ Scalar multiplication + zero case
- ✓ Transpose + double transpose
- ✓ Dot product + orthogonality
- ✓ Copy + independence
- ✓ Edge cases (1×1, 1×5)

---

## 2.7 Benchmarking Our Naive Implementation

### Current Performance

Our `matrix_multiply` uses the i-k-j loop order, which is reasonably cache-friendly but not optimized.

For a 512×512 matrix multiply:
- Naive: ~0.1-0.5 seconds (depends on CPU)
- Optimized (SIMD): ~0.01-0.05 seconds (5-10x faster)
- Highly optimized (BLAS): ~0.005-0.02 seconds (10-20x faster)

### Where We're Spending Time

```
Total time breakdown for matrix multiply:
┌────────────────────┬──────────────┐
│ Memory loads A     │ ~40%         │
│ Memory loads B     │ ~40%         │
│ Multiplications    │ ~10%         │
│ Additions          │ ~5%          │
│ Loop overhead      │ ~5%          │
└────────────────────┴──────────────┘
```

The key insight: **memory access dominates computation time!**

### Future Optimizations (Phase 5)

1. **SIMD**: 4-8 floats per instruction
2. **Blocking/Tiling**: Process in cache-sized chunks
3. **Loop unrolling**: Reduce loop overhead
4. **Prefetching**: Load data before needed
5. **OpenBLAS**: Use optimized vendor library

---

## What Just Happened

- **Implemented matrix multiplication**: O(n³) with cache-friendly loop order (i-k-j)
- **Added element-wise operations**: Add, scalar multiply, copy
- **Implemented transpose**: Flips rows and columns
- **Created dot product**: The core operation that powers all neural network computations
- **Built test suite**: 7 test functions covering basic ops, edge cases, and properties
- **Learned why it matters**: Cache locality, loop order, and SIMD optimization potential