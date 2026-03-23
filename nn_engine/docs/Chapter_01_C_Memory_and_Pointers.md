# Chapter 1: C Memory and Pointers

## 1.1 The C Memory Model

C provides low-level control over memory - you decide *where* and *how* data is stored. This is both powerful and dangerous.

### Memory Regions in a C Program

```
┌─────────────────────────────────────────────┐
│                 TEXT SEGMENT                │  ← Code (read-only)
├─────────────────────────────────────────────┤
│              RODATA (const)                 │  ← Read-only data
├─────────────────────────────────────────────┤
│                DATA (globals)               │  ← Initialized globals
├─────────────────────────────────────────────┤
│              BSS (uninit)                   │  ← Zero-initialized globals
├─────────────────────────────────────────────┤
│                      ↓                       │
│              (grows down)                   │
├─────────────────────────────────────────────┤
│                   HEAP                      │  ← malloc/free territory
│              (grows up)                      │
├─────────────────────────────────────────────┤
│              FREE MEMORY                    │
├─────────────────────────────────────────────┤
│              STACK                          │  ← Local variables
│              (grows down)                   │
└─────────────────────────────────────────────┘
```

Key insight: **Stack grows down, Heap grows up** - they meet in the middle!

---

## 1.2 Pointers — What They Really Are

A pointer is just a variable that stores a **memory address**.

```c
int x = 42;        // x is an integer with value 42
int* p = &x;       // p is a pointer, &x means "address of x"
printf("%d\n", *p); // *p means "value at address p" → prints 42
```

### Pointer Anatomy

```
Variable x at address 0x1000:
┌──────────┬──────────┐
│  0x1000  │   42     │
│  (addr)  │ (value)  │
└──────────┴──────────┘
     │
     │     Pointer p at address 0x2000:
     ▼     ┌──────────┬──────────┐
            │  0x2000  │  0x1000 │  ← stores address of x
            └──────────┴──────────┘
```

### Why Use Pointers?

1. **Efficiency**: Pass large data without copying (just pass address)
2. **Flexibility**: Dynamic memory allocation (malloc returns pointer)
3. **Data structures**: Build linked lists, trees, graphs
4. **Hardware access**: Memory-mapped devices, direct memory manipulation

---

## 1.3 Heap vs Stack

### The Stack (Automatic Memory)

```c
void function() {
    int x = 10;           // 'x' allocated on stack
    float arr[100];       // 'arr' allocated on stack
    // When function returns, x and arr are AUTOMATICALLY freed!
}
```

**Characteristics:**
- Fast allocation (just move stack pointer)
- Automatic cleanup (when function returns)
- Limited size (typically 1-8 MB)
- Fixed size at compile time for arrays

### The Heap (Dynamic Memory)

```c
void function() {
    int* x = (int*)malloc(sizeof(int));  // Allocate on heap
    *x = 10;
    free(x);  // MUST free manually!
    
    float* arr = (float*)malloc(100 * sizeof(float));  // Variable size
    free(arr);  // MUST free!
}
```

**Characteristics:**
- Slower allocation (complex memory management)
- Manual cleanup (you must call free!)
- Large size (limited by system RAM)
- Size determined at runtime

### Heap vs Stack — Analogy

Think of your desk vs a warehouse:

| Aspect | Stack (Desk) | Heap (Warehouse) |
|--------|--------------|------------------|
| Allocation | Instant (grab pen) | Slow (go find storage unit) |
| Cleanup | Automatic (clean desk) | Manual (return storage key) |
| Size | Limited space | Huge building |
| When to use | Local variables | Data needing lifetime beyond function |

---

## 1.4 Our Matrix Data Structure

Our Matrix struct uses a **flat array** approach:

```c
typedef struct {
    float* data;   // Pointer to heap-allocated array
    int rows;      // Number of rows
    int cols;      // Number of columns
    int stride;    // Distance between elements in memory
} Matrix;
```

### Creating a Matrix

```c
Matrix m = matrix_create(3, 4);  // 3 rows, 4 columns
```

What happens:
1. `malloc(12 * sizeof(float))` - allocate 12 floats on heap
2. `memset()` - zero-initialize the memory
3. Return Matrix struct with rows=3, cols=4, stride=4

### Freeing a Matrix

```c
matrix_free(&m);  // Pass address because we modify the struct
```

What happens:
1. `free(m.data)` - return heap memory to system
2. Set all fields to 0 (prevent use-after-free bugs)

---

## 1.5 Memory Layout — Why Flat Arrays Beat 2D Arrays

### The Problem with `float[][]`

```c
float matrix[3][4];  // Array of 3 arrays of 4 floats
```

Memory layout in C:
```
Row 0: [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]]
Row 1: [matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]]
Row 2: [matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]]
```

This looks contiguous, BUT:
- `float matrix[3][4]` requires known dimensions at compile time
- Can't easily pass "sub-matrices" to functions
- No flexibility for stride (column-major or row-major)

### Our Flat Array Approach

```c
typedef struct {
    float* data;  // Single contiguous block
    int rows;
    int cols;
    int stride;   // Configurable!
} Matrix;
```

**Standard layout** (stride = cols):
```
data[0] data[1] data[2] data[3] data[4] data[5] data[6] data[7] data[8] ...
         ↑ row 0                      ↑ row 1                      ↑ row 2
```

**Column-major layout** (stride = 1):
```
data[0] data[4] data[8] data[1] data[5] data[9] ...
         ↑ row 0              ↑ row 1
```

### Stride Enables Views Without Copying

```c
// Create a "view" into a column of a matrix
Matrix col_view;
col_view.data = &original.data[2];  // Start at column 2
col_view.rows = original.rows;
col_view.cols = 1;
col_view.stride = original.cols;    // Skip full row to get to next column
```

This is critical for neural networks - we often need to slice matrices efficiently!

---

## 1.6 Common Memory Bugs in C (and How to Avoid Them)

### Bug 1: Memory Leak

```c
// WRONG: allocates but never frees
void create_and_forget() {
    float* data = malloc(1000 * sizeof(float));
    // forgot to free!
}
// Called in a loop = memory grows until crash
```

**Fix**: Always pair malloc with free:

```c
void create_and_free() {
    float* data = malloc(1000 * sizeof(float));
    if (data == NULL) return ERROR;
    
    use_data(data);
    
    free(data);  // Always free!
}
```

**Detect with Valgrind**:
```bash
valgrind --leak-check=full ./nn_engine
```
Expected output with leak:
```
==12345== LEAK SUMMARY:
==12345==    definitely lost: 4000 bytes in 1 blocks
```

### Bug 2: Use-After-Free

```c
// WRONG: use pointer after freeing
char* p = malloc(100);
free(p);
printf("%s", p);  // CRASH! Memory already returned to system
```

**Fix**: Set pointer to NULL after freeing:

```c
free(p);
p = NULL;  // Defensive: now dereferencing crashes safely
```

### Bug 3: Double Free

```c
// WRONG: free same memory twice
free(p);
free(p);  // Undefined behavior!
```

**Fix**: Nullify after free, check before free:

```c
free(p);
p = NULL;
// Later...
if (p != NULL) free(p);  // Safe
```

### Bug 4: Buffer Overflow

```c
// WRONG: write beyond array bounds
float arr[10];
for (int i = 0; i < 20; i++) arr[i] = i;  // OOPS!

// RIGHT: always respect bounds
for (int i = 0; i < 10; i++) arr[i] = i;
```

### Bug 5: Not Checking NULL

```c
// WRONG: assume malloc succeeds
float* data = malloc(1000000 * sizeof(float));
data[0] = 5;  // CRASH if allocation failed!

// RIGHT: always check
float* data = malloc(1000000 * sizeof(float));
if (data == NULL) {
    fprintf(stderr, "Out of memory!\n");
    return ERROR;
}
```

### Compiler Flags That Catch Bugs

Our Makefile uses `-Wall -Wextra` which catches:
- `-Wall`: Basic warnings (unused variables, missing return)
- `-Wextra`: Extra warnings (unused parameters, comparison issues)

---

## 1.7 Weight Initialization in Neural Networks

### The Problem

Neural network weights must be carefully initialized:

- **Too small** → signals vanish through layers (output ≈ 0)
- **Too large** → signals explode (output = NaN/infinity)

### Xavier Initialization (2010)

Proposed by Xavier Glorot and Yoshua Bengio:

```
W ~ Normal(0, sqrt(1/n))    for tanh/sigmoid
W ~ Normal(0, sqrt(2/n))    for ReLU (He initialization)
```

Where `n` = number of input neurons (fan-in).

**Why it works**: Keeps variance consistent layer-to-layer!

### Mathematical Intuition

For a layer: `y = Wx + b`

If `x` has variance `σ²` and `W` has variance `σ_w²`:
```
Var(y) = n * σ² * σ_w²  (output variance)
```

For stable propagation: `n * σ_w² = 1`

So: `σ_w = 1/sqrt(n)` → Xavier initialization!

### Our Implementation

```c
void matrix_random(Matrix* m) {
    int fan_in = m->cols;
    int fan_out = m->rows;
    float scale = sqrtf(2.0f / (fan_in + fan_out));
    
    for each element:
        value = random(-1, 1) * scale;
}
```

This gives us well-scaled initial weights ready for training (or inference with a trained model).

---

## What Just Happened

- **Created Matrix struct**: A flat float array with rows/cols/stride metadata
- **Implemented heap allocation**: Used malloc/free correctly with NULL checks
- **Added stride concept**: Enables flexible memory layouts without copying data
- **Implemented Xavier init**: Properly scaled random weights to prevent vanishing/exploding gradients
- **Learned memory bugs**: Found how to avoid leaks, double-free, and buffer overflows
- **Built working code**: Matrix creation, initialization, printing, and cleanup all tested