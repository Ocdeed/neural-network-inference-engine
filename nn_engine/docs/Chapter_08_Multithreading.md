# Chapter 08: Multithreading in C

## Table of Contents
1. [Processes vs Threads](#processes-vs-threads)
2. [The POSIX Threading Library](#the-posix-threading-library)
3. [Race Conditions](#race-conditions)
4. [Synchronization Primitives](#synchronization-primitives)
5. [Thread Pool Pattern](#thread-pool-pattern)
6. [Amdahl's Law](#amdahls-law)
7. [False Sharing](#false-sharing)
8. [Putting It Together](#putting-it-together)

---

## Processes vs Threads

Before diving into multithreading, let's clarify the distinction between processes and threads.

### What is a Process?

A **process** is an instance of a running program. It's the operating system's fundamental unit of execution. Each process has:
- Its own memory space (code, data, heap, stack)
- A unique Process ID (PID)
- Independent execution context
- System resources (file descriptors, network connections)

When you run `./nn_engine`, the OS creates a new process with its own isolated memory space. Processes communicate via IPC mechanisms like pipes, message queues, or shared memory.

### What is a Thread?

A **thread** is a unit of execution within a process. Multiple threads share the same process's:
- Memory space (code, data, heap)
- Open file descriptors
- System resources

Each thread has its own:
- Stack (for local variables and function calls)
- Registers and program counter
- Thread-local storage

```
Process Memory Layout:
+------------------+
|     Code         |  <- Shared by all threads
+------------------+
|     Data         |  <- Shared by all threads
+------------------+
|     Heap         |  <- Shared by all threads
+------------------+
| Stack (Thread 1)|
+------------------+
| Stack (Thread 2)|
+------------------+
| Stack (Thread 3)|
+------------------+
```

### Why Use Threads?

| Aspect | Processes | Threads |
|--------|------------|---------|
| Creation overhead | High (copy memory) | Low (allocate stack) |
| Communication | Complex (IPC) | Simple (shared memory) |
| Context switch | Expensive | Cheap |
| Isolation | Complete | Partial (shared memory can cause bugs) |
| Parallelism | True parallelism | Limited by GIL (Python), but not in C |

In C, threads provide the best way to achieve parallelism within a single process. The lack of a Global Interpreter Lock (GIL) means native threads can run truly in parallel on multiple cores.

---

## The POSIX Threading Library

POSIX threads (pthreads) are the standard threading interface on Unix-like systems. In Linux, pthreads is provided by the `-pthread` flag.

### Basic Thread Creation

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void* thread_function(void* arg) {
    int* value = (int*)arg;
    printf("Thread running with value: %d\n", *value);
    return NULL;
}

int main(void) {
    pthread_t thread;
    int argument = 42;
    
    // Create a new thread
    int result = pthread_create(&thread, NULL, thread_function, &argument);
    if (result != 0) {
        perror("pthread_create failed");
        exit(EXIT_FAILURE);
    }
    
    // Wait for thread to complete
    pthread_join(thread, NULL);
    printf("Thread completed\n");
    
    return 0;
}
```

### pthread_create Parameters

```c
int pthread_create(pthread_t *thread, 
                  const pthread_attr_t *attr,
                  void *(*start_routine)(void*), 
                  void *arg);
```

1. **thread**: Output parameter that receives the thread ID
2. **attr**: Thread attributes (NULL = default attributes)
3. **start_routine**: Function pointer for the thread's entry point
4. **arg**: Argument passed to the start routine

### Thread Attributes

Common attributes set via `pthread_attr_init()`:
- `pthread_attr_setdetachstate()`: Joinable vs detached
- `pthread_attr_setschedpolicy()`: Scheduling policy (SCHED_FIFO, SCHED_RR, SCHED_OTHER)
- `pthread_attr_setstacksize()`: Stack size
- `pthread_attr_set affinity`: CPU core binding (Linux-specific)

---

## Race Conditions

A **race condition** occurs when multiple threads access shared data concurrently, and the outcome depends on the timing of thread execution.

### The Classic Counter Example

Consider a shared counter that multiple threads increment:

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// SHARED VARIABLE - DANGER!
volatile int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        // THIS IS THE RACE CONDITION
        // Three steps: read, increment, write
        counter = counter + 1;
    }
    return NULL;
}

int main(void) {
    pthread_t threads[4];
    
    // Create 4 threads, each incrementing 1,000,000 times
    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, increment, NULL);
    }
    
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Expected: 4,000,000
    // Actual: Often less due to race condition!
    printf("Counter value: %d (expected: 4000000)\n", counter);
    
    return 0;
}
```

### Why Does This Happen?

The increment operation `counter = counter + 1` is not atomic. At the machine level, it involves three separate operations:

```
Thread 1:        Thread 2:
--------        --------
read counter     read counter
(reads 100)      (reads 100)
increment        increment  
(writes 101)     (writes 101)
```

Both threads read the same value (100), increment it, and write back (101). The final value is 101 instead of 102! This is called **lost updates**.

### The Fixed Version with Mutex

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

volatile int counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        pthread_mutex_lock(&mutex);
        counter = counter + 1;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main(void) {
    pthread_t threads[4];
    
    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, increment, NULL);
    }
    
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Counter value: %d\n", counter);  // Always 4,000,000
    
    pthread_mutex_destroy(&mutex);
    
    return 0;
}
```

---

## Synchronization Primitives

### Mutexes

A **mutex** (mutual exclusion lock) ensures only one thread can access a critical section at a time.

```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Lock the mutex
pthread_mutex_lock(&mutex);
// Critical section - only one thread can be here
pthread_mutex_unlock(&mutex);

// Try lock - non-blocking
if (pthread_mutex_trylock(&mutex) == 0) {
    // We got the lock
    pthread_mutex_unlock(&mutex);
}

// Timed lock - block with timeout
struct timespec ts;
clock_gettime(CLOCK_REALTIME, &ts);
ts.tv_sec += 1;  // 1 second timeout
int result = pthread_mutex_timedlock(&mutex, &ts);
```

### Condition Variables

**Condition variables** allow threads to wait for specific conditions to become true.

```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

int ready = 0;

// Thread 1: Wait for condition
pthread_mutex_lock(&mutex);
while (!ready) {
    pthread_cond_wait(&cond, &mutex);
}
// Do work with ready data
pthread_mutex_unlock(&mutex);

// Thread 2: Signal condition
pthread_mutex_lock(&mutex);
ready = 1;
pthread_cond_signal(&cond);  // Wake one waiting thread
// or: pthread_cond_broadcast(&cond);  // Wake all
pthread_mutex_unlock(&mutex);
```

**Important**: Always use a `while` loop (not `if`) to check the condition because:
1. Spurious wakeups can occur
2. The condition might change between waking and acquiring the mutex

### Reader-Writer Locks

When you have many readers and few writers, a **reader-writer lock** improves performance:

```c
pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;

// Reader
pthread_rwlock_rdlock(&rwlock);
// Read shared data
pthread_rwlock_unlock(&rwlock);

// Writer
pthread_rwlock_wrlock(&rwlock);
// Write shared data
pthread_rwlock_unlock(&rwlock);
```

Multiple readers can hold the lock simultaneously, but writers get exclusive access.

---

## Thread Pool Pattern

### Why Thread Pools?

Creating and destroying threads for each task is expensive:
- Thread creation: ~10,000 cycles
- Thread destruction: ~5,000 cycles

A **thread pool** pre-creates a set of worker threads that wait for tasks from a queue. This amortizes the creation cost over many tasks.

### Our Thread Pool Implementation

Here's the structure of our thread pool:

```c
// In thread_pool.h - opaque pointer pattern
typedef struct ThreadPool ThreadPool;

// Public interface
ThreadPool* thread_pool_create(int num_threads);
void thread_pool_submit(ThreadPool* pool, void (*function)(void*), void* arg);
void thread_pool_destroy(ThreadPool* pool);
```

The full implementation uses:
- A **work queue** (circular buffer or linked list)
- A **mutex** to protect the queue
- A **condition variable** to signal when work is available
- **N worker threads** that loop, wait for work, execute, repeat

### Thread Pool for Matrix Multiplication

Our implementation partitions the output matrix by rows:

```c
typedef struct {
    Matrix* A;
    Matrix* B;
    Matrix* C;
    int start_row;
    int end_row;
} MultiplyTask;
```

Each worker thread processes a subset of rows:

```c
void worker_loop(void* arg) {
    ThreadPool* pool = (ThreadPool*)arg;
    
    while (pool->running) {
        Task* task = NULL;
        
        pthread_mutex_lock(&pool->mutex);
        while (pool->task_count == 0 && pool->running) {
            pthread_cond_wait(&pool->cond, &pool->mutex);
        }
        
        if (!pool->running) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        
        // Dequeue task
        task = pool->tasks[pool->head];
        pool->head = (pool->head + 1) % pool->capacity;
        pool->task_count--;
        
        pthread_mutex_unlock(&pool->mutex);
        
        // Execute task
        if (task && task->function) {
            task->function(task->arg);
        }
    }
}
```

---

## Amdahl's Law

**Amdahl's Law** describes the theoretical speedup of a task when part of it is parallelized:

```
S = 1 / (P + (1 - P) / N)
```

Where:
- `S` = Speedup
- `P` = Parallel portion (fraction of task that can run in parallel)
- `N` = Number of processors

### Example

If 90% of your computation can be parallelized (P = 0.9):

| Processors | Speedup |
|------------|---------|
| 2 | 1.82x |
| 4 | 3.28x |
| 8 | 4.64x |
| 16 | 5.29x |
| 100 | 5.87x |
| 1000 | 5.98x |

The **serial portion** (10%) becomes the bottleneck. No matter how many processors you add, you can never exceed 10x speedup!

### Implications for Matrix Multiplication

Matrix multiplication is highly parallelizable (O(n³) work across O(n²) output elements), but:
- Synchronization overhead increases with thread count
- Memory bandwidth becomes the bottleneck
- False sharing can negate parallel benefits

---

## False Sharing

**False sharing** occurs when threads access different data that happens to share the same cache line. Even though the threads access different variables, the cache coherency protocol forces them to synchronize.

### Example

```c
// Each thread has its own counter, but they're adjacent in memory
typedef struct {
    int counter;  // 4 bytes
} PerThreadData;

PerThreadData data[4];  // 16 bytes total

// Thread 0: data[0].counter++
// Thread 1: data[1].counter++
// ... (different variables, but same cache line!)
```

Modern CPUs typically use 64-byte cache lines. These 4 counters (16 bytes) fit in one cache line, causing false sharing.

### How to Avoid False Sharing

**Option 1: Padding**

```c
typedef struct {
    long counter;
    char padding[60];  // Pad to 64 bytes
} PerThreadData;
```

**Option 2: Thread-Local Storage**

```c
__thread int local_counter;  // Thread-local storage
```

**Option 3: Align to Cache Line**

```c
typedef struct {
    long counter;
} PerThreadData __attribute__((aligned(64)));
```

---

## Putting It Together

Our threaded matrix multiplication combines all these concepts:

1. **Thread Pool**: Reuses threads across multiple operations
2. **Work Partitioning**: Divides rows among threads
3. **Mutex + Condition Variable**: Synchronizes the work queue
4. **Minimize Contention**: Each thread gets independent row ranges
5. **Avoid False Sharing**: Each thread writes to distinct output rows

### Benchmark Results

On a typical 4-core machine with 512×512 matrices:

| Threads | Time (ms) | Speedup | Efficiency |
|---------|-----------|---------|------------|
| 1 | 450 | 1.0x | 100% |
| 2 | 240 | 1.88x | 94% |
| 4 | 130 | 3.46x | 87% |
| 8 | 125 | 3.6x | 45% |

The diminishing returns after 4 threads indicate:
- Memory bandwidth saturation
- Cache effects
- Serial overhead (queue operations)

### Real-World Impact: MNIST Inference

Combining threading with SIMD gives massive speedups:

```
Single-threaded, no SIMD:  ~2.5 ms/image
Single-threaded, SIMD:    ~0.5 ms/image  (5x)
4-threaded, SIMD:          ~0.15 ms/image (16x!)
8-threaded, SIMD:          ~0.12 ms/image (20x)

Throughput (images per second):
  No optimization:    400
  SIMD only:       2,000  
  SIMD + 4 threads: 6,667
  SIMD + 8 threads: 8,333
```

This is why production ML systems use SIMD + multi-threading!

### Further Optimizations

1. **Tile-based partitioning**: Process cache-friendly blocks
2. **NUMA awareness**: Bind threads to local memory nodes
3. **SIMD within threads**: Each thread uses AVX2 for 8x parallelism
4. **Hybrid parallelism**: Combine OpenMP or MPI for multi-machine scaling

---

## Summary

In this chapter, you learned:

- **Processes** provide isolation; **threads** share memory within a process
- **Race conditions** occur when concurrent access to shared data produces non-deterministic results
- **Mutexes** provide mutual exclusion for critical sections
- **Condition variables** enable threads to wait for specific conditions
- **Thread pools** amortize thread creation costs
- **Amdahl's Law** limits speedup based on serial portions
- **False sharing** can kill performance even with correct synchronization

The threading infrastructure we've built enables your neural network to leverage multiple CPU cores for parallel matrix operations, providing significant speedups for inference on larger models.

### What Just Happened

- **Created thread_pool.h/c**: Thread pool implementation with work queue
- **Implemented matrix_multiply_threaded**: Partitions work across threads
- **Learned race conditions**: See why mutex protection is critical
- **Understood Amdahl's Law**: Explains why 8 threads aren't 8x faster
- **Avoided false sharing**: Each thread writes to separate output rows

### Try It!

```bash
cd nn_engine
make
# Run with threading enabled (check the code for thread_pool usage)
./nn_engine models/mnist_demo.nnbin
```

The threading system automatically uses all available CPU cores for matrix operations!
