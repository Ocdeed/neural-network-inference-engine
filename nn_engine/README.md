# Neural Network Inference Engine

![build: passing](https://img.shields.io/badge/build-passing-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)
![Language: C](https://img.shields.io/badge/Language-C-blue)

A high-performance neural network inference engine written from scratch in pure C, implementing matrix operations, activation functions, dense layers, and both SIMD and multithreading optimizations.

## Overview

This project demonstrates how neural networks work under the hood by building one from scratch in C. It includes comprehensive documentation explaining each component, making it an excellent learning resource for understanding deep learning infrastructure.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neural Network Inference                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input          Layer 1          Layer 2         Output        │
│  (784)    →   [Dense 256]   →   [Dense 10]   →  [Predictions]  │
│               ReLU激活         Softmax                               │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     Core Components                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Matrix   │  │Activations│  │ Dense    │  │ Network  │        │
│  │ Ops      │  │ Functions │  │ Layers   │  │ Container│        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    Optimizations                                │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   SIMD (AVX2)       │    │  Multithreading     │            │
│  │  8 floats/cycle    │    │  4 worker threads   │            │
│  └─────────────────────┘    └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Matrix Operations**: Create, multiply, transpose, dot product with optimized layouts
- **Activation Functions**: ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax
- **Dense Layers**: Fully connected layers with Xavier initialization
- **Neural Network Container**: Chain layers together, run inference, make predictions
- **Model Persistence**: Save/load models in custom binary format (.nnbin)
- **SIMD Optimization**: AVX2 intrinsics for 8x parallelism in vector operations
- **Multithreading**: Thread pool with work queue for parallel matrix multiplication
- **Comprehensive Docs**: Each phase includes detailed explanations

## Building

```bash
cd nn_engine
make          # Build the main engine
make test     # Run unit tests
make benchmark_full  # Run performance benchmarks
```

## Benchmark Results

| Matrix Size | Naive (ms) | Threaded (ms) | Speedup |
|-------------|------------|---------------|---------|
| 128×128     | 8.60       | 2.00          | 4.30×   |
| 256×256     | 100.20     | 14.52         | 6.90×   |
| 512×512     | 1241.58    | 154.96        | 8.01×   |

*Results from Intel Core i5-3210M, 4 threads*

## Project Structure

```
nn_engine/
├── src/
│   ├── matrix.h/c        # Matrix data structure & operations
│   ├── activations.h/c   # ReLU, Sigmoid, Tanh, Softmax
│   ├── layers.h/c        # Dense (fully connected) layers
│   ├── network.h/c       # Neural network container
│   ├── loader.h/c        # Model save/load (.nnbin format)
│   ├── simd_ops.h/c      # AVX2 SIMD operations
│   ├── thread_pool.h/c   # Thread pool for parallelism
│   └── main.c            # Demo program
├── tests/
│   ├── test_matrix.c     # Matrix unit tests
│   ├── test_activations.c
│   ├── test_layers.c
│   ├── benchmark_full.c  # Full benchmark suite
│   └── benchmark_threads.c
├── docs/
│   ├── Chapter_01_C_Memory_and_Pointers.md
│   ├── Chapter_02_Matrix_Math_in_C.md
│   ├── Chapter_03_Activation_Functions.md
│   ├── Chapter_04_Forward_Pass.md
│   ├── Chapter_05_Network_Architecture.md
│   ├── Chapter_06_Model_File_Formats.md
│   ├── Chapter_07_SIMD_Optimizations.md
│   ├── Chapter_08_Multithreading.md
│   └── Chapter_09_Benchmarking_and_Profiling.md
├── Makefile
└── README.md
```

## What You Learned

This project covers fundamental systems programming concepts:

### Core Skills
- **C Memory Management**: malloc, free, pointers, heap vs stack
- **Data Structures**: Flat arrays with stride for matrix representation
- **Pointer Arithmetic**: Efficient matrix access patterns

### Neural Network Fundamentals
- **Matrix Multiplication**: The core operation (O(n³))
- **Activation Functions**: Non-linearities that enable deep learning
- **Forward Propagation**: How data flows through networks

### Performance Optimization
- **SIMD Programming**: Using AVX2 intrinsics for 8x parallelism
- **Multithreading**: Thread pools, mutexes, condition variables
- **Profiling**: gprof, perf, valgrind --callgrind

## Future Improvements

- **Quantization**: INT8 inference for mobile deployment
- **GPU Support**: OpenCL/CUDA for massive parallelism
- **Transformer Support**: Attention mechanisms for LLMs
- **Convolution**: CNN layers for image processing
- **Operators**: BatchNorm, Dropout, Pooling

## License

MIT License - feel free to use for learning or as a base for your projects!

## Contributing

This is an educational project. Feel free to submit issues or PRs to improve the documentation or add features.
