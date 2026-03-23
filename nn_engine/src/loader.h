/**
 * @file loader.h
 * @brief Model loading and saving utilities
 * 
 * This module handles saving/loading neural networks to/from disk.
 * 
 * We implement two approaches:
 * 1. Our custom .nnbin format - simple and educational
 * 2. GGUF header analysis - shows how real frameworks (llama.cpp) work
 */

#ifndef LOADER_H
#define LOADER_H

#include <stddef.h>
#include "network.h"

/**
 * @brief Activation type enum for file format
 * 
 * Must match order in activations.h!
 */
typedef enum {
    ACT_NONE = 0,    /**< Linear (no activation) */
    ACT_RELU = 1,    /**< ReLU */
    ACT_SIGMOID = 2, /**< Sigmoid */
    ACT_TANH = 3,    /**< Tanh */
    ACT_SOFTMAX = 4  /**< Softmax */
} ActivationType;

/**
 * @brief Save neural network to binary file
 * @param net Network to save
 * @param filepath Output file path
 * @return 0 on success, -1 on error
 * 
 * File format (.nnbin):
 *   [4 bytes] magic: 0x4E4E4249 ("NNBI")
 *   [4 bytes] version: 1
 *   [4 bytes] num_layers
 *   For each layer:
 *     [4 bytes] input_size
 *     [4 bytes] output_size  
 *     [4 bytes] activation_type
 *     [input*output*4 bytes] weights (float32)
 *     [output*4 bytes] biases (float32)
 */
int network_save(NeuralNetwork* net, const char* filepath);

/**
 * @brief Load neural network from binary file
 * @param filepath Input file path
 * @return Newly allocated NeuralNetwork, or NULL on error
 */
NeuralNetwork* network_load(const char* filepath);

/**
 * @brief Analyze GGUF file header (without full loading)
 * @param filepath Path to .gguf file
 * 
 * GGUF is the format used by llama.cpp for LLM models.
 * This function shows what's in the header to demystify the format.
 * 
 * NOTE: This is read-only analysis, NOT a full loader!
 */
void show_gguf_header_info(const char* filepath);

#endif /* LOADER_H */