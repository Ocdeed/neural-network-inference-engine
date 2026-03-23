/**
 * @file loader.c
 * @brief Implementation of model loading/saving
 * 
 * FILE I/O EXPLAINED:
 * ==================
 * 
 * There are two main ways to read/write files in C:
 * 
 * 1. TEXT I/O (fprintf, fscanf, fgets):
 *    - Human readable (can open in text editor)
 *    - Slower (conversion to/from text)
 *    - Use for: config files, logs, CSVs
 * 
 * 2. BINARY I/O (fread, fwrite):
 *    - Raw bytes, no conversion
 *    - Faster, smaller files
 *    - Use for: model weights, images, any numeric data
 * 
 * Why binary for models?
 * - 100K floats as text = ~1MB
 * - 100K floats as binary = ~400KB
 * - Also MUCH faster to read!
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "loader.h"
#include "network.h"
#include "layers.h"
#include "activations.h"
#include "matrix.h"

/**
 * Magic number for our .nnbin format
 * "NNBI" in ASCII = 0x4E4E4249
 */
#define NNBIN_MAGIC 0x4E4E4249
#define NNBIN_VERSION 1

/* Helper to convert ActivationFn to enum */
static ActivationType activation_to_enum(ActivationFn fn)
{
    if (fn == NULL) return ACT_NONE;
    if (fn == activation_relu) return ACT_RELU;
    if (fn == activation_sigmoid) return ACT_SIGMOID;
    if (fn == activation_tanh_act) return ACT_TANH;
    if (fn == activation_softmax) return ACT_SOFTMAX;
    return ACT_NONE;
}

/* Helper to convert enum back to ActivationFn */
static ActivationFn enum_to_activation(ActivationType act)
{
    switch (act) {
        case ACT_NONE: return NULL;
        case ACT_RELU: return activation_relu;
        case ACT_SIGMOID: return activation_sigmoid;
        case ACT_TANH: return activation_tanh_act;
        case ACT_SOFTMAX: return activation_softmax;
        default: return NULL;
    }
}

/**
 * Save network to binary file
 * 
 * BINARY WRITING:
 * ===============
 * 
 * fwrite(data, size, count, file) writes raw bytes.
 * 
 * Example:
 *   int x = 42;
 *   fwrite(&x, sizeof(int), 1, file);  // Writes 4 bytes: 0x00 0x00 0x00 0x2A
 * 
 * We use uint32_t for integers to ensure consistent 4-byte size.
 */
int network_save(NeuralNetwork* net, const char* filepath)
{
    if (net == NULL || filepath == NULL) {
        fprintf(stderr, "ERROR: NULL network or filepath\n");
        return -1;
    }
    
    FILE* f = fopen(filepath, "wb");  /* "wb" = write binary */
    if (f == NULL) {
        fprintf(stderr, "ERROR: Could not open %s for writing\n", filepath);
        return -1;
    }
    
    /* Write magic number */
    uint32_t magic = NNBIN_MAGIC;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    
    /* Write version */
    uint32_t version = NNBIN_VERSION;
    fwrite(&version, sizeof(uint32_t), 1, f);
    
    /* Write number of layers */
    uint32_t num_layers = (uint32_t)net->num_layers;
    fwrite(&num_layers, sizeof(uint32_t), 1, f);
    
    printf("Saving %d layers...\n", net->num_layers);
    
    /* Write each layer */
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer* layer = net->layers[i];
        
        /* Layer dimensions */
        uint32_t input_size = (uint32_t)layer->input_size;
        uint32_t output_size = (uint32_t)layer->output_size;
        uint32_t act_type = (uint32_t)activation_to_enum(layer->activation);
        
        fwrite(&input_size, sizeof(uint32_t), 1, f);
        fwrite(&output_size, sizeof(uint32_t), 1, f);
        fwrite(&act_type, sizeof(uint32_t), 1, f);
        
        /* Write weights */
        size_t num_weights = (size_t)layer->output_size * (size_t)layer->input_size;
        fwrite(layer->weights->data, sizeof(float), num_weights, f);
        
        /* Write biases */
        fwrite(layer->biases->data, sizeof(float), (size_t)layer->output_size, f);
        
        printf("  Layer %d: %d×%d, saved %.1f KB\n", 
               i + 1, input_size, output_size,
               (num_weights + layer->output_size) * sizeof(float) / 1024.0);
    }
    
    fclose(f);
    printf("✓ Saved to %s\n", filepath);
    
    return 0;
}

/**
 * Load network from binary file
 * 
 * BINARY READING:
 * ==============
 * 
 * fread(data, size, count, file) reads raw bytes.
 * 
 * Example:
 *   int x;
 *   fread(&x, sizeof(int), 1, file);  // Reads 4 bytes into x
 * 
 * IMPORTANT: Must match the write order exactly!
 */
NeuralNetwork* network_load(const char* filepath)
{
    if (filepath == NULL) {
        fprintf(stderr, "ERROR: NULL filepath\n");
        return NULL;
    }
    
    FILE* f = fopen(filepath, "rb");  /* "rb" = read binary */
    if (f == NULL) {
        fprintf(stderr, "ERROR: Could not open %s for reading\n", filepath);
        return NULL;
    }
    
    /* Read and verify magic number */
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, f);
    
    if (magic != NNBIN_MAGIC) {
        fprintf(stderr, "ERROR: Invalid file format. Expected magic 0x%08X, got 0x%08X\n",
                NNBIN_MAGIC, magic);
        fclose(f);
        return NULL;
    }
    
    /* Read version */
    uint32_t version;
    fread(&version, sizeof(uint32_t), 1, f);
    
    if (version != NNBIN_VERSION) {
        fprintf(stderr, "ERROR: Unsupported version %d\n", version);
        fclose(f);
        return NULL;
    }
    
    /* Read number of layers */
    uint32_t num_layers;
    fread(&num_layers, sizeof(uint32_t), 1, f);
    
    printf("Loading %d layers from %s...\n", num_layers, filepath);
    
    /* Create network */
    NeuralNetwork* net = network_create("Loaded_Network");
    
    /* Read each layer */
    for (int i = 0; i < (int)num_layers; i++) {
        uint32_t input_size, output_size, act_type;
        
        fread(&input_size, sizeof(uint32_t), 1, f);
        fread(&output_size, sizeof(uint32_t), 1, f);
        fread(&act_type, sizeof(uint32_t), 1, f);
        
        printf("  Layer %d: %d×%d, type=%d\n", i + 1, input_size, output_size, act_type);
        
        /* Create layer */
        ActivationFn activation = enum_to_activation((ActivationType)act_type);
        DenseLayer* layer = dense_layer_create((int)input_size, (int)output_size, activation);
        
        /* Read weights */
        size_t num_weights = (size_t)output_size * (size_t)input_size;
        fread(layer->weights->data, sizeof(float), num_weights, f);
        
        /* Read biases */
        fread(layer->biases->data, sizeof(float), (size_t)output_size, f);
        
        /* Add to network */
        network_add_layer(net, layer);
    }
    
    fclose(f);
    printf("✓ Loaded successfully!\n");
    
    return net;
}

/**
 * Analyze GGUF file header (read-only, doesn't load full model)
 * 
 * GGUF (GGML Unified Format) is used by llama.cpp.
 * 
 * Format structure (simplified):
 * - Magic number: "GGUF" at offset 0
 * - Version: uint32 at offset 4
 * - Tensor count: uint64 at offset 8
 * - Metadata key-value pairs follow
 * 
 * This is just a glimpse into how real model files work!
 */
void show_gguf_header_info(const char* filepath)
{
    if (filepath == NULL) {
        printf("ERROR: NULL filepath\n");
        return;
    }
    
    FILE* f = fopen(filepath, "rb");
    if (f == NULL) {
        printf("Could not open %s - file may not exist yet\n", filepath);
        printf("(GGUF analysis requires a real .gguf file to analyze)\n");
        return;
    }
    
    /* Read magic number (first 4 bytes) */
    char magic[5] = {0};
    fread(magic, 1, 4, f);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  GGUF File Header Analysis: %s\n", filepath);
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /* Check if it's actually GGUF */
    if (strncmp(magic, "GGUF", 4) != 0) {
        printf("  Not a GGUF file! Magic: %.4s (expected 'GGUF')\n", magic);
        fclose(f);
        return;
    }
    
    printf("  ✓ Magic: '%.4s' (valid GGUF)\n", magic);
    
    /* Read version */
    uint32_t version;
    fread(&version, sizeof(uint32_t), 1, f);
    printf("  ✓ Version: %u\n", version);
    
    /* Read tensor count */
    uint64_t tensor_count;
    fread(&tensor_count, sizeof(uint64_t), 1, f);
    printf("  ✓ Tensor count: %llu\n", (unsigned long long)tensor_count);
    
    /* GGUF stores metadata as key-value pairs after header
     * We won't parse them fully, just show they exist */
    printf("  ✓ (metadata and tensor data follows, %llu tensors)\n", 
           (unsigned long long)tensor_count);
    
    printf("\n  GGUF Format Features:\n");
    printf("    - Supports quantization (Q4_0, Q5_0, Q8_0, etc.)\n");
    printf("    - Memory-mapped loading for large models\n");
    printf("    - Self-contained (includes everything needed)\n");
    printf("    - Used by llama.cpp for LLaMA and other LLMs\n");
    
    printf("\n  To learn more: https://github.com/ggerganov/ggml/tree/master/examples/llama.cpp\n");
    
    fclose(f);
}