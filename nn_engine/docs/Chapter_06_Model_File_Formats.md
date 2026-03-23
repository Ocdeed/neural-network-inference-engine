# Chapter 6: Model File Formats

## 6.1 Why Model Files Exist — Separating Training from Inference

Training a neural network takes huge computational resources:
- Thousands of dollars in cloud GPU time
- Days or weeks of computation
- Large datasets that must be curated

But inference (using the model) should be:
- Fast (milliseconds per prediction)
- Cheap (can run on CPU or edge device)
- Independent of training

**Model files** store the trained weights so inference can happen anywhere, anytime, without retraining.

```
┌─────────────────┐     Save weights     ┌─────────────────┐
│   Training      │ ──────────────────▶  │   Inference     │
│   (GPU cluster) │    to file           │   (any device)  │
└─────────────────┘                      └─────────────────┘
        │                                      │
        │  - Weeks of training                 │  - Fast
        │  - Large dataset                     │  - Lightweight
        │  - Expensive                         │  - Just forward pass
```

---

## 6.2 Binary vs Text File Formats

### Text Format (JSON, CSV)

```json
{
  "weights": [
    0.1234, -0.5678, 0.9012, ...
  ]
}
```

**Pros:** Human readable, easy to debug
**Cons:** Large (each digit is a character!), slow to parse

### Binary Format

```
0x3F 0x9E 0xE6 0xB6  (raw float32 bytes)
```

**Pros:** Compact (4 bytes per float), fast to read, exact precision
**Cons:** Not human readable, need to know the format

For model weights: **binary is the standard choice**

---

## 6.3 Our Custom .nnbin Format — Design Decisions

We designed a simple format to understand the principles:

```
┌─────────────────────────────────────────────────────────────┐
│                      .nnbin File Format                     │
├─────────────────────────────────────────────────────────────┤
│ Offset │ Size    │ Field              │ Description        │
├────────┼─────────┼────────────────────┼────────────────────┤
│ 0      │ 4 bytes │ magic              │ 0x4E4E4249 ("NNBI")│
│ 4      │ 4 bytes │ version            │ 1                  │
│ 8      │ 4 bytes │ num_layers         │ Number of layers   │
│ 12     │ ...     │ Layer 1 data       │ (see below)        │
│ ...    │ ...     │ Layer 2 data       │                    │
│ ...    │ ...     │ Layer 3 data       │                    │
└─────────────────────────────────────────────────────────────┘

Per-layer data:
  [4 bytes] input_size
  [4 bytes] output_size
  [4 bytes] activation_type
  [output*input*4 bytes] weights (float32)
  [output*4 bytes] biases (float32)
```

### Design Decisions

1. **Magic number**: Identifies file type, prevents loading wrong format
2. **Version**: Allows future format changes without breaking old code
3. **Fixed-size integers**: uint32_t ensures consistent 4-byte size
4. **No compression**: Simple to implement, fast to load

---

## 6.4 fread and fwrite — Binary I/O in C

### Writing (fwrite)

```c
float weights[4] = {0.1, 0.2, 0.3, 0.4};
FILE* f = fopen("model.bin", "wb");  // "wb" = write binary
fwrite(weights, sizeof(float), 4, f);  // Write 4 floats
fclose(f);
```

This writes 16 bytes directly to disk:
```
0x3DCCCCCD 0x3ECCCCCD 0x3F19999A 0x3F666666
```

### Reading (fread)

```c
float weights[4];
FILE* f = fopen("model.bin", "rb");  // "rb" = read binary
fread(weights, sizeof(float), 4, f);  // Read 4 floats
fclose(f);
```

### Key Differences

| Function | Purpose | Data Type |
|----------|---------|-----------|
| fprintf | Text output | Converts to string |
| fscanf | Text input | Parses from string |
| fwrite | Binary output | Raw bytes |
| fread | Binary input | Raw bytes |

---

## 6.5 Endianness — The Hidden Gotcha

**Endianness** = byte order in multi-byte values.

### Little Endian (most common on x86, ARM)

```
Value 0x12345678 stored as: 78 56 34 12
                            ↑
                          Address 0
```

### Big Endian (network order, some ARM)

```
Value 0x12345678 stored as: 12 34 56 78
                            ↑
                          Address 0
```

### Why This Matters

If you save on a little-endian machine and load on big-endian:
```
Saved: 0x3F800000 (1.0 in float32)
Loaded: 0x0000803F (completely wrong!)
```

### The Fix

Real formats like GGUF specify endianness. Most modern systems are little-endian, so this usually "just works", but it's a known issue for portable formats.

---

## 6.6 The GGUF Format — How llama.cpp Does It

**GGUF** (GGML Unified Format) is used by llama.cpp for large language models.

### Header Structure

```
Offset 0-3:   "GGUF" (magic)
Offset 4-7:   version (uint32)
Offset 8-15:  tensor_count (uint64)
Offset 16+:   metadata (key-value pairs)
              tensor data
```

### Key Features

1. **Magic number**: "GGUF" identifies format
2. **Version**: Allows format evolution
3. **Tensor count**: Tells how many weight matrices to expect
4. **Metadata**: Arbitrary key-value data (tokenizer, architecture, etc.)
5. **Quantization support**: Store weights in smaller formats

### Real GGUF Files

When you download a Llama model (~4GB), it's GGUF format. The header contains:
- Model type ("llama", "mistral", etc.)
- Vocabulary size
- Embedding dimension
- Number of layers
- And much more!

---

## 6.7 Quantization — Shrinking Models Without Losing Too Much

**Quantization** = reducing precision of weights to save space.

### Precision Options

| Format | Bits | Size | Quality |
|--------|------|------|---------|
| float32 | 32 | 100% | Perfect |
| float16 | 16 | 50% | Near perfect |
| int8 | 8 | 25% | Good |
| int4 | 4 | 12.5% | Acceptable |

### How It Works

```
float32: 0.12345678 (32 bits)
int8:    Round to 0-255 → ~0.12 (8 bits)
```

### Why LLaMA Uses It

A 7B parameter model in float32 = 28 GB!
Using int4 quantization = ~3.5 GB - fits in RAM!

llama.cpp supports: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and more.

---

## 6.8 Memory-Mapped I/O — The Fast Way to Load Large Files

### The Problem

Loading a 4GB model into RAM takes time:
```
Read 4GB from disk → Copy to RAM → Process
```

### Memory Mapping (mmap)

Instead of copying:
1. Ask OS to map file to virtual memory
2. Access memory normally - OS loads pages on-demand
3. Don't load what you don't use!

```
┌─────────────────────────────────────────┐
│  RAM                                    │
│  ┌─────────────────────────────────┐    │
│  │ Page tables point to file on    │    │
│  │ disk, not RAM!                  │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
        ↓ (lazy load)
┌─────────────────────────────────────────┐
│  DISK (model.gguf)                       │
└─────────────────────────────────────────┘
```

### Benefits

- **Fast startup**: Don't load everything, just what's accessed
- **Low memory**: Unused parts stay on disk
- **Shared**: Multiple processes can share same file

llama.cpp uses mmap to load huge models efficiently!

---

## What Just Happened

- **Created .nnbin format**: Binary format with magic, version, layers, weights, biases
- **Implemented fwrite/fread**: Raw binary I/O instead of text
- **Added save/load**: network_save() and network_load() work correctly
- **Analyzed GGUF headers**: show_gguf_header_info() shows magic, version, tensor count
- **Learned about quantization**: How llama.cpp shrinks 7B models to 4GB or less
- **Understood mmap**: Memory-mapped I/O loads large files without eating RAM