# Neural Network Inference Engine - Makefile
# ===========================================

CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -pthread
# AVX2 flag enables 256-bit SIMD instructions (8 floats at once)
# Removed for compatibility - simd_ops.c has fallback
AVX2_FLAGS = 
LDFLAGS = -lm -pthread

TARGET = nn_engine
SRCDIR = src
BUILDDIR = build

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
HEADERS = $(wildcard $(SRCDIR)/*.h)

# Test configuration
TESTDIR = tests

.PHONY: all clean test_matrix test_activations run_tests benchmark benchmark_threads benchmark_full

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -o $@ $^ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(HEADERS)
	@echo "Compiling $<..."
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -c -o $@ $< -I$(SRCDIR)

# Test targets
test_matrix: $(BUILDDIR)/matrix.o
	@echo "Building test_matrix..."
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -o $(TESTDIR)/test_matrix $(TESTDIR)/test_matrix.c $(BUILDDIR)/matrix.o -I$(SRCDIR) $(LDFLAGS)
	@echo "Running test_matrix..."
	@$(TESTDIR)/test_matrix

test_activations: $(BUILDDIR)/matrix.o $(BUILDDIR)/activations.o
	@echo "Building test_activations..."
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -o $(TESTDIR)/test_activations $(TESTDIR)/test_activations.c $(BUILDDIR)/matrix.o $(BUILDDIR)/activations.o -I$(SRCDIR) $(LDFLAGS)
	@echo "Running test_activations..."
	@$(TESTDIR)/test_activations

test_layers: $(BUILDDIR)/matrix.o $(BUILDDIR)/activations.o $(BUILDDIR)/layers.o
	@echo "Building test_layers..."
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -o $(TESTDIR)/test_layers $(TESTDIR)/test_layers.c $(BUILDDIR)/matrix.o $(BUILDDIR)/activations.o $(BUILDDIR)/layers.o -I$(SRCDIR) $(LDFLAGS)
	@echo "Running test_layers..."
	@$(TESTDIR)/test_layers

benchmark: $(BUILDDIR)/simd_ops.o
	@echo "Building SIMD benchmark..."
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -o $(TESTDIR)/benchmark_simd $(TESTDIR)/benchmark_simd.c $(BUILDDIR)/simd_ops.o -I$(SRCDIR) $(LDFLAGS)
	@echo "Running benchmark..."
	@$(TESTDIR)/benchmark_simd

benchmark_threads: $(BUILDDIR)/matrix.o $(BUILDDIR)/thread_pool.o $(BUILDDIR)/simd_ops.o
	@echo "Building threaded benchmark..."
	$(CC) $(CFLAGS) -o $(TESTDIR)/benchmark_threads $(TESTDIR)/benchmark_threads.c $(BUILDDIR)/matrix.o $(BUILDDIR)/thread_pool.o $(BUILDDIR)/simd_ops.o -I$(SRCDIR) $(LDFLAGS)
	@echo "Running benchmark..."
	@$(TESTDIR)/benchmark_threads

benchmark_full: $(BUILDDIR)/matrix.o $(BUILDDIR)/layers.o $(BUILDDIR)/network.o $(BUILDDIR)/activations.o $(BUILDDIR)/thread_pool.o $(BUILDDIR)/simd_ops.o
	@echo "Building full benchmark..."
	$(CC) $(CFLAGS) -o $(TESTDIR)/benchmark_full $(TESTDIR)/benchmark_full.c $(BUILDDIR)/matrix.o $(BUILDDIR)/layers.o $(BUILDDIR)/network.o $(BUILDDIR)/activations.o $(BUILDDIR)/thread_pool.o $(BUILDDIR)/simd_ops.o -I$(SRCDIR) $(LDFLAGS)
	@echo "Running benchmark..."
	@$(TESTDIR)/benchmark_full

run_tests: test_matrix test_activations test_layers

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILDDIR) $(TARGET)
	rm -f $(TESTDIR)/test_matrix $(TESTDIR)/test_activations $(TESTDIR)/test_layers $(TESTDIR)/benchmark_simd $(TESTDIR)/benchmark_threads $(TESTDIR)/benchmark_full