/**
 * @file profiling_guide.c
 * @brief Profiling Guide - How to profile your neural network engine
 * 
 * This file demonstrates how to use Linux profiling tools to find
 * performance bottlenecks in your code.
 * 
 * COMPILE WITH PROFILING:
 * ======================
 * 
 * For gprof: Add -pg flag to both CFLAGS and LDFLAGS
 *   CFLAGS = -Wall -Wextra -O2 -g -pg
 *   LDFLAGS = -lm -pg
 * 
 * Then compile and run:
 *   make clean && make CFLAGS="-Wall -Wextra -O2 -g -pg"
 *   ./nn_engine
 *   gprof nn_engine gmon.out > profile.txt
 * 
 * ============================================================================
 * 
 * GPROF OUTPUT EXAMPLE:
 * =====================
 * 
 * Looking at gprof output:
 * 
 * Each sample counts as 0.01 seconds.
 * %       cumulative    self              self     total
 * time    seconds     seconds    calls   s/call   s/call  name
 * 45.12     4.52       4.52    100000     0.00     0.00  matrix_multiply
 * 30.05     7.23       2.71    500000     0.00     0.00  simd_dot_product
 * 15.00     8.73       1.50        10     0.15     0.15  network_forward
 * 
 * KEY COLUMNS:
 * - % time: Percentage of total execution time in this function
 * - cumulative seconds: Running total up to this function
 * - self seconds: Time spent in this function (excluding children)
 * - calls: How many times this function was called
 * - self s/call: Average time per call (excluding children)
 * - total s/call: Average time per call (including children)
 * 
 * ============================================================================
 * 
 * PERF: Linux Performance Counters
 * =================================
 * 
 * Install: sudo apt install linux-tools-common linux-tools-generic
 * 
 * Basic usage:
 *   perf stat ./nn_engine
 * 
 * This shows:
 *   - Instructions per cycle (IPC)
 *   - Cache misses (L1, LLC)
 *   - Branch mispredictions
 *   - CPU migrations
 * 
 * Example output:
 *   Performance counter stats for './nn_engine':
 *     1,234,567 instructions              #    0.85  insn per cycle
 *         12,345 cycles                   #    0.00  GHz
 *          2,345 branch-misses            #    0.19% of all branches
 *          1,234 L1-dcache-load-misses    #    0.10% of all L1 dcache loads
 *        234 LLC-load-misses               #    0.02% of all LLC loads
 * 
 * Recording with perf record:
 *   perf record -g ./nn_engine
 *   perf report
 * 
 * This creates a call graph showing hot paths.
 * 
 * ============================================================================
 * 
 * VALGRIND --CALLGRIND: Cache Analysis
 * ====================================
 * 
 * Install: sudo apt install valgrind
 * 
 * Run with callgrind:
 *   valgrind --tool=callgrind ./nn_engine
 * 
 * This creates callgrind.out.* files.
 * View with kcachegrind:
 *   kcachegrind callgrind.out.12345
 * 
 * OR analyze with callgrind_annotate:
 *   callgrind_annotate callgrind.out.12345
 * 
 * KEY METRICS:
 * - Ir: Instruction reads (total instructions executed)
 * - Dr/Dw: Data reads/writes
 * - I1mr/LLmr: L1 and Last-level cache misses
 * 
 * Example analysis:
 *   Ir         Dr         Dw         I1mr        LLmr        D1mr        LLmw    Function
 *   1,234,567  123,456    98,765    100         50          200         100     matrix_multiply
 *     500,000   50,000    50,000     10          5           20          10      simd_dot_product
 * 
 * ============================================================================
 * 
 * QUICK PROFILING CHECKLIST:
 * =========================
 * 
 * 1. Start with perf stat - identify if CPU-bound or memory-bound
 *    - High IPC (>1.0) = CPU bound
 *    - High cache miss rate = memory bound
 * 
 * 2. Use perf record -g to find hot functions
 * 
 * 3. For deep analysis, use valgrind --tool=callgrind
 * 
 * 4. Focus on the hottest functions first (Pareto principle)
 *    - Fixing 20% of code often gives 80% of speedup
 * 
 * ============================================================================
 */

#include <stdio.h>

int main(void) {
    /**
     * To profile this program:
     * 
     * Method 1 - gprof:
     *   1. Compile: gcc -pg -o myprogram myprogram.c -lm
     *   2. Run: ./myprogram (this generates gmon.out)
     *   3. Analyze: gprof myprogram gmon.out
     * 
     * Method 2 - perf:
     *   1. Run: perf stat ./myprogram
     *   2. Record: perf record -g ./myprogram
     *   3. View: perf report
     * 
     * Method 3 - valgrind:
     *   1. Run: valgrind --tool=callgrind ./myprogram
     *   2. View: callgrind_annotate callgrind.out.12345
     */
    printf("Run with profiling tools to analyze performance!\n");
    return 0;
}
