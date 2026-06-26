/* bafe/jit.h - JIT cache: compile, dlopen, dispatch
 *
 * The JIT layer:
 *   1. Hashes (graph structure + shapes) -> SHA-256
 *   2. Looks up in cache; on hit, returns the cached function pointer
 *   3. On miss:
 *      - emit C source
 *      - compile with `cc -shared -fPIC -O2 -o <cache>.so -`
 *      - dlopen() the .so
 *      - dlsym() the kernel function
 *      - cache the function pointer
 *
 * Cache location: $BAFE_CACHE_DIR (default: .bafecache under cwd)
 */
#ifndef BAFE_JIT_H
#define BAFE_JIT_H

#include "bafe/ir.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function pointer type for emitted kernels.
 *
 * The actual signature depends on the graph's input/output types, e.g.:
 *   void bafe_kernel(const float *A, const float *B, float *out)
 *
 * Callers must cast this to the correct signature before invoking.
 * For convenience, the Python binding handles this automatically via
 * ctypes; C callers should define their own typedef matching the graph.
 */
typedef void (*bafe_kernel_fn)(void);

/* Compute the SHA-256 of the graph (canonicalized form) as a hex string.
 * Writes 64 hex chars + null terminator into out (which must be >= 65 bytes). */
int bafe_jit_hash_graph(const bafe_graph *g, char *out, size_t out_size);

/* JIT-compile the graph and return a function pointer.
 * On cache hit, returns the cached pointer.
 * On miss, emits C, compiles, dlopen, caches.
 * Returns NULL on error and writes an error message to err_buf. */
bafe_kernel_fn bafe_jit_get_or_compile(const bafe_graph *g,
                                        char *err_buf, size_t err_buf_size);

/* Set the cache directory (overrides BAFE_CACHE_DIR env var).
 * Must be called before any get_or_compile. */
void bafe_jit_set_cache_dir(const char *dir);

/* Get the current cache directory. */
const char *bafe_jit_cache_dir(void);

/* Stats: number of cache hits, misses, compiles. */
typedef struct {
    int hits;
    int misses;
    int compiles;
    int compile_failures;
} bafe_jit_stats;

bafe_jit_stats bafe_jit_get_stats(void);

/* For testing: clear all caches. */
void bafe_jit_clear(void);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_JIT_H */
