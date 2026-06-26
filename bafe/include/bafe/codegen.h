/* bafe/codegen.h - emit C99 source code for a BAFE graph
 *
 * The codegen walks the graph in topological order and emits a C99
 * function `void bafe_kernel(<inputs>..., <output>)` that performs the
 * computation using plain nested loops.
 *
 * The emitted code uses:
 *   - row-major layout (C order)
 *   - direct indexing: ptr[i * stride + j]
 *   - tiled matmul (with a fixed tile size for Phase 1)
 *   - fused kernels where the op is a fused_* variant
 */
#ifndef BAFE_CODEGEN_H
#define BAFE_CODEGEN_H

#include "bafe/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Emit a C99 source string for the graph.
 *
 * The function signature is:
 *   void bafe_kernel(const float* A, const float* B, ..., float* out)
 * Inputs appear in graph.inputs order; output appears last.
 *
 * `kernel_name` is the name of the emitted function (e.g. "bafe_kernel_<hash>").
 *
 * Returns the number of bytes written (excluding the null terminator),
 * or a negative value on error.
 */
int bafe_codegen_emit(const bafe_graph *g, const char *kernel_name,
                      char *out, size_t out_size);

/* Convenience: emit to a freshly malloc'd buffer (caller frees). */
char *bafe_codegen_emit_alloc(const bafe_graph *g, const char *kernel_name);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_CODEGEN_H */
