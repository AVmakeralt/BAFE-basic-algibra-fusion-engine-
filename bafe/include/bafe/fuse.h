/* bafe/fuse.h - cross-kernel fusion (Phase 3, issue #7)
 *
 * When two jitted functions are always called in sequence (f's output
 * feeds g's input), BAFE can compile a single fused kernel that avoids
 * materializing the intermediate tensor.
 *
 * Pipeline:
 *   1. Concatenate the two optimized graphs (g's input is rewired to
 *      consume f's output node directly)
 *   2. Run the normal optimize + JIT pipeline on the fused graph
 *   3. The cost model gives a fusion bonus (no intermediate write)
 *   4. Cache the fused kernel keyed by (f_hash, g_hash)
 *
 * The fused kernel takes f's inputs and produces g's output.
 */
#ifndef BAFE_FUSE_H
#define BAFE_FUSE_H

#include "bafe/ir.h"
#include "bafe/jit.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Concatenate two graphs: graph_a's output becomes an internal node
 * that feeds graph_b's first input.
 *
 * The result graph has:
 *   - inputs: graph_a's inputs + graph_b's inputs[1:]
 *   - output: graph_b's output (with its first input rewired to graph_a's output)
 *
 * Returns 0 on success, non-zero on error.
 *
 * The caller passes two graphs (already optimized) and receives the
 * concatenated graph in `out`.
 */
int bafe_fuse_concat(const bafe_graph *graph_a, const bafe_graph *graph_b,
                     bafe_graph *out, char *err_buf, size_t err_buf_size);

/* Optimize + JIT compile a fused graph from two graphs.
 * Returns the function pointer, or NULL on error. */
bafe_kernel_fn bafe_fuse_compile(const bafe_graph *graph_a,
                                  const bafe_graph *graph_b,
                                  char *err_buf, size_t err_buf_size);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_FUSE_H */
