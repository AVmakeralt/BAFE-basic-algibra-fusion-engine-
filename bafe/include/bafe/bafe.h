/* bafe/bafe.h - top-level BAFE API
 *
 * The full optimization pipeline:
 *   1. Build IR graph
 *   2. Find rewrite alternatives
 *   3. Import graph into e-graph
 *   4. Apply alternatives to e-graph
 *   5. Rebuild e-graph (congruence closure)
 *   6. Run cost model + extractor
 *   7. Build optimized graph from extraction
 *   8. JIT compile and return function pointer
 */
#ifndef BAFE_BAFE_H
#define BAFE_BAFE_H

#include "bafe/ir.h"
#include "bafe/egraph.h"
#include "bafe/cost.h"
#include "bafe/extract.h"
#include "bafe/jit.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Run the full optimization pipeline on a graph.
 *
 * - `optimize` (out): the optimized graph (caller-allocated, will be init'd)
 * - Returns 0 on success, non-zero on error (message in err_buf).
 */
int bafe_optimize(const bafe_graph *input, bafe_graph *optimized,
                  char *err_buf, size_t err_buf_size);

/* Optimize + JIT compile in one shot.
 * Returns a function pointer to the compiled kernel.
 */
bafe_kernel_fn bafe_optimize_and_compile(const bafe_graph *input,
                                          char *err_buf, size_t err_buf_size);

/* Print a human-readable optimization report to stdout.
 * Useful for debugging and demos. */
void bafe_optimize_debug(const bafe_graph *input);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_BAFE_H */
