/* bafe/cost.h - hardware-aware cost model
 *
 * Estimates the cost of a node in arbitrary "cost units" (lower = better).
 * Used by the extractor to pick the cheapest program from the e-graph.
 *
 * Components (Phase 1):
 *   - FLOPs estimate            (alpha * flops)
 *   - Memory traffic estimate   (beta * bytes)
 *   - Intermediate tensor cost  (gamma * n_intermediates)
 *   - Fusion bonus              (-delta * fused)
 *
 * Phase 2 (planned):
 *   - structural cost (reuse distance, fusion opportunity loss)
 *   - hardware cost (cache size, occupancy)
 *   - observed feedback (runtime measurements)
 */
#ifndef BAFE_COST_H
#define BAFE_COST_H

#include "bafe/ir.h"
#include "bafe/egraph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double alpha_flops;       /* weight per FLOP */
    double beta_bytes;        /* weight per byte of memory traffic */
    double gamma_intermediate;/* weight per materialized intermediate */
    double delta_fuse;        /* bonus subtracted for fusion */
} bafe_cost_model;

bafe_cost_model bafe_cost_model_default(void);

/* Estimate FLOPs for an op given input shapes. */
double bafe_cost_flops(const char *op_name, const bafe_shape *inputs, int n_inputs,
                       const bafe_op_attrs *attrs, const bafe_shape *out_shape);

/* Estimate memory traffic (bytes read + written) for an op. */
double bafe_cost_bytes(const char *op_name, const bafe_shape *inputs, int n_inputs,
                       const bafe_shape *out_shape, bafe_dtype dtype);

/* Total cost of a node. */
double bafe_cost_node(const bafe_cost_model *m, const char *op_name,
                      const bafe_shape *inputs, int n_inputs,
                      const bafe_op_attrs *attrs,
                      const bafe_shape *out_shape, bafe_dtype dtype);

/* Cost of an entire graph (sum over nodes). */
double bafe_cost_graph(const bafe_cost_model *m, const bafe_graph *g);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_COST_H */
