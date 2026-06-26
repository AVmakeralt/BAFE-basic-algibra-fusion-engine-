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
 * Phase 2 (added):
 *   - Layout conversion cost    (epsilon * bytes_when_layout_mismatches)
 *   - Layout-compatible fusion bonus (-zeta * when_fused_inputs_share_layout)
 *   - Cache-friendliness bonus  (eta * when_access_pattern_is_contiguous)
 */
#ifndef BAFE_COST_H
#define BAFE_COST_H

#include "bafe/ir.h"
#include "bafe/egraph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double alpha_flops;        /* weight per FLOP */
    double beta_bytes;         /* weight per byte of memory traffic */
    double gamma_intermediate; /* weight per materialized intermediate */
    double delta_fuse;         /* bonus subtracted for fusion */
    /* Phase 2 layout weights */
    double epsilon_layout_conv;/* cost per byte when an op needs layout conversion */
    double zeta_layout_fuse;   /* bonus when fused inputs share a layout */
    double eta_contiguous;     /* bonus per byte when access is contiguous (cache-friendly) */
} bafe_cost_model;

bafe_cost_model bafe_cost_model_default(void);

/* Estimate FLOPs for an op given input shapes. */
double bafe_cost_flops(const char *op_name, const bafe_shape *inputs, int n_inputs,
                       const bafe_op_attrs *attrs, const bafe_shape *out_shape);

/* Estimate memory traffic (bytes read + written) for an op. */
double bafe_cost_bytes(const char *op_name, const bafe_shape *inputs, int n_inputs,
                       const bafe_shape *out_shape, bafe_dtype dtype);

/* Phase 2: Estimate layout conversion cost for an op.
 * Returns 0 if no conversion is needed, or epsilon * bytes_converted otherwise. */
double bafe_cost_layout_conversion(const bafe_cost_model *m,
                                    const char *op_name,
                                    const bafe_layout *input_layouts, int n_inputs,
                                    bafe_layout output_layout,
                                    const bafe_shape *inputs, int n_input_shapes);

/* Total cost of a node (Phase 1 + Phase 2 layout terms). */
double bafe_cost_node(const bafe_cost_model *m, const char *op_name,
                      const bafe_shape *inputs, int n_inputs,
                      const bafe_op_attrs *attrs,
                      const bafe_shape *out_shape, bafe_dtype dtype);

/* Phase 2: Total cost of a node with layout information. */
double bafe_cost_node_with_layout(const bafe_cost_model *m, const char *op_name,
                                   const bafe_shape *inputs, int n_inputs,
                                   const bafe_op_attrs *attrs,
                                   const bafe_shape *out_shape, bafe_dtype dtype,
                                   const bafe_layout *input_layouts, int n_input_layouts,
                                   bafe_layout output_layout);

/* Cost of an entire graph (sum over nodes). */
double bafe_cost_graph(const bafe_cost_model *m, const bafe_graph *g);

/* Phase 3 (issue #5): Calibrate a cost model using a learned model.
 *
 * Takes a static cost model + a learned model (from profiling.h) and
 * produces a calibrated cost model where the per-node weights are
 * adjusted based on what the learned model discovered about actual
 * runtime correlations.
 *
 * Mapping:
 *   learned.weights[4] (log_flops)     -> scales alpha_flops
 *   learned.weights[5] (log_bytes)     -> scales beta_bytes
 *   learned.weights[3] (num_fused)     -> scales delta_fuse
 *   learned.weights[6] (num_intermediates) -> scales gamma_intermediate
 *
 * The scaling is relative to the learned model's average weight, so
 * features that matter more than average get amplified and vice versa.
 *
 * If the learned model is not valid, returns a copy of the static model.
 */
bafe_cost_model bafe_cost_model_calibrate(const bafe_cost_model *static_model,
                                           const double *learned_weights,
                                           int n_weights,
                                           double learned_bias);

/* Convenience: calibrate from the global learned model (from profiling.h).
 * Returns a calibrated cost model, or the default if no learned model. */
bafe_cost_model bafe_cost_model_calibrated_default(void);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_COST_H */
