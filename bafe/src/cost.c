/* bafe/cost.c - cost model implementation
 *
 * Numbers here are deliberately simple and concrete (not tuned to any
 * specific hardware). Phase 2 will replace these with hardware-aware
 * estimates.
 */
#include "bafe/cost.h"
#include "bafe/ops.h"
#include "bafe/profiling.h"
#include <string.h>
#include <math.h>

bafe_cost_model bafe_cost_model_default(void) {
    bafe_cost_model m;
    m.alpha_flops = 1e-9;        /* 1 ns per FLOP ~ 1 GFLOP/s */
    m.beta_bytes = 1e-8;         /* 1 ns per byte ~ 1 GB/s */
    m.gamma_intermediate = 1.0;  /* flat 1 cost unit per intermediate */
    m.delta_fuse = 0.3;          /* 30% bonus for fusion */
    /* Phase 2 layout weights */
    m.epsilon_layout_conv = 2e-8;/* layout conversion: ~2x cost of a plain read */
    m.zeta_layout_fuse = 0.2;    /* layout-compatible fusion bonus */
    m.eta_contiguous = 5e-9;     /* contiguous access saves ~50% of memory cost */
    return m;
}

double bafe_cost_flops(const char *op_name, const bafe_shape *inputs, int n_inputs,
                       const bafe_op_attrs *attrs, const bafe_shape *out_shape) {
    (void)attrs;
    if (strcmp(op_name, "matmul") == 0 && n_inputs == 2) {
        /* M*N*K MACs = 2*M*N*K FLOPs */
        int32_t M = inputs[0].dims[inputs[0].rank - 2];
        int32_t K = inputs[0].dims[inputs[0].rank - 1];
        int32_t N = inputs[1].dims[inputs[1].rank - 2];
        return 2.0 * (double)M * (double)N * (double)K;
    }
    if (strcmp(op_name, "fused_matmul_relu") == 0 && n_inputs == 2) {
        int32_t M = inputs[0].dims[inputs[0].rank - 2];
        int32_t K = inputs[0].dims[inputs[0].rank - 1];
        int32_t N = inputs[1].dims[inputs[1].rank - 2];
        return 2.0 * (double)M * (double)N * (double)K;
    }
    if (strcmp(op_name, "fused_matmul_bias") == 0 && n_inputs == 3) {
        int32_t M = inputs[0].dims[inputs[0].rank - 2];
        int32_t K = inputs[0].dims[inputs[0].rank - 1];
        int32_t N = inputs[1].dims[inputs[1].rank - 2];
        return 2.0 * (double)M * (double)N * (double)K + (double)M * (double)N;
    }
    if (strcmp(op_name, "fused_matmul_bias_relu") == 0 && n_inputs == 3) {
        int32_t M = inputs[0].dims[inputs[0].rank - 2];
        int32_t K = inputs[0].dims[inputs[0].rank - 1];
        int32_t N = inputs[1].dims[inputs[1].rank - 2];
        return 2.0 * (double)M * (double)N * (double)K + (double)M * (double)N;
    }
    /* elementwise: 1 FLOP per output element */
    size_t out_n = bafe_shape_numel(out_shape);
    if (strcmp(op_name, "add") == 0 || strcmp(op_name, "sub") == 0 ||
        strcmp(op_name, "mul") == 0 || strcmp(op_name, "neg") == 0) {
        return (double)out_n;
    }
    if (strcmp(op_name, "relu") == 0 || strcmp(op_name, "sigmoid") == 0 ||
        strcmp(op_name, "tanh") == 0) {
        /* sigmoid/tanh are more expensive but we count as ~4 FLOPs/elem */
        if (strcmp(op_name, "sigmoid") == 0 || strcmp(op_name, "tanh") == 0) {
            return 4.0 * (double)out_n;
        }
        return (double)out_n;
    }
    if (strcmp(op_name, "bias_add") == 0) return (double)out_n;
    if (strcmp(op_name, "scale") == 0) return (double)out_n;
    if (strcmp(op_name, "transpose") == 0) return 0.0;  /* just a copy */
    if (strcmp(op_name, "reshape") == 0) return 0.0;
    if (strcmp(op_name, "broadcast_to") == 0) return 0.0;
    if (strcmp(op_name, "reduce_sum") == 0) {
        /* read input, write output */
        size_t in_n = bafe_shape_numel(&inputs[0]);
        return (double)in_n;
    }
    if (strcmp(op_name, "reduce_max") == 0) {
        size_t in_n = bafe_shape_numel(&inputs[0]);
        return (double)in_n;
    }
    if (strcmp(op_name, "fused_bias_relu") == 0) return 2.0 * (double)out_n;
    return 0.0;
}

double bafe_cost_bytes(const char *op_name, const bafe_shape *inputs, int n_inputs,
                       const bafe_shape *out_shape, bafe_dtype dtype) {
    size_t elem_size = bafe_dtype_byte_size(dtype);
    double read_bytes = 0.0;
    for (int i = 0; i < n_inputs; i++) {
        read_bytes += (double)bafe_shape_numel(&inputs[i]) * (double)elem_size;
    }
    double write_bytes = (double)bafe_shape_numel(out_shape) * (double)elem_size;
    /* fused ops don't write the intermediate */
    if (bafe_op_is_fused(op_name)) {
        /* for fused ops we already accounted for the right number of reads
         * (the inputs of the fused op) and a single write. No bonus here,
         * but we save an intermediate write that the unfused version would
         * have done. That's accounted in bafe_cost_node via delta_fuse. */
    }
    return read_bytes + write_bytes;
}

double bafe_cost_node(const bafe_cost_model *m, const char *op_name,
                      const bafe_shape *inputs, int n_inputs,
                      const bafe_op_attrs *attrs,
                      const bafe_shape *out_shape, bafe_dtype dtype) {
    /* backward-compatible: assume ROW_MAJOR everywhere (no layout info) */
    bafe_layout layouts[BAFE_MAX_CHILDREN];
    for (int i = 0; i < n_inputs && i < BAFE_MAX_CHILDREN; i++) {
        layouts[i] = BAFE_LAYOUT_ROW_MAJOR;
    }
    return bafe_cost_node_with_layout(m, op_name, inputs, n_inputs, attrs,
                                       out_shape, dtype,
                                       layouts, n_inputs, BAFE_LAYOUT_ROW_MAJOR);
}

double bafe_cost_layout_conversion(const bafe_cost_model *m,
                                    const char *op_name,
                                    const bafe_layout *input_layouts, int n_inputs,
                                    bafe_layout output_layout,
                                    const bafe_shape *inputs, int n_input_shapes) {
    (void)output_layout;
    /* For elementwise binary ops (add/sub/mul): if input layouts differ,
     * the codegen must convert one of them. Cost = epsilon * bytes_converted.
     * For matmul: A row-major + B row-major means we transpose B on the fly
     * (strided access), which is cache-unfriendly. Model this as a conversion.
     */
    if (n_inputs < 2) return 0.0;
    if (strcmp(op_name, "matmul") == 0 || bafe_op_is_fused(op_name)) {
        /* For matmul-family ops, the "natural" access is A row-major + B col-major.
         * If both are row-major, B is accessed with stride-N reads (cache-unfriendly).
         * Model this as a "virtual conversion" of B to col-major.
         */
        if (n_inputs >= 2 && input_layouts[0] == BAFE_LAYOUT_ROW_MAJOR
                          && input_layouts[1] == BAFE_LAYOUT_ROW_MAJOR) {
            /* cost = epsilon * bytes_of_B */
            if (n_input_shapes >= 2) {
                size_t b_bytes = bafe_shape_numel(&inputs[1]) * 4; /* assume f32 */
                return m->epsilon_layout_conv * (double)b_bytes;
            }
        }
        return 0.0;
    }
    /* elementwise: if layouts differ, conversion needed */
    if (strcmp(op_name, "add") == 0 || strcmp(op_name, "sub") == 0 ||
        strcmp(op_name, "mul") == 0 || strcmp(op_name, "bias_add") == 0) {
        bafe_layout l0 = input_layouts[0];
        for (int i = 1; i < n_inputs; i++) {
            if (input_layouts[i] != l0) {
                /* conversion needed: cost = epsilon * bytes_of_smaller_input */
                if (i < n_input_shapes) {
                    size_t bytes = bafe_shape_numel(&inputs[i]) * 4; /* assume f32 */
                    return m->epsilon_layout_conv * (double)bytes;
                }
            }
        }
    }
    return 0.0;
}

double bafe_cost_node_with_layout(const bafe_cost_model *m, const char *op_name,
                                   const bafe_shape *inputs, int n_inputs,
                                   const bafe_op_attrs *attrs,
                                   const bafe_shape *out_shape, bafe_dtype dtype,
                                   const bafe_layout *input_layouts, int n_input_layouts,
                                   bafe_layout output_layout) {
    double flops = bafe_cost_flops(op_name, inputs, n_inputs, attrs, out_shape);
    double bytes = bafe_cost_bytes(op_name, inputs, n_inputs, out_shape, dtype);
    double cost = m->alpha_flops * flops + m->beta_bytes * bytes;
    /* intermediate penalty */
    if (!bafe_op_is_fused(op_name) &&
        strcmp(op_name, "input") != 0 &&
        strcmp(op_name, "constant") != 0 &&
        strcmp(op_name, "transpose") != 0 &&
        strcmp(op_name, "reshape") != 0 &&
        strcmp(op_name, "broadcast_to") != 0 &&
        strcmp(op_name, "layout_transform") != 0) {
        cost += m->gamma_intermediate;
    }
    /* fusion bonus */
    if (bafe_op_is_fused(op_name)) {
        cost -= m->delta_fuse;
        /* Phase 2: extra bonus if all inputs share the same layout */
        if (n_input_layouts >= 2) {
            bool all_same = true;
            bafe_layout l0 = input_layouts[0];
            for (int i = 1; i < n_input_layouts; i++) {
                if (input_layouts[i] != l0) { all_same = false; break; }
            }
            if (all_same && l0 == output_layout) {
                cost -= m->zeta_layout_fuse;
            }
        }
    }
    /* Phase 2: layout conversion cost */
    cost += bafe_cost_layout_conversion(m, op_name, input_layouts, n_input_layouts,
                                         output_layout, inputs, n_inputs);
    /* Phase 2: contiguous access bonus for matmul when A is row-major
     * (the inner loop walks A contiguously, which is cache-friendly). */
    if ((strcmp(op_name, "matmul") == 0 || bafe_op_is_fused(op_name)) &&
        n_input_layouts >= 1 && input_layouts[0] == BAFE_LAYOUT_ROW_MAJOR) {
        if (n_inputs >= 1) {
            size_t a_bytes = bafe_shape_numel(&inputs[0]) * bafe_dtype_byte_size(dtype);
            cost -= m->eta_contiguous * (double)a_bytes;
        }
    }
    return cost;
}

double bafe_cost_graph(const bafe_cost_model *m, const bafe_graph *g) {
    double total = 0.0;
    for (int i = 0; i < g->n_nodes; i++) {
        const bafe_node *n = &g->nodes[i];
        if (n->is_input || n->is_constant) continue;
        bafe_shape inputs[BAFE_MAX_CHILDREN];
        bafe_layout input_layouts[BAFE_MAX_CHILDREN];
        for (int j = 0; j < n->n_children; j++) {
            inputs[j] = g->nodes[n->children[j]].shape;
            input_layouts[j] = g->nodes[n->children[j]].layout;
        }
        total += bafe_cost_node_with_layout(m, n->op_name, inputs, n->n_children,
                                              &n->attrs, &n->shape, n->dtype,
                                              input_layouts, n->n_children, n->layout);
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* Phase 3 (issue #5): Calibration                                     */
/* ------------------------------------------------------------------ */

static double _clamp_scale(double w, double avg) {
    /* If the weight is near zero, the feature has no correlation with
     * runtime — don't scale (return 1.0). */
    if (fabs(w) < avg * 0.1) return 1.0;
    double s = fabs(w) / avg;
    if (s < 0.1) s = 0.1;
    if (s > 10.0) s = 10.0;
    return s;
}

bafe_cost_model bafe_cost_model_calibrate(const bafe_cost_model *static_model,
                                           const double *learned_weights,
                                           int n_weights,
                                           double learned_bias) {
    bafe_cost_model out = static_model ? *static_model : bafe_cost_model_default();
    if (!learned_weights || n_weights < 8) return out;

    double sum_abs = 0.0;
    for (int i = 0; i < n_weights; i++) sum_abs += fabs(learned_weights[i]);
    double avg_abs = sum_abs / (double)n_weights;
    if (avg_abs < 1e-9) return out;

    /* feature[4] = log_flops -> alpha_flops */
    double s_flops = _clamp_scale(learned_weights[4], avg_abs);
    out.alpha_flops = static_model->alpha_flops * s_flops;

    /* feature[5] = log_bytes -> beta_bytes */
    double s_bytes = _clamp_scale(learned_weights[5], avg_abs);
    if (learned_weights[5] >= 0) {
        out.beta_bytes = static_model->beta_bytes * s_bytes;
    } else {
        out.beta_bytes = static_model->beta_bytes / s_bytes;
    }

    /* feature[3] = num_fused -> delta_fuse */
    double s_fused = _clamp_scale(learned_weights[3], avg_abs);
    if (learned_weights[3] < 0) {
        out.delta_fuse = static_model->delta_fuse * s_fused;
    } else {
        out.delta_fuse = static_model->delta_fuse / s_fused;
    }

    /* feature[6] = num_intermediates -> gamma_intermediate */
    double s_inter = _clamp_scale(learned_weights[6], avg_abs);
    if (learned_weights[6] >= 0) {
        out.gamma_intermediate = static_model->gamma_intermediate * s_inter;
    } else {
        out.gamma_intermediate = static_model->gamma_intermediate / s_inter;
    }

    (void)learned_bias;
    return out;
}

bafe_cost_model bafe_cost_model_calibrated_default(void) {
    bafe_cost_model stat = bafe_cost_model_default();
    const bafe_learned_cost_model *lm = bafe_profiling_get_model();
    if (!lm || !lm->valid) return stat;
    return bafe_cost_model_calibrate(&stat, lm->weights, 8, lm->bias);
}
