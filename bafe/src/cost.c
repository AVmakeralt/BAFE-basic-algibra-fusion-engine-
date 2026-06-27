/* bafe/cost.c - roofline-based cost model implementation
 *
 * The cost of a node is:
 *
 *   cost = max(flops / peak_flops, bytes / effective_bandwidth)
 *        + gamma_intermediate        (if materialized)
 *        - delta_fuse                (if fused)
 *        + layout_conversion_cost
 *        - contiguous_access_bonus
 *        - simd_vectorization_bonus
 *
 * The effective_bandwidth depends on which cache level the working set
 * fits in (L1 > L2 > L3 > DRAM).
 */
#include "bafe/cost.h"
#include "bafe/ops.h"
#include "bafe/profiling.h"
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Hardware models                                                     */
/* ------------------------------------------------------------------ */

bafe_hardware_model bafe_hardware_model_default(void) {
    bafe_hardware_model h;
    /* Modern x86-64 desktop (Skylake-class, AVX2) */
    h.peak_gflops = 64.0;       /* 8 doubles * 2 (FMA) * 4 GHz = ~64 GFLOPS */
    h.clock_ghz = 4.0;
    h.simd_width = 8;           /* AVX2: 8 F32 lanes */
    h.l1_cache_size = 32768;    /* 32 KB */
    h.l2_cache_size = 262144;   /* 256 KB */
    h.l3_cache_size = 8388608;  /* 8 MB */
    h.l1_bandwidth_gbs = 1000.0;
    h.l2_bandwidth_gbs = 500.0;
    h.l3_bandwidth_gbs = 200.0;
    h.dram_bandwidth_gbs = 50.0;
    h.f32_flop_rate = 1.0;
    h.f64_flop_rate = 0.5;
    h.f16_flop_rate = 2.0;
    h.bf16_flop_rate = 2.0;
    h.i32_flop_rate = 1.0;
    h.i64_flop_rate = 0.5;
    return h;
}

bafe_hardware_model bafe_hardware_model_server(void) {
    bafe_hardware_model h;
    /* High-end server CPU (Ice Lake, AVX-512) */
    h.peak_gflops = 128.0;      /* 16 F32 * 2 (FMA) * 4 GHz */
    h.clock_ghz = 4.0;
    h.simd_width = 16;          /* AVX-512: 16 F32 lanes */
    h.l1_cache_size = 49152;    /* 48 KB */
    h.l2_cache_size = 1048576;  /* 1 MB */
    h.l3_cache_size = 33554432; /* 32 MB */
    h.l1_bandwidth_gbs = 2000.0;
    h.l2_bandwidth_gbs = 800.0;
    h.l3_bandwidth_gbs = 300.0;
    h.dram_bandwidth_gbs = 100.0;
    h.f32_flop_rate = 1.0;
    h.f64_flop_rate = 0.5;
    h.f16_flop_rate = 2.0;
    h.bf16_flop_rate = 2.0;     /* AMX gives higher but we're conservative */
    h.i32_flop_rate = 1.0;
    h.i64_flop_rate = 0.5;
    return h;
}

/* ------------------------------------------------------------------ */
/* Cost model from hardware                                            */
/* ------------------------------------------------------------------ */

bafe_cost_model bafe_cost_model_from_hardware(const bafe_hardware_model *hw) {
    bafe_cost_model m;
    if (!hw) hw = &(bafe_hardware_model){0}; /* fallback — but we always pass non-null */

    /* Roofline: alpha = 1 / peak_flops, beta = 1 / peak_bandwidth */
    double peak_flops_per_sec = hw->peak_gflops * 1e9;
    double peak_bytes_per_sec = hw->dram_bandwidth_gbs * 1e9;

    m.alpha_flops = 1.0 / peak_flops_per_sec;     /* cost per FLOP */
    m.beta_bytes = 1.0 / peak_bytes_per_sec;       /* cost per byte (DRAM baseline) */
    m.gamma_intermediate = 1.0;
    m.delta_fuse = 0.3;

    /* Layout weights */
    m.epsilon_layout_conv = 2.0 / peak_bytes_per_sec;  /* layout conv = 2x read */
    m.zeta_layout_fuse = 0.2;
    m.eta_contiguous = 0.5 / peak_bytes_per_sec;       /* contiguous = 50% bandwidth bonus */

    /* Cache hierarchy thresholds (in bytes) */
    m.l1_threshold = (double)hw->l1_cache_size;
    m.l2_threshold = (double)hw->l2_cache_size;
    m.l3_threshold = (double)hw->l3_cache_size;

    /* Bandwidth weights: higher = faster (lower cost per byte) */
    m.l1_bw_weight = hw->l1_bandwidth_gbs / hw->dram_bandwidth_gbs;   /* ~20x */
    m.l2_bw_weight = hw->l2_bandwidth_gbs / hw->dram_bandwidth_gbs;   /* ~10x */
    m.l3_bw_weight = hw->l3_bandwidth_gbs / hw->dram_bandwidth_gbs;   /* ~4x */
    m.dram_bw_weight = 1.0;

    /* SIMD: bonus per element when the op is vectorizable */
    m.simd_bonus = 0.1 / hw->simd_width;  /* each SIMD lane saves ~10% of scalar cost */
    m.simd_width = hw->simd_width;

    return m;
}

bafe_cost_model bafe_cost_model_default(void) {
    bafe_hardware_model hw = bafe_hardware_model_default();
    return bafe_cost_model_from_hardware(&hw);
}

/* ------------------------------------------------------------------ */
/* FLOPs estimation                                                    */
/* ------------------------------------------------------------------ */

double bafe_cost_flops(const char *op_name, const bafe_shape *inputs, int n_inputs,
                       const bafe_op_attrs *attrs, const bafe_shape *out_shape) {
    (void)attrs;
    if (strcmp(op_name, "matmul") == 0 && n_inputs == 2) {
        int32_t M = inputs[0].dims[inputs[0].rank - 2];
        int32_t K = inputs[0].dims[inputs[0].rank - 1];
        int32_t N = inputs[1].dims[inputs[1].rank - 2];
        /* batch FLOPs */
        int32_t batch = 1;
        for (int i = 0; i < inputs[0].rank - 2; i++) batch *= inputs[0].dims[i];
        return 2.0 * (double)M * (double)N * (double)K * (double)batch;
    }
    if (strcmp(op_name, "fused_matmul_relu") == 0 && n_inputs == 2) {
        return bafe_cost_flops("matmul", inputs, n_inputs, attrs, out_shape);
    }
    if (strcmp(op_name, "fused_matmul_bias") == 0 && n_inputs == 3) {
        double mm_flops = bafe_cost_flops("matmul", inputs, 2, attrs, out_shape);
        return mm_flops + (double)bafe_shape_numel(out_shape);
    }
    if (strcmp(op_name, "fused_matmul_bias_relu") == 0 && n_inputs == 3) {
        double mm_flops = bafe_cost_flops("matmul", inputs, 2, attrs, out_shape);
        return mm_flops + (double)bafe_shape_numel(out_shape);
    }
    size_t out_n = bafe_shape_numel(out_shape);
    if (strcmp(op_name, "add") == 0 || strcmp(op_name, "sub") == 0 ||
        strcmp(op_name, "mul") == 0 || strcmp(op_name, "neg") == 0) {
        return (double)out_n;
    }
    if (strcmp(op_name, "relu") == 0) return (double)out_n;
    if (strcmp(op_name, "sigmoid") == 0 || strcmp(op_name, "tanh") == 0) {
        return 4.0 * (double)out_n;  /* transcendental = ~4 FLOPs */
    }
    if (strcmp(op_name, "bias_add") == 0 || strcmp(op_name, "scale") == 0) {
        return (double)out_n;
    }
    if (strcmp(op_name, "transpose") == 0 || strcmp(op_name, "reshape") == 0 ||
        strcmp(op_name, "broadcast_to") == 0 || strcmp(op_name, "layout_transform") == 0) {
        return 0.0;
    }
    if (strcmp(op_name, "reduce_sum") == 0 || strcmp(op_name, "reduce_max") == 0) {
        return (double)bafe_shape_numel(&inputs[0]);
    }
    if (strcmp(op_name, "fused_bias_relu") == 0) return 2.0 * (double)out_n;
    return 0.0;
}

/* ------------------------------------------------------------------ */
/* Memory traffic estimation                                           */
/* ------------------------------------------------------------------ */

double bafe_cost_bytes(const char *op_name, const bafe_shape *inputs, int n_inputs,
                       const bafe_shape *out_shape, bafe_dtype dtype) {
    size_t elem_size = bafe_dtype_byte_size(dtype);
    double read_bytes = 0.0;
    for (int i = 0; i < n_inputs; i++) {
        read_bytes += (double)bafe_shape_numel(&inputs[i]) * (double)elem_size;
    }
    double write_bytes = (double)bafe_shape_numel(out_shape) * (double)elem_size;
    return read_bytes + write_bytes;
}

/* ------------------------------------------------------------------ */
/* Arithmetic intensity + cache level                                  */
/* ------------------------------------------------------------------ */

double bafe_cost_arithmetic_intensity(const char *op_name,
                                       const bafe_shape *inputs, int n_inputs,
                                       const bafe_op_attrs *attrs,
                                       const bafe_shape *out_shape, bafe_dtype dtype) {
    double flops = bafe_cost_flops(op_name, inputs, n_inputs, attrs, out_shape);
    double bytes = bafe_cost_bytes(op_name, inputs, n_inputs, out_shape, dtype);
    if (bytes < 1.0) return 0.0;
    return flops / bytes;
}

int bafe_cost_cache_level(size_t working_set_bytes, const bafe_cost_model *m) {
    if ((double)working_set_bytes <= m->l1_threshold) return 0;  /* L1 */
    if ((double)working_set_bytes <= m->l2_threshold) return 1;  /* L2 */
    if ((double)working_set_bytes <= m->l3_threshold) return 2;  /* L3 */
    return 3;  /* DRAM */
}

double bafe_cost_effective_bandwidth(size_t working_set_bytes, const bafe_cost_model *m) {
    int level = bafe_cost_cache_level(working_set_bytes, m);
    switch (level) {
        case 0: return m->l1_bw_weight;
        case 1: return m->l2_bw_weight;
        case 2: return m->l3_bw_weight;
        default: return m->dram_bw_weight;
    }
}

bool bafe_cost_is_vectorizable(const char *op_name) {
    if (!op_name) return false;
    /* Elementwise ops on contiguous data are vectorizable */
    if (strcmp(op_name, "add") == 0 || strcmp(op_name, "sub") == 0 ||
        strcmp(op_name, "mul") == 0 || strcmp(op_name, "relu") == 0 ||
        strcmp(op_name, "sigmoid") == 0 || strcmp(op_name, "tanh") == 0 ||
        strcmp(op_name, "neg") == 0 || strcmp(op_name, "bias_add") == 0 ||
        strcmp(op_name, "scale") == 0 || strcmp(op_name, "fused_bias_relu") == 0) {
        return true;
    }
    return false;
}

/* ------------------------------------------------------------------ */
/* Layout conversion cost                                              */
/* ------------------------------------------------------------------ */

double bafe_cost_layout_conversion(const bafe_cost_model *m,
                                    const char *op_name,
                                    const bafe_layout *input_layouts, int n_inputs,
                                    bafe_layout output_layout,
                                    const bafe_shape *inputs, int n_input_shapes) {
    (void)output_layout;
    const size_t elem_size = 4;
    if (n_inputs < 2) return 0.0;
    if (strcmp(op_name, "matmul") == 0 || bafe_op_is_fused(op_name)) {
        if (n_inputs >= 2 && input_layouts[0] == BAFE_LAYOUT_ROW_MAJOR
                          && input_layouts[1] == BAFE_LAYOUT_ROW_MAJOR) {
            if (n_input_shapes >= 2) {
                size_t b_bytes = bafe_shape_numel(&inputs[1]) * elem_size;
                return m->epsilon_layout_conv * (double)b_bytes;
            }
        }
        return 0.0;
    }
    if (strcmp(op_name, "add") == 0 || strcmp(op_name, "sub") == 0 ||
        strcmp(op_name, "mul") == 0 || strcmp(op_name, "bias_add") == 0) {
        bafe_layout l0 = input_layouts[0];
        for (int i = 1; i < n_inputs; i++) {
            if (input_layouts[i] != l0) {
                if (i < n_input_shapes) {
                    size_t bytes = bafe_shape_numel(&inputs[i]) * elem_size;
                    return m->epsilon_layout_conv * (double)bytes;
                }
            }
        }
    }
    return 0.0;
}

/* ------------------------------------------------------------------ */
/* Dtype-specific FLOP rate                                            */
/* ------------------------------------------------------------------ */

static double _dtype_flop_rate(bafe_dtype dtype) {
    switch (dtype) {
        case BAFE_DTYPE_F32: return 1.0;
        case BAFE_DTYPE_F64: return 0.5;
        case BAFE_DTYPE_F16: return 2.0;
        case BAFE_DTYPE_BF16: return 2.0;
        case BAFE_DTYPE_I32: return 1.0;
        case BAFE_DTYPE_I64: return 0.5;
        default: return 1.0;
    }
}

/* ------------------------------------------------------------------ */
/* Main cost computation (roofline model)                              */
/* ------------------------------------------------------------------ */

double bafe_cost_node_with_layout(const bafe_cost_model *m, const char *op_name,
                                   const bafe_shape *inputs, int n_inputs,
                                   const bafe_op_attrs *attrs,
                                   const bafe_shape *out_shape, bafe_dtype dtype,
                                   const bafe_layout *input_layouts, int n_input_layouts,
                                   bafe_layout output_layout) {
    double flops = bafe_cost_flops(op_name, inputs, n_inputs, attrs, out_shape);
    double bytes = bafe_cost_bytes(op_name, inputs, n_inputs, out_shape, dtype);

    /* Adjust FLOP cost by dtype rate (F16 is 2x faster, F64 is 0.5x) */
    double flop_rate = _dtype_flop_rate(dtype);
    double compute_cost = (flops / flop_rate) * m->alpha_flops;

    /* Determine effective bandwidth based on working set size */
    size_t working_set = 0;
    for (int i = 0; i < n_inputs; i++) {
        working_set += bafe_shape_nbytes(&inputs[i], dtype);
    }
    working_set += bafe_shape_nbytes(out_shape, dtype);
    double bw_multiplier = bafe_cost_effective_bandwidth(working_set, m);
    double memory_cost = bytes * m->beta_bytes / bw_multiplier;

    /* Roofline: cost = max(compute, memory) + penalties */
    double cost = compute_cost > memory_cost ? compute_cost : memory_cost;

    /* Intermediate materialization penalty */
    if (!bafe_op_is_fused(op_name) &&
        strcmp(op_name, "input") != 0 &&
        strcmp(op_name, "constant") != 0 &&
        strcmp(op_name, "transpose") != 0 &&
        strcmp(op_name, "reshape") != 0 &&
        strcmp(op_name, "broadcast_to") != 0 &&
        strcmp(op_name, "layout_transform") != 0) {
        cost += m->gamma_intermediate;
    }

    /* Fusion bonus */
    if (bafe_op_is_fused(op_name)) {
        cost -= m->delta_fuse;
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

    /* Layout conversion cost */
    cost += bafe_cost_layout_conversion(m, op_name, input_layouts, n_input_layouts,
                                         output_layout, inputs, n_inputs);

    /* Contiguous access bonus for matmul (A row-major = K contiguous) */
    if ((strcmp(op_name, "matmul") == 0 || bafe_op_is_fused(op_name)) &&
        n_input_layouts >= 1 && input_layouts[0] == BAFE_LAYOUT_ROW_MAJOR) {
        if (n_inputs >= 1) {
            size_t a_bytes = bafe_shape_numel(&inputs[0]) * bafe_dtype_byte_size(dtype);
            cost -= m->eta_contiguous * (double)a_bytes;
        }
    }

    /* SIMD vectorization bonus for elementwise ops */
    if (bafe_cost_is_vectorizable(op_name) && m->simd_width > 1) {
        size_t out_n = bafe_shape_numel(out_shape);
        /* Each SIMD instruction processes simd_width elements, saving
         * (simd_width - 1) / simd_width of the scalar cost */
        double simd_savings = (double)(m->simd_width - 1) / (double)m->simd_width;
        cost -= m->simd_bonus * (double)out_n * simd_savings;
    }

    return cost;
}

double bafe_cost_node(const bafe_cost_model *m, const char *op_name,
                      const bafe_shape *inputs, int n_inputs,
                      const bafe_op_attrs *attrs,
                      const bafe_shape *out_shape, bafe_dtype dtype) {
    bafe_layout layouts[BAFE_MAX_CHILDREN];
    for (int i = 0; i < n_inputs && i < BAFE_MAX_CHILDREN; i++) {
        layouts[i] = BAFE_LAYOUT_ROW_MAJOR;
    }
    return bafe_cost_node_with_layout(m, op_name, inputs, n_inputs, attrs,
                                       out_shape, dtype,
                                       layouts, n_inputs, BAFE_LAYOUT_ROW_MAJOR);
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
/* Calibration                                                         */
/* ------------------------------------------------------------------ */

static double _clamp_scale(double w, double avg) {
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

    double s_flops = _clamp_scale(learned_weights[4], avg_abs);
    out.alpha_flops = static_model->alpha_flops * s_flops;

    double s_bytes = _clamp_scale(learned_weights[5], avg_abs);
    if (learned_weights[5] >= 0) {
        out.beta_bytes = static_model->beta_bytes * s_bytes;
    } else {
        out.beta_bytes = static_model->beta_bytes / s_bytes;
    }

    double s_fused = _clamp_scale(learned_weights[3], avg_abs);
    if (learned_weights[3] < 0) {
        out.delta_fuse = static_model->delta_fuse * s_fused;
    } else {
        out.delta_fuse = static_model->delta_fuse / s_fused;
    }

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
