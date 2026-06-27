/* bafe/cost.h - hardware-aware cost model based on the roofline model
 *
 *   runtime = max(flops / peak_flops, bytes / peak_bandwidth)
 *
 * Where:
 *   - peak_flops depends on SIMD width, clock speed, and dtype
 *   - peak_bandwidth depends on which cache level the data fits in
 *
 * Additional terms:
 *   - Intermediate materialization penalty (gamma * n_intermediates)
 *   - Fusion bonus (-delta * fused) — saves a write + read
 *   - Layout conversion cost (epsilon * bytes_when_layout_mismatches)
 *   - Cache-friendliness bonus (eta * when_access_pattern_is_contiguous)
 *   - SIMD vectorization bonus (simd_bonus * vectorizable_elements)
 */
#ifndef BAFE_COST_H
#define BAFE_COST_H

#include "bafe/ir.h"
#include "bafe/egraph.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Hardware model: describes the target CPU. */
typedef struct {
    double peak_gflops;        /* peak FLOP/s (e.g. 64.0 for AVX2 FMA) */
    double clock_ghz;
    int    simd_width;         /* SIMD lanes (8 for AVX2 F32, 16 for AVX-512) */
    int    l1_cache_size;      /* bytes */
    int    l2_cache_size;
    int    l3_cache_size;
    double l1_bandwidth_gbs;
    double l2_bandwidth_gbs;
    double l3_bandwidth_gbs;
    double dram_bandwidth_gbs;
    double f32_flop_rate;      /* relative to peak (1.0) */
    double f64_flop_rate;      /* 0.5 */
    double f16_flop_rate;      /* 2.0 */
    double bf16_flop_rate;     /* 2.0 */
    double i32_flop_rate;
    double i64_flop_rate;
} bafe_hardware_model;

bafe_hardware_model bafe_hardware_model_default(void);
bafe_hardware_model bafe_hardware_model_server(void);

typedef struct {
    double alpha_flops;
    double beta_bytes;
    double gamma_intermediate;
    double delta_fuse;
    double epsilon_layout_conv;
    double zeta_layout_fuse;
    double eta_contiguous;
    /* Roofline: cache-level bandwidth weights */
    double l1_threshold;
    double l2_threshold;
    double l3_threshold;
    double l1_bw_weight;
    double l2_bw_weight;
    double l3_bw_weight;
    double dram_bw_weight;
    /* SIMD */
    double simd_bonus;
    int    simd_width;
} bafe_cost_model;

bafe_cost_model bafe_cost_model_default(void);
bafe_cost_model bafe_cost_model_from_hardware(const bafe_hardware_model *hw);

double bafe_cost_flops(const char *op_name, const bafe_shape *inputs, int n_inputs,
                       const bafe_op_attrs *attrs, const bafe_shape *out_shape);
double bafe_cost_bytes(const char *op_name, const bafe_shape *inputs, int n_inputs,
                       const bafe_shape *out_shape, bafe_dtype dtype);
double bafe_cost_layout_conversion(const bafe_cost_model *m,
                                    const char *op_name,
                                    const bafe_layout *input_layouts, int n_inputs,
                                    bafe_layout output_layout,
                                    const bafe_shape *inputs, int n_input_shapes);
double bafe_cost_node(const bafe_cost_model *m, const char *op_name,
                      const bafe_shape *inputs, int n_inputs,
                      const bafe_op_attrs *attrs,
                      const bafe_shape *out_shape, bafe_dtype dtype);
double bafe_cost_node_with_layout(const bafe_cost_model *m, const char *op_name,
                                   const bafe_shape *inputs, int n_inputs,
                                   const bafe_op_attrs *attrs,
                                   const bafe_shape *out_shape, bafe_dtype dtype,
                                   const bafe_layout *input_layouts, int n_input_layouts,
                                   bafe_layout output_layout);
double bafe_cost_graph(const bafe_cost_model *m, const bafe_graph *g);

/* Arithmetic intensity = FLOPs / bytes (higher = compute-bound, lower = memory-bound) */
double bafe_cost_arithmetic_intensity(const char *op_name,
                                       const bafe_shape *inputs, int n_inputs,
                                       const bafe_op_attrs *attrs,
                                       const bafe_shape *out_shape, bafe_dtype dtype);

/* Determine which cache level the working set fits in.
 * Returns 0=L1, 1=L2, 2=L3, 3=DRAM. */
int bafe_cost_cache_level(size_t working_set_bytes,
                           const bafe_cost_model *m);

/* Get the effective bandwidth weight for a given working set size. */
double bafe_cost_effective_bandwidth(size_t working_set_bytes,
                                      const bafe_cost_model *m);

/* Check if an op is SIMD-vectorizable (elementwise on contiguous data). */
bool bafe_cost_is_vectorizable(const char *op_name);

bafe_cost_model bafe_cost_model_calibrate(const bafe_cost_model *static_model,
                                           const double *learned_weights,
                                           int n_weights, double learned_bias);
bafe_cost_model bafe_cost_model_calibrated_default(void);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_COST_H */
