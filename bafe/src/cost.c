/* bafe/cost.c - cost model implementation
 *
 * Numbers here are deliberately simple and concrete (not tuned to any
 * specific hardware). Phase 2 will replace these with hardware-aware
 * estimates.
 */
#include "bafe/cost.h"
#include "bafe/ops.h"
#include <string.h>

bafe_cost_model bafe_cost_model_default(void) {
    bafe_cost_model m;
    m.alpha_flops = 1e-9;        /* 1 ns per FLOP ~ 1 GFLOP/s */
    m.beta_bytes = 1e-8;         /* 1 ns per byte ~ 1 GB/s */
    m.gamma_intermediate = 1.0;  /* flat 1 cost unit per intermediate */
    m.delta_fuse = 0.3;          /* 30% bonus for fusion */
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
    double flops = bafe_cost_flops(op_name, inputs, n_inputs, attrs, out_shape);
    double bytes = bafe_cost_bytes(op_name, inputs, n_inputs, out_shape, dtype);
    double cost = m->alpha_flops * flops + m->beta_bytes * bytes;
    /* intermediate penalty: 1 per non-fused op (since they materialize) */
    if (!bafe_op_is_fused(op_name) &&
        strcmp(op_name, "input") != 0 &&
        strcmp(op_name, "constant") != 0 &&
        strcmp(op_name, "transpose") != 0 &&
        strcmp(op_name, "reshape") != 0 &&
        strcmp(op_name, "broadcast_to") != 0) {
        cost += m->gamma_intermediate;
    }
    /* fusion bonus */
    if (bafe_op_is_fused(op_name)) {
        cost -= m->delta_fuse;
    }
    return cost;
}

double bafe_cost_graph(const bafe_cost_model *m, const bafe_graph *g) {
    double total = 0.0;
    for (int i = 0; i < g->n_nodes; i++) {
        const bafe_node *n = &g->nodes[i];
        if (n->is_input || n->is_constant) continue;
        bafe_shape inputs[BAFE_MAX_CHILDREN];
        for (int j = 0; j < n->n_children; j++) inputs[j] = g->nodes[n->children[j]].shape;
        total += bafe_cost_node(m, n->op_name, inputs, n->n_children, &n->attrs, &n->shape, n->dtype);
    }
    return total;
}
