/* bafe/ir.h - IR Graph: nodes and DAG structure
 *
 * A Graph is a DAG of Nodes. Each Node has:
 *   - id        : stable int identifier (unique within the graph)
 *   - op_name   : registered op name
 *   - attrs     : op attributes (axes, perm, shape, etc.)
 *   - children  : array of child Node ids
 *   - shape     : inferred Shape (cached)
 *   - dtype     : Dtype
 *
 * Graphs are mutable working data structures. The e-graph provides the
 * immutable, hashable form used during search.
 */
#ifndef BAFE_IR_H
#define BAFE_IR_H

#include "bafe/types.h"
#include "bafe/ops.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BAFE_MAX_CHILDREN 4
#define BAFE_MAX_NODES    512

typedef int32_t bafe_node_id;

typedef struct {
    bafe_node_id   id;
    const char    *op_name;     /* borrowed from registry */
    bafe_op_attrs  attrs;
    int            n_children;
    bafe_node_id   children[BAFE_MAX_CHILDREN];
    bafe_shape     shape;
    bafe_dtype     dtype;
    bafe_layout    layout;       /* Phase 2: memory layout of this node's output */
    char           input_name[BAFE_MAX_ATTR_LEN]; /* for input nodes */
    bool           is_input;
    bool           is_constant;
    double         const_value;  /* for constant nodes */
} bafe_node;

typedef struct {
    bafe_node   nodes[BAFE_MAX_NODES];
    int         n_nodes;
    bafe_node_id inputs[BAFE_MAX_NODES];
    int         n_inputs;
    bafe_node_id outputs[BAFE_MAX_NODES];
    int         n_outputs;
} bafe_graph;

/* lifecycle */
void bafe_graph_init(bafe_graph *g);

/* construction */
bafe_node_id bafe_graph_add_input(bafe_graph *g, const char *name,
                                   const bafe_shape *shape, bafe_dtype dtype);
bafe_node_id bafe_graph_add_input_with_layout(bafe_graph *g, const char *name,
                                              const bafe_shape *shape,
                                              bafe_dtype dtype,
                                              bafe_layout layout);
bafe_node_id bafe_graph_add_constant(bafe_graph *g, double value,
                                      const bafe_shape *shape, bafe_dtype dtype);
bafe_node_id bafe_graph_add(bafe_graph *g, const char *op_name,
                             const bafe_node_id *children, int n_children,
                             const bafe_op_attrs *attrs);

/* Set the layout of an existing node (used by the layout rewrite engine).
 * Returns 0 on success, non-zero on error. */
int bafe_graph_set_node_layout(bafe_graph *g, bafe_node_id id, bafe_layout layout);

/* Get the layout of a node. */
bafe_layout bafe_graph_get_node_layout(const bafe_graph *g, bafe_node_id id);

/* convenience wrappers (return new node id, or -1 on error) */
bafe_node_id bafe_graph_matmul(bafe_graph *g, bafe_node_id a, bafe_node_id b);
bafe_node_id bafe_graph_add_op(bafe_graph *g, bafe_node_id a, bafe_node_id b);
bafe_node_id bafe_graph_mul(bafe_graph *g, bafe_node_id a, bafe_node_id b);
bafe_node_id bafe_graph_sub(bafe_graph *g, bafe_node_id a, bafe_node_id b);
bafe_node_id bafe_graph_bias_add(bafe_graph *g, bafe_node_id a, bafe_node_id b);
bafe_node_id bafe_graph_relu(bafe_graph *g, bafe_node_id x);
bafe_node_id bafe_graph_sigmoid(bafe_graph *g, bafe_node_id x);
bafe_node_id bafe_graph_tanh(bafe_graph *g, bafe_node_id x);
bafe_node_id bafe_graph_neg(bafe_graph *g, bafe_node_id x);
bafe_node_id bafe_graph_transpose(bafe_graph *g, bafe_node_id x,
                                   const int32_t *perm, int32_t n_perm);
bafe_node_id bafe_graph_reduce_sum(bafe_graph *g, bafe_node_id x,
                                    const int32_t *axes, int32_t n_axes,
                                    bool keepdims);
bafe_node_id bafe_graph_reduce_max(bafe_graph *g, bafe_node_id x,
                                    const int32_t *axes, int32_t n_axes,
                                    bool keepdims);
bafe_node_id bafe_graph_reshape(bafe_graph *g, bafe_node_id x,
                                 const int32_t *shape, int32_t n_shape);
bafe_node_id bafe_graph_broadcast_to(bafe_graph *g, bafe_node_id x,
                                      const int32_t *shape, int32_t n_shape);

void bafe_graph_set_output(bafe_graph *g, bafe_node_id id);

/* traversal */
int bafe_graph_topo_order(bafe_graph *g, bafe_node_id *out, int max_out);

/* debug */
void bafe_graph_print(bafe_graph *g, char *buf, size_t buf_size);
int  bafe_graph_summary(const bafe_graph *g, char *buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_IR_H */
