/* bafe/ir.c - IR Graph implementation */
#include "bafe/ir.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void bafe_graph_init(bafe_graph *g) {
    memset(g, 0, sizeof(*g));
}

static bafe_node_id _new_id(bafe_graph *g) {
    if (g->n_nodes >= BAFE_MAX_NODES) return -1;
    bafe_node_id id = g->n_nodes++;
    bafe_node *n = &g->nodes[id];
    memset(n, 0, sizeof(*n));
    n->id = id;
    n->attrs = bafe_op_attrs_default();
    n->dtype = BAFE_DTYPE_F32;
    return id;
}

static void _set_attrs_from_va(bafe_op_attrs *attrs, const bafe_op_attrs *src) {
    if (src) *attrs = *src;
}

bafe_node_id bafe_graph_add_input(bafe_graph *g, const char *name,
                                   const bafe_shape *shape, bafe_dtype dtype) {
    return bafe_graph_add_input_with_layout(g, name, shape, dtype, BAFE_LAYOUT_ROW_MAJOR);
}

bafe_node_id bafe_graph_add_input_with_layout(bafe_graph *g, const char *name,
                                              const bafe_shape *shape,
                                              bafe_dtype dtype,
                                              bafe_layout layout) {
    bafe_node_id id = _new_id(g);
    if (id < 0) return -1;
    bafe_node *n = &g->nodes[id];
    n->op_name = "input";
    n->is_input = true;
    n->shape = *shape;
    n->dtype = dtype;
    n->layout = layout;
    if (name) {
        strncpy(n->input_name, name, BAFE_MAX_ATTR_LEN - 1);
        n->input_name[BAFE_MAX_ATTR_LEN - 1] = '\0';
        /* also store in attrs.name so the e-graph can distinguish inputs */
        strncpy(n->attrs.name, name, BAFE_MAX_ATTR_LEN - 1);
        n->attrs.name[BAFE_MAX_ATTR_LEN - 1] = '\0';
    }
    g->inputs[g->n_inputs++] = id;
    return id;
}

int bafe_graph_set_node_layout(bafe_graph *g, bafe_node_id id, bafe_layout layout) {
    if (id < 0 || id >= g->n_nodes) return -1;
    g->nodes[id].layout = layout;
    return 0;
}

bafe_layout bafe_graph_get_node_layout(const bafe_graph *g, bafe_node_id id) {
    if (id < 0 || id >= g->n_nodes) return BAFE_LAYOUT_ROW_MAJOR;
    return g->nodes[id].layout;
}

bafe_node_id bafe_graph_add_constant(bafe_graph *g, double value,
                                      const bafe_shape *shape, bafe_dtype dtype) {
    bafe_node_id id = _new_id(g);
    if (id < 0) return -1;
    bafe_node *n = &g->nodes[id];
    n->op_name = "constant";
    n->is_constant = true;
    n->shape = *shape;
    n->dtype = dtype;
    n->layout = BAFE_LAYOUT_ROW_MAJOR;
    n->const_value = value;
    return id;
}

/* Layout propagation: infer the output layout of an op based on its type
 * and the layouts of its children.
 *
 * Rules (Phase 2 MVP):
 *   - input/constant: from caller (already set)
 *   - layout_transform(x): from attrs.name (parsed as a layout string)
 *   - transpose(x, perm): if rank-2 with perm=(1,0), flip ROW<->COL;
 *     otherwise inherit x's layout
 *   - all other ops: inherit from first child
 */
static bafe_layout _infer_layout(const char *op_name, const bafe_op_attrs *attrs,
                                  const bafe_node *children[], int n_children) {
    if (bafe_op_is_layout_transform(op_name)) {
        /* parse attrs.name as a layout string */
        if (attrs && attrs->name[0]) {
            if (strcmp(attrs->name, "row") == 0) return BAFE_LAYOUT_ROW_MAJOR;
            if (strcmp(attrs->name, "col") == 0) return BAFE_LAYOUT_COL_MAJOR;
            if (strcmp(attrs->name, "blocked") == 0) return BAFE_LAYOUT_BLOCKED;
            if (strcmp(attrs->name, "tc") == 0) return BAFE_LAYOUT_TENSOR_CORE;
        }
        return BAFE_LAYOUT_ROW_MAJOR;
    }
    if (strcmp(op_name, "transpose") == 0 && n_children == 1 && children[0]) {
        /* for rank-2 transpose with perm=(1,0), flip the layout */
        const bafe_node *x = children[0];
        if (x->shape.rank == 2 && attrs && attrs->n_perm == 2 &&
            attrs->perm[0] == 1 && attrs->perm[1] == 0) {
            if (x->layout == BAFE_LAYOUT_ROW_MAJOR) return BAFE_LAYOUT_COL_MAJOR;
            if (x->layout == BAFE_LAYOUT_COL_MAJOR) return BAFE_LAYOUT_ROW_MAJOR;
        }
        return x->layout;
    }
    if (n_children > 0 && children[0]) return children[0]->layout;
    return BAFE_LAYOUT_ROW_MAJOR;
}

bafe_node_id bafe_graph_add(bafe_graph *g, const char *op_name,
                             const bafe_node_id *children, int n_children,
                             const bafe_op_attrs *attrs) {
    const bafe_op *op = bafe_op_get(op_name);
    if (!op) return -1;
    if (n_children != op->arity) return -1;
    /* validate children */
    for (int i = 0; i < n_children; i++) {
        if (children[i] < 0 || children[i] >= g->n_nodes) return -1;
    }
    bafe_node_id id = _new_id(g);
    if (id < 0) return -1;
    bafe_node *n = &g->nodes[id];
    n->op_name = op->name;
    n->n_children = n_children;
    for (int i = 0; i < n_children; i++) n->children[i] = children[i];
    _set_attrs_from_va(&n->attrs, attrs);
    /* infer shape */
    bafe_shape child_shapes[BAFE_MAX_CHILDREN];
    const bafe_node *child_nodes[BAFE_MAX_CHILDREN];
    for (int i = 0; i < n_children; i++) {
        child_shapes[i] = g->nodes[children[i]].shape;
        child_nodes[i] = &g->nodes[children[i]];
    }
    n->shape = op->shape_fn(child_shapes, n_children, &n->attrs);
    /* dtype: take from first child */
    if (n_children > 0) n->dtype = g->nodes[children[0]].dtype;
    /* layout: propagate from children */
    n->layout = _infer_layout(op_name, &n->attrs, child_nodes, n_children);
    return id;
}

bafe_node_id bafe_graph_matmul(bafe_graph *g, bafe_node_id a, bafe_node_id b) {
    bafe_node_id c[2] = {a, b};
    return bafe_graph_add(g, "matmul", c, 2, NULL);
}
bafe_node_id bafe_graph_add_op(bafe_graph *g, bafe_node_id a, bafe_node_id b) {
    bafe_node_id c[2] = {a, b};
    return bafe_graph_add(g, "add", c, 2, NULL);
}
bafe_node_id bafe_graph_mul(bafe_graph *g, bafe_node_id a, bafe_node_id b) {
    bafe_node_id c[2] = {a, b};
    return bafe_graph_add(g, "mul", c, 2, NULL);
}
bafe_node_id bafe_graph_sub(bafe_graph *g, bafe_node_id a, bafe_node_id b) {
    bafe_node_id c[2] = {a, b};
    return bafe_graph_add(g, "sub", c, 2, NULL);
}
bafe_node_id bafe_graph_bias_add(bafe_graph *g, bafe_node_id a, bafe_node_id b) {
    bafe_node_id c[2] = {a, b};
    return bafe_graph_add(g, "bias_add", c, 2, NULL);
}
bafe_node_id bafe_graph_relu(bafe_graph *g, bafe_node_id x) {
    return bafe_graph_add(g, "relu", &x, 1, NULL);
}
bafe_node_id bafe_graph_sigmoid(bafe_graph *g, bafe_node_id x) {
    return bafe_graph_add(g, "sigmoid", &x, 1, NULL);
}
bafe_node_id bafe_graph_tanh(bafe_graph *g, bafe_node_id x) {
    return bafe_graph_add(g, "tanh", &x, 1, NULL);
}
bafe_node_id bafe_graph_neg(bafe_graph *g, bafe_node_id x) {
    return bafe_graph_add(g, "neg", &x, 1, NULL);
}
bafe_node_id bafe_graph_transpose(bafe_graph *g, bafe_node_id x,
                                   const int32_t *perm, int32_t n_perm) {
    bafe_op_attrs a = bafe_op_attrs_default();
    a.n_perm = n_perm > BAFE_MAX_ATTR_LEN ? BAFE_MAX_ATTR_LEN : n_perm;
    for (int32_t i = 0; i < a.n_perm; i++) a.perm[i] = perm[i];
    return bafe_graph_add(g, "transpose", &x, 1, &a);
}
bafe_node_id bafe_graph_reduce_sum(bafe_graph *g, bafe_node_id x,
                                    const int32_t *axes, int32_t n_axes,
                                    bool keepdims) {
    bafe_op_attrs a = bafe_op_attrs_default();
    a.n_axes = n_axes > BAFE_MAX_ATTR_LEN ? BAFE_MAX_ATTR_LEN : n_axes;
    for (int32_t i = 0; i < a.n_axes; i++) a.axes[i] = axes[i];
    a.keepdims = keepdims;
    return bafe_graph_add(g, "reduce_sum", &x, 1, &a);
}
bafe_node_id bafe_graph_reduce_max(bafe_graph *g, bafe_node_id x,
                                    const int32_t *axes, int32_t n_axes,
                                    bool keepdims) {
    bafe_op_attrs a = bafe_op_attrs_default();
    a.n_axes = n_axes > BAFE_MAX_ATTR_LEN ? BAFE_MAX_ATTR_LEN : n_axes;
    for (int32_t i = 0; i < a.n_axes; i++) a.axes[i] = axes[i];
    a.keepdims = keepdims;
    return bafe_graph_add(g, "reduce_max", &x, 1, &a);
}
bafe_node_id bafe_graph_reshape(bafe_graph *g, bafe_node_id x,
                                 const int32_t *shape, int32_t n_shape) {
    bafe_op_attrs a = bafe_op_attrs_default();
    a.n_shape = n_shape > BAFE_MAX_ATTR_LEN ? BAFE_MAX_ATTR_LEN : n_shape;
    for (int32_t i = 0; i < a.n_shape; i++) a.shape[i] = shape[i];
    return bafe_graph_add(g, "reshape", &x, 1, &a);
}
bafe_node_id bafe_graph_broadcast_to(bafe_graph *g, bafe_node_id x,
                                      const int32_t *shape, int32_t n_shape) {
    bafe_op_attrs a = bafe_op_attrs_default();
    a.n_shape = n_shape > BAFE_MAX_ATTR_LEN ? BAFE_MAX_ATTR_LEN : n_shape;
    for (int32_t i = 0; i < a.n_shape; i++) a.shape[i] = shape[i];
    return bafe_graph_add(g, "broadcast_to", &x, 1, &a);
}

void bafe_graph_set_output(bafe_graph *g, bafe_node_id id) {
    if (id < 0 || id >= g->n_nodes) return;
    if (g->n_outputs >= BAFE_MAX_NODES) return;
    g->outputs[g->n_outputs++] = id;
}

/* iterative DFS topological sort */
int bafe_graph_topo_order(bafe_graph *g, bafe_node_id *out, int max_out) {
    bool visited[BAFE_MAX_NODES] = {false};
    bool on_stack[BAFE_MAX_NODES] = {false};
    /* iterative DFS, post-order */
    bafe_node_id stack[BAFE_MAX_NODES * 2];
    int sp = 0;
    /* push roots: outputs first, then everything else */
    for (int i = 0; i < g->n_outputs; i++) {
        stack[sp++] = g->outputs[i];
    }
    for (int i = 0; i < g->n_nodes; i++) {
        stack[sp++] = i;
    }
    /* we need post-order reversed; emit children before parent.
       Use a second stack to reverse. */
    bafe_node_id order[BAFE_MAX_NODES];
    int order_n = 0;
    while (sp > 0) {
        bafe_node_id cur = stack[--sp];
        if (cur < 0 || cur >= g->n_nodes) continue;
        if (visited[cur]) continue;
        if (on_stack[cur]) {
            /* already expanded, emit */
            on_stack[cur] = false;
            visited[cur] = true;
            order[order_n++] = cur;
            continue;
        }
        /* mark as "to expand": push self with on_stack marker, then children */
        on_stack[cur] = true;
        stack[sp++] = cur;
        bafe_node *n = &g->nodes[cur];
        for (int i = 0; i < n->n_children; i++) {
            if (!visited[n->children[i]] && !on_stack[n->children[i]]) {
                stack[sp++] = n->children[i];
            }
        }
    }
    /* order is children-first (post-order DFS emits children before parents).
     * We want children-first for topo order, so keep as-is. */
    int n_out = order_n < max_out ? order_n : max_out;
    for (int i = 0; i < n_out; i++) out[i] = order[i];
    return n_out;
}

int bafe_graph_summary(const bafe_graph *g, char *buf, size_t buf_size) {
    int n_input = 0, n_const = 0, n_op = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].is_input) n_input++;
        else if (g->nodes[i].is_constant) n_const++;
        else n_op++;
    }
    return snprintf(buf, buf_size,
        "Graph(nodes=%d, inputs=%d, constants=%d, ops=%d, outputs=%d)",
        g->n_nodes, n_input, n_const, n_op, g->n_outputs);
}

void bafe_graph_print(bafe_graph *g, char *buf, size_t buf_size) {
    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, buf_size - pos, "Graph:\n");
    for (int i = 0; i < g->n_nodes && pos < buf_size; i++) {
        bafe_node *n = &g->nodes[i];
        char sh[64];
        bafe_shape_snprintf(sh, sizeof(sh), &n->shape);
        pos += (size_t)snprintf(buf + pos, buf_size - pos,
            "  n%d = %s", i, n->op_name);
        if (n->is_input) pos += (size_t)snprintf(buf + pos, buf_size - pos, " name=%s", n->input_name);
        if (n->n_children > 0) {
            pos += (size_t)snprintf(buf + pos, buf_size - pos, "(");
            for (int j = 0; j < n->n_children; j++) {
                pos += (size_t)snprintf(buf + pos, buf_size - pos,
                    "%s%d", j == 0 ? "" : ", ", n->children[j]);
            }
            pos += (size_t)snprintf(buf + pos, buf_size - pos, ")");
        }
        pos += (size_t)snprintf(buf + pos, buf_size - pos, " : %s\n", sh);
    }
}
