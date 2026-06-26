/* bafe/rewrite.c - rewrite rules and engine
 *
 * Rules are encoded as C functions. Each rule:
 *   - checks if it applies to a node
 *   - if yes, builds a bafe_alternative describing the equivalent expression
 *
 * The engine walks the graph in topological order and runs every rule on
 * every node.
 *
 * Rules may need to add NEW intermediate nodes to the graph (e.g. when
 * distributing mul over add, we need new mul nodes). This is done with
 * bafe_graph_add_*, and the resulting node ids are stored in the
 * alternative's children array.
 *
 * IMPORTANT: rule functions take a MUTABLE graph pointer because they may
 * need to add nodes. The original node is NOT mutated.
 */
#include "bafe/rewrite.h"
#include <string.h>
#include <stdio.h>

static const bafe_node *_child(const bafe_graph *g, bafe_node_id id) {
    return &g->nodes[id];
}

static bool _is_op(const bafe_node *n, const char *name) {
    return strcmp(n->op_name, name) == 0;
}

/* ------------------------------------------------------------------ */
/* Rule implementations                                               */
/* ------------------------------------------------------------------ */

/* add(a, b) -> add(b, a) */
static bool _add_comm_applies(const bafe_graph *g, const bafe_node *n) {
    (void)g;
    return _is_op(n, "add");
}
static int _add_comm(const bafe_graph *g, const bafe_node *n, bafe_alternative *out) {
    (void)g;
    out->original_node_id = n->id;
    out->op_name = "add";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 2;
    out->children[0] = n->children[1];
    out->children[1] = n->children[0];
    return 1;
}

/* add(add(a, b), c) -> add(a, add(b, c)) */
static bool _add_assoc_left_applies(const bafe_graph *g, const bafe_node *n) {
    return _is_op(n, "add") && _is_op(_child(g, n->children[0]), "add");
}
static int _add_assoc_left(const bafe_graph *g_, const bafe_node *n, bafe_alternative *out) {
    bafe_graph *g = (bafe_graph *)g_;  /* cast away const to add new nodes */
    bafe_node_id left = n->children[0];
    bafe_node_id a = g->nodes[left].children[0];
    bafe_node_id b = g->nodes[left].children[1];
    bafe_node_id c = n->children[1];
    bafe_node_id new_inner = bafe_graph_add_op(g, b, c);
    if (new_inner < 0) return 0;
    out->original_node_id = n->id;
    out->op_name = "add";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 2;
    out->children[0] = a;
    out->children[1] = new_inner;
    return 1;
}

/* add(a, add(b, c)) -> add(add(a, b), c) */
static bool _add_assoc_right_applies(const bafe_graph *g, const bafe_node *n) {
    return _is_op(n, "add") && _is_op(_child(g, n->children[1]), "add");
}
static int _add_assoc_right(const bafe_graph *g_, const bafe_node *n, bafe_alternative *out) {
    bafe_graph *g = (bafe_graph *)g_;
    bafe_node_id a = n->children[0];
    bafe_node_id right = n->children[1];
    bafe_node_id b = g->nodes[right].children[0];
    bafe_node_id c = g->nodes[right].children[1];
    bafe_node_id new_inner = bafe_graph_add_op(g, a, b);
    if (new_inner < 0) return 0;
    out->original_node_id = n->id;
    out->op_name = "add";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 2;
    out->children[0] = new_inner;
    out->children[1] = c;
    return 1;
}

/* mul(a, b) -> mul(b, a) */
static bool _mul_comm_applies(const bafe_graph *g, const bafe_node *n) {
    (void)g;
    return _is_op(n, "mul");
}
static int _mul_comm(const bafe_graph *g, const bafe_node *n, bafe_alternative *out) {
    (void)g;
    out->original_node_id = n->id;
    out->op_name = "mul";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 2;
    out->children[0] = n->children[1];
    out->children[1] = n->children[0];
    return 1;
}

/* mul(add(a, b), c) -> add(mul(a, c), mul(b, c)) */
static bool _distribute_mul_left_applies(const bafe_graph *g, const bafe_node *n) {
    return _is_op(n, "mul") && _is_op(_child(g, n->children[0]), "add");
}
static int _distribute_mul_left(const bafe_graph *g_, const bafe_node *n, bafe_alternative *out) {
    bafe_graph *g = (bafe_graph *)g_;
    bafe_node_id left = n->children[0];
    bafe_node_id a = g->nodes[left].children[0];
    bafe_node_id b = g->nodes[left].children[1];
    bafe_node_id c = n->children[1];
    bafe_node_id m1 = bafe_graph_mul(g, a, c);
    bafe_node_id m2 = bafe_graph_mul(g, b, c);
    if (m1 < 0 || m2 < 0) return 0;
    out->original_node_id = n->id;
    out->op_name = "add";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 2;
    out->children[0] = m1;
    out->children[1] = m2;
    return 1;
}

/* mul(c, add(a, b)) -> add(mul(c, a), mul(c, b)) */
static bool _distribute_mul_right_applies(const bafe_graph *g, const bafe_node *n) {
    return _is_op(n, "mul") && _is_op(_child(g, n->children[1]), "add");
}
static int _distribute_mul_right(const bafe_graph *g_, const bafe_node *n, bafe_alternative *out) {
    bafe_graph *g = (bafe_graph *)g_;
    bafe_node_id right = n->children[1];
    bafe_node_id c = n->children[0];
    bafe_node_id a = g->nodes[right].children[0];
    bafe_node_id b = g->nodes[right].children[1];
    bafe_node_id m1 = bafe_graph_mul(g, c, a);
    bafe_node_id m2 = bafe_graph_mul(g, c, b);
    if (m1 < 0 || m2 < 0) return 0;
    out->original_node_id = n->id;
    out->op_name = "add";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 2;
    out->children[0] = m1;
    out->children[1] = m2;
    return 1;
}

/* sub(a, sub(b, c)) -> add(sub(a, b), c) */
static bool _sub_sub_applies(const bafe_graph *g, const bafe_node *n) {
    return _is_op(n, "sub") && _is_op(_child(g, n->children[1]), "sub");
}
static int _sub_sub(const bafe_graph *g_, const bafe_node *n, bafe_alternative *out) {
    bafe_graph *g = (bafe_graph *)g_;
    bafe_node_id a = n->children[0];
    bafe_node_id right = n->children[1];
    bafe_node_id b = g->nodes[right].children[0];
    bafe_node_id c = g->nodes[right].children[1];
    bafe_node_id new_sub = bafe_graph_sub(g, a, b);
    if (new_sub < 0) return 0;
    out->original_node_id = n->id;
    out->op_name = "add";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 2;
    out->children[0] = new_sub;
    out->children[1] = c;
    return 1;
}

/* transpose(transpose(x, p1), p2) -> x if p2 = inverse(p1), else composed perm */
static bool _transpose_transpose_applies(const bafe_graph *g, const bafe_node *n) {
    return _is_op(n, "transpose") && _is_op(_child(g, n->children[0]), "transpose");
}
static int _transpose_transpose(const bafe_graph *g, const bafe_node *n, bafe_alternative *out) {
    const bafe_node *inner = _child(g, n->children[0]);
    int32_t p1[BAFE_MAX_ATTR_LEN];
    int32_t p2[BAFE_MAX_ATTR_LEN];
    int n1 = inner->attrs.n_perm;
    int n2 = n->attrs.n_perm;
    if (n1 != n2 || n1 > BAFE_MAX_ATTR_LEN) return 0;
    for (int i = 0; i < n1; i++) { p1[i] = inner->attrs.perm[i]; p2[i] = n->attrs.perm[i]; }
    /* check inverse: p1[p2[i]] == i */
    bool is_inv = true;
    for (int i = 0; i < n1; i++) if (p1[p2[i]] != i) { is_inv = false; break; }
    out->original_node_id = n->id;
    out->attrs = bafe_op_attrs_default();
    out->n_children = 1;
    out->children[0] = inner->children[0];
    if (is_inv) {
        /* equivalent to identity: use reshape with same shape */
        out->op_name = "reshape";
        bafe_shape src = g->nodes[inner->children[0]].shape;
        out->attrs.n_shape = src.rank;
        for (int i = 0; i < src.rank; i++) out->attrs.shape[i] = src.dims[i];
    } else {
        /* compose: p1[p2[i]] */
        out->op_name = "transpose";
        out->attrs.n_perm = n1;
        for (int i = 0; i < n1; i++) out->attrs.perm[i] = p1[p2[i]];
    }
    return 1;
}

/* relu(matmul(A, B)) -> fused_matmul_relu(A, B) */
static bool _fuse_matmul_relu_applies(const bafe_graph *g, const bafe_node *n) {
    return _is_op(n, "relu") && _is_op(_child(g, n->children[0]), "matmul");
}
static int _fuse_matmul_relu(const bafe_graph *g, const bafe_node *n, bafe_alternative *out) {
    const bafe_node *mm = _child(g, n->children[0]);
    out->original_node_id = n->id;
    out->op_name = "fused_matmul_relu";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 2;
    out->children[0] = mm->children[0];
    out->children[1] = mm->children[1];
    return 1;
}

/* relu(bias_add(X, bias)) -> fused_bias_relu(X, bias) */
static bool _fuse_bias_relu_applies(const bafe_graph *g, const bafe_node *n) {
    return _is_op(n, "relu") && _is_op(_child(g, n->children[0]), "bias_add");
}
static int _fuse_bias_relu(const bafe_graph *g, const bafe_node *n, bafe_alternative *out) {
    const bafe_node *ba = _child(g, n->children[0]);
    out->original_node_id = n->id;
    out->op_name = "fused_bias_relu";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 2;
    out->children[0] = ba->children[0];
    out->children[1] = ba->children[1];
    return 1;
}

/* add(matmul(A, B), bias_vector) -> fused_matmul_bias(A, B, bias)
 * Condition: the right operand is rank-1 (a bias vector).
 */
static bool _fuse_matmul_bias_applies(const bafe_graph *g, const bafe_node *n) {
    if (!_is_op(n, "add")) return false;
    if (!_is_op(_child(g, n->children[0]), "matmul")) return false;
    const bafe_node *rhs = _child(g, n->children[1]);
    return rhs->shape.rank == 1;
}
static int _fuse_matmul_bias(const bafe_graph *g, const bafe_node *n, bafe_alternative *out) {
    const bafe_node *mm = _child(g, n->children[0]);
    out->original_node_id = n->id;
    out->op_name = "fused_matmul_bias";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 3;
    out->children[0] = mm->children[0];
    out->children[1] = mm->children[1];
    out->children[2] = n->children[1];
    return 1;
}

/* relu(add(matmul(A, B), bias)) -> fused_matmul_bias_relu(A, B, bias) */
static bool _fuse_matmul_bias_relu_applies(const bafe_graph *g, const bafe_node *n) {
    if (!_is_op(n, "relu")) return false;
    const bafe_node *inner = _child(g, n->children[0]);
    if (!_is_op(inner, "add")) return false;
    if (!_is_op(_child(g, inner->children[0]), "matmul")) return false;
    const bafe_node *rhs = _child(g, inner->children[1]);
    return rhs->shape.rank == 1;
}
static int _fuse_matmul_bias_relu(const bafe_graph *g, const bafe_node *n, bafe_alternative *out) {
    const bafe_node *add_node = _child(g, n->children[0]);
    const bafe_node *mm = _child(g, add_node->children[0]);
    out->original_node_id = n->id;
    out->op_name = "fused_matmul_bias_relu";
    out->attrs = bafe_op_attrs_default();
    out->n_children = 3;
    out->children[0] = mm->children[0];
    out->children[1] = mm->children[1];
    out->children[2] = add_node->children[1];
    return 1;
}

/* ------------------------------------------------------------------ */
/* Rule table                                                         */
/* ------------------------------------------------------------------ */

typedef bool (*rule_applies_fn)(const bafe_graph *g, const bafe_node *n);
typedef int  (*rule_apply_fn)(const bafe_graph *g, const bafe_node *n, bafe_alternative *out);

typedef struct {
    const char       *name;
    rule_applies_fn   applies;
    rule_apply_fn     apply;
} rule_def;

static const rule_def _RULES[] = {
    {"add_commutative",            _add_comm_applies,            _add_comm},
    {"add_assoc_left",             _add_assoc_left_applies,      _add_assoc_left},
    {"add_assoc_right",            _add_assoc_right_applies,     _add_assoc_right},
    {"mul_commutative",            _mul_comm_applies,            _mul_comm},
    {"distribute_mul_over_add_l",  _distribute_mul_left_applies, _distribute_mul_left},
    {"distribute_mul_over_add_r",  _distribute_mul_right_applies,_distribute_mul_right},
    {"sub_sub_to_sub_add",         _sub_sub_applies,             _sub_sub},
    {"transpose_transpose",        _transpose_transpose_applies, _transpose_transpose},
    {"fuse_matmul_relu",           _fuse_matmul_relu_applies,    _fuse_matmul_relu},
    {"fuse_bias_relu",             _fuse_bias_relu_applies,      _fuse_bias_relu},
    {"fuse_matmul_bias",           _fuse_matmul_bias_applies,    _fuse_matmul_bias},
    {"fuse_matmul_bias_relu",      _fuse_matmul_bias_relu_applies,_fuse_matmul_bias_relu},
};

static const int _N_RULES = (int)(sizeof(_RULES) / sizeof(_RULES[0]));

int bafe_rewrite_default_count(void) { return _N_RULES; }
const char *bafe_rewrite_default_name(int i) {
    if (i < 0 || i >= _N_RULES) return NULL;
    return _RULES[i].name;
}

int bafe_rewrite_find(const bafe_graph *g, bafe_alt_list *out) {
    out->n = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        const bafe_node *n = &g->nodes[i];
        if (n->is_input || n->is_constant) continue;
        for (int r = 0; r < _N_RULES; r++) {
            if (!_RULES[r].applies(g, n)) continue;
            if (out->n >= BAFE_MAX_ALTERNATIVES) return out->n;
            bafe_alternative *alt = &out->items[out->n];
            if (_RULES[r].apply(g, n, alt)) {
                out->n++;
            }
        }
    }
    return out->n;
}

bafe_node_id bafe_rewrite_materialize(bafe_graph *g, const bafe_alternative *alt) {
    return bafe_graph_add(g, alt->op_name, alt->children, alt->n_children, &alt->attrs);
}
