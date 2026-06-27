/* bafe/extract.c - min-cost extraction from e-graph
 *
 * Iterative DP because the e-graph can contain cycles (rare, but possible
 * after aggressive rewrites). We initialize each class's cost to +inf,
 * then iterate Bellman-Ford-style until no improvement.
 */
#include "bafe/extract.h"
#include "bafe/ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static bafe_eclass_id _find_nonconst(bafe_egraph *eg, bafe_eclass_id x) {
    /* mutable find for the cost computation (we cast away const in caller) */
    return bafe_egraph_find(eg, x);
}

/* Build an eclass -> shape mapping from the graph + node_to_eclass.
 * For each eclass, we use the shape of the first graph node that maps to it. */
static void _build_eclass_shapes(const bafe_egraph *eg,
                                  const bafe_graph *g,
                                  const bafe_eclass_id *node_to_eclass,
                                  bafe_shape *eclass_shapes, /* size = n_total_classes_allocated */
                                  bafe_dtype *eclass_dtypes) {
    for (int i = 0; i < eg->n_total_classes_allocated; i++) {
        eclass_shapes[i] = bafe_shape_scalar();
        eclass_dtypes[i] = BAFE_DTYPE_F32;
    }
    if (!node_to_eclass || !g) return;
    for (int nid = 0; nid < g->n_nodes; nid++) {
        bafe_eclass_id cid = bafe_egraph_find((bafe_egraph *)eg, node_to_eclass[nid]);
        if (cid >= 0 && cid < eg->n_total_classes_allocated) {
            /* use the first node's shape (they should all be equivalent) */
            if (bafe_shape_numel(&eclass_shapes[cid]) <= 1) {
                eclass_shapes[cid] = g->nodes[nid].shape;
                eclass_dtypes[cid] = g->nodes[nid].dtype;
            }
        }
    }
}

static double _node_cost(const bafe_cost_model *m,
                         const bafe_enode *e,
                         const bafe_shape *eclass_shapes,
                         const bafe_dtype *eclass_dtypes) {
    bafe_shape inputs[BAFE_MAX_CHILDREN];
    bafe_shape out_shape = bafe_shape_scalar();
    bafe_dtype dtype = BAFE_DTYPE_F32;

    for (int i = 0; i < e->n_children && i < BAFE_MAX_CHILDREN; i++) {
        bafe_eclass_id child = e->children[i];
        if (child >= 0 && child < 1024) {  /* bounds check */
            inputs[i] = eclass_shapes[child];
            if (i == 0) dtype = eclass_dtypes[child];
        } else {
            inputs[i] = bafe_shape_make_2(1, 1);
        }
    }

    /* infer output shape using the op's shape function */
    const bafe_op *op = bafe_op_get(e->op_name);
    if (op && op->shape_fn) {
        out_shape = op->shape_fn(inputs, e->n_children, &e->attrs);
    }

    return bafe_cost_node(m, e->op_name, inputs, e->n_children,
                          &e->attrs, &out_shape, dtype);
}

void bafe_extract_run(const bafe_egraph *eg_, const bafe_cost_model *m,
                      const bafe_graph *g_for_shapes,
                      const bafe_eclass_id *node_to_eclass,
                      bafe_plan *plan, int *eclass_to_plan) {
    bafe_egraph *eg = (bafe_egraph *)eg_;  /* cast away const for find() */
    plan->n = 0;
    int n_classes = eg->n_total_classes_allocated;

    /* Build eclass -> shape mapping for shape-aware cost computation.
     * Heap-allocated because BAFE_EG_MAX_CLASSES can be large. */
    bafe_shape *eclass_shapes = (bafe_shape *)malloc(sizeof(bafe_shape) * BAFE_EG_MAX_CLASSES);
    bafe_dtype *eclass_dtypes = (bafe_dtype *)malloc(sizeof(bafe_dtype) * BAFE_EG_MAX_CLASSES);
    if (!eclass_shapes || !eclass_dtypes) {
        free(eclass_shapes); free(eclass_dtypes);
        return;
    }
    _build_eclass_shapes(eg, g_for_shapes, node_to_eclass,
                         eclass_shapes, eclass_dtypes);

    /* Initialize each class's plan with +inf cost. */
    for (int cid = 0; cid < n_classes; cid++) {
        if (bafe_egraph_find(eg, cid) != cid) {
            eclass_to_plan[cid] = -1;
            continue;
        }
        bafe_plan_node *p = &plan->items[plan->n];
        memset(p, 0, sizeof(*p));
        p->cost = INFINITY;
        eclass_to_plan[cid] = plan->n;
        plan->n++;
    }

    /* Bellman-Ford-style relaxation: iterate until no improvement.
     * Each iteration, for every class, try every e-node and compute
     * cost = node_cost + sum(child_costs). */
    bool changed = true;
    int iters = 0;
    while (changed && iters < 100) {
        changed = false;
        iters++;
        for (int cid = 0; cid < n_classes; cid++) {
            if (bafe_egraph_find(eg, cid) != cid) continue;
            int plan_idx = eclass_to_plan[cid];
            if (plan_idx < 0) continue;
            bafe_plan_node *best = &plan->items[plan_idx];
            const bafe_eclass *cls = &eg->classes[cid];
            for (int i = 0; i < cls->n_nodes; i++) {
                const bafe_enode *e = &cls->nodes[i];
                /* compute cost: node_cost + sum of child best costs */
                double child_total = 0.0;
                bool children_ok = true;
                int child_plan[BAFE_MAX_CHILDREN];
                for (int j = 0; j < e->n_children; j++) {
                    bafe_eclass_id child_root = bafe_egraph_find(eg, e->children[j]);
                    if (child_root < 0 || child_root >= eg->n_total_classes_allocated) {
                        children_ok = false; break;
                    }
                    int child_idx = eclass_to_plan[child_root];
                    if (child_idx < 0) { children_ok = false; break; }
                    double c = plan->items[child_idx].cost;
                    if (!isfinite(c)) { children_ok = false; break; }
                    child_total += c;
                    child_plan[j] = child_idx;
                }
                if (!children_ok) continue;
                double node_c = _node_cost(m, e, eclass_shapes, eclass_dtypes);
                double total = node_c + child_total;
                if (total < best->cost) {
                    best->enode = *e;
                    best->cost = total;
                    for (int j = 0; j < e->n_children; j++) best->child_plan[j] = child_plan[j];
                    changed = true;
                }
            }
        }
    }
    (void)g_for_shapes;
    free(eclass_shapes);
    free(eclass_dtypes);
}

bafe_node_id bafe_extract_build_graph(const bafe_egraph *eg,
                                       const bafe_plan *plan,
                                       const int *eclass_to_plan,
                                       bafe_eclass_id root,
                                       bafe_graph *out_graph,
                                       int *eclass_visited,
                                       int visit_marker) {
    bafe_eclass_id root_canon = bafe_egraph_find((bafe_egraph *)eg, root);
    int plan_idx = eclass_to_plan[root_canon];
    if (plan_idx < 0) return -1;
    if (eclass_visited[root_canon] == visit_marker) {
        /* already built; we need to find the existing node id */
        /* For Phase 1 we don't track this; rely on the caller to not
         * have DAG sharing. We'll re-build. This is acceptable for the
         * small graphs we deal with. */
    }
    const bafe_plan_node *p = &plan->items[plan_idx];
    /* build children first */
    bafe_node_id children[BAFE_MAX_CHILDREN];
    for (int j = 0; j < p->enode.n_children; j++) {
        bafe_eclass_id child_root = bafe_egraph_find((bafe_egraph *)eg, p->enode.children[j]);
        bafe_node_id child_id = bafe_extract_build_graph(eg, plan, eclass_to_plan,
                                                          child_root, out_graph,
                                                          eclass_visited, visit_marker);
        if (child_id < 0) return -1;
        children[j] = child_id;
    }
    /* add this node */
    bafe_node_id new_id;
    if (strcmp(p->enode.op_name, "input") == 0) {
        /* Should not happen at this point; bafe_optimize handles inputs
         * via the original graph. Fall through to a generic add. */
        new_id = -1;
    } else if (strcmp(p->enode.op_name, "constant") == 0) {
        bafe_shape scalar = bafe_shape_scalar();
        new_id = bafe_graph_add_constant(out_graph, p->enode.attrs.scalar_value,
                                          &scalar, BAFE_DTYPE_F32);
    } else {
        new_id = bafe_graph_add(out_graph, p->enode.op_name, children, p->enode.n_children, &p->enode.attrs);
    }
    if (new_id >= 0) eclass_visited[root_canon] = visit_marker;
    return new_id;
}
