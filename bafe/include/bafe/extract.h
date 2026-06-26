/* bafe/extract.h - extract the min-cost program from an e-graph
 *
 * Standard e-graph extraction: dynamic programming over e-classes.
 * For each e-class, find the cheapest e-node, where the cost of an e-node
 * is its node cost plus the cost of its children's best e-classes.
 *
 * Cycles in the e-graph (rare but possible after aggressive rewrites)
 * are handled by iteratively updating costs until convergence.
 */
#ifndef BAFE_EXTRACT_H
#define BAFE_EXTRACT_H

#include "bafe/ir.h"
#include "bafe/egraph.h"
#include "bafe/cost.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A plan node: chosen e-node + its children's chosen plan indices. */
typedef struct {
    bafe_enode    enode;          /* chosen e-node (with canonical children) */
    double        cost;
    int           child_plan[BAFE_MAX_CHILDREN];  /* index into plan array */
} bafe_plan_node;

typedef struct {
    bafe_plan_node items[BAFE_EG_MAX_CLASSES];
    int n;
} bafe_plan;

/* Run extraction. Returns a plan covering all e-classes.
 * Each e-class id maps to its best plan via the `eclass_to_plan` array
 * (caller-allocated, size = eg->n_total_classes_allocated). */
void bafe_extract_run(const bafe_egraph *eg, const bafe_cost_model *m,
                      const bafe_graph *g_for_shapes,
                      bafe_plan *plan, int *eclass_to_plan);

/* Build a fresh graph containing only the chosen e-nodes, rooted at the
 * given e-class. Returns the new output node id (or -1 on error). */
bafe_node_id bafe_extract_build_graph(const bafe_egraph *eg,
                                       const bafe_plan *plan,
                                       const int *eclass_to_plan,
                                       bafe_eclass_id root,
                                       bafe_graph *out_graph,
                                       int *eclass_visited, /* scratch, size = n_total_classes_allocated */
                                       int visit_marker);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_EXTRACT_H */
