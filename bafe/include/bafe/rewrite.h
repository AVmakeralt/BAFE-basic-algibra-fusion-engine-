/* bafe/rewrite.h - rewrite rules and engine
 *
 * A rewrite rule produces an "alternative" expression for a node:
 *   original_node_id === (op_name, attrs, child_ids...)
 *
 * The engine walks the graph once and collects all alternatives. The
 * e-graph consumes them to build equivalence classes.
 */
#ifndef BAFE_REWRITE_H
#define BAFE_REWRITE_H

#include "bafe/ir.h"
#include "bafe/ops.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BAFE_MAX_RULES 64
#define BAFE_MAX_ALTERNATIVES 1024

typedef struct {
    bafe_node_id   original_node_id;   /* the node being rewritten */
    const char    *op_name;            /* borrowed from registry */
    bafe_op_attrs  attrs;
    int            n_children;
    bafe_node_id   children[BAFE_MAX_CHILDREN];
} bafe_alternative;

typedef struct {
    bafe_alternative items[BAFE_MAX_ALTERNATIVES];
    int n;
} bafe_alt_list;

/* Walk the graph once and collect all rewrite matches. */
int bafe_rewrite_find(const bafe_graph *g, bafe_alt_list *out);

/* Default rule set (algebraic + fusion). */
int bafe_rewrite_default_count(void);
const char *bafe_rewrite_default_name(int i);

/* Helper to materialize an alternative into the graph (returns new node id). */
bafe_node_id bafe_rewrite_materialize(bafe_graph *g, const bafe_alternative *alt);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_REWRITE_H */
