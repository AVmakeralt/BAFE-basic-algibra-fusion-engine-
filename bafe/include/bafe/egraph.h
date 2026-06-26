/* bafe/egraph.h - e-graph (congruence closure over IR)
 *
 * Components:
 *   - ENode   : (op_name, attrs, child_eclass_ids)  -- hashable
 *   - EClass  : a set of equivalent ENodes (identified by an int id)
 *   - Union-find over EClass ids
 *   - Congruence table: canonical ENode -> EClassId
 *
 * Supports:
 *   - add enode (returns eclass id, dedups structurally identical ones)
 *   - declare equivalence (union two eclasses)
 *   - rebuild (saturate congruence closure)
 *   - find (canonical id)
 */
#ifndef BAFE_EGRAPH_H
#define BAFE_EGRAPH_H

#include "bafe/ir.h"
#include "bafe/ops.h"
#include "bafe/rewrite.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BAFE_EG_MAX_NODES    1024
#define BAFE_EG_MAX_CLASSES  1024
#define BAFE_EG_MAX_ENODES_PER_CLASS 16
#define BAFE_EG_MAX_PENDING  512

typedef int32_t bafe_eclass_id;

typedef struct {
    const char    *op_name;       /* borrowed from registry */
    bafe_op_attrs  attrs;
    int            n_children;
    bafe_eclass_id children[BAFE_MAX_CHILDREN];
} bafe_enode;

typedef struct {
    bafe_enode nodes[BAFE_EG_MAX_ENODES_PER_CLASS];
    int n_nodes;
} bafe_eclass;

typedef struct {
    /* union-find */
    bafe_eclass_id parent[BAFE_EG_MAX_CLASSES];
    int32_t        rank[BAFE_EG_MAX_CLASSES];

    /* classes: indexed by canonical id (leader). Non-leaders have empty
     * node lists; their nodes are migrated to the leader after rebuild. */
    bafe_eclass classes[BAFE_EG_MAX_CLASSES];
    int n_classes;

    /* congruence table: maps a canonical enode to its eclass id.
     * We use linear probing. */
    struct {
        bafe_enode key;
        bafe_eclass_id value;
        bool used;
    } congruence[BAFE_EG_MAX_NODES * 2];
    int congruence_cap;

    /* pending unions from declare_equivalent / new congruences */
    struct {
        bafe_eclass_id a, b;
    } pending[BAFE_EG_MAX_PENDING];
    int n_pending;

    int n_total_classes_allocated;  /* monotonic counter */
} bafe_egraph;

/* lifecycle */
void bafe_egraph_init(bafe_egraph *eg);

/* add an enode (children must already be canonical). returns its eclass id. */
bafe_eclass_id bafe_egraph_add(bafe_egraph *eg, const char *op_name,
                                const bafe_op_attrs *attrs,
                                const bafe_eclass_id *children, int n_children);

/* declare two eclasses equivalent (records a pending union). */
void bafe_egraph_declare_equiv(bafe_egraph *eg, bafe_eclass_id a, bafe_eclass_id b);

/* saturate congruence closure. returns iteration count. */
int bafe_egraph_rebuild(bafe_egraph *eg, int max_iters);

/* canonical id of an eclass. */
bafe_eclass_id bafe_egraph_find(bafe_egraph *eg, bafe_eclass_id x);

/* import a graph into the egraph. returns a mapping nodeId -> eclass_id
 * (caller-allocated array of size g->n_nodes). */
void bafe_egraph_import_graph(bafe_egraph *eg, const bafe_graph *g,
                              bafe_eclass_id *node_to_eclass);

/* apply all alternatives from the rewrite engine to the egraph. */
void bafe_egraph_apply_alternatives(bafe_egraph *eg, const bafe_graph *g,
                                    const bafe_eclass_id *node_to_eclass,
                                    const bafe_alt_list *alts);

/* stats */
int bafe_egraph_num_classes(const bafe_egraph *eg);
int bafe_egraph_num_enodes(const bafe_egraph *eg);

/* debug dump */
int bafe_egraph_dump(const bafe_egraph *eg, char *buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_EGRAPH_H */
