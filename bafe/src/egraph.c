/* bafe/egraph.c - e-graph implementation
 *
 * Textbook congruence closure with union-find. ENodes are hashed by
 * (op_name, attrs, canonical_children). When two eclasses are unioned,
 * we re-canonicalize all enodes and detect new congruences.
 */
#include "bafe/egraph.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* ------------------------------------------------------------------ */
/* ENode hashing/equality                                             */
/* ------------------------------------------------------------------ */

static bool _attrs_eq(const bafe_op_attrs *a, const bafe_op_attrs *b) {
    if (a->n_axes != b->n_axes) return false;
    if (a->n_perm != b->n_perm) return false;
    if (a->n_shape != b->n_shape) return false;
    if (a->keepdims != b->keepdims) return false;
    if (a->has_scalar != b->has_scalar) return false;
    if (a->has_scalar && a->scalar_value != b->scalar_value) return false;
    for (int i = 0; i < a->n_axes; i++) if (a->axes[i] != b->axes[i]) return false;
    for (int i = 0; i < a->n_perm; i++) if (a->perm[i] != b->perm[i]) return false;
    for (int i = 0; i < a->n_shape; i++) if (a->shape[i] != b->shape[i]) return false;
    /* input name must distinguish different inputs */
    if (strncmp(a->name, b->name, BAFE_MAX_ATTR_LEN) != 0) return false;
    return true;
}

static bool _enode_eq(const bafe_enode *a, const bafe_enode *b) {
    if (a->op_name != b->op_name) {
        /* op_names are interned via the registry; but we should still
         * compare by string to be safe across imports. */
        if (!a->op_name || !b->op_name) return false;
        if (strcmp(a->op_name, b->op_name) != 0) return false;
    }
    if (a->n_children != b->n_children) return false;
    if (!_attrs_eq(&a->attrs, &b->attrs)) return false;
    for (int i = 0; i < a->n_children; i++) {
        if (a->children[i] != b->children[i]) return false;
    }
    return true;
}

/* FNV-1a hash */
static uint32_t _hash_enode(const bafe_enode *e) {
    uint32_t h = 2166136261u;
    const char *s = e->op_name ? e->op_name : "";
    while (*s) { h ^= (uint8_t)*s++; h *= 16777619u; }
    /* hash attrs as bytes, INCLUDING the name field (which distinguishes
     * different input nodes from each other) */
    const uint8_t *p = (const uint8_t *)&e->attrs;
    size_t attr_hash_len = sizeof(bafe_op_attrs);
    for (size_t i = 0; i < attr_hash_len; i++) {
        h ^= p[i]; h *= 16777619u;
    }
    for (int i = 0; i < e->n_children; i++) {
        h ^= (uint32_t)e->children[i]; h *= 16777619u;
    }
    return h;
}

/* ------------------------------------------------------------------ */
/* Init                                                               */
/* ------------------------------------------------------------------ */

void bafe_egraph_init(bafe_egraph *eg) {
    memset(eg, 0, sizeof(*eg));
    eg->congruence_cap = (int)(sizeof(eg->congruence) / sizeof(eg->congruence[0]));
}

/* ------------------------------------------------------------------ */
/* Union-find                                                         */
/* ------------------------------------------------------------------ */

static bafe_eclass_id _uf_find(bafe_egraph *eg, bafe_eclass_id x) {
    if (x < 0 || x >= eg->n_total_classes_allocated) return x;
    bafe_eclass_id root = x;
    while (eg->parent[root] != root) root = eg->parent[root];
    /* path compression */
    bafe_eclass_id cur = x;
    while (eg->parent[cur] != root) {
        bafe_eclass_id nxt = eg->parent[cur];
        eg->parent[cur] = root;
        cur = nxt;
    }
    return root;
}

bafe_eclass_id bafe_egraph_find(bafe_egraph *eg, bafe_eclass_id x) {
    return _uf_find(eg, x);
}

static bafe_eclass_id _uf_make(bafe_egraph *eg) {
    if (eg->n_total_classes_allocated >= BAFE_EG_MAX_CLASSES) return -1;
    bafe_eclass_id id = eg->n_total_classes_allocated++;
    eg->parent[id] = id;
    eg->rank[id] = 0;
    eg->classes[id].n_nodes = 0;
    return id;
}

static void _uf_union(bafe_egraph *eg, bafe_eclass_id a, bafe_eclass_id b) {
    bafe_eclass_id ra = _uf_find(eg, a);
    bafe_eclass_id rb = _uf_find(eg, b);
    if (ra == rb) return;
    if (eg->rank[ra] < eg->rank[rb]) { bafe_eclass_id t = ra; ra = rb; rb = t; }
    eg->parent[rb] = ra;
    if (eg->rank[ra] == eg->rank[rb]) eg->rank[ra]++;
    /* migrate e-nodes from the absorbed class (rb) into the new leader (ra) */
    bafe_eclass *src = &eg->classes[rb];
    bafe_eclass *dst = &eg->classes[ra];
    for (int i = 0; i < src->n_nodes; i++) {
        if (dst->n_nodes >= BAFE_EG_MAX_ENODES_PER_CLASS) break;
        dst->nodes[dst->n_nodes++] = src->nodes[i];
    }
    src->n_nodes = 0;
}

void bafe_egraph_declare_equiv(bafe_egraph *eg, bafe_eclass_id a, bafe_eclass_id b) {
    if (eg->n_pending >= BAFE_EG_MAX_PENDING) return;
    eg->pending[eg->n_pending].a = a;
    eg->pending[eg->n_pending].b = b;
    eg->n_pending++;
}

/* ------------------------------------------------------------------ */
/* Congruence table                                                   */
/* ------------------------------------------------------------------ */

static bafe_eclass_id _congruence_lookup(bafe_egraph *eg, const bafe_enode *e) {
    uint32_t h = _hash_enode(e);
    for (int probe = 0; probe < eg->congruence_cap; probe++) {
        int idx = (int)((h + (uint32_t)probe) % (uint32_t)eg->congruence_cap);
        if (!eg->congruence[idx].used) return -1;
        if (_enode_eq(&eg->congruence[idx].key, e)) return eg->congruence[idx].value;
    }
    return -1;
}

static void _congruence_insert(bafe_egraph *eg, const bafe_enode *e, bafe_eclass_id cid) {
    uint32_t h = _hash_enode(e);
    for (int probe = 0; probe < eg->congruence_cap; probe++) {
        int idx = (int)((h + (uint32_t)probe) % (uint32_t)eg->congruence_cap);
        if (!eg->congruence[idx].used) {
            eg->congruence[idx].key = *e;
            eg->congruence[idx].value = cid;
            eg->congruence[idx].used = true;
            return;
        }
        if (_enode_eq(&eg->congruence[idx].key, e)) {
            /* already present; just update value */
            eg->congruence[idx].value = cid;
            return;
        }
    }
    /* table full; silently drop. Should not happen for our sizes. */
}

/* ------------------------------------------------------------------ */
/* Add enode                                                          */
/* ------------------------------------------------------------------ */

bafe_eclass_id bafe_egraph_add(bafe_egraph *eg, const char *op_name,
                                const bafe_op_attrs *attrs,
                                const bafe_eclass_id *children, int n_children) {
    /* canonicalize children */
    bafe_enode e;
    memset(&e, 0, sizeof(e));
    e.op_name = op_name;
    e.attrs = attrs ? *attrs : bafe_op_attrs_default();
    e.n_children = n_children;
    for (int i = 0; i < n_children; i++) e.children[i] = _uf_find(eg, children[i]);

    /* check congruence */
    bafe_eclass_id existing = _congruence_lookup(eg, &e);
    if (existing >= 0) return _uf_find(eg, existing);

    /* new class */
    bafe_eclass_id cid = _uf_make(eg);
    if (cid < 0) return -1;
    if (eg->classes[cid].n_nodes >= BAFE_EG_MAX_ENODES_PER_CLASS) return -1;
    eg->classes[cid].nodes[eg->classes[cid].n_nodes++] = e;
    _congruence_insert(eg, &e, cid);
    return cid;
}

/* ------------------------------------------------------------------ */
/* Rebuild (congruence closure saturation)                            */
/* ------------------------------------------------------------------ */

int bafe_egraph_rebuild(bafe_egraph *eg, int max_iters) {
    int iters = 0;
    while (iters < max_iters) {
        iters++;
        /* 1. process pending unions */
        bool any_union = false;
        if (eg->n_pending > 0) {
            for (int i = 0; i < eg->n_pending; i++) {
                _uf_union(eg, eg->pending[i].a, eg->pending[i].b);
                any_union = true;
            }
            eg->n_pending = 0;
        }

        /* 2. rebuild congruence table with canonical children */
        memset(eg->congruence, 0, sizeof(eg->congruence));
        for (int cid = 0; cid < eg->n_total_classes_allocated; cid++) {
            if (_uf_find(eg, cid) != cid) continue;
            bafe_eclass *cls = &eg->classes[cid];
            for (int i = 0; i < cls->n_nodes; i++) {
                bafe_enode canon = cls->nodes[i];
                for (int j = 0; j < canon.n_children; j++) {
                    canon.children[j] = _uf_find(eg, canon.children[j]);
                }
                bafe_eclass_id existing = _congruence_lookup(eg, &canon);
                if (existing >= 0 && existing != cid) {
                    /* congruence detected! schedule union */
                    if (eg->n_pending < BAFE_EG_MAX_PENDING) {
                        eg->pending[eg->n_pending].a = existing;
                        eg->pending[eg->n_pending].b = cid;
                        eg->n_pending++;
                        any_union = true;
                    }
                } else {
                    _congruence_insert(eg, &canon, cid);
                }
            }
        }

        if (!any_union) break;
    }
    return iters;
}

/* ------------------------------------------------------------------ */
/* Stats + dump                                                       */
/* ------------------------------------------------------------------ */

int bafe_egraph_num_classes(const bafe_egraph *eg) {
    int n = 0;
    /* count distinct canonical roots */
    bool seen[BAFE_EG_MAX_CLASSES] = {false};
    for (int i = 0; i < eg->n_total_classes_allocated; i++) {
        /* we can't call _uf_find on a const eg, so replicate */
        bafe_eclass_id root = i;
        while (eg->parent[root] != root) root = eg->parent[root];
        if (!seen[root]) { seen[root] = true; n++; }
    }
    return n;
}

int bafe_egraph_num_enodes(const bafe_egraph *eg) {
    int n = 0;
    for (int i = 0; i < eg->n_total_classes_allocated; i++) {
        bafe_eclass_id root = i;
        while (eg->parent[root] != root) root = eg->parent[root];
        if (root == i) n += eg->classes[i].n_nodes;
    }
    return n;
}

int bafe_egraph_dump(const bafe_egraph *eg, char *buf, size_t buf_size) {
    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, buf_size - pos,
        "EGraph: %d classes, %d enodes\n",
        bafe_egraph_num_classes(eg), bafe_egraph_num_enodes(eg));
    int printed = 0;
    for (int cid = 0; cid < eg->n_total_classes_allocated && printed < 50; cid++) {
        bafe_eclass_id root = cid;
        while (eg->parent[root] != root) root = eg->parent[root];
        if (root != cid) continue;
        if (eg->classes[cid].n_nodes == 0) continue;
        printed++;
        pos += (size_t)snprintf(buf + pos, buf_size - pos, "  eclass %d:\n", cid);
        for (int i = 0; i < eg->classes[cid].n_nodes; i++) {
            const bafe_enode *e = &eg->classes[cid].nodes[i];
            pos += (size_t)snprintf(buf + pos, buf_size - pos,
                "    (%s", e->op_name);
            for (int j = 0; j < e->n_children; j++) {
                pos += (size_t)snprintf(buf + pos, buf_size - pos,
                    " %d", e->children[j]);
            }
            pos += (size_t)snprintf(buf + pos, buf_size - pos, ")\n");
        }
    }
    return (int)pos;
}

/* ------------------------------------------------------------------ */
/* Import from Graph                                                  */
/* ------------------------------------------------------------------ */

void bafe_egraph_import_graph(bafe_egraph *eg, const bafe_graph *g,
                              bafe_eclass_id *node_to_eclass) {
    bafe_node_id order[BAFE_MAX_NODES];
    int n = bafe_graph_topo_order((bafe_graph *)g, order, BAFE_MAX_NODES);
    for (int i = 0; i < n; i++) {
        bafe_node_id nid = order[i];
        const bafe_node *node = &g->nodes[nid];
        bafe_eclass_id children[BAFE_MAX_CHILDREN];
        for (int j = 0; j < node->n_children; j++) children[j] = node_to_eclass[node->children[j]];
        bafe_eclass_id cid = bafe_egraph_add(eg, node->op_name, &node->attrs, children, node->n_children);
        node_to_eclass[nid] = cid;
    }
}

void bafe_egraph_apply_alternatives(bafe_egraph *eg, const bafe_graph *g,
                                    const bafe_eclass_id *node_to_eclass_in,
                                    const bafe_alt_list *alts) {
    /* node_to_eclass_in may be missing entries for nodes that the rewrite
     * engine added to the graph *after* we imported it. We need a mutable
     * copy so we can fill in those entries on demand. */
    bafe_eclass_id nte[BAFE_MAX_NODES];
    for (int i = 0; i < g->n_nodes; i++) nte[i] = node_to_eclass_in[i];

    for (int i = 0; i < alts->n; i++) {
        const bafe_alternative *alt = &alts->items[i];
        bafe_eclass_id target = nte[alt->original_node_id];
        bafe_eclass_id children[BAFE_MAX_CHILDREN];
        bool ok = true;
        for (int j = 0; j < alt->n_children; j++) {
            bafe_node_id child_nid = alt->children[j];
            if (child_nid < 0 || child_nid >= g->n_nodes) { ok = false; break; }
            if (nte[child_nid] < 0) {
                /* This child was added by the rewrite engine but not yet
                 * imported into the e-graph. Import it now. */
                const bafe_node *cn = &g->nodes[child_nid];
                bafe_eclass_id cc[BAFE_MAX_CHILDREN];
                for (int k = 0; k < cn->n_children; k++) {
                    if (nte[cn->children[k]] < 0) { ok = false; break; }
                    cc[k] = nte[cn->children[k]];
                }
                if (!ok) break;
                bafe_eclass_id cid = bafe_egraph_add(eg, cn->op_name, &cn->attrs, cc, cn->n_children);
                nte[child_nid] = cid;
            }
            children[j] = nte[child_nid];
        }
        if (!ok) continue;
        bafe_eclass_id alt_class = bafe_egraph_add(eg, alt->op_name, &alt->attrs, children, alt->n_children);
        if (alt_class >= 0 && target >= 0) {
            bafe_egraph_declare_equiv(eg, target, alt_class);
        }
    }
}
