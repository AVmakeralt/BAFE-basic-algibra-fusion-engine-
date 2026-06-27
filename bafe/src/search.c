/* bafe/search.c - stochastic search layer implementation
 *
 * Multi-pass rewrite exploration with budget control.
 *
 * Algorithm:
 *   for iter in 0..max_iters:
 *     1. Find all deterministic alternatives on current graph
 *     2. For each alternative, compute a "score" (cost delta)
 *     3. Sample which alternatives to materialize using temperature:
 *        P(materialize) = exp(-cost_delta / T)  (Boltzmann)
 *        At high T, all alternatives are roughly equally likely.
 *        At low T, only cost-reducing alternatives are materialized.
 *     4. Materialize selected alternatives by adding them to the graph
 *     5. The new nodes unlock new rule matches for the next pass
 *
 * The PRNG is xorshift128 for reproducibility.
 */
#define _POSIX_C_SOURCE 200809L
#include "bafe/search.h"
#include "bafe/cost.h"
#include "bafe/ops.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Default budget                                                      */
/* ------------------------------------------------------------------ */

bafe_search_budget bafe_search_budget_default(void) {
    bafe_search_budget b;
    /* The superoptimizer runs automatically on every @bafe.jit call.
     * Default: single-pass deterministic rewrites + full e-graph + cost model.
     * This is fast and correct. Multi-pass stochastic search is enabled
     * via @bafe.jit(deep=True) or @bafe.jit(iters=N). */
    b.max_iters = 1;
    b.max_nodes = 4096;
    b.max_rewrites = 512;
    b.time_budget_ms = 0;
    b.temperature = 1.0;
    b.seed = 0xBAFE5EEDu;
    b.enable_multi_pass = false;
    b.deep_search = false;
    return b;
}

bafe_search_budget bafe_search_budget_deep(void) {
    bafe_search_budget b = bafe_search_budget_default();
    /* Deep search: maximum exploration for large workloads.
     * Uses more memory but finds more optimizations. */
    b.max_iters = 16;
    b.max_nodes = 16384;
    b.max_rewrites = 2000;
    b.temperature = 2.0;        /* more exploration */
    b.deep_search = true;
    return b;
}

/* ------------------------------------------------------------------ */
/* xorshift128 PRNG                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t s[4];
} prng_state;

static uint32_t _prng_next(prng_state *p) {
    /* xorshift128 (Marsaglia) */
    uint32_t t = p->s[0] ^ (p->s[0] << 11);
    p->s[0] = p->s[1];
    p->s[1] = p->s[2];
    p->s[2] = p->s[3];
    p->s[3] = p->s[3] ^ (p->s[3] >> 19) ^ (t ^ (t >> 8));
    return p->s[3];
}

static void _prng_seed(prng_state *p, uint32_t seed) {
    /* splitmix32 to expand the seed into 4 state words */
    uint32_t z = seed;
    for (int i = 0; i < 4; i++) {
        z += 0x9E3779B9u;
        uint32_t t = z;
        t = (t ^ (t >> 16)) * 0x85EBCA6Bu;
        t = (t ^ (t >> 13)) * 0xC2B2AE35u;
        t = t ^ (t >> 16);
        p->s[i] = t;
    }
    /* ensure non-zero state */
    for (int i = 0; i < 4; i++) {
        if (p->s[i] == 0) p->s[i] = 0xDEADBEEFu;
    }
}

static double _prng_uniform(prng_state *p) {
    /* uniform in [0, 1) */
    return (double)(_prng_next(p) >> 8) / (double)(1u << 24);
}

/* ------------------------------------------------------------------ */
/* Time helper                                                         */
/* ------------------------------------------------------------------ */

static double _now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------ */
/* Alternative scoring                                                 */
/* ------------------------------------------------------------------ */

/* Estimate the cost delta of materializing an alternative.
 * We compare the cost of the original node vs. the cost of the
 * alternative's op (with dummy shapes). Negative delta = improvement. */
static double _alt_cost_delta(const bafe_graph *g, const bafe_alternative *alt) {
    bafe_cost_model cm = bafe_cost_model_default();
    const bafe_node *orig = &g->nodes[alt->original_node_id];

    /* Use real shapes from the graph nodes */
    bafe_shape orig_inputs[BAFE_MAX_CHILDREN];
    bafe_shape alt_inputs[BAFE_MAX_CHILDREN];
    for (int i = 0; i < orig->n_children && i < BAFE_MAX_CHILDREN; i++) {
        orig_inputs[i] = g->nodes[orig->children[i]].shape;
    }
    for (int i = 0; i < alt->n_children && i < BAFE_MAX_CHILDREN; i++) {
        alt_inputs[i] = g->nodes[alt->children[i]].shape;
    }

    /* Infer output shapes using the ops' shape functions */
    const bafe_op *orig_op = bafe_op_get(orig->op_name);
    bafe_shape orig_out = orig_op && orig_op->shape_fn
        ? orig_op->shape_fn(orig_inputs, orig->n_children, &orig->attrs)
        : bafe_shape_make_2(1, 1);
    const bafe_op *alt_op = bafe_op_get(alt->op_name);
    bafe_shape alt_out = alt_op && alt_op->shape_fn
        ? alt_op->shape_fn(alt_inputs, alt->n_children, &alt->attrs)
        : bafe_shape_make_2(1, 1);

    bafe_dtype dtype = orig->n_children > 0 ? g->nodes[orig->children[0]].dtype : BAFE_DTYPE_F32;

    double orig_cost = bafe_cost_node(&cm, orig->op_name, orig_inputs,
                                       orig->n_children, &orig->attrs,
                                       &orig_out, dtype);
    double alt_cost = bafe_cost_node(&cm, alt->op_name, alt_inputs,
                                      alt->n_children, &alt->attrs,
                                      &alt_out, dtype);
    return alt_cost - orig_cost;
}

/* ------------------------------------------------------------------ */
/* Stochastic search                                                   */
/* ------------------------------------------------------------------ */

int bafe_rewrite_stochastic(bafe_graph *g, bafe_alt_list *out,
                             const bafe_search_budget *budget) {
    bafe_search_stats stats;
    return bafe_rewrite_stochastic_stats(g, out, budget, &stats);
}

int bafe_rewrite_stochastic_stats(bafe_graph *g, bafe_alt_list *out,
                                   const bafe_search_budget *budget_in,
                                   bafe_search_stats *stats) {
    bafe_search_budget budget = budget_in ? *budget_in : bafe_search_budget_default();
    memset(stats, 0, sizeof(*stats));
    out->n = 0;

    prng_state prng;
    _prng_seed(&prng, budget.seed);

    double start_ms = _now_ms();
    int nodes_at_start = g->n_nodes;
    int rewrites_done = 0;

    for (int iter = 0; iter < budget.max_iters; iter++) {
        stats->iters_done = iter + 1;

        /* time budget check */
        if (budget.time_budget_ms > 0) {
            double elapsed = _now_ms() - start_ms;
            if (elapsed > budget.time_budget_ms) break;
        }

        /* node budget check */
        if (g->n_nodes >= budget.max_nodes) break;
        if (rewrites_done >= budget.max_rewrites) break;

        /* Step 1: find all deterministic alternatives on the current graph */
        bafe_alt_list pass_alts;
        int n_found = bafe_rewrite_find(g, &pass_alts);
        if (n_found == 0) break;  /* no more rules apply -> converged */

        /* Step 2: copy new alternatives to the output list (dedup by
         * original_node_id + op_name, since the deterministic pass may
         * re-discover the same alts on unchanged parts of the graph). */
        for (int i = 0; i < n_found && out->n < BAFE_MAX_ALTERNATIVES; i++) {
            const bafe_alternative *a = &pass_alts.items[i];
            /* check if we already have this alternative */
            bool dup = false;
            for (int j = 0; j < out->n; j++) {
                const bafe_alternative *b = &out->items[j];
                if (b->original_node_id == a->original_node_id &&
                    b->op_name == a->op_name &&
                    b->n_children == a->n_children) {
                    /* same target op + same original node -> assume dup */
                    bool same = true;
                    for (int k = 0; k < a->n_children; k++) {
                        if (b->children[k] != a->children[k]) { same = false; break; }
                    }
                    if (same) { dup = true; break; }
                }
            }
            if (!dup) {
                out->items[out->n++] = *a;
                stats->alts_found++;
            }
        }

        if (!budget.enable_multi_pass) break;  /* deterministic mode */

        /* Step 3: sample which alternatives to materialize.
         * P(materialize) = exp(-cost_delta / T) clamped to [0.05, 1.0].
         * At T=0 (greedy), only materialize cost-reducing alts (delta < 0).
         * At high T, materialize ~all alts (exploration). */
        for (int i = 0; i < n_found; i++) {
            if (rewrites_done >= budget.max_rewrites) break;
            if (g->n_nodes >= budget.max_nodes) break;

            const bafe_alternative *a = &pass_alts.items[i];
            double delta = _alt_cost_delta(g, a);
            double T = budget.temperature > 0 ? budget.temperature : 1e-6;
            double p = exp(-delta / T);
            if (p < 0.05) p = 0.05;  /* always at least 5% chance, to explore */
            if (p > 1.0) p = 1.0;

            if (_prng_uniform(&prng) < p) {
                /* materialize: add the alternative as a new node in the graph.
                 * The next iteration's bafe_rewrite_find will see this new
                 * node and may match rules against it. */
                bafe_node_id new_id = bafe_rewrite_materialize(g, a);
                if (new_id >= 0) {
                    rewrites_done++;
                    stats->alts_materialized++;
                    stats->nodes_added = g->n_nodes - nodes_at_start;
                }
            }
        }
    }

    stats->elapsed_ms = _now_ms() - start_ms;
    return out->n;
}
