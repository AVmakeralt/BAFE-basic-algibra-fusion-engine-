/* bafe/search.h - stochastic search layer (Phase 2, issue #1)
 *
 * The deterministic rewrite engine (bafe_rewrite_find) does ONE pass over
 * the graph: it applies every rule to every node once, collecting all
 * single-step alternatives. It never re-applies rules to the NEW nodes
 * created by previous rewrites, so multi-step transformations are missed.
 *
 * The stochastic search layer fixes this by doing multiple passes:
 *   1. Find all current alternatives (deterministic pass)
 *   2. Randomly select a subset (temperature-controlled)
 *   3. Materialize the selected alternatives (add them to the graph)
 *   4. The new nodes unlock new rule matches for the next pass
 *
 * This discovers "non-obvious" rewrites that require intermediate forms.
 *
 * Budget controller prevents combinatorial explosion:
 *   - max_iters: how many stochastic passes
 *   - max_nodes: don't let the graph grow beyond this
 *   - max_rewrites: cap on total rewrites materialized
 *   - time_budget_ms: wall-clock limit (0 = no limit)
 *   - temperature: high = explore randomly, low = exploit cost-reducing
 *   - seed: for reproducibility
 */
#ifndef BAFE_SEARCH_H
#define BAFE_SEARCH_H

#include "bafe/ir.h"
#include "bafe/rewrite.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int     max_iters;        /* number of stochastic passes (default 4) */
    int     max_nodes;        /* graph size cap (default 256) */
    int     max_rewrites;     /* total rewrites to materialize (default 64) */
    int     time_budget_ms;   /* wall-clock limit, 0 = no limit (default 0) */
    double  temperature;      /* 0.0 = greedy, high = random (default 1.0) */
    uint32_t seed;            /* PRNG seed (default 0xBAFE5EED) */
    bool    enable_multi_pass;/* if false, degrades to deterministic (default true) */
} bafe_search_budget;

/* Default budget: moderate exploration, reproducible. */
bafe_search_budget bafe_search_budget_default(void);

/* Run stochastic search on the graph.
 *
 * MUTATES the graph: materializes selected alternatives by adding new
 * nodes (via bafe_graph_add). Returns the full list of alternatives
 * found across all passes (including ones not materialized).
 *
 * The caller feeds these alternatives into the e-graph just like
 * deterministic ones.
 *
 * Returns the number of alternatives found, or -1 on error.
 */
int bafe_rewrite_stochastic(bafe_graph *g, bafe_alt_list *out,
                             const bafe_search_budget *budget);

/* Convenience: run stochastic search and report stats.
 * Writes a human-readable summary to buf. */
typedef struct {
    int iters_done;
    int alts_found;
    int alts_materialized;
    int nodes_added;
    double elapsed_ms;
} bafe_search_stats;

int bafe_rewrite_stochastic_stats(bafe_graph *g, bafe_alt_list *out,
                                   const bafe_search_budget *budget,
                                   bafe_search_stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_SEARCH_H */
