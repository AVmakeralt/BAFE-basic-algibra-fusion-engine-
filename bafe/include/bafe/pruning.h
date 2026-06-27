/* bafe/pruning.h - multi-tier pruning with time budget (Phase 2, issue #4)
 *
 * A resource-bounded search controller that converts a wall-clock time
 * budget into structured per-stage limits. Decides which rewrite
 * alternatives survive at each tier.
 *
 * Tiers:
 *   A. Hard structural pruning (invalid shapes, illegal fusion)  [always]
 *   B. Heuristic scoring + threshold cut                        [always]
 *   C. Beam search (keep top-k alternatives)                     [>=10ms]
 *   D. Stochastic survival (Boltzmann sampling)                  [>=100ms]
 *
 * Time-budget regimes:
 *   1 ms  -> greedy only (A + B, beam_width=1, no stochastic)
 *   10 ms -> light (A + B + C, beam_width=4)
 *   100 ms -> e-graph + beam (A + B + C + D, beam_width=16)
 *   1000+ ms -> deep (all tiers, beam_width=64, more iters)
 *
 * Anytime property: tracks the best-so-far optimized graph. If the
 * time budget is hit mid-search, returns the best found so far.
 *
 * Kill switches (hard caps regardless of time budget):
 *   max_nodes, max_rewrites, max_egraph_size
 */
#ifndef BAFE_PRUNING_H
#define BAFE_PRUNING_H

#include "bafe/ir.h"
#include "bafe/rewrite.h"
#include "bafe/search.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Which tiers to run, derived from the time budget. */
typedef enum {
    BAFE_REGIME_GREEDY     = 0,  /* <= 1 ms:   A + B only, beam=1 */
    BAFE_REGIME_LIGHT      = 1,  /* <= 10 ms:  A + B + C, beam=4 */
    BAFE_REGIME_BEAM       = 2,  /* <= 100 ms: A + B + C + D, beam=16 */
    BAFE_REGIME_DEEP       = 3,  /* > 100 ms:  all tiers, beam=64 */
} bafe_pruning_regime;

/* Configuration for the pruning controller. */
typedef struct {
    int time_budget_ms;          /* wall-clock limit (0 = no limit) */
    int max_nodes;               /* hard cap on graph size */
    int max_rewrites;            /* hard cap on rewrites materialized */
    int max_egraph_size;         /* hard cap on e-graph classes */
    int beam_width;              /* Level C: keep top-k alternatives (0 = auto from regime) */
    double heuristic_threshold;  /* Level B: drop alts with score below this (0.0 = keep all) */
    double temperature;          /* Level D: Boltzmann temperature (0 = greedy) */
    uint32_t seed;               /* Level D: PRNG seed */
    bool enable_anytime;         /* track best-so-far for interruption */
} bafe_pruning_config;

/* Statistics from a pruning run. */
typedef struct {
    bafe_pruning_regime regime;
    int tier_a_passed;           /* alts that passed structural pruning */
    int tier_b_passed;           /* alts that passed heuristic scoring */
    int tier_c_kept;             /* alts kept by beam search */
    int tier_d_materialized;     /* alts materialized by stochastic survival */
    int total_alts_found;
    int best_cost;               /* cost of the best alternative found (x1000) */
    double elapsed_ms;
    bool was_interrupted;        /* true if time budget was hit */
} bafe_pruning_stats;

/* Default config: no time limit, moderate beam, anytime enabled. */
bafe_pruning_config bafe_pruning_config_default(void);

/* Determine the regime from a time budget. */
bafe_pruning_regime bafe_pruning_regime_from_budget(int time_budget_ms);

/* Get the default beam width for a regime. */
int bafe_pruning_beam_width_for_regime(bafe_pruning_regime regime);

/* Get the default max_iters for a regime (stochastic passes). */
int bafe_pruning_iters_for_regime(bafe_pruning_regime regime);

/* Run the multi-tier pruning controller on a graph.
 *
 * This is the main entry point. It:
 *   1. Determines the regime from time_budget_ms
 *   2. Runs Level A (structural) on all alternatives
 *   3. Runs Level B (heuristic) on survivors
 *   4. Runs Level C (beam) on survivors if regime >= LIGHT
 *   5. Runs Level D (stochastic) if regime >= BEAM
 *   6. Tracks best-so-far for anytime property
 *   7. Enforces all kill switches
 *
 * The graph is mutated (alternatives may be materialized).
 * The surviving alternatives are written to `out`.
 *
 * Returns 0 on success, non-zero on error. Stats are written to `stats`. */
int bafe_pruning_run(bafe_graph *g, bafe_alt_list *out,
                     const bafe_pruning_config *config,
                     bafe_pruning_stats *stats);

/* Convenience: run pruning with a time budget (uses default config
 * for other params). Returns 0 on success. */
int bafe_pruning_run_with_budget(bafe_graph *g, bafe_alt_list *out,
                                  int time_budget_ms,
                                  bafe_pruning_stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_PRUNING_H */
