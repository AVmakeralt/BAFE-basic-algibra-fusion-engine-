/* bafe/pruning.c - multi-tier pruning controller implementation
 *
 * Converts a wall-clock time budget into structured per-stage limits
 * and applies 4 tiers of pruning to the rewrite alternatives.
 */
#define _POSIX_C_SOURCE 200809L
#include "bafe/pruning.h"
#include "bafe/cost.h"
#include "bafe/ops.h"
#include "bafe/search.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Defaults + regime mapping                                           */
/* ------------------------------------------------------------------ */

bafe_pruning_config bafe_pruning_config_default(void) {
    bafe_pruning_config c;
    c.time_budget_ms = 0;          /* no limit */
    c.max_nodes = 256;
    c.max_rewrites = 64;
    c.max_egraph_size = 1024;
    c.beam_width = 0;              /* auto from regime */
    c.heuristic_threshold = 0.0;   /* keep all by default */
    c.temperature = 1.0;
    c.seed = 0xBAFE5EEDu;
    c.enable_anytime = true;
    return c;
}

bafe_pruning_regime bafe_pruning_regime_from_budget(int time_budget_ms) {
    if (time_budget_ms <= 0) return BAFE_REGIME_DEEP;  /* no limit = deep */
    if (time_budget_ms <= 1)   return BAFE_REGIME_GREEDY;
    if (time_budget_ms <= 10)  return BAFE_REGIME_LIGHT;
    if (time_budget_ms <= 100) return BAFE_REGIME_BEAM;
    return BAFE_REGIME_DEEP;
}

int bafe_pruning_beam_width_for_regime(bafe_pruning_regime regime) {
    switch (regime) {
        case BAFE_REGIME_GREEDY: return 1;
        case BAFE_REGIME_LIGHT:  return 4;
        case BAFE_REGIME_BEAM:   return 16;
        case BAFE_REGIME_DEEP:   return 64;
    }
    return 4;
}

int bafe_pruning_iters_for_regime(bafe_pruning_regime regime) {
    switch (regime) {
        case BAFE_REGIME_GREEDY: return 1;   /* single pass */
        case BAFE_REGIME_LIGHT:  return 2;
        case BAFE_REGIME_BEAM:   return 4;
        case BAFE_REGIME_DEEP:   return 8;
    }
    return 4;
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
/* Level A: Hard structural pruning                                    */
/* ------------------------------------------------------------------ */

/* Check if an alternative is structurally valid:
 *   - target node exists
 *   - all children exist
 *   - child count matches the op's arity
 *   - no cycles (child is not the node itself)
 */
static bool _level_a_valid(const bafe_graph *g, const bafe_alternative *alt) {
    if (alt->original_node_id < 0 || alt->original_node_id >= g->n_nodes) return false;
    const bafe_op *op = bafe_op_get(alt->op_name);
    if (!op) return false;
    if (alt->n_children != op->arity) return false;
    for (int i = 0; i < alt->n_children; i++) {
        if (alt->children[i] < 0 || alt->children[i] >= g->n_nodes) return false;
        /* no self-loops */
        if (alt->children[i] == alt->original_node_id) return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Level B: Heuristic scoring                                          */
/* ------------------------------------------------------------------ */

/* Score an alternative: lower = better.
 * Score = estimated cost delta (alt_cost - orig_cost).
 * Negative delta = improvement (keep). */
static double _level_b_score(const bafe_graph *g, const bafe_alternative *alt) {
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
/* Level C: Beam search (keep top-k)                                   */
/* ------------------------------------------------------------------ */

/* Sort alternatives by score (ascending = best first) and keep top-k. */
typedef struct {
    int idx;
    double score;
} scored_alt;

static int _cmp_scored(const void *a, const void *b) {
    const scored_alt *sa = (const scored_alt *)a;
    const scored_alt *sb = (const scored_alt *)b;
    if (sa->score < sb->score) return -1;
    if (sa->score > sb->score) return 1;
    return 0;
}

static int _level_c_beam(bafe_alt_list *alts, const double *scores,
                          int beam_width) {
    if (alts->n <= beam_width) return alts->n;  /* keep all */

    /* build scored array */
    scored_alt *arr = (scored_alt *)malloc(sizeof(scored_alt) * alts->n);
    if (!arr) return alts->n;
    for (int i = 0; i < alts->n; i++) {
        arr[i].idx = i;
        arr[i].score = scores[i];
    }
    qsort(arr, alts->n, sizeof(scored_alt), _cmp_scored);

    /* build the survivors list (top-k) */
    bafe_alternative survivors[BAFE_MAX_ALTERNATIVES];
    int n_survivors = beam_width < alts->n ? beam_width : alts->n;
    for (int i = 0; i < n_survivors; i++) {
        survivors[i] = alts->items[arr[i].idx];
    }
    /* copy back */
    for (int i = 0; i < n_survivors; i++) {
        alts->items[i] = survivors[i];
    }
    alts->n = n_survivors;
    free(arr);
    return n_survivors;
}

/* ------------------------------------------------------------------ */
/* Level D: Stochastic survival (Boltzmann)                            */
/* ------------------------------------------------------------------ */

/* xorshift128 PRNG (same as in search.c) */
typedef struct {
    uint32_t s[4];
} _prng;

static uint32_t _prng_next(_prng *p) {
    uint32_t t = p->s[0] ^ (p->s[0] << 11);
    p->s[0] = p->s[1];
    p->s[1] = p->s[2];
    p->s[2] = p->s[3];
    p->s[3] = p->s[3] ^ (p->s[3] >> 19) ^ (t ^ (t >> 8));
    return p->s[3];
}

static void _prng_seed(_prng *p, uint32_t seed) {
    uint32_t z = seed;
    for (int i = 0; i < 4; i++) {
        z += 0x9E3779B9u;
        uint32_t t = z;
        t = (t ^ (t >> 16)) * 0x85EBCA6Bu;
        t = (t ^ (t >> 13)) * 0xC2B2AE35u;
        t = t ^ (t >> 16);
        p->s[i] = t;
    }
    for (int i = 0; i < 4; i++) if (p->s[i] == 0) p->s[i] = 0xDEADBEEFu;
}

static double _prng_uniform(_prng *p) {
    return (double)(_prng_next(p) >> 8) / (double)(1u << 24);
}

/* Materialize alternatives that survive Boltzmann sampling.
 * P(materialize) = exp(-score / T), clamped to [0.05, 1.0]. */
static int _level_d_stochastic(bafe_graph *g, bafe_alt_list *alts,
                                const double *scores,
                                double temperature, uint32_t seed,
                                int max_rewrites, int max_nodes,
                                double deadline_ms, double start_ms) {
    _prng prng;
    _prng_seed(&prng, seed);
    int materialized = 0;

    for (int i = 0; i < alts->n; i++) {
        /* kill switches */
        if (materialized >= max_rewrites) break;
        if (g->n_nodes >= max_nodes) break;
        /* time budget check */
        if (deadline_ms > 0 && (_now_ms() - start_ms) > deadline_ms) break;

        double score = scores[i];
        double T = temperature > 0 ? temperature : 1e-6;
        double p = exp(-score / T);
        if (p < 0.05) p = 0.05;
        if (p > 1.0) p = 1.0;

        if (_prng_uniform(&prng) < p) {
            bafe_node_id new_id = bafe_rewrite_materialize(g, &alts->items[i]);
            if (new_id >= 0) materialized++;
        }
    }
    return materialized;
}

/* ------------------------------------------------------------------ */
/* Main controller                                                     */
/* ------------------------------------------------------------------ */

int bafe_pruning_run(bafe_graph *g, bafe_alt_list *out,
                     const bafe_pruning_config *config_in,
                     bafe_pruning_stats *stats) {
    bafe_pruning_config config = config_in ? *config_in : bafe_pruning_config_default();
    if (stats) memset(stats, 0, sizeof(*stats));

    double start_ms = _now_ms();
    double deadline = config.time_budget_ms > 0 ? (double)config.time_budget_ms : 0.0;

    /* Determine regime */
    bafe_pruning_regime regime = bafe_pruning_regime_from_budget(config.time_budget_ms);
    if (stats) stats->regime = regime;

    int beam_width = config.beam_width > 0 ? config.beam_width :
                     bafe_pruning_beam_width_for_regime(regime);
    int max_iters = bafe_pruning_iters_for_regime(regime);

    out->n = 0;

    /* Run multi-pass: each pass finds alts, prunes them, materializes survivors.
     * The new nodes unlock new rule matches for the next pass. */
    int total_materialized = 0;
    int nodes_at_start = g->n_nodes;

    for (int iter = 0; iter < max_iters; iter++) {
        /* time budget check */
        if (deadline > 0 && (_now_ms() - start_ms) > deadline) {
            if (stats) stats->was_interrupted = true;
            break;
        }
        /* kill switch: max_nodes */
        if (g->n_nodes >= config.max_nodes) break;
        if (total_materialized >= config.max_rewrites) break;

        /* Find all deterministic alternatives on the current graph */
        bafe_alt_list pass_alts;
        int n_found = bafe_rewrite_find(g, &pass_alts);
        if (n_found == 0) break;  /* converged */

        if (stats) stats->total_alts_found += n_found;

        /* Level A: structural pruning */
        bafe_alt_list tier_a;
        tier_a.n = 0;
        for (int i = 0; i < n_found; i++) {
            if (_level_a_valid(g, &pass_alts.items[i])) {
                tier_a.items[tier_a.n++] = pass_alts.items[i];
            }
        }
        if (stats) stats->tier_a_passed += tier_a.n;

        /* Level B: heuristic scoring + threshold cut */
        double scores[BAFE_MAX_ALTERNATIVES];
        bafe_alt_list tier_b;
        tier_b.n = 0;
        for (int i = 0; i < tier_a.n; i++) {
            double s = _level_b_score(g, &tier_a.items[i]);
            if (s <= config.heuristic_threshold || config.heuristic_threshold == 0.0) {
                tier_b.items[tier_b.n] = tier_a.items[i];
                scores[tier_b.n] = s;
                tier_b.n++;
            }
        }
        if (stats) stats->tier_b_passed += tier_b.n;

        /* Level C: beam search (keep top-k) */
        bafe_alt_list tier_c = tier_b;
        if (regime >= BAFE_REGIME_LIGHT && tier_c.n > beam_width) {
            _level_c_beam(&tier_c, scores, beam_width);
        }
        if (stats) stats->tier_c_kept += tier_c.n;

        /* copy survivors to output (dedup) */
        for (int i = 0; i < tier_c.n && out->n < BAFE_MAX_ALTERNATIVES; i++) {
            bool dup = false;
            for (int j = 0; j < out->n; j++) {
                if (out->items[j].original_node_id == tier_c.items[i].original_node_id &&
                    out->items[j].op_name == tier_c.items[i].op_name) {
                    dup = true; break;
                }
            }
            if (!dup) {
                out->items[out->n++] = tier_c.items[i];
            }
        }

        /* Level D: stochastic survival (materialize) */
        if (regime >= BAFE_REGIME_BEAM) {
            int mat = _level_d_stochastic(g, &tier_c, scores,
                                           config.temperature, config.seed + iter,
                                           config.max_rewrites - total_materialized,
                                           config.max_nodes,
                                           deadline, start_ms);
            total_materialized += mat;
            if (stats) stats->tier_d_materialized += mat;
        } else if (regime >= BAFE_REGIME_LIGHT) {
            /* In LIGHT regime, materialize the best beam alternative (greedy) */
            if (tier_c.n > 0 && total_materialized < config.max_rewrites &&
                g->n_nodes < config.max_nodes) {
                /* materialize the lowest-score (best) alt */
                bafe_node_id new_id = bafe_rewrite_materialize(g, &tier_c.items[0]);
                if (new_id >= 0) total_materialized++;
            }
        } else {
            /* GREEDY regime: no materialization, just collect alts */
        }
    }

    /* compute best cost (use the first alt's score as a proxy) */
    if (stats && out->n > 0) {
        double best = _level_b_score(g, &out->items[0]);
        for (int i = 1; i < out->n; i++) {
            double s = _level_b_score(g, &out->items[i]);
            if (s < best) best = s;
        }
        stats->best_cost = (int)(best * 1000);
    }

    if (stats) {
        stats->elapsed_ms = _now_ms() - start_ms;
    }
    (void)nodes_at_start;
    return 0;
}

int bafe_pruning_run_with_budget(bafe_graph *g, bafe_alt_list *out,
                                  int time_budget_ms,
                                  bafe_pruning_stats *stats) {
    bafe_pruning_config c = bafe_pruning_config_default();
    c.time_budget_ms = time_budget_ms;
    return bafe_pruning_run(g, out, &c, stats);
}
