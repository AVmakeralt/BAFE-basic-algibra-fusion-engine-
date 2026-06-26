/* bafe/profiling.h - auto-tuning loop with profiling feedback (Phase 3, issue #6)
 *
 * Closes the feedback cycle: compile → measure → refit → re-optimize.
 *
 * Pipeline:
 *   1. JIT compiles a kernel (first call) or hits cache
 *   2. Autotune layer measures kernel runtime
 *   3. Logs (graph_hash, features, predicted_cost, observed_runtime)
 *   4. After N samples, refits the cost model via linear regression
 *   5. If the best kernel for a shape changed, invalidates stale cache entries
 *   6. Next call re-optimizes with the learned cost model
 *
 * Feature vector (8 features per kernel):
 *   0: num_inputs
 *   1: num_ops
 *   2: num_matmuls
 *   3: num_fused_ops
 *   4: log(total_flops + 1)
 *   5: log(total_bytes + 1)
 *   6: num_intermediates
 *   7: has_col_major_input (0 or 1)
 *
 * The learned model predicts log(runtime_ms) as a linear function of features.
 */
#ifndef BAFE_PROFILING_H
#define BAFE_PROFILING_H

#include "bafe/ir.h"
#include "bafe/jit.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BAFE_NUM_FEATURES 8
#define BAFE_PROFILING_LOG_SIZE 4096

/* A single profiling record: one kernel execution. */
typedef struct {
    char     graph_hash[65];          /* SHA-256 hex of the graph */
    double   features[BAFE_NUM_FEATURES];
    double   predicted_cost;          /* what the cost model predicted */
    double   observed_ms;             /* measured wall-clock runtime */
    int      kernel_id;               /* index into the JIT cache */
} bafe_profiling_record;

/* The in-memory log (ring buffer). */
typedef struct {
    bafe_profiling_record records[BAFE_PROFILING_LOG_SIZE];
    int n;
    int head;             /* next write slot */
    bool wrapped;         /* true if we've overwritten old records */
} bafe_profiling_log;

/* A learned cost model: linear regression on features.
 * predicts: log(runtime_ms) = sum(w[i] * features[i]) + bias */
typedef struct {
    double weights[BAFE_NUM_FEATURES];
    double bias;
    double r_squared;       /* quality of fit, 0..1 */
    int    n_samples;       /* samples used for the fit */
    bool   valid;           /* true after a successful refit */
} bafe_learned_cost_model;

/* Stats for the autotune loop. */
typedef struct {
    int    total_calls;
    int    total_compiles;
    int    total_refits;
    int    total_invalidations;
    double last_refit_r_squared;
    int    log_size;
} bafe_autotune_stats;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                          */
/* ------------------------------------------------------------------ */

/* Initialize the profiling subsystem. Idempotent. */
void bafe_profiling_init(void);

/* Reset all profiling state (log, learned model, stats). */
void bafe_profiling_reset(void);

/* ------------------------------------------------------------------ */
/* Feature extraction                                                 */
/* ------------------------------------------------------------------ */

/* Extract the 8-feature vector from a graph.
 * Writes into `features` (caller-allocated, size >= BAFE_NUM_FEATURES). */
void bafe_profiling_extract_features(const bafe_graph *g, double *features);

/* ------------------------------------------------------------------ */
/* Logging                                                            */
/* ------------------------------------------------------------------ */

/* Log a kernel execution. Called by the autotune dispatch wrapper.
 * If the in-memory log is full, the oldest record is overwritten. */
void bafe_profiling_add(const char *graph_hash,
                         const double *features,
                         double predicted_cost,
                         double observed_ms,
                         int kernel_id);

/* Get a pointer to the in-memory log (for inspection / refit). */
const bafe_profiling_log *bafe_profiling_get_log(void);

/* Write the log to a JSONL file. Returns number of records written, or -1. */
int bafe_profiling_dump_jsonl(const char *path);

/* ------------------------------------------------------------------ */
/* Refit                                                              */
/* ------------------------------------------------------------------ */

/* Refit the learned cost model from the in-memory log.
 * Uses closed-form least squares on log(observed_ms).
 * Stores the result in the global learned model.
 * Returns 0 on success, non-zero if not enough samples. */
int bafe_profiling_refit(void);

/* Get the current learned cost model. */
const bafe_learned_cost_model *bafe_profiling_get_model(void);

/* Predict log(runtime_ms) for a graph using the learned model.
 * Returns 0 if the model is not valid (no refit yet). */
double bafe_profiling_predict_log(const double *features);

/* Predict runtime_ms (exp of the log prediction). */
double bafe_profiling_predict_ms(const double *features);

/* ------------------------------------------------------------------ */
/* Autotune config + dispatch                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    bool   enabled;
    int    refit_threshold;        /* refit after this many new samples */
    double invalidation_drift;     /* invalidate if prediction drifts > this ratio */
    int    warmup_calls;           /* skip timing for the first N calls */
    int    timing_iters;           /* average over this many kernel invocations */
} bafe_autotune_config;

bafe_autotune_config bafe_autotune_config_default(void);

void bafe_autotune_configure(const bafe_autotune_config *cfg);

bafe_autotune_config bafe_autotune_get_config(void);

/* Get current autotune stats. */
bafe_autotune_stats bafe_autotune_get_stats(void);

/* Dispatch wrapper: call a kernel with timing + logging.
 *
 * This is the core of the autotune loop. It:
 *   1. Optionally warms up the cache (warmup_calls)
 *   2. Times the kernel over timing_iters invocations
 *   3. Logs the (features, predicted, observed) record
 *   4. If enough new samples, refits the cost model
 *   5. If predictions drifted, invalidates stale cache entries
 *
 * `fn` is the kernel function pointer.
 * `args` is an array of (n_inputs+1) void pointers (inputs + output).
 * `graph` is the optimized graph (for feature extraction + hash).
 *
 * Returns the observed runtime in ms, or -1 on error. */
double bafe_autotune_dispatch(bafe_kernel_fn fn,
                               void **args, int n_args,
                               const bafe_graph *graph,
                               const char *graph_hash,
                               double predicted_cost);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_PROFILING_H */
