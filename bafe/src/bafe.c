/* bafe/bafe.c - top-level BAFE API: optimize + compile
 *
 * Pipeline:
 *   1. (Optional) Run stochastic multi-pass search on a working copy
 *   2. Import the (possibly expanded) graph into e-graph
 *   3. Find rewrite alternatives (deterministic, on the working copy)
 *   4. Apply alternatives (declare equivalences)
 *   5. Rebuild (congruence closure)
 *   6. Extract min-cost program (DP)
 *   7. Build optimized graph from extraction
 *   8. JIT compile
 */
#include "bafe/bafe.h"
#include "bafe/rewrite.h"
#include "bafe/codegen.h"
#include "bafe/search.h"
#include "bafe/pruning.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int bafe_optimize_with_budget(const bafe_graph *input, bafe_graph *optimized,
                               const bafe_search_budget *budget_in,
                               char *err_buf, size_t err_buf_size) {
    if (err_buf && err_buf_size > 0) err_buf[0] = '\0';
    bafe_search_budget budget = budget_in ? *budget_in : bafe_search_budget_default();

    bafe_graph_init(optimized);

    /* Step 0: make a working copy of the input graph (heap-allocated — the
     * struct is too large for the stack with BAFE_MAX_NODES=4096). */
    bafe_graph *work = (bafe_graph *)malloc(sizeof(bafe_graph));
    bafe_alt_list *alts = (bafe_alt_list *)malloc(sizeof(bafe_alt_list));
    if (!work || !alts) {
        if (err_buf) snprintf(err_buf, err_buf_size, "out of memory");
        free(work); free(alts);
        return 7;
    }
    bafe_graph_init(work);
    alts->n = 0;
    /* copy nodes from input -> work */
    for (int i = 0; i < input->n_nodes; i++) {
        work->nodes[i] = input->nodes[i];
    }
    work->n_nodes = input->n_nodes;
    for (int i = 0; i < input->n_inputs; i++) work->inputs[i] = input->inputs[i];
    work->n_inputs = input->n_inputs;
    for (int i = 0; i < input->n_outputs; i++) work->outputs[i] = input->outputs[i];
    work->n_outputs = input->n_outputs;

    /* Step 1: run search on the working copy.
     *
     * Phase 3 (issue #4): if a time_budget_ms > 0 is specified, use the
     * multi-tier pruning controller (Levels A/B/C/D with anytime property).
     * Otherwise, fall back to the existing stochastic search.
     */
    if (budget.time_budget_ms > 0) {
        /* Use the pruning controller */
        bafe_pruning_config pc = bafe_pruning_config_default();
        pc.time_budget_ms = budget.time_budget_ms;
        pc.max_nodes = budget.max_nodes;
        pc.max_rewrites = budget.max_rewrites;
        pc.temperature = budget.temperature;
        pc.seed = budget.seed;
        bafe_pruning_stats pstats;
        bafe_pruning_run(work, alts, &pc, &pstats);
#ifdef BAFE_DEBUG
        printf("[bafe_optimize] pruning controller: regime=%d, alts=%d, "
               "tier_a=%d, tier_b=%d, tier_c=%d, tier_d=%d, elapsed=%.2fms\n",
               pstats.regime, pstats.total_alts_found,
               pstats.tier_a_passed, pstats.tier_b_passed,
               pstats.tier_c_kept, pstats.tier_d_materialized, pstats.elapsed_ms);
#endif
    } else {
        /* Use the existing stochastic search */
        bafe_search_stats search_stats;
        int n_alts = bafe_rewrite_stochastic_stats(work, alts, &budget, &search_stats);
        (void)n_alts;
        (void)search_stats;
#ifdef BAFE_DEBUG
        printf("[bafe_optimize] stochastic search: %d iters, %d alts found, %d materialized\n",
               search_stats.iters_done, search_stats.alts_found,
               search_stats.alts_materialized);
#endif
    }

    /* Step 2: import the (possibly expanded) working graph into e-graph */
    bafe_egraph *eg = (bafe_egraph *)malloc(sizeof(bafe_egraph));
    if (!eg) {
        if (err_buf) snprintf(err_buf, err_buf_size, "out of memory");
        return 7;
    }
    bafe_egraph_init(eg);
    bafe_eclass_id node_to_eclass[BAFE_MAX_NODES];
    for (int i = 0; i < BAFE_MAX_NODES; i++) node_to_eclass[i] = -1;
    bafe_egraph_import_graph(eg, work, node_to_eclass);

#ifdef BAFE_DEBUG
    {
        char dbg[8192];
        bafe_egraph_dump(eg, dbg, sizeof(dbg));
        printf("[bafe_optimize] after import:\n%s\n", dbg);
    }
#endif

    /* Step 3: apply alternatives (declare equivalences) */
    bafe_egraph_apply_alternatives(eg, work, node_to_eclass, alts);

    /* Step 4: rebuild (congruence closure) */
    int iters = bafe_egraph_rebuild(eg, 100);
    (void)iters;

#ifdef BAFE_DEBUG
    {
        char dbg[8192];
        bafe_egraph_dump(eg, dbg, sizeof(dbg));
        printf("[bafe_optimize] after rebuild (%d iters):\n%s\n", iters, dbg);
    }
#endif

    /* Step 5: extract min-cost program.
     * Phase 3 (issue #5): use the calibrated cost model if a learned model
     * is available; otherwise fall back to the static default. */
    bafe_cost_model cm = bafe_cost_model_calibrated_default();
    bafe_plan *plan = (bafe_plan *)malloc(sizeof(bafe_plan));
    if (!plan) { free(eg); free(work); free(alts); if (err_buf) snprintf(err_buf, err_buf_size, "oom"); return 7; }
    int eclass_to_plan[BAFE_EG_MAX_CLASSES];
    bafe_extract_run(eg, &cm, work, node_to_eclass, plan, eclass_to_plan);

    /* Step 6: build optimized graph from extraction */
    if (work->n_outputs == 0) {
        if (err_buf) snprintf(err_buf, err_buf_size, "input has no outputs");
        free(eg);
        return 1;
    }
    bafe_eclass_id root_eclass = node_to_eclass[work->outputs[0]];

    bafe_eclass_id new_eclass_to_node[BAFE_EG_MAX_CLASSES];
    for (int i = 0; i < BAFE_EG_MAX_CLASSES; i++) new_eclass_to_node[i] = -1;

    /* map from eclass -> original input node id (for shape/name recovery) */
    bafe_node_id eclass_to_input_origin[BAFE_EG_MAX_CLASSES];
    for (int i = 0; i < BAFE_EG_MAX_CLASSES; i++) eclass_to_input_origin[i] = -1;
    for (int i = 0; i < work->n_inputs; i++) {
        bafe_node_id nid = work->inputs[i];
        bafe_eclass_id cid = bafe_egraph_find(eg, node_to_eclass[nid]);
        eclass_to_input_origin[cid] = nid;
    }

    /* iterative DFS to build the optimized graph */
    typedef struct {
        bafe_eclass_id eclass;
        int state;
    } stack_entry;
    stack_entry stack[512];
    int sp = 0;
    stack[sp].eclass = bafe_egraph_find(eg, root_eclass);
    stack[sp].state = 0;
    sp++;

    while (sp > 0) {
        stack_entry *top = &stack[sp - 1];
        bafe_eclass_id cid = top->eclass;
        if (new_eclass_to_node[cid] >= 0) {
            sp--;
            continue;
        }
        int plan_idx = eclass_to_plan[cid];
        if (plan_idx < 0) {
            if (err_buf) snprintf(err_buf, err_buf_size, "no plan for eclass %d", cid);
            free(eg);
            return 2;
        }
        const bafe_plan_node *p = &plan->items[plan_idx];
        if (p->enode.op_name == NULL) {
            if (err_buf) snprintf(err_buf, err_buf_size, "no enode chosen for eclass %d", cid);
            free(eg);
            return 8;
        }
        if (top->state == 0) {
            if (strcmp(p->enode.op_name, "input") == 0) {
                bafe_node_id orig = eclass_to_input_origin[cid];
                if (orig < 0) {
                    if (err_buf) snprintf(err_buf, err_buf_size, "no original input for eclass %d", cid);
                    free(eg);
                    return 3;
                }
                const bafe_node *orig_node = &work->nodes[orig];
                bafe_node_id new_id = bafe_graph_add_input_with_layout(
                    optimized, orig_node->input_name,
                    &orig_node->shape, orig_node->dtype, orig_node->layout);
                new_eclass_to_node[cid] = new_id;
                sp--;
                continue;
            }
            if (strcmp(p->enode.op_name, "constant") == 0) {
                bafe_shape scalar = bafe_shape_scalar();
                bafe_node_id new_id = bafe_graph_add_constant(optimized, p->enode.attrs.scalar_value,
                                                                &scalar, BAFE_DTYPE_F32);
                new_eclass_to_node[cid] = new_id;
                sp--;
                continue;
            }
            top->state = 1;
            for (int j = p->enode.n_children - 1; j >= 0; j--) {
                bafe_eclass_id child_root = bafe_egraph_find(eg, p->enode.children[j]);
                if (new_eclass_to_node[child_root] < 0) {
                    if (sp >= 512) {
                        if (err_buf) snprintf(err_buf, err_buf_size, "stack overflow");
                        free(eg);
                        return 4;
                    }
                    stack[sp].eclass = child_root;
                    stack[sp].state = 0;
                    sp++;
                }
            }
            continue;
        }
        bafe_node_id children[BAFE_MAX_CHILDREN];
        for (int j = 0; j < p->enode.n_children; j++) {
            bafe_eclass_id child_root = bafe_egraph_find(eg, p->enode.children[j]);
            children[j] = new_eclass_to_node[child_root];
            if (children[j] < 0) {
                if (err_buf) snprintf(err_buf, err_buf_size, "child not built for eclass %d", cid);
                free(eg);
                return 5;
            }
        }
        bafe_node_id new_id = bafe_graph_add(optimized, p->enode.op_name, children,
                                              p->enode.n_children, &p->enode.attrs);
        new_eclass_to_node[cid] = new_id;
        sp--;
    }

    bafe_node_id root_node = new_eclass_to_node[bafe_egraph_find(eg, root_eclass)];
    if (root_node < 0) {
        if (err_buf) snprintf(err_buf, err_buf_size, "root not built");
        free(eg);
        return 6;
    }
    bafe_graph_set_output(optimized, root_node);
    free(eg);
    free(work);
    free(alts);
    free(plan);
    return 0;
}

int bafe_optimize(const bafe_graph *input, bafe_graph *optimized,
                  char *err_buf, size_t err_buf_size) {
    /* The superoptimizer runs automatically with aggressive defaults:
     * multi-pass stochastic search with 8 iterations, 4096 node limit,
     * and 512 rewrite materializations. This is NOT a single-pass
     * deterministic mode — the full optimization pipeline runs by default. */
    bafe_search_budget b = bafe_search_budget_default();
    return bafe_optimize_with_budget(input, optimized, &b, err_buf, err_buf_size);
}

bafe_kernel_fn bafe_optimize_and_compile(const bafe_graph *input,
                                          char *err_buf, size_t err_buf_size) {
    if (err_buf && err_buf_size > 0) err_buf[0] = '\0';
    bafe_graph optimized;
    if (bafe_optimize(input, &optimized, err_buf, err_buf_size) != 0) {
        return NULL;
    }
    return bafe_jit_get_or_compile(&optimized, err_buf, err_buf_size);
}

bafe_kernel_fn bafe_optimize_and_compile_with_budget(const bafe_graph *input,
                                                      const bafe_search_budget *budget,
                                                      char *err_buf, size_t err_buf_size) {
    if (err_buf && err_buf_size > 0) err_buf[0] = '\0';
    bafe_graph optimized;
    if (bafe_optimize_with_budget(input, &optimized, budget, err_buf, err_buf_size) != 0) {
        return NULL;
    }
    return bafe_jit_get_or_compile(&optimized, err_buf, err_buf_size);
}

void bafe_optimize_debug(const bafe_graph *input) {
    printf("=== Input graph ===\n");
    char buf[8192];
    bafe_graph_summary(input, buf, sizeof(buf));
    printf("%s\n", buf);

    bafe_graph optimized;
    char err[256];
    int rc = bafe_optimize(input, &optimized, err, sizeof(err));
    if (rc != 0) {
        printf("optimize failed: %s\n", err);
        return;
    }
    printf("\n=== Optimized graph ===\n");
    bafe_graph_summary(&optimized, buf, sizeof(buf));
    printf("%s\n", buf);
    bafe_graph_print((bafe_graph *)&optimized, buf, sizeof(buf));
    printf("%s\n", buf);

    /* cost comparison */
    bafe_cost_model cm = bafe_cost_model_default();
    double in_cost = bafe_cost_graph(&cm, input);
    double opt_cost = bafe_cost_graph(&cm, &optimized);
    printf("\nCost: input=%.4f -> optimized=%.4f\n", in_cost, opt_cost);

    /* e-graph stats */
    bafe_egraph *eg = (bafe_egraph *)malloc(sizeof(bafe_egraph));
    if (!eg) { printf("out of memory\n"); return; }
    bafe_egraph_init(eg);
    bafe_eclass_id node_to_eclass[BAFE_MAX_NODES];
    bafe_egraph_import_graph(eg, input, node_to_eclass);
    bafe_alt_list alts_dbg;
    bafe_rewrite_find(input, &alts_dbg);
    bafe_egraph_apply_alternatives(eg, input, node_to_eclass, &alts_dbg);
    int iters = bafe_egraph_rebuild(eg, 100);
    printf("E-graph: %d classes, %d enodes, %d rebuild iters, %d alternatives\n",
           bafe_egraph_num_classes(eg), bafe_egraph_num_enodes(eg), iters, alts_dbg.n);
    free(eg);

    /* show emitted C source */
    char *src = bafe_codegen_emit_alloc(&optimized, "bafe_kernel");
    if (src) {
        printf("\n=== Emitted C source ===\n%s\n", src);
        free(src);
    }
}
