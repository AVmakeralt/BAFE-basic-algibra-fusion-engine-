/* bafe/bafe.c - top-level BAFE API: optimize + compile
 *
 * Pipeline:
 *   1. Import input graph into e-graph
 *   2. Find rewrite alternatives
 *   3. Apply alternatives (declare equivalences)
 *   4. Rebuild (congruence closure)
 *   5. Extract min-cost program (DP)
 *   6. Build optimized graph from extraction
 *   7. JIT compile
 */
#include "bafe/bafe.h"
#include "bafe/rewrite.h"
#include "bafe/codegen.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int bafe_optimize(const bafe_graph *input, bafe_graph *optimized,
                  char *err_buf, size_t err_buf_size) {
    if (err_buf && err_buf_size > 0) err_buf[0] = '\0';

    bafe_graph_init(optimized);

    /* Step 1: import input graph into e-graph (heap-allocated, ~12MB) */
    bafe_egraph *eg = (bafe_egraph *)malloc(sizeof(bafe_egraph));
    if (!eg) {
        if (err_buf) snprintf(err_buf, err_buf_size, "out of memory");
        return 7;
    }
    bafe_egraph_init(eg);
    bafe_eclass_id node_to_eclass[BAFE_MAX_NODES];
    for (int i = 0; i < input->n_nodes; i++) node_to_eclass[i] = -1;
    bafe_egraph_import_graph(eg, input, node_to_eclass);

    /* Step 2: find rewrite alternatives */
    bafe_alt_list alts;
    bafe_rewrite_find(input, &alts);
#ifdef BAFE_DEBUG
    printf("[bafe_optimize] found %d alternatives:\n", alts.n);
    for (int i = 0; i < alts.n; i++) {
        printf("  alt %d: node %d -> %s(", i, alts.items[i].original_node_id, alts.items[i].op_name);
        for (int j = 0; j < alts.items[i].n_children; j++) {
            printf("%s%d", j == 0 ? "" : ",", alts.items[i].children[j]);
        }
        printf(")\n");
    }
    {
        char dbg[8192];
        bafe_egraph_dump(eg, dbg, sizeof(dbg));
        printf("[bafe_optimize] after import (before alternatives applied):\n%s\n", dbg);
    }
#endif

    /* Step 3: apply alternatives (declare equivalences) */
    bafe_egraph_apply_alternatives(eg, input, node_to_eclass, &alts);

    /* Step 4: rebuild (saturate congruence closure) */
    int iters = bafe_egraph_rebuild(eg, 100);
    (void)iters;
#ifdef BAFE_DEBUG
    {
        char dbg[8192];
        bafe_egraph_dump(eg, dbg, sizeof(dbg));
        printf("[bafe_optimize] after rebuild (%d iters):\n%s\n", iters, dbg);
        printf("[bafe_optimize] node_to_eclass:");
        for (int i = 0; i < input->n_nodes; i++) {
            printf(" n%d->e%d(find=%d)", i, node_to_eclass[i], bafe_egraph_find(eg, node_to_eclass[i]));
        }
        printf("\n");
    }
#endif

    /* Step 5: extract min-cost program */
    bafe_cost_model cm = bafe_cost_model_default();
    bafe_plan plan;
    int eclass_to_plan[BAFE_EG_MAX_CLASSES];
    bafe_extract_run(eg, &cm, input, &plan, eclass_to_plan);

    /* Step 6: build optimized graph from extraction, rooted at the
     * e-class of the input's output node. */
    if (input->n_outputs == 0) {
        if (err_buf) snprintf(err_buf, err_buf_size, "input has no outputs");
        free(eg);
        return 1;
    }
    bafe_eclass_id root_eclass = node_to_eclass[input->outputs[0]];

    /* We need to copy the input/constant nodes from the original graph
     * into the optimized graph, and map e-class ids to the new node ids.
     * Strategy: build a fresh graph from scratch by walking the plan.
     * We need a separate mapping from original input node ids to new
     * input node ids. */

    /* First, add all inputs to the optimized graph (same shapes/names). */
    bafe_eclass_id new_eclass_to_node[BAFE_EG_MAX_CLASSES];
    for (int i = 0; i < BAFE_EG_MAX_CLASSES; i++) new_eclass_to_node[i] = -1;
    int eclass_visited[BAFE_EG_MAX_CLASSES];
    for (int i = 0; i < BAFE_EG_MAX_CLASSES; i++) eclass_visited[i] = 0;

    /* We need a mapping from the original graph's input nodes to the
     * e-class ids they were imported as, so that when we encounter an
     * input e-node in the plan, we can find its original shape/name. */
    bafe_node_id eclass_to_input_origin[BAFE_EG_MAX_CLASSES];
    for (int i = 0; i < BAFE_EG_MAX_CLASSES; i++) eclass_to_input_origin[i] = -1;
    for (int i = 0; i < input->n_inputs; i++) {
        bafe_node_id nid = input->inputs[i];
        bafe_eclass_id cid = bafe_egraph_find(eg, node_to_eclass[nid]);
        eclass_to_input_origin[cid] = nid;
    }

    /* Walk the plan recursively and build the optimized graph. */
    /* Custom recursion that also handles inputs and constants. */
    /* We'll do an explicit stack-based DFS. */
    typedef struct {
        bafe_eclass_id eclass;
        int state;  /* 0 = unvisited, 1 = children done */
    } stack_entry;
    stack_entry stack[256];
    int sp = 0;
    stack[sp].eclass = bafe_egraph_find(eg, root_eclass);
    stack[sp].state = 0;
    sp++;

    while (sp > 0) {
        stack_entry *top = &stack[sp - 1];
        bafe_eclass_id cid = top->eclass;
        if (new_eclass_to_node[cid] >= 0) {
            sp--;  /* already built */
            continue;
        }
        int plan_idx = eclass_to_plan[cid];
        if (plan_idx < 0) {
            if (err_buf) snprintf(err_buf, err_buf_size, "no plan for eclass %d", cid);
            free(eg);
            return 2;
        }
        const bafe_plan_node *p = &plan.items[plan_idx];
        if (p->enode.op_name == NULL) {
            if (err_buf) snprintf(err_buf, err_buf_size, "no enode chosen for eclass %d", cid);
            free(eg);
            return 8;
        }
        if (top->state == 0) {
            /* check if this is an input or constant eclass */
            if (strcmp(p->enode.op_name, "input") == 0) {
                bafe_node_id orig = eclass_to_input_origin[cid];
                if (orig < 0) {
                    if (err_buf) snprintf(err_buf, err_buf_size, "no original input for eclass %d", cid);
                    free(eg);
                    return 3;
                }
                const bafe_node *orig_node = &input->nodes[orig];
                bafe_node_id new_id = bafe_graph_add_input(optimized, orig_node->input_name,
                                                            &orig_node->shape, orig_node->dtype);
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
            /* push children */
            top->state = 1;
            for (int j = p->enode.n_children - 1; j >= 0; j--) {
                bafe_eclass_id child_root = bafe_egraph_find(eg, p->enode.children[j]);
                if (new_eclass_to_node[child_root] < 0) {
                    if (sp >= 256) {
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
        /* state == 1: children done, build this node */
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
    return 0;
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
    bafe_alt_list alts;
    bafe_rewrite_find(input, &alts);
    bafe_egraph_apply_alternatives(eg, input, node_to_eclass, &alts);
    int iters = bafe_egraph_rebuild(eg, 100);
    printf("E-graph: %d classes, %d enodes, %d rebuild iters, %d alternatives\n",
           bafe_egraph_num_classes(eg), bafe_egraph_num_enodes(eg), iters, alts.n);
    free(eg);

    /* show emitted C source */
    char *src = bafe_codegen_emit_alloc(&optimized, "bafe_kernel");
    if (src) {
        printf("\n=== Emitted C source ===\n%s\n", src);
        free(src);
    }
}
