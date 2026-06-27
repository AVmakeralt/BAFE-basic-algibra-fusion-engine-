/* bafe/fuse.c - cross-kernel fusion implementation
 *
 * Concatenates two optimized graphs so that graph_a's output feeds
 * graph_b's first input, avoiding the intermediate materialization.
 */
#include "bafe/fuse.h"
#include "bafe/bafe.h"
#include "bafe/rewrite.h"
#include <stdio.h>
#include <string.h>

int bafe_fuse_concat(const bafe_graph *graph_a, const bafe_graph *graph_b,
                     bafe_graph *out, char *err_buf, size_t err_buf_size) {
    if (err_buf && err_buf_size > 0) err_buf[0] = '\0';
    bafe_graph_init(out);

    if (graph_a->n_outputs == 0) {
        if (err_buf) snprintf(err_buf, err_buf_size, "graph_a has no outputs");
        return 1;
    }
    if (graph_b->n_inputs == 0) {
        if (err_buf) snprintf(err_buf, err_buf_size, "graph_b has no inputs");
        return 2;
    }

    /* Strategy: copy all nodes from graph_a, then all nodes from graph_b
     * (with child indices shifted by graph_a's node count). graph_b's
     * first input is replaced by graph_a's output node.
     *
     * The output is graph_b's output node (in the combined graph).
     */

    /* node id mapping: graph_a node i -> out node i
     *                  graph_b node j -> out node (graph_a->n_nodes + j) */
    int offset_a = 0;
    int offset_b = graph_a->n_nodes;

    /* copy graph_a's nodes */
    for (int i = 0; i < graph_a->n_nodes; i++) {
        if (out->n_nodes >= BAFE_MAX_NODES) {
            if (err_buf) snprintf(err_buf, err_buf_size, "fused graph too large");
            return 3;
        }
        out->nodes[out->n_nodes] = graph_a->nodes[i];
        out->nodes[out->n_nodes].id = out->n_nodes;
        /* children stay the same (they're within graph_a) */
        out->n_nodes++;
    }

    /* the output node of graph_a (will feed graph_b's first input) */
    int a_output_node = graph_a->outputs[0];  /* in graph_a's index space */

    /* copy graph_b's nodes, shifting children */
    for (int j = 0; j < graph_b->n_nodes; j++) {
        if (out->n_nodes >= BAFE_MAX_NODES) {
            if (err_buf) snprintf(err_buf, err_buf_size, "fused graph too large");
            return 4;
        }
        bafe_node *dst = &out->nodes[out->n_nodes];
        const bafe_node *src = &graph_b->nodes[j];
        *dst = *src;
        dst->id = out->n_nodes;

        /* remap children:
         * - if this is graph_b's first INPUT node and a child points to
         *   graph_b's first input, replace with graph_a's output
         * - otherwise shift by offset_b */
        for (int c = 0; c < dst->n_children; c++) {
            bafe_node_id child = dst->children[c];
            /* check if this child is graph_b's first input */
            bool is_b_first_input = false;
            for (int k = 0; k < graph_b->n_inputs; k++) {
                if (child == graph_b->inputs[k]) {
                    if (k == 0) {
                        /* rewired to graph_a's output */
                        dst->children[c] = a_output_node;
                        is_b_first_input = true;
                    } else {
                        /* graph_b's input k -> out's input (n_inputs_from_a + k - 1) */
                        dst->children[c] = child + offset_b;
                    }
                    break;
                }
            }
            if (!is_b_first_input) {
                /* normal shift */
                dst->children[c] = child + offset_b;
            }
        }
        out->n_nodes++;
    }

    /* set up inputs: graph_a's inputs + graph_b's inputs[1:] */
    for (int i = 0; i < graph_a->n_inputs; i++) {
        out->inputs[out->n_inputs++] = graph_a->inputs[i] + offset_a;
    }
    for (int i = 1; i < graph_b->n_inputs; i++) {
        out->inputs[out->n_inputs++] = graph_b->inputs[i] + offset_b;
    }

    /* set output: graph_b's output node (shifted) */
    out->outputs[0] = graph_b->outputs[0] + offset_b;
    out->n_outputs = 1;

    (void)offset_a;
    return 0;
}

bafe_kernel_fn bafe_fuse_compile(const bafe_graph *graph_a,
                                  const bafe_graph *graph_b,
                                  char *err_buf, size_t err_buf_size) {
    bafe_graph fused;
    if (bafe_fuse_concat(graph_a, graph_b, &fused, err_buf, err_buf_size) != 0) {
        return NULL;
    }

    /* optimize the fused graph (may trigger new fusion rewrites) */
    bafe_graph optimized;
    if (bafe_optimize(&fused, &optimized, err_buf, err_buf_size) != 0) {
        return NULL;
    }

    /* JIT compile */
    return bafe_jit_get_or_compile(&optimized, err_buf, err_buf_size);
}
