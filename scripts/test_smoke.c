/* Smoke test: build a graph, optimize, codegen, compile, run.
 * Compile with: cc -std=c11 -Ibafe/include test_smoke.c bafe/build/*.o -ldl -lm -o test_smoke
 */
#include "bafe/bafe.h"
#include "bafe/codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    /* Build: relu(matmul(A, B) + bias)
     * A: 32x32, B: 32x32, bias: 32 */
    bafe_graph g;
    bafe_graph_init(&g);
    bafe_shape shape_a = bafe_shape_make_2(32, 32);
    bafe_shape shape_b = bafe_shape_make_2(32, 32);
    bafe_shape shape_bias = bafe_shape_make_1(32);
    bafe_node_id a = bafe_graph_add_input(&g, "A", &shape_a, BAFE_DTYPE_F32);
    bafe_node_id b = bafe_graph_add_input(&g, "B", &shape_b, BAFE_DTYPE_F32);
    bafe_node_id bias = bafe_graph_add_input(&g, "bias", &shape_bias, BAFE_DTYPE_F32);
    bafe_node_id mm = bafe_graph_matmul(&g, a, b);
    bafe_node_id ad = bafe_graph_add_op(&g, mm, bias);
    bafe_node_id out = bafe_graph_relu(&g, ad);
    bafe_graph_set_output(&g, out);

    printf("Input graph:\n");
    char buf[8192];
    bafe_graph_summary(&g, buf, sizeof(buf));
    printf("  %s\n", buf);

    /* Optimize */
    bafe_graph opt;
    char err[256];
    if (bafe_optimize(&g, &opt, err, sizeof(err)) != 0) {
        fprintf(stderr, "optimize failed: %s\n", err);
        return 1;
    }
    printf("\nOptimized graph:\n");
    bafe_graph_summary(&opt, buf, sizeof(buf));
    printf("  %s\n", buf);
    bafe_graph_print(&opt, buf, sizeof(buf));
    printf("%s\n", buf);

    /* Emit C source */
    char *src = bafe_codegen_emit_alloc(&opt, "bafe_kernel");
    if (!src) { fprintf(stderr, "codegen failed\n"); return 1; }
    printf("\n=== Emitted C source ===\n%s\n", src);

    /* JIT compile and run */
    bafe_kernel_fn fn = bafe_jit_get_or_compile(&opt, err, sizeof(err));
    if (!fn) { fprintf(stderr, "jit failed: %s\n", err); free(src); return 1; }

    /* allocate input/output buffers */
    float A[32*32], B[32*32], Bias[32], Out[32*32];
    for (int i = 0; i < 32*32; i++) { A[i] = (float)(i % 7) * 0.1f; B[i] = (float)(i % 5) * 0.1f; }
    for (int i = 0; i < 32; i++) Bias[i] = (float)i * 0.01f;

    /* The emitted kernel takes pointers directly: void bafe_kernel(A, B, bias, out) */
    typedef void (*kernel_sig)(const float *, const float *, const float *, float *);
    ((kernel_sig)fn)(A, B, Bias, Out);

    /* compute reference */
    float ref[32*32];
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 32; k++) acc += A[i*32+k] * B[k*32+j];
            acc += Bias[j];
            ref[i*32+j] = acc > 0 ? acc : 0;
        }
    }

    /* compare */
    int mismatches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < 32*32; i++) {
        float d = Out[i] - ref[i];
        if (d < 0) d = -d;
        if (d > max_err) max_err = d;
        if (d > 1e-5f) mismatches++;
    }
    printf("\nResults: %d mismatches, max error = %e\n", mismatches, max_err);
    if (mismatches == 0) printf("PASS\n");
    else { printf("FAIL\n"); return 1; }

    /* JIT stats */
    bafe_jit_stats stats = bafe_jit_get_stats();
    printf("\nJIT stats: hits=%d misses=%d compiles=%d failures=%d\n",
           stats.hits, stats.misses, stats.compiles, stats.compile_failures);

    free(src);
    return 0;
}
