/* bafe/ops.c - op registry and shape inference
 *
 * We register all ops at startup. The registry is a static array; lookups
 * are by linear scan (we have ~20 ops, no need for a hash table).
 */
#include "bafe/ops.h"
#include <string.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/* Attrs                                                              */
/* ------------------------------------------------------------------ */

bafe_op_attrs bafe_op_attrs_default(void) {
    bafe_op_attrs a;
    memset(&a, 0, sizeof(a));
    return a;
}

/* ------------------------------------------------------------------ */
/* Shape inference functions                                          */
/* ------------------------------------------------------------------ */

static bafe_shape _matmul(const bafe_shape *in, int n, const bafe_op_attrs *a) {
    (void)a; (void)n;  /* n == 2 */
    const bafe_shape *A = &in[0];
    const bafe_shape *B = &in[1];
    if (A->rank < 2 || B->rank < 2) return bafe_shape_scalar();
    int32_t M = A->dims[A->rank - 2];
    int32_t K1 = A->dims[A->rank - 1];
    int32_t K2 = B->dims[B->rank - 2];
    int32_t N = B->dims[B->rank - 1];
    (void)K1; (void)K2;  /* mismatch checked at validation, not here */
    if (A->rank == 2 && B->rank == 2) return bafe_shape_make_2(M, N);
    /* batched: broadcast leading dims */
    int32_t ba = A->rank - 2;
    int32_t bb = B->rank - 2;
    int32_t nbatch = ba > bb ? ba : bb;
    bafe_shape batched_a, batched_b;
    batched_a.rank = ba; batched_b.rank = bb;
    for (int32_t i = 0; i < ba; i++) batched_a.dims[i] = A->dims[i];
    for (int32_t i = 0; i < bb; i++) batched_b.dims[i] = B->dims[i];
    bafe_shape batch = bafe_shape_broadcast(&batched_a, &batched_b);
    if (nbatch + 2 > BAFE_MAX_RANK) return bafe_shape_make_2(M, N);
    bafe_shape out;
    out.rank = nbatch + 2;
    for (int32_t i = 0; i < nbatch; i++) out.dims[i] = batch.dims[i];
    out.dims[nbatch] = M;
    out.dims[nbatch + 1] = N;
    return out;
}

static bafe_shape _binary_broadcast(const bafe_shape *in, int n, const bafe_op_attrs *a) {
    (void)a; (void)n;
    return bafe_shape_broadcast(&in[0], &in[1]);
}

static bafe_shape _scale(const bafe_shape *in, int n, const bafe_op_attrs *a) {
    (void)a; (void)n;
    /* scale(tensor, scalar) -> tensor */
    return in[0];
}

static bafe_shape _bias_add(const bafe_shape *in, int n, const bafe_op_attrs *a) {
    (void)a; (void)n;
    /* bias_add([M,N], [N]) -> [M,N] */
    return in[0];
}

static bafe_shape _unary(const bafe_shape *in, int n, const bafe_op_attrs *a) {
    (void)a; (void)n;
    return in[0];
}

static bafe_shape _transpose(const bafe_shape *in, int n, const bafe_op_attrs *a) {
    (void)n;
    return bafe_shape_transpose(&in[0], a->perm);
}

static bafe_shape _reduce(const bafe_shape *in, int n, const bafe_op_attrs *a) {
    (void)n;
    return bafe_shape_reduce(&in[0], a->axes, a->n_axes, a->keepdims);
}

static bafe_shape _reshape(const bafe_shape *in, int n, const bafe_op_attrs *a) {
    (void)n;
    /* resolve -1 in target shape */
    int32_t target[BAFE_MAX_RANK];
    int32_t ntarget = a->n_shape;
    size_t in_numel = bafe_shape_numel(&in[0]);
    for (int32_t i = 0; i < ntarget; i++) target[i] = a->shape[i];
    int32_t neg = -1;
    int32_t known = 1;
    for (int32_t i = 0; i < ntarget; i++) {
        if (target[i] == -1) {
            if (neg != -1) return bafe_shape_scalar(); /* multiple -1 invalid */
            neg = i;
        } else {
            known *= target[i];
        }
    }
    if (neg != -1) {
        if (known == 0 || in_numel % (size_t)known != 0) return bafe_shape_scalar();
        target[neg] = (int32_t)(in_numel / (size_t)known);
    }
    return bafe_shape_make(ntarget, target);
}

static bafe_shape _broadcast_to(const bafe_shape *in, int n, const bafe_op_attrs *a) {
    (void)n; (void)in;
    return bafe_shape_make(a->n_shape, a->shape);
}

static bafe_shape _fused_matmul_bias(const bafe_shape *in, int n, const bafe_op_attrs *a) {
    (void)n; (void)a;
    /* fused_matmul_bias(A, B, bias) -> matmul(A, B) */
    bafe_shape mm_in[2] = {in[0], in[1]};
    return _matmul(mm_in, 2, a);
}

/* ------------------------------------------------------------------ */
/* Registry                                                           */
/* ------------------------------------------------------------------ */

static const bafe_op _OPS[] = {
    {"input",   0, false, "bafe_input",   _unary},  /* shape comes from caller */
    {"constant",0, false, "bafe_constant",_unary},
    {"matmul",  2, false, "bafe_matmul",  _matmul},
    {"add",     2, false, "bafe_add",     _binary_broadcast},
    {"mul",     2, false, "bafe_mul",     _binary_broadcast},
    {"sub",     2, false, "bafe_sub",     _binary_broadcast},
    {"scale",   2, false, "bafe_scale",   _scale},
    {"bias_add",2, false, "bafe_bias_add",_bias_add},
    {"relu",    1, false, "bafe_relu",    _unary},
    {"sigmoid", 1, false, "bafe_sigmoid", _unary},
    {"tanh",    1, false, "bafe_tanh",    _unary},
    {"neg",     1, false, "bafe_neg",     _unary},
    {"layout_transform", 1, false, "bafe_layout_transform", _unary},
    {"transpose",1,false, "bafe_transpose",_transpose},
    {"reduce_sum",1,false,"bafe_reduce_sum",_reduce},
    {"reduce_max",1,false,"bafe_reduce_max",_reduce},
    {"reshape", 1, false, "bafe_reshape", _reshape},
    {"broadcast_to",1,false,"bafe_broadcast_to",_broadcast_to},
    /* fused ops */
    {"fused_matmul_relu",      2, true, "bafe_fused_matmul_relu",      _matmul},
    {"fused_matmul_bias",      3, true, "bafe_fused_matmul_bias",      _fused_matmul_bias},
    {"fused_matmul_bias_relu", 3, true, "bafe_fused_matmul_bias_relu", _fused_matmul_bias},
    {"fused_bias_relu",        2, true, "bafe_fused_bias_relu",        _bias_add},
};

static const int _N_OPS = (int)(sizeof(_OPS) / sizeof(_OPS[0]));

const bafe_op *bafe_op_get(const char *name) {
    for (int i = 0; i < _N_OPS; i++) {
        if (strcmp(_OPS[i].name, name) == 0) return &_OPS[i];
    }
    return NULL;
}

int bafe_op_count(void) { return _N_OPS; }

const bafe_op *bafe_op_at(int i) {
    if (i < 0 || i >= _N_OPS) return NULL;
    return &_OPS[i];
}

bool bafe_op_is_fused(const char *op_name) {
    if (!op_name) return false;
    return strncmp(op_name, "fused_", 6) == 0;
}

bool bafe_op_is_layout_transform(const char *op_name) {
    if (!op_name) return false;
    return strcmp(op_name, "layout_transform") == 0;
}

void bafe_op_registry_init(void) {
    /* nothing to do; the registry is statically initialized */
}
