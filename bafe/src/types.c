/* bafe/types.c - implementations */
#include "bafe/types.h"
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Dtype                                                              */
/* ------------------------------------------------------------------ */

static const char *const _c_names[] = {"float", "double", "int32_t", "int64_t", "uint16_t", "uint16_t"};
static const char *const _np_names[] = {"float32", "float64", "int32", "int64", "float16", "bfloat16"};
static const size_t   _bytes[]      = {4, 8, 4, 8, 2, 2};

const char *bafe_dtype_c_name(bafe_dtype d) {
    if ((int)d < 0 || (int)d > 5) return "float";
    return _c_names[(int)d];
}

const char *bafe_dtype_numpy_name(bafe_dtype d) {
    if ((int)d < 0 || (int)d > 5) return "float32";
    return _np_names[(int)d];
}

size_t bafe_dtype_byte_size(bafe_dtype d) {
    if ((int)d < 0 || (int)d > 5) return 4;
    return _bytes[(int)d];
}

bafe_dtype bafe_dtype_from_str(const char *s) {
    if (!s) return BAFE_DTYPE_F32;
    if (!strcmp(s, "f32") || !strcmp(s, "float32") || !strcmp(s, "float")) return BAFE_DTYPE_F32;
    if (!strcmp(s, "f64") || !strcmp(s, "float64") || !strcmp(s, "double")) return BAFE_DTYPE_F64;
    if (!strcmp(s, "i32") || !strcmp(s, "int32") || !strcmp(s, "int"))     return BAFE_DTYPE_I32;
    if (!strcmp(s, "i64") || !strcmp(s, "int64") || !strcmp(s, "long"))    return BAFE_DTYPE_I64;
    if (!strcmp(s, "f16") || !strcmp(s, "float16") || !strcmp(s, "half"))  return BAFE_DTYPE_F16;
    if (!strcmp(s, "bf16") || !strcmp(s, "bfloat16"))                      return BAFE_DTYPE_BF16;
    return BAFE_DTYPE_F32;
}

/* ------------------------------------------------------------------ */
/* Shape                                                              */
/* ------------------------------------------------------------------ */

bafe_shape bafe_shape_scalar(void) {
    bafe_shape s; s.rank = 0; return s;
}

bafe_shape bafe_shape_make_1(int32_t d0) {
    bafe_shape s; s.rank = 1; s.dims[0] = d0; return s;
}

bafe_shape bafe_shape_make_2(int32_t d0, int32_t d1) {
    bafe_shape s; s.rank = 2; s.dims[0] = d0; s.dims[1] = d1; return s;
}

bafe_shape bafe_shape_make_3(int32_t d0, int32_t d1, int32_t d2) {
    bafe_shape s; s.rank = 3; s.dims[0] = d0; s.dims[1] = d1; s.dims[2] = d2; return s;
}

bafe_shape bafe_shape_make(int32_t rank, const int32_t *dims) {
    bafe_shape s;
    if (rank < 0) rank = 0;
    if (rank > BAFE_MAX_RANK) rank = BAFE_MAX_RANK;
    s.rank = rank;
    for (int32_t i = 0; i < rank; i++) s.dims[i] = dims[i];
    return s;
}

bool bafe_shape_is_scalar(const bafe_shape *s) { return s->rank == 0; }

bool bafe_shape_is_empty(const bafe_shape *s) {
    for (int32_t i = 0; i < s->rank; i++) if (s->dims[i] == 0) return true;
    return false;
}

int32_t bafe_shape_rank(const bafe_shape *s) { return s->rank; }

size_t bafe_shape_numel(const bafe_shape *s) {
    if (s->rank == 0) return 1;
    size_t n = 1;
    for (int32_t i = 0; i < s->rank; i++) {
        if (s->dims[i] < 0) return 0;
        n *= (size_t)s->dims[i];
    }
    return n;
}

size_t bafe_shape_nbytes(const bafe_shape *s, bafe_dtype d) {
    return bafe_shape_numel(s) * bafe_dtype_byte_size(d);
}

int32_t bafe_shape_dim(const bafe_shape *s, int32_t i) {
    if (i < 0) i += s->rank;
    if (i < 0 || i >= s->rank) return 0;
    return s->dims[i];
}

bafe_shape bafe_shape_broadcast(const bafe_shape *a, const bafe_shape *b) {
    bafe_shape out;
    int32_t na = a->rank, nb = b->rank;
    int32_t n = na > nb ? na : nb;
    if (n > BAFE_MAX_RANK) n = BAFE_MAX_RANK;
    out.rank = n;
    for (int32_t i = 0; i < n; i++) {
        int32_t da = (i < n - na) ? 1 : a->dims[i - (n - na)];
        int32_t db = (i < n - nb) ? 1 : b->dims[i - (n - nb)];
        if (da == 1) out.dims[i] = db;
        else if (db == 1 || db == da) out.dims[i] = da;
        else { out.dims[i] = -1; /* signal error */ }
    }
    return out;
}

bafe_shape bafe_shape_reduce(const bafe_shape *s, const int32_t *axes,
                              int32_t n_axes, bool keepdims) {
    bafe_shape out;
    out.rank = 0;
    /* build a mask */
    bool mask[BAFE_MAX_RANK] = {false};
    if (n_axes == 0) {
        for (int32_t i = 0; i < s->rank; i++) mask[i] = true;
    } else {
        for (int32_t i = 0; i < n_axes; i++) {
            int32_t a = axes[i];
            if (a < 0) a += s->rank;
            if (a >= 0 && a < s->rank) mask[a] = true;
        }
    }
    for (int32_t i = 0; i < s->rank; i++) {
        if (mask[i]) {
            if (keepdims) out.dims[out.rank++] = 1;
        } else {
            out.dims[out.rank++] = s->dims[i];
        }
    }
    return out;
}

bafe_shape bafe_shape_transpose(const bafe_shape *s, const int32_t *perm) {
    bafe_shape out;
    out.rank = s->rank;
    for (int32_t i = 0; i < s->rank; i++) out.dims[i] = s->dims[perm[i]];
    return out;
}

bool bafe_shape_eq(const bafe_shape *a, const bafe_shape *b) {
    if (a->rank != b->rank) return false;
    for (int32_t i = 0; i < a->rank; i++) if (a->dims[i] != b->dims[i]) return false;
    return true;
}

int bafe_shape_snprintf(char *buf, size_t buf_size, const bafe_shape *s) {
    if (!buf || buf_size == 0) return 0;
    size_t pos = 0;
    if (pos < buf_size) buf[pos++] = '(';
    for (int32_t i = 0; i < s->rank; i++) {
        if (i > 0 && pos < buf_size) buf[pos++] = ',';
        int n = snprintf(buf + pos, buf_size - pos, "%d", s->dims[i]);
        if (n < 0) break;
        pos += (size_t)n;
    }
    if (s->rank == 1 && pos < buf_size) buf[pos++] = ',';
    if (pos < buf_size) buf[pos++] = ')';
    if (pos < buf_size) buf[pos] = '\0';
    else if (buf_size > 0) buf[buf_size - 1] = '\0';
    return (int)pos;
}

/* ------------------------------------------------------------------ */
/* Layout                                                             */
/* ------------------------------------------------------------------ */

const char *bafe_layout_name(bafe_layout l) {
    switch (l) {
        case BAFE_LAYOUT_ROW_MAJOR:   return "row";
        case BAFE_LAYOUT_COL_MAJOR:   return "col";
        case BAFE_LAYOUT_BLOCKED:     return "blocked";
        case BAFE_LAYOUT_TENSOR_CORE: return "tc";
    }
    return "?";
}
