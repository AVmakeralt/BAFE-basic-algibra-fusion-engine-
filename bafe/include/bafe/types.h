/* bafe/types.h - core type definitions for the BAFE IR
 *
 * Defines:
 *   - bafe_dtype        element types (F32, F64, I32, I64)
 *   - bafe_shape        tensor shape (immutable tuple of dims)
 *   - bafe_layout       memory layout enum
 *   - helper functions  shape inference, broadcasting, reduction
 *
 * All types are POD (plain-old-data) for cheap passing across the FFI
 * boundary. Shapes are owned by the caller; the library does not allocate
 * shape objects on its own.
 */
#ifndef BAFE_TYPES_H
#define BAFE_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Dtype                                                              */
/* ------------------------------------------------------------------ */

typedef enum {
    BAFE_DTYPE_F32 = 0,
    BAFE_DTYPE_F64 = 1,
    BAFE_DTYPE_I32 = 2,
    BAFE_DTYPE_I64 = 3,
    BAFE_DTYPE_F16 = 4,     /* IEEE half-precision (16-bit float) */
    BAFE_DTYPE_BF16 = 5,    /* Google Brain float (16-bit, 8 exp, 7 mantissa) */
} bafe_dtype;

/* C99 type name for a dtype, e.g. "float" or "int32_t". */
const char *bafe_dtype_c_name(bafe_dtype d);

/* numpy dtype string for a dtype, e.g. "float32". */
const char *bafe_dtype_numpy_name(bafe_dtype d);

/* bytes per element */
size_t bafe_dtype_byte_size(bafe_dtype d);

/* parse a dtype from a string like "f32" or "float32" */
bafe_dtype bafe_dtype_from_str(const char *s);

/* ------------------------------------------------------------------ */
/* Shape                                                              */
/* ------------------------------------------------------------------ */

#define BAFE_MAX_RANK 8

typedef struct {
    int32_t dims[BAFE_MAX_RANK];
    int32_t rank;
} bafe_shape;

/* constructors */
bafe_shape bafe_shape_scalar(void);                  /* rank 0 */
bafe_shape bafe_shape_make_1(int32_t d0);
bafe_shape bafe_shape_make_2(int32_t d0, int32_t d1);
bafe_shape bafe_shape_make_3(int32_t d0, int32_t d1, int32_t d2);
bafe_shape bafe_shape_make(int32_t rank, const int32_t *dims);

/* queries */
bool      bafe_shape_is_scalar(const bafe_shape *s);
bool      bafe_shape_is_empty(const bafe_shape *s);  /* any dim == 0 */
int32_t   bafe_shape_rank(const bafe_shape *s);
size_t    bafe_shape_numel(const bafe_shape *s);     /* 1 for scalar, 0 if any 0 */
size_t    bafe_shape_nbytes(const bafe_shape *s, bafe_dtype d);
int32_t   bafe_shape_dim(const bafe_shape *s, int32_t i);  /* with bounds check */

/* shape algebra */
bafe_shape bafe_shape_broadcast(const bafe_shape *a, const bafe_shape *b);
bafe_shape bafe_shape_reduce(const bafe_shape *s, const int32_t *axes,
                              int32_t n_axes, bool keepdims);
bafe_shape bafe_shape_transpose(const bafe_shape *s, const int32_t *perm);
bool       bafe_shape_eq(const bafe_shape *a, const bafe_shape *b);

/* string form, e.g. "(64,64)" -- writes into buf, returns length */
int bafe_shape_snprintf(char *buf, size_t buf_size, const bafe_shape *s);

/* ------------------------------------------------------------------ */
/* Layout                                                             */
/* ------------------------------------------------------------------ */

typedef enum {
    BAFE_LAYOUT_ROW_MAJOR = 0,    /* C order */
    BAFE_LAYOUT_COL_MAJOR = 1,    /* Fortran order */
    BAFE_LAYOUT_BLOCKED   = 2,    /* Phase 2 */
    BAFE_LAYOUT_TENSOR_CORE = 3,  /* Phase 2, GPU */
} bafe_layout;

const char *bafe_layout_name(bafe_layout l);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_TYPES_H */
