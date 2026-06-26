/* bafe/ops.h - op registry and shape inference
 *
 * Each op has:
 *   - a name (str)
 *   - an arity (number of tensor inputs)
 *   - a shape inference function
 *
 * The registry is built at startup in ops.c and is read-only afterwards.
 */
#ifndef BAFE_OPS_H
#define BAFE_OPS_H

#include "bafe/types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BAFE_MAX_ATTRS 4
#define BAFE_MAX_ATTR_LEN 32

typedef struct {
    int32_t     n_axes;
    int32_t     axes[BAFE_MAX_ATTR_LEN];
    int32_t     n_perm;
    int32_t     perm[BAFE_MAX_ATTR_LEN];
    int32_t     n_shape;
    int32_t     shape[BAFE_MAX_ATTR_LEN];
    bool        keepdims;
    double      scalar_value;   /* for constant folding */
    bool        has_scalar;
    char        name[BAFE_MAX_ATTR_LEN]; /* input name, etc. */
} bafe_op_attrs;

typedef bafe_shape (*bafe_shape_fn)(const bafe_shape *inputs, int n_inputs,
                                     const bafe_op_attrs *attrs);

typedef struct {
    const char       *name;
    int               arity;
    bool              has_fusion_form;
    const char       *c_name;     /* kernel function name */
    bafe_shape_fn     shape_fn;
} bafe_op;

/* Returns the registered op or NULL if unknown. */
const bafe_op *bafe_op_get(const char *name);

/* Returns the number of registered ops. */
int bafe_op_count(void);

/* Returns op at index i (for iteration). */
const bafe_op *bafe_op_at(int i);

/* Returns true if op_name starts with "fused_". */
bool bafe_op_is_fused(const char *op_name);

/* Initializes the registry. Idempotent. Called automatically on first use. */
void bafe_op_registry_init(void);

/* Default attrs (all-zero). */
bafe_op_attrs bafe_op_attrs_default(void);

#ifdef __cplusplus
}
#endif

#endif /* BAFE_OPS_H */
