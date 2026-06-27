"""BAFE - ctypes binding to libbafe.so.

This module loads libbafe.so and exposes the C API as Python functions.
The public API (matmul, add, relu, @jit, ...) is in __init__.py; this
file is the low-level FFI.

The library is searched in this order:
  1. $BAFE_LIB environment variable (full path to .so)
  2. ./bafe/build/libbafe.so (development)
  3. ./build/libbafe.so
  4. system library paths (via ctypes.util.find_library)
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from ctypes import (
    c_int, c_int32, c_uint32, c_size_t, c_double, c_bool, c_char, c_char_p, c_void_p,
    POINTER, Structure, byref, cast, string_at,
)

# ---------------------------------------------------------------------------
# Path resolution for libbafe.so
# ---------------------------------------------------------------------------

def _find_library() -> str:
    env = os.environ.get("BAFE_LIB")
    if env and Path(env).exists():
        return env
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "bafe" / "build" / "libbafe.so",
        Path(__file__).resolve().parent.parent / "build" / "libbafe.so",
        Path.cwd() / "bafe" / "build" / "libbafe.so",
        Path.cwd() / "build" / "libbafe.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # last resort: let ctypes try system paths
    found = ctypes.util.find_library("bafe")  # type: ignore[attr-defined]
    if found:
        return found
    raise RuntimeError(
        "libbafe.so not found. Set BAFE_LIB to its path, or run `make` "
        "in the project root to build it."
    )


# ---------------------------------------------------------------------------
# C struct definitions (must match bafe/types.h, bafe/ir.h, bafe/ops.h, etc.)
# ---------------------------------------------------------------------------

class BafeShape(Structure):
    _fields_ = [
        ("dims", c_int32 * 8),    # BAFE_MAX_RANK = 8
        ("rank", c_int32),
    ]


BAFE_MAX_CHILDREN = 4
BAFE_MAX_ATTR_LEN = 32


class BafeOpAttrs(Structure):
    _fields_ = [
        ("n_axes", c_int32),
        ("axes", c_int32 * BAFE_MAX_ATTR_LEN),
        ("n_perm", c_int32),
        ("perm", c_int32 * BAFE_MAX_ATTR_LEN),
        ("n_shape", c_int32),
        ("shape", c_int32 * BAFE_MAX_ATTR_LEN),
        ("keepdims", c_bool),
        ("scalar_value", c_double),
        ("has_scalar", c_bool),
        ("name", c_char * BAFE_MAX_ATTR_LEN),
    ]


BAFE_MAX_NODES = 512


class BafeNode(Structure):
    _fields_ = [
        ("id", c_int32),
        ("op_name", c_char_p),
        ("attrs", BafeOpAttrs),
        ("n_children", c_int),
        ("children", c_int32 * BAFE_MAX_CHILDREN),
        ("shape", BafeShape),
        ("dtype", c_int),
        ("layout", c_int),  # Phase 2: bafe_layout enum
        ("input_name", c_char * BAFE_MAX_ATTR_LEN),
        ("is_input", c_bool),
        ("is_constant", c_bool),
        ("const_value", c_double),
    ]


class BafeGraph(Structure):
    _fields_ = [
        ("nodes", BafeNode * BAFE_MAX_NODES),
        ("n_nodes", c_int),
        ("inputs", c_int32 * BAFE_MAX_NODES),
        ("n_inputs", c_int),
        ("outputs", c_int32 * BAFE_MAX_NODES),
        ("n_outputs", c_int),
    ]


# Phase 2: rewrite alternatives (used by tests)
BAFE_MAX_ALTERNATIVES = 512


class BafeAlternative(Structure):
    _fields_ = [
        ("original_node_id", c_int32),
        ("op_name", c_char_p),
        ("attrs", BafeOpAttrs),
        ("n_children", c_int),
        ("children", c_int32 * BAFE_MAX_CHILDREN),
    ]


class BafeAltList(Structure):
    _fields_ = [
        ("items", BafeAlternative * BAFE_MAX_ALTERNATIVES),
        ("n", c_int),
    ]


# Phase 2: cost model struct (used by tests)
class BafeCostModel(Structure):
    _fields_ = [
        ("alpha_flops", c_double),
        ("beta_bytes", c_double),
        ("gamma_intermediate", c_double),
        ("delta_fuse", c_double),
        ("epsilon_layout_conv", c_double),
        ("zeta_layout_fuse", c_double),
        ("eta_contiguous", c_double),
    ]


# ---------------------------------------------------------------------------
# Load library and set up function prototypes
# ---------------------------------------------------------------------------

_lib_path = _find_library()
_lib = ctypes.CDLL(_lib_path)

# Phase 2: rewrite + cost bindings (used by tests)
_lib.bafe_cost_model_default.argtypes = []
_lib.bafe_cost_model_default.restype = BafeCostModel
_lib.bafe_cost_graph.argtypes = [POINTER(BafeCostModel), POINTER(BafeGraph)]
_lib.bafe_cost_graph.restype = c_double
_lib.bafe_cost_model_calibrate.argtypes = [POINTER(BafeCostModel), POINTER(c_double), c_int, c_double]
_lib.bafe_cost_model_calibrate.restype = BafeCostModel
_lib.bafe_cost_model_calibrated_default.argtypes = []
_lib.bafe_cost_model_calibrated_default.restype = BafeCostModel
_lib.bafe_rewrite_find.argtypes = [POINTER(BafeGraph), POINTER(BafeAltList)]
_lib.bafe_rewrite_find.restype = c_int
_lib.bafe_rewrite_default_count.argtypes = []
_lib.bafe_rewrite_default_count.restype = c_int


# Phase 2 (issue #1): stochastic search budget + stats
class BafeSearchBudget(Structure):
    _fields_ = [
        ("max_iters", c_int),
        ("max_nodes", c_int),
        ("max_rewrites", c_int),
        ("time_budget_ms", c_int),
        ("temperature", c_double),
        ("seed", c_uint32),
        ("enable_multi_pass", c_bool),
    ]


class BafeSearchStats(Structure):
    _fields_ = [
        ("iters_done", c_int),
        ("alts_found", c_int),
        ("alts_materialized", c_int),
        ("nodes_added", c_int),
        ("elapsed_ms", c_double),
    ]


_lib.bafe_search_budget_default.argtypes = []
_lib.bafe_search_budget_default.restype = BafeSearchBudget
_lib.bafe_rewrite_stochastic.argtypes = [POINTER(BafeGraph), POINTER(BafeAltList),
                                          POINTER(BafeSearchBudget)]
_lib.bafe_rewrite_stochastic.restype = c_int
_lib.bafe_rewrite_stochastic_stats.argtypes = [POINTER(BafeGraph), POINTER(BafeAltList),
                                                POINTER(BafeSearchBudget),
                                                POINTER(BafeSearchStats)]
_lib.bafe_rewrite_stochastic_stats.restype = c_int
_lib.bafe_optimize_with_budget.argtypes = [POINTER(BafeGraph), POINTER(BafeGraph),
                                            POINTER(BafeSearchBudget),
                                            c_char_p, c_size_t]
_lib.bafe_optimize_with_budget.restype = c_int
_lib.bafe_optimize_and_compile_with_budget.argtypes = [POINTER(BafeGraph),
                                                        POINTER(BafeSearchBudget),
                                                        c_char_p, c_size_t]
_lib.bafe_optimize_and_compile_with_budget.restype = c_void_p


# Phase 3 (issue #6): auto-tuning + profiling
BAFE_NUM_FEATURES = 8
BAFE_PROFILING_LOG_SIZE = 4096


class BafeProfilingRecord(Structure):
    _fields_ = [
        ("graph_hash", c_char * 65),
        ("features", c_double * BAFE_NUM_FEATURES),
        ("predicted_cost", c_double),
        ("observed_ms", c_double),
        ("kernel_id", c_int),
    ]


class BafeProfilingLog(Structure):
    _fields_ = [
        ("records", BafeProfilingRecord * BAFE_PROFILING_LOG_SIZE),
        ("n", c_int),
        ("head", c_int),
        ("wrapped", c_bool),
    ]


class BafeLearnedCostModel(Structure):
    _fields_ = [
        ("weights", c_double * BAFE_NUM_FEATURES),
        ("bias", c_double),
        ("r_squared", c_double),
        ("n_samples", c_int),
        ("valid", c_bool),
    ]


class BafeAutotuneConfig(Structure):
    _fields_ = [
        ("enabled", c_bool),
        ("refit_threshold", c_int),
        ("invalidation_drift", c_double),
        ("warmup_calls", c_int),
        ("timing_iters", c_int),
    ]


class BafeAutotuneStats(Structure):
    _fields_ = [
        ("total_calls", c_int),
        ("total_compiles", c_int),
        ("total_refits", c_int),
        ("total_invalidations", c_int),
        ("last_refit_r_squared", c_double),
        ("log_size", c_int),
    ]


_lib.bafe_profiling_init.argtypes = []
_lib.bafe_profiling_init.restype = None
_lib.bafe_profiling_reset.argtypes = []
_lib.bafe_profiling_reset.restype = None
_lib.bafe_profiling_extract_features.argtypes = [POINTER(BafeGraph), POINTER(c_double)]
_lib.bafe_profiling_extract_features.restype = None
_lib.bafe_profiling_add.argtypes = [c_char_p, POINTER(c_double), c_double, c_double, c_int]
_lib.bafe_profiling_add.restype = None
_lib.bafe_profiling_get_log.argtypes = []
_lib.bafe_profiling_get_log.restype = POINTER(BafeProfilingLog)
_lib.bafe_profiling_dump_jsonl.argtypes = [c_char_p]
_lib.bafe_profiling_dump_jsonl.restype = c_int
_lib.bafe_profiling_refit.argtypes = []
_lib.bafe_profiling_refit.restype = c_int
_lib.bafe_profiling_get_model.argtypes = []
_lib.bafe_profiling_get_model.restype = POINTER(BafeLearnedCostModel)
_lib.bafe_profiling_predict_ms.argtypes = [POINTER(c_double)]
_lib.bafe_profiling_predict_ms.restype = c_double
_lib.bafe_autotune_config_default.argtypes = []
_lib.bafe_autotune_config_default.restype = BafeAutotuneConfig
_lib.bafe_autotune_configure.argtypes = [POINTER(BafeAutotuneConfig)]
_lib.bafe_autotune_configure.restype = None
_lib.bafe_autotune_get_config.argtypes = []
_lib.bafe_autotune_get_config.restype = BafeAutotuneConfig
_lib.bafe_autotune_get_stats.argtypes = []
_lib.bafe_autotune_get_stats.restype = BafeAutotuneStats
_lib.bafe_jit_invalidate_memory_cache.argtypes = []
_lib.bafe_jit_invalidate_memory_cache.restype = None


# Phase 3 (issue #7): cross-kernel fusion
_lib.bafe_fuse_concat.argtypes = [POINTER(BafeGraph), POINTER(BafeGraph),
                                   POINTER(BafeGraph), c_char_p, c_size_t]
_lib.bafe_fuse_concat.restype = c_int
_lib.bafe_fuse_compile.argtypes = [POINTER(BafeGraph), POINTER(BafeGraph),
                                    c_char_p, c_size_t]
_lib.bafe_fuse_compile.restype = c_void_p


# Phase 3 (issue #4): multi-tier pruning with time budget
class BafePruningConfig(Structure):
    _fields_ = [
        ("time_budget_ms", c_int),
        ("max_nodes", c_int),
        ("max_rewrites", c_int),
        ("max_egraph_size", c_int),
        ("beam_width", c_int),
        ("heuristic_threshold", c_double),
        ("temperature", c_double),
        ("seed", c_uint32),
        ("enable_anytime", c_bool),
    ]


class BafePruningStats(Structure):
    _fields_ = [
        ("regime", c_int),
        ("tier_a_passed", c_int),
        ("tier_b_passed", c_int),
        ("tier_c_kept", c_int),
        ("tier_d_materialized", c_int),
        ("total_alts_found", c_int),
        ("best_cost", c_int),
        ("elapsed_ms", c_double),
        ("was_interrupted", c_bool),
    ]


_lib.bafe_pruning_config_default.argtypes = []
_lib.bafe_pruning_config_default.restype = BafePruningConfig
_lib.bafe_pruning_regime_from_budget.argtypes = [c_int]
_lib.bafe_pruning_regime_from_budget.restype = c_int
_lib.bafe_pruning_beam_width_for_regime.argtypes = [c_int]
_lib.bafe_pruning_beam_width_for_regime.restype = c_int
_lib.bafe_pruning_iters_for_regime.argtypes = [c_int]
_lib.bafe_pruning_iters_for_regime.restype = c_int
_lib.bafe_pruning_run.argtypes = [POINTER(BafeGraph), POINTER(BafeAltList),
                                   POINTER(BafePruningConfig), POINTER(BafePruningStats)]
_lib.bafe_pruning_run.restype = c_int
_lib.bafe_pruning_run_with_budget.argtypes = [POINTER(BafeGraph), POINTER(BafeAltList),
                                               c_int, POINTER(BafePruningStats)]
_lib.bafe_pruning_run_with_budget.restype = c_int


# types
_lib.bafe_dtype_c_name.argtypes = [c_int]
_lib.bafe_dtype_c_name.restype = c_char_p
_lib.bafe_dtype_numpy_name.argtypes = [c_int]
_lib.bafe_dtype_numpy_name.restype = c_char_p
_lib.bafe_dtype_byte_size.argtypes = [c_int]
_lib.bafe_dtype_byte_size.restype = c_size_t
_lib.bafe_dtype_from_str.argtypes = [c_char_p]
_lib.bafe_dtype_from_str.restype = c_int

_lib.bafe_shape_make.argtypes = [c_int32, POINTER(c_int32)]
_lib.bafe_shape_make.restype = BafeShape
_lib.bafe_shape_numel.argtypes = [POINTER(BafeShape)]
_lib.bafe_shape_numel.restype = c_size_t
_lib.bafe_shape_broadcast.argtypes = [POINTER(BafeShape), POINTER(BafeShape)]
_lib.bafe_shape_broadcast.restype = BafeShape
_lib.bafe_shape_reduce.argtypes = [POINTER(BafeShape), POINTER(c_int32), c_int32, c_bool]
_lib.bafe_shape_reduce.restype = BafeShape
_lib.bafe_shape_transpose.argtypes = [POINTER(BafeShape), POINTER(c_int32)]
_lib.bafe_shape_transpose.restype = BafeShape
_lib.bafe_shape_eq.argtypes = [POINTER(BafeShape), POINTER(BafeShape)]
_lib.bafe_shape_eq.restype = c_bool
_lib.bafe_shape_is_scalar.argtypes = [POINTER(BafeShape)]
_lib.bafe_shape_is_scalar.restype = c_bool
_lib.bafe_shape_is_empty.argtypes = [POINTER(BafeShape)]
_lib.bafe_shape_is_empty.restype = c_bool
_lib.bafe_shape_rank.argtypes = [POINTER(BafeShape)]
_lib.bafe_shape_rank.restype = c_int32
_lib.bafe_shape_nbytes.argtypes = [POINTER(BafeShape), c_int]
_lib.bafe_shape_nbytes.restype = c_size_t
_lib.bafe_shape_dim.argtypes = [POINTER(BafeShape), c_int32]
_lib.bafe_shape_dim.restype = c_int32
_lib.bafe_layout_name.argtypes = [c_int]
_lib.bafe_layout_name.restype = c_char_p

# ops
_lib.bafe_op_get.argtypes = [c_char_p]
_lib.bafe_op_get.restype = c_void_p

# ir
_lib.bafe_graph_init.argtypes = [POINTER(BafeGraph)]
_lib.bafe_graph_init.restype = None
_lib.bafe_graph_add_input.argtypes = [POINTER(BafeGraph), c_char_p, POINTER(BafeShape), c_int]
_lib.bafe_graph_add_input.restype = c_int32
_lib.bafe_graph_add_input_with_layout.argtypes = [POINTER(BafeGraph), c_char_p, POINTER(BafeShape), c_int, c_int]
_lib.bafe_graph_add_input_with_layout.restype = c_int32
_lib.bafe_graph_set_node_layout.argtypes = [POINTER(BafeGraph), c_int32, c_int]
_lib.bafe_graph_set_node_layout.restype = c_int
_lib.bafe_graph_get_node_layout.argtypes = [POINTER(BafeGraph), c_int32]
_lib.bafe_graph_get_node_layout.restype = c_int
_lib.bafe_graph_add_constant.argtypes = [POINTER(BafeGraph), c_double, POINTER(BafeShape), c_int]
_lib.bafe_graph_add_constant.restype = c_int32
_lib.bafe_graph_add.argtypes = [
    POINTER(BafeGraph), c_char_p, POINTER(c_int32), c_int, POINTER(BafeOpAttrs)
]
_lib.bafe_graph_add.restype = c_int32

_lib.bafe_graph_matmul.argtypes = [POINTER(BafeGraph), c_int32, c_int32]
_lib.bafe_graph_matmul.restype = c_int32
_lib.bafe_graph_add_op.argtypes = [POINTER(BafeGraph), c_int32, c_int32]
_lib.bafe_graph_add_op.restype = c_int32
_lib.bafe_graph_mul.argtypes = [POINTER(BafeGraph), c_int32, c_int32]
_lib.bafe_graph_mul.restype = c_int32
_lib.bafe_graph_sub.argtypes = [POINTER(BafeGraph), c_int32, c_int32]
_lib.bafe_graph_sub.restype = c_int32
_lib.bafe_graph_bias_add.argtypes = [POINTER(BafeGraph), c_int32, c_int32]
_lib.bafe_graph_bias_add.restype = c_int32
_lib.bafe_graph_relu.argtypes = [POINTER(BafeGraph), c_int32]
_lib.bafe_graph_relu.restype = c_int32
_lib.bafe_graph_sigmoid.argtypes = [POINTER(BafeGraph), c_int32]
_lib.bafe_graph_sigmoid.restype = c_int32
_lib.bafe_graph_tanh.argtypes = [POINTER(BafeGraph), c_int32]
_lib.bafe_graph_tanh.restype = c_int32
_lib.bafe_graph_neg.argtypes = [POINTER(BafeGraph), c_int32]
_lib.bafe_graph_neg.restype = c_int32
_lib.bafe_graph_transpose.argtypes = [POINTER(BafeGraph), c_int32, POINTER(c_int32), c_int32]
_lib.bafe_graph_transpose.restype = c_int32
_lib.bafe_graph_reduce_sum.argtypes = [POINTER(BafeGraph), c_int32, POINTER(c_int32), c_int32, c_int]
_lib.bafe_graph_reduce_sum.restype = c_int32
_lib.bafe_graph_reduce_max.argtypes = [POINTER(BafeGraph), c_int32, POINTER(c_int32), c_int32, c_int]
_lib.bafe_graph_reduce_max.restype = c_int32
_lib.bafe_graph_reshape.argtypes = [POINTER(BafeGraph), c_int32, POINTER(c_int32), c_int32]
_lib.bafe_graph_reshape.restype = c_int32
_lib.bafe_graph_broadcast_to.argtypes = [POINTER(BafeGraph), c_int32, POINTER(c_int32), c_int32]
_lib.bafe_graph_broadcast_to.restype = c_int32

_lib.bafe_graph_set_output.argtypes = [POINTER(BafeGraph), c_int32]
_lib.bafe_graph_set_output.restype = None

# bafe (top-level)
_lib.bafe_optimize.argtypes = [POINTER(BafeGraph), POINTER(BafeGraph), c_char_p, c_size_t]
_lib.bafe_optimize.restype = c_int
_lib.bafe_optimize_and_compile.argtypes = [POINTER(BafeGraph), c_char_p, c_size_t]
_lib.bafe_optimize_and_compile.restype = c_void_p

# jit
_lib.bafe_jit_get_or_compile.argtypes = [POINTER(BafeGraph), c_char_p, c_size_t]
_lib.bafe_jit_get_or_compile.restype = c_void_p
_lib.bafe_jit_set_cache_dir.argtypes = [c_char_p]
_lib.bafe_jit_set_cache_dir.restype = None
_lib.bafe_jit_cache_dir.argtypes = []
_lib.bafe_jit_cache_dir.restype = c_char_p

class BafeJitStats(Structure):
    _fields_ = [
        ("hits", c_int),
        ("misses", c_int),
        ("compiles", c_int),
        ("compile_failures", c_int),
    ]

_lib.bafe_jit_get_stats.argtypes = []
_lib.bafe_jit_get_stats.restype = BafeJitStats


# ---------------------------------------------------------------------------
# Convenience helpers for the public API
# ---------------------------------------------------------------------------

def make_shape(dims):
    """Build a BafeShape from a Python tuple/list of ints."""
    n = len(dims)
    arr = (c_int32 * max(n, 1))(*dims)
    return _lib.bafe_shape_make(c_int32(n), arr)


def make_attrs(**kw):
    """Build a BafeOpAttrs from keyword args."""
    a = BafeOpAttrs()
    # zero it
    ctypes.memset(byref(a), 0, ctypes.sizeof(a))
    if "axes" in kw:
        ax = list(kw["axes"])
        a.n_axes = len(ax)
        for i, v in enumerate(ax):
            a.axes[i] = v
    if "perm" in kw:
        p = list(kw["perm"])
        a.n_perm = len(p)
        for i, v in enumerate(p):
            a.perm[i] = v
    if "shape" in kw:
        s = list(kw["shape"])
        a.n_shape = len(s)
        for i, v in enumerate(s):
            a.shape[i] = v
    if "keepdims" in kw:
        a.keepdims = 1 if kw["keepdims"] else 0
    if "scalar" in kw:
        a.scalar_value = float(kw["scalar"])
        a.has_scalar = 1
    if "name" in kw:
        nm = kw["name"].encode("utf-8")[:BAFE_MAX_ATTR_LEN-1]
        a.name = nm
    return a


def graph_summary(g: BafeGraph) -> str:
    """Read summary string from a BafeGraph."""
    buf = ctypes.create_string_buffer(8192)
    # bafe_graph_summary is in ir.h
    _lib.bafe_graph_summary.argtypes = [POINTER(BafeGraph), c_char_p, c_size_t]
    _lib.bafe_graph_summary.restype = c_int
    _lib.bafe_graph_summary(byref(g), buf, c_size_t(len(buf)))
    return buf.value.decode("utf-8")


__all__ = [
    "_lib", "_lib_path",
    "BafeShape", "BafeOpAttrs", "BafeNode", "BafeGraph", "BafeJitStats",
    "BAFE_MAX_NODES", "BAFE_MAX_CHILDREN", "BAFE_MAX_ATTR_LEN",
    "make_shape", "make_attrs", "graph_summary",
]
