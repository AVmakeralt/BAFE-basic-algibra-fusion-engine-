"""BAFE - Basic Algebra Fusion Engine.

Public API:
    import bafe

    @bafe.jit
    def f(A, B, C):
        return bafe.relu(bafe.matmul(A, B) + C)

The decorator traces the function (which builds an IR graph via the op
functions), runs the BAFE optimization pipeline, JIT-compiles the result,
and returns a callable that invokes the compiled kernel directly.
"""
from __future__ import annotations

import ctypes
import functools
import os
from typing import Callable, List, Tuple, Any

import numpy as np

from bafe._binding import (
    _lib, BafeGraph, BafeShape, BafeOpAttrs, BafeNode,
    make_shape, make_attrs, graph_summary,
    BAFE_MAX_NODES, BAFE_MAX_CHILDREN,
)

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

_NP_TO_BAFE = {
    np.dtype("float32"): 0,  # BAFE_DTYPE_F32
    np.dtype("float64"): 1,  # BAFE_DTYPE_F64
    np.dtype("int32"):   2,  # BAFE_DTYPE_I32
    np.dtype("int64"):   3,  # BAFE_DTYPE_I64
    np.dtype("float16"): 4,  # BAFE_DTYPE_F16
}

# BF16 is not a native numpy dtype, so we use a custom representation.
# We store BF16 data as uint16 arrays, with the dtype name "bfloat16".
# The binding handles the conversion at the FFI boundary.
_BAFE_TO_NP = {v: k for k, v in _NP_TO_BAFE.items()}
_BAFE_TO_NP[5] = np.dtype("float32")  # BF16: we'll convert to/from float32 at the boundary


# ---------------------------------------------------------------------------
# Tracing context
# ---------------------------------------------------------------------------

class _TraceContext:
    """Holds the in-progress graph and the name->node_id mapping."""
    def __init__(self):
        self.graph = BafeGraph()
        _lib.bafe_graph_init(ctypes.byref(self.graph))
        self.inputs: List[Tuple[str, "Tensor"]] = []
        self.input_names: List[str] = []

    def add_input(self, name: str, shape: Tuple[int, ...], dtype: int,
                  layout: int = 0) -> int:
        """Add an input. layout is a bafe_layout enum value (0=row, 1=col)."""
        sh = make_shape(shape)
        if layout == 0:
            # default row-major: use the plain add_input for backward compat
            nid = _lib.bafe_graph_add_input(
                ctypes.byref(self.graph),
                name.encode("utf-8"),
                ctypes.byref(sh),
                ctypes.c_int(dtype),
            )
        else:
            nid = _lib.bafe_graph_add_input_with_layout(
                ctypes.byref(self.graph),
                name.encode("utf-8"),
                ctypes.byref(sh),
                ctypes.c_int(dtype),
                ctypes.c_int(layout),
            )
        if nid < 0:
            raise RuntimeError(f"failed to add input {name}")
        self.input_names.append(name)
        return nid

    def add_op(self, op_name: str, children: List[int], **attrs) -> int:
        n = len(children)
        arr = (ctypes.c_int32 * max(n, 1))(*children) if n else None
        a = make_attrs(**attrs) if attrs else None
        nid = _lib.bafe_graph_add(
            ctypes.byref(self.graph),
            op_name.encode("utf-8"),
            arr,
            ctypes.c_int(n),
            ctypes.byref(a) if a else None,
        )
        if nid < 0:
            raise RuntimeError(f"failed to add op {op_name} (children={children})")
        return nid

    def set_output(self, nid: int) -> None:
        _lib.bafe_graph_set_output(ctypes.byref(self.graph), ctypes.c_int32(nid))


_TRACE: _TraceContext | None = None


def _ensure_trace() -> _TraceContext:
    global _TRACE
    if _TRACE is None:
        raise RuntimeError(
            "no active trace - bafe ops can only be called inside a "
            "function decorated with @bafe.jit"
        )
    return _TRACE


# ---------------------------------------------------------------------------
# Tensor handle (symbolic during tracing)
# ---------------------------------------------------------------------------

class Tensor:
    """A symbolic tensor handle.

    During tracing (inside an @bafe.jit function), Tensor instances refer
    to IR graph nodes. They support Python operator overloads so users
    can write natural math expressions.
    """
    __slots__ = ("node_id", "shape", "dtype", "_name")

    def __init__(self, node_id: int, shape: Tuple[int, ...], dtype: int, name: str | None = None):
        self.node_id = node_id
        self.shape = shape
        self.dtype = dtype
        self._name = name

    def __add__(self, other: "Tensor") -> "Tensor":
        return _binop("add", self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return _binop("sub", self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        return _binop("mul", self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        tc = _ensure_trace()
        # Support both rank-2 and batched (rank > 2) matmul
        if len(self.shape) < 2 or len(other.shape) < 2:
            raise ValueError("matmul requires rank>=2 tensors")
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(
                f"matmul shape mismatch: {self.shape} @ {other.shape} (K dim must match)"
            )
        if len(self.shape) == 2 and len(other.shape) == 2:
            out_shape = (self.shape[0], other.shape[1])
        else:
            # batched: broadcast leading dims, output = (batch..., M, N)
            out_shape = tuple(max(s1, s2) for s1, s2 in zip(
                self.shape[:-2], other.shape[:-2]
            )) + (self.shape[-2], other.shape[-1])
        nid = tc.add_op("matmul", [self.node_id, other.node_id])
        return Tensor(nid, out_shape, self.dtype)

    @property
    def name(self) -> str | None:
        return self._name

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={_BAFE_TO_NP[self.dtype]})"


def _binop(op: str, a: Tensor, b: Tensor) -> Tensor:
    tc = _ensure_trace()
    # broadcasting: simple version, just take the larger shape
    out_shape = _broadcast_shapes(a.shape, b.shape)
    nid = tc.add_op(op, [a.node_id, b.node_id])
    return Tensor(nid, out_shape, a.dtype)


def _broadcast_shapes(a, b):
    if len(a) == len(b):
        return tuple(max(x, y) for x, y in zip(a, b))
    if len(a) < len(b):
        a, b = b, a
    # pad b
    bpad = (1,) * (len(a) - len(b)) + tuple(b)
    return tuple(max(x, y) for x, y in zip(a, bpad))


# ---------------------------------------------------------------------------
# Op functions (used inside @bafe.jit functions)
# ---------------------------------------------------------------------------

def input(shape: Tuple[int, ...], dtype: str | np.dtype = "float32",
          name: str = "x", layout: str = "row") -> Tensor:
    """Declare an input tensor.

    Usually called automatically by @bafe.jit based on the function's
    arguments, but can also be called explicitly.

    layout: "row" (default, C order) or "col" (Fortran order).
    The layout tag tells BAFE how the input data is stored in memory,
    which affects codegen (access patterns) and the cost model
    (conversion penalties, fusion bonuses).
    """
    tc = _ensure_trace()
    np_dt = np.dtype(dtype)
    if np_dt not in _NP_TO_BAFE:
        raise ValueError(f"unsupported dtype {dtype}; supported: {list(_NP_TO_BAFE)}")
    bafe_dt = _NP_TO_BAFE[np_dt]
    layout_map = {"row": 0, "col": 1, "blocked": 2, "tc": 3}
    if layout not in layout_map:
        raise ValueError(f"unsupported layout {layout!r}; supported: {list(layout_map)}")
    bafe_layout = layout_map[layout]
    nid = tc.add_input(name, tuple(int(s) for s in shape), bafe_dt, layout=bafe_layout)
    return Tensor(nid, tuple(int(s) for s in shape), bafe_dt, name=name)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a @ b


def add(a: Tensor, b: Tensor) -> Tensor:
    return a + b


def sub(a: Tensor, b: Tensor) -> Tensor:
    return a - b


def mul(a: Tensor, b: Tensor) -> Tensor:
    return a * b


def relu(x: Tensor) -> Tensor:
    tc = _ensure_trace()
    nid = tc.add_op("relu", [x.node_id])
    return Tensor(nid, x.shape, x.dtype)


def sigmoid(x: Tensor) -> Tensor:
    tc = _ensure_trace()
    nid = tc.add_op("sigmoid", [x.node_id])
    return Tensor(nid, x.shape, x.dtype)


def tanh(x: Tensor) -> Tensor:
    tc = _ensure_trace()
    nid = tc.add_op("tanh", [x.node_id])
    return Tensor(nid, x.shape, x.dtype)


def bias_add(x: Tensor, bias: Tensor) -> Tensor:
    tc = _ensure_trace()
    nid = tc.add_op("bias_add", [x.node_id, bias.node_id])
    return Tensor(nid, x.shape, x.dtype)


def transpose(x: Tensor, perm: Tuple[int, ...]) -> Tensor:
    tc = _ensure_trace()
    nid = tc.add_op("transpose", [x.node_id], perm=list(perm))
    out_shape = tuple(x.shape[p] for p in perm)
    return Tensor(nid, out_shape, x.dtype)


def reduce_sum(x: Tensor, axes: Tuple[int, ...], keepdims: bool = False) -> Tensor:
    tc = _ensure_trace()
    nid = tc.add_op("reduce_sum", [x.node_id], axes=list(axes), keepdims=keepdims)
    out_shape = _reduce_shape(x.shape, axes, keepdims)
    return Tensor(nid, out_shape, x.dtype)


def reduce_max(x: Tensor, axes: Tuple[int, ...], keepdims: bool = False) -> Tensor:
    tc = _ensure_trace()
    nid = tc.add_op("reduce_max", [x.node_id], axes=list(axes), keepdims=keepdims)
    out_shape = _reduce_shape(x.shape, axes, keepdims)
    return Tensor(nid, out_shape, x.dtype)


def reshape(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    tc = _ensure_trace()
    nid = tc.add_op("reshape", [x.node_id], shape=list(shape))
    return Tensor(nid, tuple(shape), x.dtype)


def broadcast_to(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    tc = _ensure_trace()
    nid = tc.add_op("broadcast_to", [x.node_id], shape=list(shape))
    return Tensor(nid, tuple(shape), x.dtype)


def _reduce_shape(in_shape, axes, keepdims):
    axes_set = {a % len(in_shape) for a in axes}
    out = []
    for i, d in enumerate(in_shape):
        if i in axes_set:
            if keepdims:
                out.append(1)
        else:
            out.append(d)
    return tuple(out)


# ---------------------------------------------------------------------------
# @bafe.jit decorator
# ---------------------------------------------------------------------------

class JittedFunction:
    """A JIT-compiled BAFE function.

    On first call with concrete numpy arrays, it:
      1. Inspects the input shapes/dtypes
      2. Runs the user function under a trace
      3. Calls bafe_optimize + bafe_jit_get_or_compile
      4. Builds a ctypes function pointer with the correct signature
      5. Invokes the kernel

    Subsequent calls with the same shapes/dtypes hit the JIT cache.

    Phase 2 (issue #1): if a search budget is set, uses stochastic
    multi-pass search to discover deeper rewrites.
    """

    def __init__(self, fn: Callable, budget=None, autotune=False, time_budget_ms=None):
        self._fn = fn
        self._budget = budget  # BafeSearchBudget or None (deterministic)
        self._autotune = autotune  # Phase 3: enable autotune loop
        self._time_budget_ms = time_budget_ms  # Phase 3 (issue #4): pruning time budget
        self._compiled = {}  # key: (shapes, dtypes, layouts) -> compiled tuple
        self._call_count = {}  # key -> call number (for warmup tracking)
        functools.update_wrapper(self, fn)

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        # validate
        if not args:
            raise TypeError("jitted function requires at least one input")
        for a in args:
            if not isinstance(a, np.ndarray):
                raise TypeError(f"expected numpy array, got {type(a)}")

        # build cache key (Phase 2: include layout so row-major vs col-major
        # inputs produce different compiled kernels)
        def _layout_of(a):
            if a.ndim >= 2 and a.flags["F_CONTIGUOUS"] and not a.flags["C_CONTIGUOUS"]:
                return "col"
            return "row"
        key = tuple((a.shape, str(a.dtype), _layout_of(a)) for a in args)

        if key not in self._compiled:
            self._compiled[key] = self._compile(args)
            # Phase 3 (issue #6): if autotune is enabled and this is a new
            # compile, increment the compile counter
            if self._autotune_enabled():
                stats = _lib.bafe_autotune_get_stats()
                # we can't easily mutate the C stats from here, but the C
                # side tracks compiles via bafe_jit_get_or_compile
                pass

        fn_ptr, sig, in_dtypes, out_shape, out_dtype, opt_graph, graph_hash, predicted_cost = self._compiled[key]

        # allocate output
        out = np.zeros(out_shape, dtype=out_dtype)

        # build arg list: inputs + output pointer
        # Phase 2: preserve the input's memory layout — if we compiled for
        # col-major, the array must stay col-major (don't ascontiguousarray it).
        c_args = []
        for a, dt in zip(args, in_dtypes):
            if a.dtype == dt:
                arr = a
            else:
                # dtype conversion needed; preserve layout
                if a.flags["F_CONTIGUOUS"] and not a.flags["C_CONTIGUOUS"]:
                    arr = np.asfortranarray(a, dtype=dt)
                else:
                    arr = np.ascontiguousarray(a, dtype=dt)
            c_args.append(arr.ctypes.data_as(ctypes.c_void_p))
        c_args.append(out.ctypes.data_as(ctypes.c_void_p))

        # call: sig(fn_ptr) creates the callable; then pass the c_args
        kernel = sig(fn_ptr)

        # Phase 3 (issue #6): if autotune is enabled, time the kernel + log
        if self._autotune_enabled():
            import time
            config = _lib.bafe_autotune_get_config()
            stats = _lib.bafe_autotune_get_stats()
            # warmup: skip timing for the first N calls
            # (stats.total_calls is incremented on the C side, but since
            # we're driving the loop from Python, we track it here)
            call_num = self._call_count.get(key, 0) + 1
            self._call_count[key] = call_num

            if call_num <= config.warmup_calls:
                kernel(*c_args)
            else:
                # time over multiple iterations
                iters = config.timing_iters if config.timing_iters > 0 else 1
                t0 = time.perf_counter()
                for _ in range(iters):
                    kernel(*c_args)
                t1 = time.perf_counter()
                observed_ms = (t1 - t0) * 1000.0 / iters

                # extract features + log
                features = (ctypes.c_double * 8)()
                _lib.bafe_profiling_extract_features(
                    ctypes.byref(opt_graph), features
                )
                _lib.bafe_profiling_add(
                    graph_hash.encode("utf-8") if isinstance(graph_hash, str) else graph_hash,
                    features,
                    ctypes.c_double(predicted_cost),
                    ctypes.c_double(observed_ms),
                    ctypes.c_int(0),
                )

                # check if we should refit
                log = _lib.bafe_profiling_get_log().contents
                if log.n > 0 and log.n % config.refit_threshold == 0:
                    _lib.bafe_profiling_refit()
                    # Phase 3 (issue #5): after refit, invalidate the
                    # Python-level compiled cache so the next call
                    # re-optimizes with the calibrated cost model.
                    # The C-level JIT cache is also invalidated so
                    # dlopen'd handles are released.
                    self._compiled.clear()
                    _lib.bafe_jit_invalidate_memory_cache()
        else:
            kernel(*c_args)

        return out

    def _autotune_enabled(self) -> bool:
        """Check if autotune is enabled for this function."""
        return getattr(self, "_autotune", False)

    def _compile(self, args: Tuple[np.ndarray, ...]):
        global _TRACE
        # run the trace
        _TRACE = _TraceContext()
        try:
            # build input tensors
            in_tensors = []
            for i, a in enumerate(args):
                np_dt = np.dtype(a.dtype)
                if np_dt not in _NP_TO_BAFE:
                    raise TypeError(f"unsupported dtype {np_dt}")
                bafe_dt = _NP_TO_BAFE[np_dt]
                name = self._fn.__code__.co_varnames[i] if i < len(self._fn.__code__.co_varnames) else f"in{i}"
                # Phase 2: auto-detect input layout from numpy array flags
                # If the array is Fortran-contiguous (col-major), tag it as "col".
                # Otherwise default to "row" (C order).
                if a.ndim >= 2 and a.flags["F_CONTIGUOUS"] and not a.flags["C_CONTIGUOUS"]:
                    layout = "col"
                else:
                    layout = "row"
                t = input(a.shape, np_dt, name=name, layout=layout)
                in_tensors.append(t)

            # call the user function
            result = self._fn(*in_tensors)
            if not isinstance(result, Tensor):
                raise TypeError(
                    f"jitted function must return a Tensor, got {type(result)}"
                )
            _TRACE.set_output(result.node_id)

            # snapshot the input graph (the optimize call may add nodes via rewrites)
            in_graph = _TRACE.graph
        finally:
            _TRACE = None

        # optimize + compile
        # Phase 2 (issue #1): if a budget is set, use stochastic multi-pass search
        # Phase 3 (issue #4): if time_budget_ms is set, use the pruning controller
        optimized = BafeGraph()
        err_buf = ctypes.create_string_buffer(256)
        if self._budget is not None or self._time_budget_ms is not None:
            # build a budget struct
            if self._budget is not None:
                budget = self._budget
            else:
                budget = _lib.bafe_search_budget_default()
            if self._time_budget_ms is not None:
                budget.time_budget_ms = int(self._time_budget_ms)
            rc = _lib.bafe_optimize_with_budget(
                ctypes.byref(in_graph),
                ctypes.byref(optimized),
                ctypes.byref(budget),
                err_buf,
                ctypes.c_size_t(len(err_buf)),
            )
        else:
            rc = _lib.bafe_optimize(
                ctypes.byref(in_graph),
                ctypes.byref(optimized),
                err_buf,
                ctypes.c_size_t(len(err_buf)),
            )
        if rc != 0:
            raise RuntimeError(
                f"bafe_optimize failed (code {rc}): {err_buf.value.decode()}"
            )

        # JIT compile
        fn_ptr = _lib.bafe_jit_get_or_compile(
            ctypes.byref(optimized),
            err_buf,
            ctypes.c_size_t(len(err_buf)),
        )
        if not fn_ptr:
            raise RuntimeError(
                f"bafe_jit_get_or_compile failed: {err_buf.value.decode()}"
            )

        # build ctypes signature: void name(const T1* in1, const T2* in2, ..., T* out)
        in_dtypes = [a.dtype for a in args]
        # all args are pointers (void*) for simplicity, ctypes will cast
        sig = ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * (len(args) + 1)))

        # output shape from the optimized graph's output node
        out_node = optimized.nodes[optimized.outputs[0]]
        out_shape = tuple(out_node.shape.dims[i] for i in range(out_node.shape.rank))
        out_np_dt = _BAFE_TO_NP[out_node.dtype]

        # Phase 3 (issue #6): compute graph hash + predicted cost for autotune logging
        graph_hash_buf = ctypes.create_string_buffer(65)
        _lib.bafe_jit_hash_graph(ctypes.byref(optimized), graph_hash_buf, ctypes.c_size_t(65))
        graph_hash = graph_hash_buf.value.decode("utf-8")

        # predicted cost = total graph cost from the cost model
        cm = _lib.bafe_cost_model_default()
        predicted_cost = _lib.bafe_cost_graph(ctypes.byref(cm), ctypes.byref(optimized))

        # keep a copy of the optimized graph for feature extraction during autotune
        # (we store it as a ctypes object so it doesn't get GC'd)
        opt_graph_copy = BafeGraph()
        ctypes.memmove(ctypes.byref(opt_graph_copy), ctypes.byref(optimized), ctypes.sizeof(BafeGraph))

        return (fn_ptr, sig, in_dtypes, out_shape, out_np_dt,
                opt_graph_copy, graph_hash, predicted_cost)


def jit(fn: Callable = None, *, budget=None, iters: int = None,
        temperature: float = None, seed: int = None, autotune: bool = False,
        time_budget_ms: int = None):
    """Decorator: trace + optimize + JIT-compile a tensor function.

    Phase 2 (issue #1): optional stochastic search parameters.
    Phase 3 (issue #4): optional time-budget pruning.
    Phase 3 (issue #6): optional autotune loop.

    Args:
        budget: a BafeSearchBudget for full control.
        iters: number of stochastic passes (default 4 if budget mode on).
        temperature: 0.0 = greedy, high = explore randomly (default 1.0).
        seed: PRNG seed for reproducibility (default 0xBAFE5EED).
        autotune: if True, enable the auto-tuning loop.
        time_budget_ms: wall-clock limit for the optimization search.
            Controls the pruning regime:
              <= 1 ms:   greedy (Level A+B only, beam=1)
              <= 10 ms:  light (A+B+C, beam=4)
              <= 100 ms: beam (A+B+C+D, beam=16)
              > 100 ms:  deep (all tiers, beam=64)
            0 or None means no limit (uses stochastic search).

    Examples:
        @bafe.jit
        def f(A, B): ...                      # deterministic (default)

        @bafe.jit(time_budget_ms=100)
        def f(A, B): ...                      # 100ms pruning budget

        @bafe.jit(time_budget_ms=1000, autotune=True)
        def f(A, B): ...                      # 1s budget + autotune
    """
    if fn is not None:
        return JittedFunction(fn, autotune=autotune, time_budget_ms=time_budget_ms)

    def deco(fn):
        b = budget
        if b is None and (iters is not None or temperature is not None or seed is not None):
            b = make_search_budget(
                max_iters=iters if iters is not None else 4,
                temperature=temperature if temperature is not None else 1.0,
                seed=seed if seed is not None else 0xBAFE5EED,
            )
        if b is not None and time_budget_ms is not None:
            b.time_budget_ms = int(time_budget_ms)
        return JittedFunction(fn, budget=b, autotune=autotune, time_budget_ms=time_budget_ms)
    return deco


def make_search_budget(max_iters: int = 4, max_nodes: int = 256,
                       max_rewrites: int = 64, time_budget_ms: int = 0,
                       temperature: float = 1.0, seed: int = 0xBAFE5EED,
                       enable_multi_pass: bool = True):
    """Build a BafeSearchBudget for use with @bafe.jit(budget=...).

    The budget controls the stochastic search layer:
      - max_iters: how many stochastic passes (each pass re-applies rules
        to newly-created nodes, discovering deeper rewrites)
      - max_nodes: hard cap on graph size during search
      - max_rewrites: cap on total rewrites materialized
      - time_budget_ms: wall-clock limit (0 = no limit)
      - temperature: 0.0 = greedy (only cost-reducing rewrites),
                     high = explore randomly
      - seed: PRNG seed for reproducibility
      - enable_multi_pass: if False, degrades to deterministic single-pass
    """
    from bafe._binding import BafeSearchBudget
    b = BafeSearchBudget()
    b.max_iters = int(max_iters)
    b.max_nodes = int(max_nodes)
    b.max_rewrites = int(max_rewrites)
    b.time_budget_ms = int(time_budget_ms)
    b.temperature = float(temperature)
    b.seed = int(seed) & 0xFFFFFFFF
    b.enable_multi_pass = bool(enable_multi_pass)
    return b


# expose the optimize function for low-level use
def optimize(graph: BafeGraph) -> BafeGraph:
    """Run the BAFE optimization pipeline on a graph."""
    out = BafeGraph()
    err = ctypes.create_string_buffer(256)
    rc = _lib.bafe_optimize(byref(graph), byref(out), err, c_size_t(len(err)))
    if rc != 0:
        raise RuntimeError(f"optimize failed: {err.value.decode()}")
    return out


# ---------------------------------------------------------------------------
# Phase 3 (issue #6): autotune API
# ---------------------------------------------------------------------------

def configure_autotune(refit_threshold: int = 20,
                       invalidation_drift: float = 0.25,
                       warmup_calls: int = 2,
                       timing_iters: int = 5):
    """Configure the global autotune settings.

    Args:
        refit_threshold: refit the cost model after this many new samples.
        invalidation_drift: invalidate cached kernels when predictions
                           drift by more than this ratio (0.25 = 25%).
        warmup_calls: skip timing for the first N calls (cache effects).
        timing_iters: average the kernel runtime over this many invocations.
    """
    cfg = _lib.bafe_autotune_config_default()
    cfg.enabled = True
    cfg.refit_threshold = int(refit_threshold)
    cfg.invalidation_drift = float(invalidation_drift)
    cfg.warmup_calls = int(warmup_calls)
    cfg.timing_iters = int(timing_iters)
    _lib.bafe_autotune_configure(ctypes.byref(cfg))


def autotune_stats() -> dict:
    """Get current autotune statistics.

    Returns a dict with:
        total_calls, total_compiles, total_refits, total_invalidations,
        last_refit_r_squared, log_size
    """
    s = _lib.bafe_autotune_get_stats()
    return {
        "total_calls": s.total_calls,
        "total_compiles": s.total_compiles,
        "total_refits": s.total_refits,
        "total_invalidations": s.total_invalidations,
        "last_refit_r_squared": s.last_refit_r_squared,
        "log_size": s.log_size,
    }


def autotune_refit() -> int:
    """Manually trigger a cost model refit.

    Returns 0 on success, non-zero if not enough samples.
    """
    return _lib.bafe_profiling_refit()


def autotune_model() -> dict:
    """Get the current learned cost model.

    Returns a dict with:
        weights (list of 8 floats), bias, r_squared, n_samples, valid
    """
    m = _lib.bafe_profiling_get_model().contents
    return {
        "weights": [m.weights[i] for i in range(8)],
        "bias": m.bias,
        "r_squared": m.r_squared,
        "n_samples": m.n_samples,
        "valid": bool(m.valid),
    }


def autotune_dump_log(path: str) -> int:
    """Dump the profiling log to a JSONL file. Returns number of records."""
    return _lib.bafe_profiling_dump_jsonl(path.encode("utf-8"))


def autotune_reset():
    """Reset all profiling state (log, learned model, stats)."""
    _lib.bafe_profiling_reset()


def calibrate():
    """Build a calibrated cost model from the current learned model.

    Returns a BafeCostModel that has its per-node weights adjusted based
    on what the learned model discovered about actual runtime correlations.

    If no learned model is available (no refit has happened yet), returns
    the static default cost model.

    The calibrated model is automatically used by bafe_optimize for all
    subsequent extractions.
    """
    return _lib.bafe_cost_model_calibrated_default()


def calibrated_cost_model() -> dict:
    """Inspect the calibrated cost model (for debugging).

    Returns a dict with the calibrated weights:
        alpha_flops, beta_bytes, gamma_intermediate, delta_fuse,
        epsilon_layout_conv, zeta_layout_fuse, eta_contiguous
    """
    cm = _lib.bafe_cost_model_calibrated_default()
    return {
        "alpha_flops": cm.alpha_flops,
        "beta_bytes": cm.beta_bytes,
        "gamma_intermediate": cm.gamma_intermediate,
        "delta_fuse": cm.delta_fuse,
        "epsilon_layout_conv": cm.epsilon_layout_conv,
        "zeta_layout_fuse": cm.zeta_layout_fuse,
        "eta_contiguous": cm.eta_contiguous,
    }


# ---------------------------------------------------------------------------
# Phase 3 (issue #7): Cross-kernel fusion
# ---------------------------------------------------------------------------

class FusedFunction:
    """A fused kernel combining two jitted functions.

    When you call `h = bafe.fuse(f, g)`, BAFE concatenates the two
    optimized graphs (f's output feeds g's first input) and compiles
    a single kernel. This avoids materializing the intermediate tensor.

    The fused kernel takes f's inputs followed by g's inputs[1:].
    """

    def __init__(self, func_a: "JittedFunction", func_b: "JittedFunction"):
        self._func_a = func_a
        self._func_b = func_b
        self._compiled = {}  # key: (shapes, dtypes, layouts) -> compiled tuple
        # for fuse chaining: expose a _fn-like object with the combined arg count
        # n_total = n_a_inputs + n_b_inputs - 1
        class _FnShim:
            def __init__(self, a_fn, b_fn):
                self.__code__ = type("_Code", (), {
                    "co_argcount": a_fn.__code__.co_argcount + b_fn.__code__.co_argcount - 1,
                    "co_varnames": a_fn.__code__.co_varnames[:a_fn.__code__.co_argcount] + \
                                   tuple(b_fn.__code__.co_varnames[1:b_fn.__code__.co_argcount]),
                })()
        self._fn = _FnShim(func_a._fn, func_b._fn)
        functools.update_wrapper(self, func_a)

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        if not args:
            raise TypeError("fused function requires at least one input")

        # split args: first len(a_inputs) go to f, rest go to g[1:]
        # we need to know how many inputs f takes — infer from the first compile
        # Actually, f takes some inputs, g takes some inputs, and f's output
        # feeds g's first input. So total args = n_f_inputs + n_g_inputs - 1.
        # We need to figure out the split.

        # build cache key
        def _layout_of(a):
            if a.ndim >= 2 and a.flags["F_CONTIGUOUS"] and not a.flags["C_CONTIGUOUS"]:
                return "col"
            return "row"
        key = tuple((a.shape, str(a.dtype), _layout_of(a)) for a in args)

        if key not in self._compiled:
            self._compiled[key] = self._compile(args)

        fn_ptr, sig, in_dtypes, out_shape, out_dtype, _opt_graph, _hash, _pred = self._compiled[key]

        # allocate output
        out = np.zeros(out_shape, dtype=out_dtype)

        # build arg list
        c_args = []
        for a, dt in zip(args, in_dtypes):
            if a.dtype == dt:
                arr = a
            else:
                if a.flags["F_CONTIGUOUS"] and not a.flags["C_CONTIGUOUS"]:
                    arr = np.asfortranarray(a, dtype=dt)
                else:
                    arr = np.ascontiguousarray(a, dtype=dt)
            c_args.append(arr.ctypes.data_as(ctypes.c_void_p))
        c_args.append(out.ctypes.data_as(ctypes.c_void_p))

        kernel = sig(fn_ptr)
        kernel(*c_args)
        return out

    def _compile(self, args):
        """Compile the fused function and return the 8-tuple matching
        JittedFunction._compile (for fuse chaining)."""
        # Figure out n_f_inputs by looking at f's signature
        f_code = self._func_a._fn.__code__
        n_f_inputs = f_code.co_argcount
        g_code = self._func_b._fn.__code__
        n_g_inputs = g_code.co_argcount

        if len(args) != n_f_inputs + n_g_inputs - 1:
            raise TypeError(
                f"fused function expects {n_f_inputs + n_g_inputs - 1} args "
                f"({n_f_inputs} for f + {n_g_inputs - 1} for g, since g's "
                f"first input is f's output), got {len(args)}"
            )

        f_args = args[:n_f_inputs]
        g_args = args[n_f_inputs:]

        # compile f to get its optimized graph
        f_result = self._func_a._compile(f_args)
        f_fn_ptr, f_sig, f_in_dtypes, f_out_shape, f_out_dt, f_opt_graph, f_hash, f_pred = f_result

        # For g, compile with a dummy first input matching f's output
        dummy_f_out = np.zeros(f_out_shape, dtype=f_out_dt)
        g_full_args = (dummy_f_out,) + g_args
        g_result = self._func_b._compile(g_full_args)
        g_opt_graph = g_result[5]

        # Concatenate the two optimized graphs via the C API
        fused_graph = BafeGraph()
        err_buf = ctypes.create_string_buffer(256)
        rc = _lib.bafe_fuse_concat(
            ctypes.byref(f_opt_graph),
            ctypes.byref(g_opt_graph),
            ctypes.byref(fused_graph),
            err_buf,
            ctypes.c_size_t(len(err_buf)),
        )
        if rc != 0:
            raise RuntimeError(
                f"bafe_fuse_concat failed (code {rc}): {err_buf.value.decode()}"
            )

        # Optimize + JIT compile the fused graph
        optimized = BafeGraph()
        rc = _lib.bafe_optimize(
            ctypes.byref(fused_graph),
            ctypes.byref(optimized),
            err_buf,
            ctypes.c_size_t(len(err_buf)),
        )
        if rc != 0:
            raise RuntimeError(
                f"bafe_optimize failed for fused graph (code {rc}): {err_buf.value.decode()}"
            )

        fn_ptr = _lib.bafe_jit_get_or_compile(
            ctypes.byref(optimized),
            err_buf,
            ctypes.c_size_t(len(err_buf)),
        )
        if not fn_ptr:
            raise RuntimeError(
                f"JIT compile failed for fused graph: {err_buf.value.decode()}"
            )

        # build the ctypes signature for the fused kernel
        n_total_inputs = n_f_inputs + n_g_inputs - 1
        sig = ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * (n_total_inputs + 1)))

        in_dtypes = [a.dtype for a in args]

        out_node = optimized.nodes[optimized.outputs[0]]
        out_shape = tuple(out_node.shape.dims[i] for i in range(out_node.shape.rank))
        out_dt = _BAFE_TO_NP[out_node.dtype]

        # keep a copy of the optimized graph for autotune feature extraction
        opt_graph_copy = BafeGraph()
        ctypes.memmove(ctypes.byref(opt_graph_copy), ctypes.byref(optimized), ctypes.sizeof(BafeGraph))

        graph_hash_buf = ctypes.create_string_buffer(65)
        _lib.bafe_jit_hash_graph(ctypes.byref(optimized), graph_hash_buf, ctypes.c_size_t(65))
        graph_hash = graph_hash_buf.value.decode("utf-8")

        cm = _lib.bafe_cost_model_calibrated_default()
        predicted = _lib.bafe_cost_graph(ctypes.byref(cm), ctypes.byref(optimized))

        return (fn_ptr, sig, in_dtypes, out_shape, out_dt,
                opt_graph_copy, graph_hash, predicted)


def fuse(func_a, func_b):
    """Fuse two jitted functions into a single kernel.

    When `h = bafe.fuse(f, g)`, calling `h(a, b, c)` is equivalent to
    `g(f(a, b), c)` but compiled as a single kernel — the intermediate
    tensor (f's output) is never materialized.

    Args:
        func_a: a @bafe.jit-decorated function or FusedFunction (producer)
        func_b: a @bafe.jit-decorated function or FusedFunction (consumer;
                its first argument receives f's output)

    Returns:
        A FusedFunction that takes f's inputs + g's inputs[1:].
    """
    # Accept both JittedFunction and FusedFunction (for chaining)
    if not isinstance(func_a, (JittedFunction, FusedFunction)):
        raise TypeError("fuse() requires @bafe.jit-decorated or fused functions")
    if not isinstance(func_b, (JittedFunction, FusedFunction)):
        raise TypeError("fuse() requires @bafe.jit-decorated or fused functions")
    return FusedFunction(func_a, func_b)


__all__ = [
    "Tensor", "jit", "optimize", "make_search_budget", "fuse",
    "input", "matmul", "add", "sub", "mul", "relu", "sigmoid", "tanh",
    "bias_add", "transpose", "reduce_sum", "reduce_max", "reshape", "broadcast_to",
    "graph_summary",
    "configure_autotune", "autotune_stats", "autotune_refit",
    "autotune_model", "autotune_dump_log", "autotune_reset",
    "calibrate", "calibrated_cost_model",
    "pruning_regime_name", "pruning_beam_width", "pruning_iters",
    "__version__",
]


# ---------------------------------------------------------------------------
# Phase 3 (issue #4): Pruning controller helpers
# ---------------------------------------------------------------------------

_PRUNING_REGIMES = {
    0: "greedy",
    1: "light",
    2: "beam",
    3: "deep",
}


def pruning_regime_name(time_budget_ms: int) -> str:
    """Get the regime name for a time budget.

    Returns one of: "greedy" (<=1ms), "light" (<=10ms), "beam" (<=100ms),
    "deep" (>100ms). 0 or None returns "deep" (no limit).
    """
    if time_budget_ms is None or time_budget_ms <= 0:
        return "deep"
    regime = _lib.bafe_pruning_regime_from_budget(int(time_budget_ms))
    return _PRUNING_REGIMES.get(regime, "unknown")


def pruning_beam_width(time_budget_ms: int) -> int:
    """Get the beam width for a time budget's regime."""
    if time_budget_ms is None or time_budget_ms <= 0:
        time_budget_ms = 0
    regime = _lib.bafe_pruning_regime_from_budget(int(time_budget_ms))
    return _lib.bafe_pruning_beam_width_for_regime(regime)


def pruning_iters(time_budget_ms: int) -> int:
    """Get the number of stochastic iterations for a time budget's regime."""
    if time_budget_ms is None or time_budget_ms <= 0:
        time_budget_ms = 0
    regime = _lib.bafe_pruning_regime_from_budget(int(time_budget_ms))
    return _lib.bafe_pruning_iters_for_regime(regime)
