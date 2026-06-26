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
}

_BAFE_TO_NP = {v: k for k, v in _NP_TO_BAFE.items()}


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
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul requires rank-2 tensors")
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"matmul shape mismatch: {self.shape} @ {other.shape}"
            )
        out_shape = (self.shape[0], other.shape[1])
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
    """

    def __init__(self, fn: Callable):
        self._fn = fn
        self._compiled = {}  # key: (shapes, dtypes) -> (fn_ptr, sig, out_shape)
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

        fn_ptr, sig, in_dtypes, out_shape, out_dtype = self._compiled[key]

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
        kernel(*c_args)
        return out

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
        optimized = BafeGraph()
        err_buf = ctypes.create_string_buffer(256)
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

        return (fn_ptr, sig, in_dtypes, out_shape, out_np_dt)


def jit(fn: Callable) -> JittedFunction:
    """Decorator: trace + optimize + JIT-compile a tensor function."""
    return JittedFunction(fn)


# expose the optimize function for low-level use
def optimize(graph: BafeGraph) -> BafeGraph:
    """Run the BAFE optimization pipeline on a graph."""
    out = BafeGraph()
    err = ctypes.create_string_buffer(256)
    rc = _lib.bafe_optimize(byref(graph), byref(out), err, c_size_t(len(err)))
    if rc != 0:
        raise RuntimeError(f"optimize failed: {err.value.decode()}")
    return out


__all__ = [
    "Tensor", "jit", "optimize",
    "input", "matmul", "add", "sub", "mul", "relu", "sigmoid", "tanh",
    "bias_add", "transpose", "reduce_sum", "reduce_max", "reshape", "broadcast_to",
    "graph_summary",
    "__version__",
]
