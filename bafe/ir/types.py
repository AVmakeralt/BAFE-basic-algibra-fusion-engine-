"""Type system for BAFE IR: dtypes, shapes, layouts.

These are immutable, hashable value objects. They are used throughout the
IR, the rewrite engine, the e-graph, the cost model, and the backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Iterable


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------

class Dtype(Enum):
    """Element types supported by the IR and the C backend.

    The .c_name attribute gives the matching C99 type. The .numpy_name
    attribute gives the matching numpy dtype string.
    """
    F32 = "f32"
    F64 = "f64"
    I32 = "i32"
    I64 = "i64"

    @property
    def c_name(self) -> str:
        return _DTYPE_C[self]

    @property
    def numpy_name(self) -> str:
        return _DTYPE_NP[self]

    @property
    def byte_size(self) -> int:
        """Bytes per element."""
        return _DTYPE_BYTES[self]


_DTYPE_C = {
    Dtype.F32: "float",
    Dtype.F64: "double",
    Dtype.I32: "int32_t",
    Dtype.I64: "int64_t",
}

_DTYPE_NP = {
    Dtype.F32: "float32",
    Dtype.F64: "float64",
    Dtype.I32: "int32",
    Dtype.I64: "int64",
}

_DTYPE_BYTES = {
    Dtype.F32: 4,
    Dtype.F64: 8,
    Dtype.I32: 4,
    Dtype.I64: 8,
}


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """A tensor shape: an immutable tuple of non-negative ints (0 = scalar).

    A rank-0 shape () represents a scalar. A dimension of 0 means "unknown /
    symbolic" — used only by the symbolic frontend during shape inference.
    Once inferred, all dims are concrete non-negative integers.
    """
    dims: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.dims, tuple):
            object.__setattr__(self, "dims", tuple(self.dims))
        for d in self.dims:
            if not isinstance(d, int) or isinstance(d, bool):
                raise TypeError(f"shape dims must be ints, got {d!r}")
            if d < 0:
                raise ValueError(f"shape dim must be >= 0, got {d}")

    @classmethod
    def of(cls, *dims: int) -> "Shape":
        return cls(dims=tuple(int(d) for d in dims))

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def is_scalar(self) -> bool:
        return self.rank == 0

    @property
    def is_empty(self) -> bool:
        return any(d == 0 for d in self.dims)

    @property
    def numel(self) -> int:
        """Number of elements. Returns 1 for scalar, 0 if any dim is 0."""
        if self.is_scalar:
            return 1
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self, dtype: Dtype = Dtype.F32) -> int:
        return self.numel * dtype.byte_size

    def nbytes_for(self, dtype: Dtype) -> int:
        return self.numel * dtype.byte_size

    def __iter__(self):
        return iter(self.dims)

    def __len__(self):
        return self.rank

    def __getitem__(self, i):
        return self.dims[i]

    def __mul__(self, other: "Shape") -> "Shape":
        """Concatenate shapes (used for batching)."""
        return Shape(dims=self.dims + other.dims)

    def __str__(self) -> str:
        return "(" + ",".join(str(d) for d in self.dims) + ("," if self.rank == 1 else "") + ")"

    def __repr__(self) -> str:
        return f"Shape{self.dims!r}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

class Layout(Enum):
    """Logical memory layout for a tensor.

    In Phase 1 we only support two physical layouts: row-major (C order) and
    column-major (Fortran order). The 'Blocked' and 'TensorCore' layouts are
    reserved for the Phase 2 layout superoptimizer and are not yet codegen-
    ready.

    Why include them now? Because the cost model and the rewrite engine
    already need to *reason about* layout as a first-class variable, even
    before the backend can emit code for all variants.
    """
    ROW_MAJOR = "row"     # C order, last dim contiguous
    COL_MAJOR = "col"     # Fortran order, first dim contiguous
    BLOCKED = "blocked"   # tile-major (Phase 2)
    TENSOR_CORE = "tc"    # tensor-core-friendly (Phase 2, GPU only)

    @property
    def c_order(self) -> bool:
        """True if numpy/strides match C order."""
        return self == Layout.ROW_MAJOR


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def broadcast_shapes(a: Shape, b: Shape) -> Shape:
    """Numpy-style broadcasting of two shapes.

    Returns the broadcast shape, or raises ValueError if incompatible.
    """
    if a.is_scalar:
        return b
    if b.is_scalar:
        return a
    n = max(a.rank, b.rank)
    a_pad = (1,) * (n - a.rank) + a.dims
    b_pad = (1,) * (n - b.rank) + b.dims
    out = []
    for da, db in zip(a_pad, b_pad):
        if da == 1:
            out.append(db)
        elif db == 1 or db == da:
            out.append(da)
        else:
            raise ValueError(
                f"cannot broadcast shapes {a.dims} and {b.dims}: "
                f"dim mismatch {da} vs {db}"
            )
    return Shape(dims=tuple(out))


def reduce_shape(s: Shape, axes: Iterable[int], keepdims: bool = False) -> Shape:
    """Compute the output shape of a reduction over `axes`."""
    axes_set = {ax % s.rank for ax in axes}
    if not axes_set:
        return s
    out = []
    for i, d in enumerate(s.dims):
        if i in axes_set:
            if keepdims:
                out.append(1)
        else:
            out.append(d)
    return Shape(dims=tuple(out))
