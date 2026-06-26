"""Op definitions for BAFE IR.

Each op has:
  - a name (str, used in IR and C codegen)
  - an arity (number of tensor inputs)
  - extra attributes (axes, scale factor, etc.) encoded as a frozen dict
  - a shape-inference function

Ops are pure value objects: two Op instances with the same name and attrs
are equal and hash-equal. This matters for the e-graph, which hashes
ENodes (op + child eclass ids).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple, Callable, Any

from bafe.ir.types import Shape, Dtype, broadcast_shapes, reduce_shape


# ---------------------------------------------------------------------------
# Op registry
# ---------------------------------------------------------------------------

ShapeFn = Callable[[Tuple[Shape, ...], Mapping[str, Any]], Shape]


@dataclass(frozen=True)
class Op:
    """An op definition (template), not an op instance."""
    name: str
    arity: int
    attrs_schema: Tuple[str, ...] = field(default_factory=tuple)
    shape_fn: ShapeFn = None
    has_fusion_form: bool = False
    c_name: str = ""

    def __post_init__(self):
        if not self.c_name:
            object.__setattr__(self, "c_name", f"bafe_{self.name}")

    def infer_shape(self, shapes: Tuple[Shape, ...], attrs: Mapping[str, Any]) -> Shape:
        if self.shape_fn is None:
            raise NotImplementedError(f"op {self.name} has no shape_fn")
        return self.shape_fn(shapes, attrs)


_REGISTRY: dict[str, Op] = {}


def register(
    name: str,
    arity: int,
    *,
    attrs_schema: Tuple[str, ...] = (),
    has_fusion_form: bool = False,
    c_name: str = "",
) -> Callable[[ShapeFn], ShapeFn]:
    def deco(fn: ShapeFn) -> ShapeFn:
        op = Op(
            name=name,
            arity=arity,
            attrs_schema=tuple(attrs_schema),
            shape_fn=fn,
            has_fusion_form=has_fusion_form,
            c_name=c_name or f"bafe_{name}",
        )
        if name in _REGISTRY:
            raise ValueError(f"op {name!r} already registered")
        _REGISTRY[name] = op
        return fn
    return deco


def get_op(name: str) -> Op:
    if name not in _REGISTRY:
        raise KeyError(f"unknown op {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_ops() -> Tuple[Op, ...]:
    return tuple(_REGISTRY.values())


def is_fused(op_name: str) -> bool:
    return op_name.startswith("fused_")


# ---------------------------------------------------------------------------
# Shape inference functions
# ---------------------------------------------------------------------------

@register("matmul", 2, c_name="bafe_matmul")
def _shape_matmul(shapes: Tuple[Shape, ...], attrs: Mapping) -> Shape:
    a, b = shapes
    if a.rank < 2 or b.rank < 2:
        raise ValueError(f"matmul needs rank>=2 inputs, got {a}, {b}")
    m, k1 = a.dims[-2], a.dims[-1]
    k2, n = b.dims[-2], b.dims[-1]
    if k1 != k2:
        raise ValueError(f"matmul K mismatch: {a} x {b}")
    if a.rank == 2 and b.rank == 2:
        return Shape.of(m, n)
    ba = a.dims[:-2]
    bb = b.dims[:-2]
    bbroad = broadcast_shapes(Shape(dims=ba), Shape(dims=bb))
    return Shape(dims=bbroad.dims + (m, n))


def _shape_binary_broadcast(shapes: Tuple[Shape, ...], attrs: Mapping) -> Shape:
    return broadcast_shapes(shapes[0], shapes[1])


register("add", 2, c_name="bafe_add")(_shape_binary_broadcast)
register("mul", 2, c_name="bafe_mul")(_shape_binary_broadcast)
register("sub", 2, c_name="bafe_sub")(_shape_binary_broadcast)


@register("scale", 2, c_name="bafe_scale")
def _shape_scale(shapes, attrs):
    t, s = shapes
    if not s.is_scalar:
        raise ValueError(f"scale second arg must be scalar, got {s}")
    return t


@register("bias_add", 2, c_name="bafe_bias_add")
def _shape_bias_add(shapes, attrs):
    t, b = shapes
    if t.rank == 0:
        raise ValueError("bias_add needs rank>=1 input")
    if b.rank != 1:
        raise ValueError(f"bias must be rank-1, got {b}")
    if t.dims[-1] != b.dims[0]:
        raise ValueError(f"bias_add last-dim mismatch: {t} vs {b}")
    return t


def _shape_unary_passthrough(shapes, attrs):
    return shapes[0]


register("relu", 1, c_name="bafe_relu")(_shape_unary_passthrough)
register("sigmoid", 1, c_name="bafe_sigmoid")(_shape_unary_passthrough)
register("tanh", 1, c_name="bafe_tanh")(_shape_unary_passthrough)
register("neg", 1, c_name="bafe_neg")(_shape_unary_passthrough)


@register("transpose", 1, attrs_schema=("perm",), c_name="bafe_transpose")
def _shape_transpose(shapes, attrs):
    s = shapes[0]
    perm = attrs["perm"]
    if len(perm) != s.rank:
        raise ValueError(f"transpose perm length {len(perm)} != rank {s.rank}")
    if sorted(perm) != list(range(s.rank)):
        raise ValueError(f"transpose perm {perm} not a permutation of range({s.rank})")
    return Shape(dims=tuple(s.dims[i] for i in perm))


def _shape_reduce(shapes, attrs):
    s = shapes[0]
    axes = attrs.get("axes", tuple(range(s.rank)))
    keepdims = attrs.get("keepdims", False)
    return reduce_shape(s, axes, keepdims=keepdims)


register("reduce_sum", 1, attrs_schema=("axes", "keepdims"), c_name="bafe_reduce_sum")(_shape_reduce)
register("reduce_max", 1, attrs_schema=("axes", "keepdims"), c_name="bafe_reduce_max")(_shape_reduce)


@register("reshape", 1, attrs_schema=("shape",), c_name="bafe_reshape")
def _shape_reshape(shapes, attrs):
    in_shape = shapes[0]
    target = Shape(dims=tuple(attrs["shape"]))
    in_numel = in_shape.numel
    if -1 in target.dims:
        known = 1
        for d in target.dims:
            if d != -1:
                known *= d
        if known == 0 or in_numel % known != 0:
            raise ValueError(f"reshape {in_shape} -> {target} with -1 invalid")
        inferred = in_numel // known
        target = Shape(dims=tuple(inferred if d == -1 else d for d in target.dims))
    if target.numel != in_numel:
        raise ValueError(f"reshape numel mismatch: {in_shape} ({in_numel}) -> {target} ({target.numel})")
    return target


@register("broadcast_to", 1, attrs_schema=("shape",), c_name="bafe_broadcast_to")
def _shape_broadcast_to(shapes, attrs):
    src = shapes[0]
    target = Shape(dims=tuple(attrs["shape"]))
    if src.rank > target.rank:
        raise ValueError(f"broadcast_to: src rank {src.rank} > target {target.rank}")
    pad = (1,) * (target.rank - src.rank) + src.dims
    for s, t in zip(pad, target.dims):
        if s != 1 and s != t:
            raise ValueError(f"broadcast_to: {src} cannot broadcast to {target}")
    return target


# ---------------------------------------------------------------------------
# Fused ops (synthesized by the rewrite engine)
# ---------------------------------------------------------------------------

register("fused_matmul_relu", 2, has_fusion_form=True, c_name="bafe_fused_matmul_relu")(_shape_matmul)


@register("fused_matmul_bias", 3, has_fusion_form=True, c_name="bafe_fused_matmul_bias")
def _shape_fused_matmul_bias(shapes, attrs):
    mm = _shape_matmul((shapes[0], shapes[1]), attrs)
    b = shapes[2]
    if b.rank != 1 or b.dims[0] != mm.dims[-1]:
        raise ValueError(f"fused_matmul_bias bias mismatch: mm={mm}, bias={b}")
    return mm


register("fused_matmul_bias_relu", 3, has_fusion_form=True, c_name="bafe_fused_matmul_bias_relu")(_shape_fused_matmul_bias)


@register("fused_bias_relu", 2, has_fusion_form=True, c_name="bafe_fused_bias_relu")
def _shape_fused_bias_relu(shapes, attrs):
    return _shape_bias_add(shapes, attrs)


DEFAULT_DTYPE = Dtype.F32
