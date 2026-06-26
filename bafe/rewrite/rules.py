"""Rewrite rules for BAFE.

Each rule is a small dataclass with:
  - name         : str
  - applies      : (Node, Graph) -> bool
  - rewrite      : (Node, Graph) -> Optional[Rewrite]

A Rewrite describes a *new way to express* the original node. It says:
  "this node is equivalent to (op_name, attrs, [child_id_0, child_id_1, ...])"
where the child_ids refer to existing nodes in the graph.

The engine does NOT mutate the graph directly. It returns a list of
alternatives which the e-graph layer consumes.

Why this design?
  - rules are pure and easy to test
  - the same rule set can be applied to a Graph or to an EGraph
  - the cost model + extractor decide which alternative wins
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Mapping

from bafe.ir.graph import Graph, Node, NodeId, _FrozenAttrs
from bafe.ir.types import Shape


# ---------------------------------------------------------------------------
# Rule data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rewrite:
    """An alternative expression for a node.

    `children` are existing NodeIds in the graph (already built). If a rule
    needs a NEW intermediate node (e.g. to introduce a transpose), it must
    add it to the graph first and reference it here.

    The e-graph layer treats the original node and this alternative as
    equivalent.
    """
    op_name: str
    attrs: Mapping[str, object]
    children: Tuple[NodeId, ...]


AppliesFn = Callable[[Node, Graph], bool]
RewriteFn = Callable[[Node, Graph], Optional[Rewrite]]


@dataclass(frozen=True)
class Rule:
    name: str
    applies: AppliesFn
    rewrite: RewriteFn

    def __call__(self, node: Node, graph: Graph) -> Optional[Rewrite]:
        if self.applies(node, graph):
            return self.rewrite(node, graph)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_op(name: str) -> AppliesFn:
    return lambda n, g: n.op_name == name


def _is_op_in(*names: str) -> AppliesFn:
    s = set(names)
    return lambda n, g: n.op_name in s


def _child(node: Node, graph: Graph, idx: int) -> Node:
    return graph.nodes[node.children[idx]]


def _make_attrs(**kw) -> Mapping[str, object]:
    return _FrozenAttrs(dict(kw))


# ---------------------------------------------------------------------------
# Algebraic rules
# ---------------------------------------------------------------------------

# add is commutative and associative
RULE_ADD_COMMUTATIVE = Rule(
    name="add_commutative",
    applies=_is_op("add"),
    rewrite=lambda n, g: Rewrite("add", _make_attrs(), (n.children[1], n.children[0])),
)

RULE_ADD_ASSOCIATIVE_LEFT = Rule(
    name="add_assoc_left",
    applies=lambda n, g: (
        n.op_name == "add"
        and _child(n, g, 0).op_name == "add"
    ),
    rewrite=lambda n, g: _assoc_left(n, g, "add"),
)

RULE_ADD_ASSOCIATIVE_RIGHT = Rule(
    name="add_assoc_right",
    applies=lambda n, g: (
        n.op_name == "add"
        and _child(n, g, 1).op_name == "add"
    ),
    rewrite=lambda n, g: _assoc_right(n, g, "add"),
)

RULE_MUL_COMMUTATIVE = Rule(
    name="mul_commutative",
    applies=_is_op("mul"),
    rewrite=lambda n, g: Rewrite("mul", _make_attrs(), (n.children[1], n.children[0])),
)

RULE_MUL_ASSOCIATIVE_LEFT = Rule(
    name="mul_assoc_left",
    applies=lambda n, g: (
        n.op_name == "mul"
        and _child(n, g, 0).op_name == "mul"
    ),
    rewrite=lambda n, g: _assoc_left(n, g, "mul"),
)

# sub is not associative, but sub(a, sub(b, c)) == add(sub(a, b), c)
RULE_SUB_SUB_TO_SUB_ADD = Rule(
    name="sub_sub_to_sub_add",
    applies=lambda n, g: (
        n.op_name == "sub"
        and _child(n, g, 1).op_name == "sub"
    ),
    rewrite=lambda n, g: _sub_sub_to_sub_add(n, g),
)

# Identity: add(x, 0) == x; mul(x, 1) == x — requires constant folding first
# we skip these for Phase 1; they are handled by constant folding in engine.

# Distributivity: mul(add(a, b), c) -> add(mul(a, c), mul(b, c))
# (and its dual)
RULE_DISTRIBUTE_MUL_OVER_ADD_LEFT = Rule(
    name="distribute_mul_over_add_left",
    applies=lambda n, g: (
        n.op_name == "mul"
        and _child(n, g, 0).op_name == "add"
    ),
    rewrite=lambda n, g: _distribute_mul_left(n, g),
)

RULE_DISTRIBUTE_MUL_OVER_ADD_RIGHT = Rule(
    name="distribute_mul_over_add_right",
    applies=lambda n, g: (
        n.op_name == "mul"
        and _child(n, g, 1).op_name == "add"
    ),
    rewrite=lambda n, g: _distribute_mul_right(n, g),
)


# Double negation: neg(neg(x)) == x
RULE_DOUBLE_NEG = Rule(
    name="double_neg",
    applies=lambda n, g: (
        n.op_name == "neg"
        and _child(n, g, 0).op_name == "neg"
    ),
    rewrite=lambda n, g: Rewrite("identity", _make_attrs(), (_child(_child(n, g, 0), g, 0).id,)),
)
# NOTE: 'identity' is not yet an op. We use a passthrough: we just return
# the grandchild as the equivalent. The e-graph treats this as:
#   neg(neg(x)) === x
# by unioning the original node's class with x's class. We'll express this
# in the engine instead. Skip this rule for Phase 1.


# ---------------------------------------------------------------------------
# Transpose + matmul interaction
# ---------------------------------------------------------------------------

# matmul(transpose(A, perm), B) -> matmul(A', B) where A' permutes back.
# Specifically: if A has shape (M,K) and transpose swaps to (K,M), then
# matmul(transpose(A), B) computes K-side correctly as long as B is (K,N).
# This is semantically the same as: matmul(A_transposed, B) - no rewrite
# needed. Instead, we expose:
#   matmul(A, B) == transpose(matmul(transpose(B, (1,0)), transpose(A, (1,0))), (1,0))
# This identity is useful when the consumer expects a transposed layout.
# For Phase 1 we skip this (it requires shape reasoning).

# Simpler: transpose(transpose(x, p1), p2) == x if p2 is the inverse perm.
RULE_TRANSPOSE_TRANSPOSE = Rule(
    name="transpose_transpose",
    applies=lambda n, g: (
        n.op_name == "transpose"
        and _child(n, g, 0).op_name == "transpose"
    ),
    rewrite=lambda n, g: _transpose_transpose(n, g),
)


# ---------------------------------------------------------------------------
# Fusion rules (the heart of the system)
# ---------------------------------------------------------------------------

# relu(matmul(A, B)) -> fused_matmul_relu(A, B)
RULE_FUSE_MATMUL_RELU = Rule(
    name="fuse_matmul_relu",
    applies=lambda n, g: (
        n.op_name == "relu"
        and _child(n, g, 0).op_name == "matmul"
    ),
    rewrite=lambda n, g: Rewrite(
        "fused_matmul_relu",
        _make_attrs(),
        _child(n, g, 0).children,
    ),
)

# relu(bias_add(X, bias)) -> fused_bias_relu(X, bias)
RULE_FUSE_BIAS_RELU = Rule(
    name="fuse_bias_relu",
    applies=lambda n, g: (
        n.op_name == "relu"
        and _child(n, g, 0).op_name == "bias_add"
    ),
    rewrite=lambda n, g: Rewrite(
        "fused_bias_relu",
        _make_attrs(),
        _child(n, g, 0).children,
    ),
)

# relu(add(matmul(A, B), bias_vector)) -> fused_matmul_bias_relu(A, B, bias)
# Note: requires that the added bias is a 1-D vector that broadcasts on last dim.
RULE_FUSE_MATMUL_BIAS_RELU = Rule(
    name="fuse_matmul_bias_relu",
    applies=lambda n, g: (
        n.op_name == "relu"
        and _child(n, g, 0).op_name == "add"
        and _child(_child(n, g, 0), g, 0).op_name == "matmul"
        and _child(_child(n, g, 0), g, 1).shape.rank == 1
    ),
    rewrite=lambda n, g: _fuse_matmul_bias_relu(n, g),
)

# add(matmul(A, B), bias_vector) -> fused_matmul_bias(A, B, bias)
RULE_FUSE_MATMUL_BIAS = Rule(
    name="fuse_matmul_bias",
    applies=lambda n, g: (
        n.op_name == "add"
        and _child(n, g, 0).op_name == "matmul"
        and _child(n, g, 1).shape.rank == 1
    ),
    rewrite=lambda n, g: Rewrite(
        "fused_matmul_bias",
        _make_attrs(),
        (_child(n, g, 0).children[0], _child(n, g, 0).children[1], n.children[1]),
    ),
)


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

def _assoc_left(n: Node, g: Graph, op: str) -> Optional[Rewrite]:
    """(a OP b) OP c -> a OP (b OP c)"""
    left = _child(n, g, 0)
    right = _child(n, g, 1)
    a = left.children[0]
    b = left.children[1]
    c = right.id
    # need a new node for (b OP c)
    new_inner = g.add(op, b, c)
    return Rewrite(op, _make_attrs(), (a, new_inner))


def _assoc_right(n: Node, g: Graph, op: str) -> Optional[Rewrite]:
    """a OP (b OP c) -> (a OP b) OP c"""
    a = n.children[0]
    right = _child(n, g, 1)
    b = right.children[0]
    c = right.children[1]
    new_inner = g.add(op, a, b)
    return Rewrite(op, _make_attrs(), (new_inner, c))


def _sub_sub_to_sub_add(n: Node, g: Graph) -> Optional[Rewrite]:
    """a - (b - c) -> (a - b) + c"""
    a = n.children[0]
    right = _child(n, g, 1)
    b = right.children[0]
    c = right.children[1]
    new_sub = g.add("sub", a, b)
    return Rewrite("add", _make_attrs(), (new_sub, c))


def _distribute_mul_left(n: Node, g: Graph) -> Optional[Rewrite]:
    """mul(add(a, b), c) -> add(mul(a, c), mul(b, c))"""
    left = _child(n, g, 0)
    c = n.children[1]
    a = left.children[0]
    b = left.children[1]
    new_mul1 = g.add("mul", a, c)
    new_mul2 = g.add("mul", b, c)
    return Rewrite("add", _make_attrs(), (new_mul1, new_mul2))


def _distribute_mul_right(n: Node, g: Graph) -> Optional[Rewrite]:
    """mul(c, add(a, b)) -> add(mul(c, a), mul(c, b))"""
    right = _child(n, g, 1)
    c = n.children[0]
    a = right.children[0]
    b = right.children[1]
    new_mul1 = g.add("mul", c, a)
    new_mul2 = g.add("mul", c, b)
    return Rewrite("add", _make_attrs(), (new_mul1, new_mul2))


def _transpose_transpose(n: Node, g: Graph) -> Optional[Rewrite]:
    """transpose(transpose(x, p1), p2) -> x if p2 = inverse(p1)"""
    inner = _child(n, g, 0)
    p1 = tuple(inner.attrs["perm"])
    p2 = tuple(n.attrs["perm"])
    # inverse: if p1[p2[i]] == i for all i, then p2 is inverse of p1
    inv = all(p1[p2[i]] == i for i in range(len(p1)))
    if not inv:
        # otherwise compute the composed permutation
        composed = tuple(p1[p2[i]] for i in range(len(p1)))
        return Rewrite("transpose", _make_attrs(perm=composed), (inner.children[0],))
    # equivalently: the result is just the inner input
    # Express as: identity(inner.children[0]). Since we have no identity op,
    # we return a passthrough via reshape with same shape.
    inner_in = inner.children[0]
    src_shape = g.nodes[inner_in].shape
    # use a no-op reshape to express identity
    return Rewrite("reshape", _make_attrs(shape=tuple(src_shape.dims)), (inner_in,))


def _fuse_matmul_bias_relu(n: Node, g: Graph) -> Optional[Rewrite]:
    add_node = _child(n, g, 0)
    mm_node = _child(add_node, g, 0)
    bias_id = add_node.children[1]
    a = mm_node.children[0]
    b = mm_node.children[1]
    return Rewrite("fused_matmul_bias_relu", _make_attrs(), (a, b, bias_id))


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------

DEFAULT_RULES: Tuple[Rule, ...] = (
    # algebraic
    RULE_ADD_COMMUTATIVE,
    RULE_ADD_ASSOCIATIVE_LEFT,
    RULE_ADD_ASSOCIATIVE_RIGHT,
    RULE_MUL_COMMUTATIVE,
    RULE_MUL_ASSOCIATIVE_LEFT,
    RULE_SUB_SUB_TO_SUB_ADD,
    RULE_DISTRIBUTE_MUL_OVER_ADD_LEFT,
    RULE_DISTRIBUTE_MUL_OVER_ADD_RIGHT,
    RULE_TRANSPOSE_TRANSPOSE,
    # fusion
    RULE_FUSE_MATMUL_RELU,
    RULE_FUSE_BIAS_RELU,
    RULE_FUSE_MATMUL_BIAS,
    RULE_FUSE_MATMUL_BIAS_RELU,
)


def all_rules() -> Tuple[Rule, ...]:
    return DEFAULT_RULES
