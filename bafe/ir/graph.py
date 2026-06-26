"""IR Graph: nodes, edges, and traversal helpers.

A Graph is a DAG of Nodes. Each Node has:
  - id          : a stable int identifier (unique within the graph)
  - op_name     : str, registered op name
  - attrs       : frozen mapping of op attributes (axes, perm, shape, ...)
  - children    : tuple of child Node ids (the inputs)
  - shape       : inferred Shape (cached after construction)
  - dtype       : Dtype

Graphs are *not* hashable or immutable — they're working data structures
used during construction. The e-graph (see bafe.egraph) provides the
immutable, hashable form used during search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Mapping, Optional, Tuple, Dict

from bafe.ir.ops import get_op, is_fused
from bafe.ir.types import Shape, Dtype


# Node ids are ints, allocated monotonically within a graph.
NodeId = int


@dataclass
class Node:
    """A single IR node."""
    id: NodeId
    op_name: str
    attrs: Mapping[str, object]      # frozen dict-like; we use a Mapping proxy
    children: Tuple[NodeId, ...]
    shape: Shape
    dtype: Dtype

    @property
    def arity(self) -> int:
        return len(self.children)

    @property
    def is_fused(self) -> bool:
        return is_fused(self.op_name)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return isinstance(other, Node) and self.id == other.id


def _freeze_attrs(attrs: Optional[Mapping[str, object]]) -> Mapping[str, object]:
    """Convert a mapping into a hashable, comparable form."""
    if attrs is None:
        return _EMPTY_ATTRS
    # ensure values are hashable (tuples, ints, floats, strs)
    out = {}
    for k, v in attrs.items():
        if isinstance(v, list):
            v = tuple(v)
        out[k] = v
    return frozenset(out.items()) if False else _FrozenAttrs(out)


@dataclass(frozen=True)
class _FrozenAttrs:
    """A hashable wrapper around a dict of attributes."""
    _d: Dict[str, object]

    def __getitem__(self, k):
        return self._d[k]

    def get(self, k, default=None):
        return self._d.get(k, default)

    def __contains__(self, k):
        return k in self._d

    def __iter__(self):
        return iter(self._d)

    def items(self):
        return self._d.items()

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def __len__(self):
        return len(self._d)

    def __hash__(self):
        return hash(frozenset(self._d.items()))

    def __eq__(self, other):
        if isinstance(other, _FrozenAttrs):
            return self._d == other._d
        if isinstance(other, Mapping):
            return dict(self._d) == dict(other)
        return NotImplemented


_EMPTY_ATTRS = _FrozenAttrs({})


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

@dataclass
class Graph:
    """A DAG of IR nodes.

    Construction:
        g = Graph()
        a = g.input("A", Shape.of(64, 64), Dtype.F32)
        b = g.input("B", Shape.of(64, 64), Dtype.F32)
        c = g.matmul(a, b)
        g.set_output(c)

    The graph stores:
      - nodes: dict[NodeId, Node]
      - inputs: list of (name, NodeId)
      - outputs: list of NodeId (in order)
    """
    nodes: Dict[NodeId, Node] = field(default_factory=dict)
    inputs: List[Tuple[str, NodeId]] = field(default_factory=list)
    outputs: List[NodeId] = field(default_factory=list)
    _next_id: NodeId = 0
    _input_names: Dict[str, NodeId] = field(default_factory=dict)

    # ----- construction -------------------------------------------------

    def _new_id(self) -> NodeId:
        i = self._next_id
        self._next_id += 1
        return i

    def input(self, name: str, shape: Shape, dtype: Dtype = Dtype.F32) -> NodeId:
        if name in self._input_names:
            raise ValueError(f"duplicate input name {name!r}")
        nid = self._new_id()
        node = Node(
            id=nid,
            op_name="input",
            attrs=_FrozenAttrs({"name": name}),
            children=(),
            shape=shape,
            dtype=dtype,
        )
        self.nodes[nid] = node
        self.inputs.append((name, nid))
        self._input_names[name] = nid
        return nid

    def constant(self, value, shape: Shape, dtype: Dtype = Dtype.F32) -> NodeId:
        """Add a constant scalar/vector node.

        The value is stored as-is in attrs['value']; the C backend emits it
        as a static initializer.
        """
        nid = self._new_id()
        node = Node(
            id=nid,
            op_name="constant",
            attrs=_FrozenAttrs({"value": value, "dtype": dtype.value}),
            children=(),
            shape=shape,
            dtype=dtype,
        )
        self.nodes[nid] = node
        return nid

    def add(self, op_name: str, *children: NodeId, attrs: Optional[Mapping] = None, dtype: Optional[Dtype] = None) -> NodeId:
        """Add an op node.

        Infers shape and dtype from children. Validates arity.
        """
        if op_name not in ("input", "constant"):
            op = get_op(op_name)
            if len(children) != op.arity:
                raise ValueError(
                    f"op {op_name!r} arity {op.arity} but got {len(children)} children"
                )
        # validate children exist
        for c in children:
            if c not in self.nodes:
                raise KeyError(f"unknown child node id {c}")

        # infer shape
        if op_name == "input":
            raise ValueError("use Graph.input() to add inputs")
        if op_name == "constant":
            raise ValueError("use Graph.constant() to add constants")

        op = get_op(op_name)
        child_shapes = tuple(self.nodes[c].shape for c in children)
        shape = op.infer_shape(child_shapes, _freeze_attrs(attrs))

        # dtype: take from first child for now (F32 default)
        if dtype is None:
            dtype = self.nodes[children[0]].dtype if children else Dtype.F32

        nid = self._new_id()
        node = Node(
            id=nid,
            op_name=op_name,
            attrs=_freeze_attrs(attrs),
            children=tuple(children),
            shape=shape,
            dtype=dtype,
        )
        self.nodes[nid] = node
        return nid

    # Convenience wrappers
    def matmul(self, a: NodeId, b: NodeId) -> NodeId:
        return self.add("matmul", a, b)

    def add_op(self, a: NodeId, b: NodeId) -> NodeId:
        return self.add("add", a, b)

    def mul(self, a: NodeId, b: NodeId) -> NodeId:
        return self.add("mul", a, b)

    def sub(self, a: NodeId, b: NodeId) -> NodeId:
        return self.add("sub", a, b)

    def relu(self, x: NodeId) -> NodeId:
        return self.add("relu", x)

    def sigmoid(self, x: NodeId) -> NodeId:
        return self.add("sigmoid", x)

    def tanh(self, x: NodeId) -> NodeId:
        return self.add("tanh", x)

    def transpose(self, x: NodeId, perm: Tuple[int, ...]) -> NodeId:
        return self.add("transpose", x, attrs={"perm": tuple(perm)})

    def reduce_sum(self, x: NodeId, axes: Tuple[int, ...], keepdims: bool = False) -> NodeId:
        return self.add("reduce_sum", x, attrs={"axes": tuple(axes), "keepdims": bool(keepdims)})

    def reduce_max(self, x: NodeId, axes: Tuple[int, ...], keepdims: bool = False) -> NodeId:
        return self.add("reduce_max", x, attrs={"axes": tuple(axes), "keepdims": bool(keepdims)})

    def reshape(self, x: NodeId, shape: Tuple[int, ...]) -> NodeId:
        return self.add("reshape", x, attrs={"shape": tuple(shape)})

    def broadcast_to(self, x: NodeId, shape: Tuple[int, ...]) -> NodeId:
        return self.add("broadcast_to", x, attrs={"shape": tuple(shape)})

    def bias_add(self, x: NodeId, bias: NodeId) -> NodeId:
        return self.add("bias_add", x, bias)

    def scale(self, x: NodeId, scalar: NodeId) -> NodeId:
        return self.add("scale", x, scalar)

    def set_output(self, nid: NodeId) -> None:
        if nid not in self.nodes:
            raise KeyError(f"unknown output node id {nid}")
        self.outputs.append(nid)

    # ----- traversal ----------------------------------------------------

    def topo_order(self) -> List[NodeId]:
        """Return node ids in topological order (inputs first)."""
        visited = set()
        order: List[NodeId] = []
        # iterative DFS
        stack: List[Tuple[NodeId, bool]] = [(nid, False) for nid in self.nodes]
        # we want to visit in id order for determinism
        stack = []
        for nid in sorted(self.nodes):
            if nid not in visited:
                stack.append((nid, False))
                while stack:
                    cur, expanded = stack.pop()
                    if cur in visited:
                        continue
                    if expanded:
                        visited.add(cur)
                        order.append(cur)
                        continue
                    stack.append((cur, True))
                    for c in self.nodes[cur].children:
                        if c not in visited:
                            stack.append((c, False))
        return order

    def reverse_topo(self) -> List[NodeId]:
        return list(reversed(self.topo_order()))

    def children_of(self, nid: NodeId) -> Tuple[NodeId, ...]:
        return self.nodes[nid].children

    def parents_of(self, nid: NodeId) -> List[NodeId]:
        return [p for p, n in self.nodes.items() if nid in n.children]

    def reachable_from(self, roots: List[NodeId]) -> List[NodeId]:
        """All nodes reachable from the given roots (children-first)."""
        seen = set()
        stack = list(roots)
        out = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            for c in self.nodes[cur].children:
                if c not in seen:
                    stack.append(c)
        return out

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, nid: NodeId) -> bool:
        return nid in self.nodes

    def __getitem__(self, nid: NodeId) -> Node:
        return self.nodes[nid]

    def __iter__(self) -> Iterator[NodeId]:
        return iter(self.nodes)

    # ----- debug --------------------------------------------------------

    def to_dot(self) -> str:
        lines = ["digraph G {"]
        for nid, node in self.nodes.items():
            label = f"{node.op_name}"
            if node.op_name == "input":
                label = f"input\\n{node.attrs['name']}"
            lines.append(f'  n{nid} [label="{label}\\n{node.shape}"];')
        for nid, node in self.nodes.items():
            for c in node.children:
                lines.append(f"  n{c} -> n{nid};")
        lines.append("}")
        return "\n".join(lines)

    def summary(self) -> str:
        n_input = sum(1 for _, n in self.nodes.items() if n.op_name == "input")
        n_const = sum(1 for _, n in self.nodes.items() if n.op_name == "constant")
        n_op = len(self.nodes) - n_input - n_const
        return (
            f"Graph(nodes={len(self.nodes)}, inputs={n_input}, "
            f"constants={n_const}, ops={n_op}, outputs={len(self.outputs)})"
        )
