"""E-graph implementation for BAFE.

Terminology (egg-compatible):
  - ENode  : (op_name, attrs, child_eclass_ids)  -- hashable
  - EClass : a set of ENodes that are all equivalent
  - Union-find over EClass ids (with path compression + union by rank)
  - Congruence table: maps a canonical ENode -> EClass id, used to detect
    that two ENodes are structurally identical and should be merged

Phase 1 features:
  - add node / add alternative
  - union
  - rebuild (congruence closure saturation)
  - find (canonical id)

Phase 2+ features (deferred):
  - cost-based extraction pruning
  - e-class analysis (lattice of facts)
  - controlled iteration with budget
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Mapping, Iterable
import sys

from bafe.ir.graph import Graph, Node, NodeId, _FrozenAttrs
from bafe.ir.types import Shape, Dtype
from bafe.ir.ops import get_op


# ---------------------------------------------------------------------------
# ENode and EClassId
# ---------------------------------------------------------------------------

EClassId = int


@dataclass(frozen=True)
class ENode:
    """An e-node: op + attrs + child eclass ids.

    ENodes are hashable and compared by structural equality. Two ENodes
    with the same op, same attrs, and same child ids are *the same* e-node.
    """
    op_name: str
    attrs: Mapping[str, object]
    children: Tuple[EClassId, ...]

    def canonicalize(self, find_fn) -> "ENode":
        """Return a new ENode whose children are replaced by their canonical ids."""
        return ENode(
            op_name=self.op_name,
            attrs=self.attrs,
            children=tuple(find_fn(c) for c in self.children),
        )

    def __str__(self) -> str:
        ch = ",".join(str(c) for c in self.children)
        a = ""
        if self.attrs and len(self.attrs) > 0:
            a = " " + " ".join(f"{k}={v}" for k, v in self.attrs.items())
        return f"({self.op_name}{a} {ch})"


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------

class UnionFind:
    """Union-find with path compression and union by rank."""

    def __init__(self):
        self._parent: Dict[EClassId, EClassId] = {}
        self._rank: Dict[EClassId, int] = {}

    def make(self, x: EClassId) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: EClassId) -> EClassId:
        # iterative path compression
        root = x
        while self._parent.get(root, root) != root:
            root = self._parent[root]
        # compress
        cur = x
        while self._parent[cur] != root:
            nxt = self._parent[cur]
            self._parent[cur] = root
            cur = nxt
        return root

    def union(self, a: EClassId, b: EClassId) -> EClassId:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        # union by rank
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return ra

    def __contains__(self, x: EClassId) -> bool:
        return x in self._parent


# ---------------------------------------------------------------------------
# EGraph
# ---------------------------------------------------------------------------

@dataclass
class EGraph:
    """An e-graph.

    Internal state:
      - uf          : union-find over EClassId
      - classes     : dict EClassId -> list of ENodes (the "leaders" hold the
                      actual nodes; non-leaders are empty after rebuild)
      - congruence  : dict ENode -> EClassId, used to detect congruence
      - pending     : list of (a, b) union operations not yet processed
      - next_id     : monotonic counter for new EClassIds
    """
    uf: UnionFind = field(default_factory=UnionFind)
    classes: Dict[EClassId, List[ENode]] = field(default_factory=dict)
    congruence: Dict[ENode, EClassId] = field(default_factory=dict)
    pending: List[Tuple[EClassId, EClassId]] = field(default_factory=list)
    next_id: EClassId = 0

    # ----- construction -------------------------------------------------

    def _new_class(self) -> EClassId:
        cid = self.next_id
        self.next_id += 1
        self.uf.make(cid)
        self.classes[cid] = []
        return cid

    def add_enode(self, enode: ENode) -> EClassId:
        """Add an ENode (with already-canonical children) to the e-graph.

        If a structurally identical ENode exists, return its existing class.
        Otherwise create a new class.
        """
        # canonicalize children
        enode = enode.canonicalize(self.find)
        existing = self.congruence.get(enode)
        if existing is not None:
            return self.find(existing)
        cid = self._new_class()
        self.classes[cid].append(enode)
        self.congruence[enode] = cid
        return cid

    def add(self, op_name: str, attrs: Optional[Mapping] = None, children: Tuple[EClassId, ...] = ()) -> EClassId:
        """Convenience: add an ENode by parts."""
        a = _FrozenAttrs(dict(attrs)) if attrs else _FrozenAttrs({})
        return self.add_enode(ENode(op_name, a, tuple(children)))

    def add_alternative(self, target_eclass: EClassId, op_name: str, attrs: Optional[Mapping], children: Tuple[EClassId, ...]) -> EClassId:
        """Declare that `target_eclass` is equivalent to (op_name, attrs, children).

        Internally: build the ENode, get-or-create its class, then union
        the two classes.
        """
        alt_class = self.add(op_name, attrs, children)
        self.union(target_eclass, alt_class)
        return self.find(target_eclass)

    def union(self, a: EClassId, b: EClassId) -> EClassId:
        return self.uf.union(a, b)

    def find(self, x: EClassId) -> EClassId:
        return self.uf.find(x)

    # ----- rebuild (congruence closure saturation) ----------------------

    def rebuild(self, max_iters: int = 100) -> int:
        """Saturate congruence closure.

        Repeatedly:
          1. process pending unions
          2. re-canonicalize all ENodes (because unions may have changed
             the canonical ids of their children)
          3. detect new congruences and add to pending unions

        Returns the number of iterations performed.
        """
        iters = 0
        while iters < max_iters:
            iters += 1
            # 1. process pending unions
            if self.pending:
                for a, b in self.pending:
                    self.uf.union(a, b)
                self.pending.clear()
            else:
                # nothing pending; we still need to re-canonicalize because
                # explicit unions may have changed canonical ids
                pass

            # 2. rebuild congruence table with canonical children
            new_congruence: Dict[ENode, EClassId] = {}
            new_pending: List[Tuple[EClassId, EClassId]] = []
            for cid, nodes in self.classes.items():
                canonical_cid = self.find(cid)
                for enode in nodes:
                    canon = enode.canonicalize(self.find)
                    if canon in new_congruence:
                        # congruence detected!
                        other = new_congruence[canon]
                        if self.find(other) != canonical_cid:
                            new_pending.append((other, canonical_cid))
                    else:
                        new_congruence[canon] = canonical_cid
            self.congruence = new_congruence
            if not new_pending:
                break
            self.pending = new_pending
        return iters

    # ----- query --------------------------------------------------------

    def enodes_in(self, cid: EClassId) -> List[ENode]:
        """Return all ENodes in the e-class with canonical id == find(cid)."""
        root = self.find(cid)
        out = []
        for k, nodes in self.classes.items():
            if self.find(k) == root:
                out.extend(nodes)
        return out

    def num_classes(self) -> int:
        """Count distinct canonical classes."""
        roots = set()
        for k in self.classes:
            roots.add(self.find(k))
        return len(roots)

    def num_enodes(self) -> int:
        return sum(len(nodes) for nodes in self.classes.values())

    # ----- import from Graph -------------------------------------------

    def from_graph(self, graph: Graph) -> Dict[NodeId, EClassId]:
        """Import a Graph into this e-graph.

        Returns a mapping from Graph NodeId -> EClassId.
        """
        node_to_eclass: Dict[NodeId, EClassId] = {}
        for nid in graph.topo_order():
            node = graph.nodes[nid]
            child_eclasses = tuple(node_to_eclass[c] for c in node.children)
            attrs = _FrozenAttrs(dict(node.attrs)) if not isinstance(node.attrs, _FrozenAttrs) else node.attrs
            enode = ENode(node.op_name, attrs, child_eclasses)
            cid = self.add_enode(enode)
            node_to_eclass[nid] = cid
        return node_to_eclass

    # ----- debug --------------------------------------------------------

    def dump(self, max_classes: int = 50) -> str:
        lines = [f"EGraph: {self.num_classes()} classes, {self.num_enodes()} enodes"]
        count = 0
        for cid, nodes in self.classes.items():
            if self.find(cid) != cid:
                continue  # skip non-leaders
            count += 1
            if count > max_classes:
                lines.append("  ...")
                break
            lines.append(f"  eclass {cid}:")
            for e in nodes:
                lines.append(f"    {e}")
        return "\n".join(lines)
