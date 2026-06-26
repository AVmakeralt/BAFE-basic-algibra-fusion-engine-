"""Rewrite engine: apply rules to a graph.

The engine walks the graph in topological order and applies every matching
rule. For each match it produces an `Alternative`, which the caller may
feed into the e-graph.

Phase 1 engine is a single forward pass. Iteration to fixpoint is the
e-graph's job (via `rebuild` + repeated rule application).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from bafe.ir.graph import Graph, Node, NodeId
from bafe.rewrite.rules import Rule, Rewrite, DEFAULT_RULES


@dataclass(frozen=True)
class Alternative:
    """An alternative expression for `original_node_id`.

    Equivalent to: `original_node_id := (op_name, attrs, children)`
    """
    original_node_id: NodeId
    rewrite: Rewrite


def find_alternatives(graph: Graph, rules: Iterable[Rule] = DEFAULT_RULES) -> List[Alternative]:
    """Walk the graph once and collect all rule matches.

    Returns a list of Alternatives (one per (node, rule) match).
    """
    out: List[Alternative] = []
    rule_list = tuple(rules)
    for nid in graph.topo_order():
        node = graph.nodes[nid]
        if node.op_name in ("input", "constant"):
            continue
        for rule in rule_list:
            rw = rule(node, graph)
            if rw is not None:
                out.append(Alternative(original_node_id=nid, rewrite=rw))
    return out


def apply_alternatives(graph: Graph, alts: List[Alternative]) -> Graph:
    """Build a new graph that materializes the alternatives.

    For each alternative we create a NEW node implementing the rewrite,
    and we record that the original node and the new node are equivalent
    (this bookkeeping is returned via the e-graph layer, not here).

    NOTE: For Phase 1 we mostly use this to verify rewrites produce valid
    graphs. The real consumer of alternatives is the e-graph.
    """
    new_graph = graph  # mutate in place for simplicity
    for alt in alts:
        new_id = new_graph.add(
            alt.rewrite.op_name,
            *alt.rewrite.children,
            attrs=dict(alt.rewrite.attrs),
        )
        # in a real system we'd union these two classes in the e-graph
    return new_graph
