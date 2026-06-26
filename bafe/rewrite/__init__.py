"""BAFE rewrite engine.

A rewrite rule is a pair (pattern, rewriter):
  - pattern: a function (node, graph) -> bool  (does this rule apply?)
  - rewriter: a function (node, graph) -> optional new (op_name, attrs, child_ids)
              that is semantically equivalent to `node`

The engine walks the graph in topological order and applies every matching
rule. It does not commit instantly — instead it produces "alternatives"
that the caller can choose to insert into the e-graph.

For Phase 1 we ship deterministic rules only. Stochastic exploration is
planned for Phase 3.
"""
