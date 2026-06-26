"""BAFE e-graph.

An e-graph represents many equivalent programs compactly. Key components:

  - ENode   : an op + attrs + child eclass ids (hashable, immutable)
  - EClass  : a set of equivalent ENodes (mutable, identified by an int id)
  - EGraph  : the container, with a union-find structure on EClass ids

The e-graph supports:
  - add(expr)         : insert an expression, return its eclass id
  - union(a, b)       : declare two eclasses equivalent (merges them)
  - rebuild()         : saturate congruence closure (merge classes that
                        have become equivalent due to prior unions)
  - find(eclass_id)   : return the canonical id of an eclass

This is a textbook e-graph (à la egg / Tarasov) but written from scratch
for BAFE's IR.
"""
