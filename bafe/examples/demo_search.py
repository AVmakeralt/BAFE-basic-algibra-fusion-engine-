"""Demo: stochastic search vs deterministic search.

Shows how multi-pass stochastic exploration discovers more rewrite
alternatives than single-pass deterministic, by re-applying rules to
nodes created during the search.

Run with:
    BAFE_LIB=bafe/build/libbafe.so PYTHONPATH=bafe/python python3 bafe/examples/demo_search.py
"""
import os
import sys
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("BAFE_LIB", str(ROOT / "bafe" / "build" / "libbafe.so"))

import numpy as np
import bafe
from bafe._binding import (
    _lib, BafeGraph, make_shape, BafeAltList,
    BafeSearchBudget, BafeSearchStats,
)


def build_graph():
    """Build a graph with multiple fusion opportunities.

    relu(add(matmul(A, B), mul(C, D)))

    This has:
      - add_commutative on the add
      - mul_commutative on the mul
      - distribute_mul_over_add (if we had the right shape)
      - various fusion opportunities that multi-pass can chain
    """
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([16, 16])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh), 0)
    c = _lib.bafe_graph_add_input(ctypes.byref(g), b"C", ctypes.byref(sh), 0)
    d = _lib.bafe_graph_add_input(ctypes.byref(g), b"D", ctypes.byref(sh), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    md = _lib.bafe_graph_mul(ctypes.byref(g), c, d)
    ad = _lib.bafe_graph_add_op(ctypes.byref(g), mm, md)
    out = _lib.bafe_graph_relu(ctypes.byref(g), ad)
    _lib.bafe_graph_set_output(ctypes.byref(g), out)
    return g


def copy_graph(g):
    g2 = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g2))
    for i in range(g.n_nodes):
        g2.nodes[i] = g.nodes[i]
    g2.n_nodes = g.n_nodes
    for i in range(g.n_inputs):
        g2.inputs[i] = g.inputs[i]
    g2.n_inputs = g.n_inputs
    for i in range(g.n_outputs):
        g2.outputs[i] = g.outputs[i]
    g2.n_outputs = g.n_outputs
    return g2


def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    banner("BAFE stochastic search demo")

    g = build_graph()
    print(f"\nInput graph: {g.n_nodes} nodes")
    print("  relu(add(matmul(A, B), mul(C, D)))")

    # ---- Deterministic ----
    banner("1. Deterministic single-pass search")
    alts_det = BafeAltList()
    n_det = _lib.bafe_rewrite_find(ctypes.byref(g), ctypes.byref(alts_det))
    print(f"\n  Alternatives found: {n_det}")
    print(f"  Graph nodes: {g.n_nodes} (unchanged)")
    for i in range(n_det):
        a = alts_det.items[i]
        children = [a.children[j] for j in range(a.n_children)]
        print(f"    alt {i}: node {a.original_node_id} -> {a.op_name.decode()}({children})")

    # ---- Stochastic ----
    banner("2. Stochastic multi-pass search (5 iters, T=2.0, seed=42)")

    g2 = copy_graph(g)
    budget = BafeSearchBudget()
    budget.max_iters = 5
    budget.max_nodes = 300
    budget.max_rewrites = 100
    budget.time_budget_ms = 0
    budget.temperature = 2.0
    budget.seed = 42
    budget.enable_multi_pass = True

    alts_stoch = BafeAltList()
    stats = BafeSearchStats()
    n_stoch = _lib.bafe_rewrite_stochastic_stats(
        ctypes.byref(g2), ctypes.byref(alts_stoch),
        ctypes.byref(budget), ctypes.byref(stats)
    )

    print(f"\n  Iters done:        {stats.iters_done}")
    print(f"  Alternatives found: {stats.alts_found}")
    print(f"  Materialized:      {stats.alts_materialized}")
    print(f"  Nodes added:       {stats.nodes_added}")
    print(f"  Graph grew to:     {g2.n_nodes} nodes (from {g.n_nodes})")
    print(f"  Elapsed:           {stats.elapsed_ms:.2f} ms")

    # Show the unique alternative op types discovered
    det_ops = set()
    for i in range(n_det):
        det_ops.add(alts_det.items[i].op_name.decode())
    stoch_ops = set()
    for i in range(n_stoch):
        stoch_ops.add(alts_stoch.items[i].op_name.decode())

    print(f"\n  Op types discovered by deterministic: {sorted(det_ops)}")
    print(f"  Op types discovered by stochastic:    {sorted(stoch_ops)}")
    print(f"  Total alts: deterministic={n_det}, stochastic={n_stoch} "
          f"({n_stoch/n_det:.1f}x more)")

    # ---- Temperature comparison ----
    banner("3. Temperature effect (low T = greedy, high T = explore)")

    for temp in [0.01, 0.5, 1.0, 2.0, 5.0]:
        g_t = copy_graph(g)
        b = BafeSearchBudget()
        b.max_iters = 5
        b.max_nodes = 300
        b.max_rewrites = 200
        b.temperature = temp
        b.seed = 42
        b.enable_multi_pass = True
        alts_t = BafeAltList()
        s_t = BafeSearchStats()
        _lib.bafe_rewrite_stochastic_stats(
            ctypes.byref(g_t), ctypes.byref(alts_t),
            ctypes.byref(b), ctypes.byref(s_t)
        )
        print(f"  T={temp:5.2f}: materialized={s_t.alts_materialized:3d}, "
              f"nodes_added={s_t.nodes_added:3d}, alts_found={s_t.alts_found:3d}")

    # ---- Reproducibility ----
    banner("4. Reproducibility (same seed = same result)")

    def run(seed):
        g_s = copy_graph(g)
        b = BafeSearchBudget()
        b.max_iters = 5
        b.max_nodes = 300
        b.max_rewrites = 100
        b.temperature = 2.0
        b.seed = seed
        b.enable_multi_pass = True
        alts_s = BafeAltList()
        s_s = BafeSearchStats()
        _lib.bafe_rewrite_stochastic_stats(
            ctypes.byref(g_s), ctypes.byref(alts_s),
            ctypes.byref(b), ctypes.byref(s_s)
        )
        return (s_s.alts_materialized, s_s.nodes_added, s_s.alts_found)

    r1 = run(42)
    r2 = run(42)
    r3 = run(999)
    print(f"\n  seed=42:   materialized={r1[0]}, nodes_added={r1[1]}, alts={r1[2]}")
    print(f"  seed=42:   materialized={r2[0]}, nodes_added={r2[1]}, alts={r2[2]}")
    print(f"  seed=999:  materialized={r3[0]}, nodes_added={r3[1]}, alts={r3[2]}")
    print(f"\n  Same seed gives identical results: {r1 == r2}")

    # ---- End-to-end with @bafe.jit ----
    banner("5. End-to-end: @bafe.jit with stochastic search")

    @bafe.jit(iters=8, temperature=2.0, seed=42)
    def f(A, B, C, D):
        return bafe.relu(bafe.matmul(A, B) + bafe.mul(C, D))

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    C = np.random.randn(16, 16).astype(np.float32)
    D = np.random.randn(16, 16).astype(np.float32)

    out = f(A, B, C, D)
    ref = np.maximum(A @ B + C * D, 0.0).astype(np.float32)
    print(f"\n  Input: relu(matmul(A,B) + mul(C,D)), shape (16,16)")
    print(f"  Output shape: {out.shape}")
    print(f"  Max error vs numpy: {np.max(np.abs(out - ref)):.2e}")
    print(f"  Correct: {np.allclose(out, ref, atol=1e-4)}")

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"""
  The stochastic search layer (issue #1) adds:
    - Multi-pass exploration: re-applies rules to newly-created nodes
    - Budget control: max_iters, max_nodes, max_rewrites, time_budget_ms
    - Temperature: 0 = greedy, high = explore randomly
    - Reproducibility: seeded xorshift128 PRNG
    - Integration: feeds alternatives into the existing e-graph

  On this graph, stochastic found {n_stoch} alternatives vs deterministic's {n_det}
  ({n_stoch/max(n_det,1):.1f}x more), by materializing rewrites and
  re-applying rules to the new nodes.
""")


if __name__ == "__main__":
    main()
