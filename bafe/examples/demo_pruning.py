"""Demo: multi-tier pruning with time budget.

Shows how different time budgets produce different optimization regimes,
from greedy (1ms) to deep exploration (1000ms).

Run with:
    BAFE_LIB=bafe/build/libbafe.so PYTHONPATH=bafe/python python3 bafe/examples/demo_pruning.py
"""
import os
import sys
import ctypes
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("BAFE_LIB", str(ROOT / "bafe" / "build" / "libbafe.so"))

import numpy as np
import bafe
from bafe._binding import (
    _lib, BafeGraph, make_shape, BafeAltList,
    BafePruningConfig, BafePruningStats,
)


def build_graph():
    """Build relu(add(matmul(A,B), mul(C,D))) — has multiple fusion paths."""
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
    banner("BAFE multi-tier pruning demo")

    g = build_graph()
    print(f"\nInput graph: {g.n_nodes} nodes")
    print("  relu(add(matmul(A, B), mul(C, D)))")

    # ---- Regime table ----
    banner("1. Time-budget regimes")
    print(f"\n  {'budget':>10s}  {'regime':>8s}  {'beam':>6s}  {'iters':>6s}  {'tiers':>12s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*12}")
    for ms in [1, 10, 100, 1000, 0]:
        regime = bafe.pruning_regime_name(ms)
        beam = bafe.pruning_beam_width(ms)
        iters = bafe.pruning_iters(ms)
        tiers = "A+B" if regime == "greedy" else \
                "A+B+C" if regime == "light" else \
                "A+B+C+D" if regime == "beam" else "A+B+C+D (deep)"
        label = f"{ms}ms" if ms > 0 else "no limit"
        print(f"  {label:>10s}  {regime:>8s}  {beam:>6d}  {iters:>6d}  {tiers:>12s}")

    # ---- Run each regime ----
    banner("2. Running pruning at each budget")

    results = []
    for ms in [1, 10, 100, 1000]:
        g_copy = copy_graph(g)
        alts = BafeAltList()
        stats = BafePruningStats()
        config = _lib.bafe_pruning_config_default()
        config.time_budget_ms = ms
        config.seed = 42
        _lib.bafe_pruning_run(ctypes.byref(g_copy), ctypes.byref(alts),
                               ctypes.byref(config), ctypes.byref(stats))

        regime = bafe.pruning_regime_name(ms)
        results.append((ms, regime, stats, g_copy.n_nodes))
        print(f"\n  {ms:4d} ms ({regime:8s}):")
        print(f"    regime:              {stats.regime} ({regime})")
        print(f"    total alts found:    {stats.total_alts_found}")
        print(f"    tier A passed:       {stats.tier_a_passed}")
        print(f"    tier B passed:       {stats.tier_b_passed}")
        print(f"    tier C kept:         {stats.tier_c_kept}")
        print(f"    tier D materialized: {stats.tier_d_materialized}")
        print(f"    graph grew to:       {g_copy.n_nodes} nodes (from {g.n_nodes})")
        print(f"    elapsed:             {stats.elapsed_ms:.3f} ms")
        print(f"    interrupted:         {stats.was_interrupted}")

    # ---- Comparison ----
    banner("3. Regime comparison")
    print(f"\n  {'budget':>8s}  {'regime':>8s}  {'alts':>6s}  {'materialized':>13s}  {'nodes':>6s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*13}  {'-'*6}")
    for ms, regime, stats, n_nodes in results:
        print(f"  {ms:6d}ms  {regime:>8s}  {stats.total_alts_found:6d}  "
              f"{stats.tier_d_materialized:13d}  {n_nodes:6d}")

    # ---- Anytime property ----
    banner("4. Anytime property (interruption is safe)")
    g_copy = copy_graph(g)
    alts = BafeAltList()
    stats = BafePruningStats()
    config = _lib.bafe_pruning_config_default()
    config.time_budget_ms = 1  # very tight
    config.enable_anytime = True
    _lib.bafe_pruning_run(ctypes.byref(g_copy), ctypes.byref(alts),
                           ctypes.byref(config), ctypes.byref(stats))
    print(f"\n  1ms budget (likely interrupted):")
    print(f"    alts found:  {stats.total_alts_found}")
    print(f"    elapsed:     {stats.elapsed_ms:.3f} ms")
    print(f"    interrupted: {stats.was_interrupted}")
    print(f"    valid result: {alts.n >= 0}")

    # ---- End-to-end with @bafe.jit ----
    banner("5. End-to-end: @bafe.jit(time_budget_ms=...)")

    @bafe.jit(time_budget_ms=100)
    def f(A, B):
        return bafe.relu(bafe.matmul(A, B))

    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)
    out = f(A, B)
    ref = np.maximum(A @ B, 0).astype(np.float32)
    print(f"\n  @bafe.jit(time_budget_ms=100)")
    print(f"  relu(matmul(32x32, 32x32))")
    print(f"  output shape: {out.shape}")
    print(f"  max error:    {np.max(np.abs(out - ref)):.2e}")
    print(f"  correct:      {np.allclose(out, ref, atol=1e-4)}")

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"""
  The multi-tier pruning controller (issue #4) converts a wall-clock
  time budget into structured per-stage limits:

    1 ms  -> greedy   (Level A+B, beam=1, no materialization)
    10 ms -> light    (A+B+C, beam=4, materialize best)
    100 ms -> beam    (A+B+C+D, beam=16, stochastic survival)
    1000+ ms -> deep  (all tiers, beam=64, 8 iterations)

  Key properties:
    - Anytime: if interrupted, returns the best-so-far (never worse
      than the deterministic baseline)
    - Kill switches: max_nodes, max_rewrites, max_egraph_size all
      enforced as hard caps
    - Tiered: cheaper tiers run first, expensive tiers only run if
      the budget allows
""")


if __name__ == "__main__":
    main()
