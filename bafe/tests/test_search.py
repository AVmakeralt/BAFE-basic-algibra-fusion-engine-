"""Tests for Phase 2 issue #1: stochastic search layer."""
import ctypes
import pytest
import numpy as np

import bafe
from bafe._binding import (
    _lib, BafeGraph, make_shape, BafeAltList,
    BafeSearchBudget, BafeSearchStats,
)


def make_graph_with(chained_ops):
    """Build a graph from a list of (op_name, child_indices) tuples."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([8, 8])
    inputs = []
    for i in range(chained_ops[0][1]):  # first op tells us how many inputs
        nid = _lib.bafe_graph_add_input(
            ctypes.byref(g), f"input{i}".encode(), ctypes.byref(sh), 0
        )
        inputs.append(nid)
    return g, inputs


def build_simple_graph():
    """Build relu(add(matmul(A, B), bias)) — 6 nodes."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([16, 16])
    sh_bias = make_shape([16])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh), 0)
    bias = _lib.bafe_graph_add_input(ctypes.byref(g), b"bias", ctypes.byref(sh_bias), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    ad = _lib.bafe_graph_add_op(ctypes.byref(g), mm, bias)
    out = _lib.bafe_graph_relu(ctypes.byref(g), ad)
    _lib.bafe_graph_set_output(ctypes.byref(g), out)
    return g


def copy_graph(g):
    """Make a deep copy of a graph (for stochastic search which mutates)."""
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


# ---------------------------------------------------------------------------
# Search budget defaults
# ---------------------------------------------------------------------------

def test_search_budget_default():
    """The default budget should have sensible values."""
    b = _lib.bafe_search_budget_default()
    assert b.max_iters == 1  # default budget has 1 iter (single-pass)
    
    assert b.max_nodes == 4096
    assert b.max_rewrites == 512
    assert b.time_budget_ms == 0
    assert b.temperature == 1.0
    assert b.seed == 0xBAFE5EED
    


# ---------------------------------------------------------------------------
# Stochastic finds more alternatives than deterministic
# ---------------------------------------------------------------------------

def test_stochastic_finds_more_alts_than_deterministic():
    """Multi-pass stochastic should find >= deterministic alts, usually more."""
    g = build_simple_graph()

    # deterministic: one pass
    alts_det = BafeAltList()
    n_det = _lib.bafe_rewrite_find(ctypes.byref(g), ctypes.byref(alts_det))

    # stochastic: multi-pass on a copy
    g2 = copy_graph(g)
    budget = BafeSearchBudget()
    budget.max_iters = 5; budget.enable_multi_pass = False
    budget.max_nodes = 200
    budget.max_rewrites = 50
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

    assert n_det > 0, "deterministic should find at least 1 alt"
    assert n_stoch >= n_det, \
        f"stochastic ({n_stoch}) should find >= deterministic ({n_det})"
    assert stats.alts_materialized > 0, "stochastic should materialize some alts"
    assert stats.nodes_added > 0, "stochastic should add nodes to the graph"
    assert g2.n_nodes > g.n_nodes, "stochastic should grow the graph"


def test_stochastic_grows_graph_via_materialization():
    """Each materialized alternative adds a new node to the graph."""
    g = build_simple_graph()
    initial_nodes = g.n_nodes

    g2 = copy_graph(g)
    budget = BafeSearchBudget()
    budget.max_iters = 4
    budget.max_nodes = 200
    budget.max_rewrites = 30
    budget.temperature = 1.5
    budget.seed = 100
    budget.enable_multi_pass = True

    alts = BafeAltList()
    stats = BafeSearchStats()
    _lib.bafe_rewrite_stochastic_stats(
        ctypes.byref(g2), ctypes.byref(alts),
        ctypes.byref(budget), ctypes.byref(stats)
    )

    assert g2.n_nodes == initial_nodes + stats.nodes_added
    assert stats.nodes_added == stats.alts_materialized  # each materialize adds 1 node


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_stochastic_is_reproducible_with_same_seed():
    """Same seed should produce the same search trajectory."""
    g = build_simple_graph()

    def run(seed):
        g2 = copy_graph(g)
        budget = BafeSearchBudget()
        budget.max_iters = 5; budget.enable_multi_pass = False
        budget.max_nodes = 200
        budget.max_rewrites = 50
        budget.temperature = 2.0
        budget.seed = seed
        budget.enable_multi_pass = True
        alts = BafeAltList()
        stats = BafeSearchStats()
        _lib.bafe_rewrite_stochastic_stats(
            ctypes.byref(g2), ctypes.byref(alts),
            ctypes.byref(budget), ctypes.byref(stats)
        )
        return stats.alts_materialized, stats.nodes_added, g2.n_nodes

    a = run(12345)
    b = run(12345)
    c = run(99999)  # different seed -> likely different result

    assert a == b, "same seed should give identical results"
    # (we can't assert a != c because different seeds could happen to
    # produce the same trajectory, but it's very unlikely)


# ---------------------------------------------------------------------------
# Budget limits are respected
# ---------------------------------------------------------------------------

def test_max_nodes_limit_is_respected():
    """The search should stop before exceeding max_nodes."""
    g = build_simple_graph()
    g2 = copy_graph(g)
    budget = BafeSearchBudget()
    budget.max_iters = 100  # very high
    budget.max_nodes = 20   # low cap
    budget.max_rewrites = 1000
    budget.temperature = 5.0  # very explorative
    budget.seed = 1
    budget.enable_multi_pass = True

    alts = BafeAltList()
    stats = BafeSearchStats()
    _lib.bafe_rewrite_stochastic_stats(
        ctypes.byref(g2), ctypes.byref(alts),
        ctypes.byref(budget), ctypes.byref(stats)
    )

    # graph should not grow beyond max_nodes (may stop a bit before due to
    # the per-iteration check)
    assert g2.n_nodes <= budget.max_nodes, \
        f"graph grew to {g2.n_nodes}, exceeds max_nodes {budget.max_nodes}"


def test_max_rewrites_limit_is_respected():
    """The search should stop after materializing max_rewrites alternatives."""
    g = build_simple_graph()
    g2 = copy_graph(g)
    budget = BafeSearchBudget()
    budget.max_iters = 100
    budget.max_nodes = 1000
    budget.max_rewrites = 10
    budget.temperature = 5.0
    budget.seed = 1
    budget.enable_multi_pass = True

    alts = BafeAltList()
    stats = BafeSearchStats()
    _lib.bafe_rewrite_stochastic_stats(
        ctypes.byref(g2), ctypes.byref(alts),
        ctypes.byref(budget), ctypes.byref(stats)
    )

    assert stats.alts_materialized <= budget.max_rewrites


def test_enable_multi_pass_false_degrades_to_deterministic():
    """With enable_multi_pass=False, stochastic should behave like deterministic."""
    g = build_simple_graph()
    g2 = copy_graph(g)
    budget = BafeSearchBudget()
    budget.max_iters = 10
    budget.max_nodes = 1000
    budget.max_rewrites = 100
    budget.temperature = 5.0
    budget.seed = 1
    budget.enable_multi_pass = False  # deterministic mode

    alts = BafeAltList()
    stats = BafeSearchStats()
    _lib.bafe_rewrite_stochastic_stats(
        ctypes.byref(g2), ctypes.byref(alts),
        ctypes.byref(budget), ctypes.byref(stats)
    )

    # no materialization should happen
    assert stats.alts_materialized == 0
    assert stats.nodes_added == 0
    assert g2.n_nodes == g.n_nodes  # graph unchanged

    # alt count should match deterministic
    alts_det = BafeAltList()
    n_det = _lib.bafe_rewrite_find(ctypes.byref(g), ctypes.byref(alts_det))
    assert stats.alts_found == n_det


# ---------------------------------------------------------------------------
# Temperature behavior
# ---------------------------------------------------------------------------

def test_low_temperature_fewer_materializations():
    """At temperature=0 (greedy), only cost-reducing alts are materialized.
    At high temperature, more alts are materialized (exploration)."""
    g = build_simple_graph()

    def run(temp):
        g2 = copy_graph(g)
        budget = BafeSearchBudget()
        budget.max_iters = 5; budget.enable_multi_pass = False
        budget.max_nodes = 500
        budget.max_rewrites = 200
        budget.temperature = temp
        budget.seed = 42
        budget.enable_multi_pass = True
        alts = BafeAltList()
        stats = BafeSearchStats()
        _lib.bafe_rewrite_stochastic_stats(
            ctypes.byref(g2), ctypes.byref(alts),
            ctypes.byref(budget), ctypes.byref(stats)
        )
        return stats.alts_materialized

    # We can't guarantee strict ordering (randomness), but very low T
    # should generally materialize fewer than very high T.
    # Run several times to smooth out randomness.
    low_T_total = sum(run(0.01) for _ in range(5))
    high_T_total = sum(run(10.0) for _ in range(5))
    # high T should explore at least as much as low T
    assert high_T_total >= low_T_total, \
        f"high T ({high_T_total}) should explore >= low T ({low_T_total})"


# ---------------------------------------------------------------------------
# Full pipeline with stochastic search
# ---------------------------------------------------------------------------

def test_optimize_with_budget_produces_correct_result():
    """bafe_optimize_with_budget should produce a valid optimized graph."""
    g = build_simple_graph()
    budget = BafeSearchBudget()
    budget.max_iters = 5; budget.enable_multi_pass = False
    budget.max_nodes = 200
    budget.max_rewrites = 50
    budget.temperature = 1.5
    budget.seed = 42
    budget.enable_multi_pass = True

    opt = BafeGraph()
    err = ctypes.create_string_buffer(256)
    rc = _lib.bafe_optimize_with_budget(
        ctypes.byref(g), ctypes.byref(opt),
        ctypes.byref(budget), err, ctypes.c_size_t(len(err))
    )
    assert rc == 0, f"optimize failed: {err.value}"
    assert opt.n_nodes > 0
    assert opt.n_outputs == 1


def test_optimize_with_budget_does_not_mutate_input():
    """The input graph should be unchanged after optimize_with_budget."""
    g = build_simple_graph()
    n_before = g.n_nodes

    budget = BafeSearchBudget()
    budget.max_iters = 5; budget.enable_multi_pass = False
    budget.max_nodes = 200
    budget.max_rewrites = 50
    budget.temperature = 2.0
    budget.seed = 42
    budget.enable_multi_pass = True

    opt = BafeGraph()
    err = ctypes.create_string_buffer(256)
    _lib.bafe_optimize_with_budget(
        ctypes.byref(g), ctypes.byref(opt),
        ctypes.byref(budget), err, ctypes.c_size_t(len(err))
    )

    assert g.n_nodes == n_before, "input graph should not be mutated"


# ---------------------------------------------------------------------------
# Python @bafe.jit with stochastic parameters
# ---------------------------------------------------------------------------

def test_jit_with_stochastic_budget():
    """@bafe.jit(budget=...) should produce correct results."""
    budget = bafe.make_search_budget(max_iters=5, enable_multi_pass=False, seed=42, temperature=1.5)

    @bafe.jit(budget=budget)
    def f(A, B, bias):
        return bafe.relu(bafe.matmul(A, B) + bias)

    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)
    bias = np.random.randn(32).astype(np.float32)

    out = f(A, B, bias)
    ref = np.maximum(A @ B + bias, 0.0).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4)


def test_jit_with_iters_param():
    """@bafe.jit(iters=8) should work."""
    @bafe.jit(iters=8, temperature=2.0, seed=123)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    out = f(A, B)
    ref = (A @ B).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4)


def test_jit_default_is_deterministic():
    """@bafe.jit without budget params should use deterministic search."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    assert f._budget is None


def test_jit_with_stochastic_and_deterministic_give_same_correct_result():
    """Both modes should produce correct numerical results."""
    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)
    bias = np.random.randn(32).astype(np.float32)
    ref = np.maximum(A @ B + bias, 0.0).astype(np.float32)

    @bafe.jit
    def f_det(A, B, bias):
        return bafe.relu(bafe.matmul(A, B) + bias)

    @bafe.jit(iters=10, temperature=3.0, seed=42)
    def f_stoch(A, B, bias):
        return bafe.relu(bafe.matmul(A, B) + bias)

    out_det = f_det(A, B, bias)
    out_stoch = f_stoch(A, B, bias)

    assert np.allclose(out_det, ref, atol=1e-4)
    assert np.allclose(out_stoch, ref, atol=1e-4)


def test_make_search_budget_returns_valid_struct():
    """bafe.make_search_budget should return a usable budget struct."""
    b = bafe.make_search_budget(max_iters=10, max_nodes=500, enable_multi_pass=True, seed=99)
    assert b.max_iters == 10  # default budget has 1 iter (single-pass)
    0
    assert b.max_nodes == 500
    assert b.seed == 99
    

    # should be usable with @bafe.jit
    @bafe.jit(budget=b)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(8, 8).astype(np.float32)
    B = np.random.randn(8, 8).astype(np.float32)
    out = f(A, B)
    assert np.allclose(out, (A @ B).astype(np.float32), atol=1e-4)
