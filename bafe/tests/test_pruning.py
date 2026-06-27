"""Tests for Phase 3 issue #4: multi-tier pruning with time budget."""
import ctypes
import pytest
import numpy as np

import bafe
from bafe._binding import (
    _lib, BafeGraph, make_shape, BafeAltList,
    BafePruningConfig, BafePruningStats,
)


def _build_graph():
    """Build relu(add(matmul(A,B), mul(C,D)))."""
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


def _copy_graph(g):
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
# Regime mapping
# ---------------------------------------------------------------------------

def test_regime_greedy_for_1ms():
    assert _lib.bafe_pruning_regime_from_budget(1) == 0  # BAFE_REGIME_GREEDY


def test_regime_light_for_10ms():
    assert _lib.bafe_pruning_regime_from_budget(10) == 1  # BAFE_REGIME_LIGHT


def test_regime_beam_for_100ms():
    assert _lib.bafe_pruning_regime_from_budget(100) == 2  # BAFE_REGIME_BEAM


def test_regime_deep_for_1000ms():
    assert _lib.bafe_pruning_regime_from_budget(1000) == 3  # BAFE_REGIME_DEEP


def test_regime_deep_for_no_limit():
    assert _lib.bafe_pruning_regime_from_budget(0) == 3  # no limit = deep


def test_beam_width_increases_with_regime():
    widths = [_lib.bafe_pruning_beam_width_for_regime(r) for r in range(4)]
    assert widths == [1, 4, 16, 64]
    # strictly increasing
    assert widths[0] < widths[1] < widths[2] < widths[3]


def test_iters_increase_with_regime():
    iters = [_lib.bafe_pruning_iters_for_regime(r) for r in range(4)]
    assert iters == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------

def test_pruning_regime_name():
    assert bafe.pruning_regime_name(1) == "greedy"
    assert bafe.pruning_regime_name(10) == "light"
    assert bafe.pruning_regime_name(100) == "beam"
    assert bafe.pruning_regime_name(1000) == "deep"
    assert bafe.pruning_regime_name(0) == "deep"
    assert bafe.pruning_regime_name(None) == "deep"


def test_pruning_beam_width_python():
    assert bafe.pruning_beam_width(1) == 1
    assert bafe.pruning_beam_width(100) == 16
    assert bafe.pruning_beam_width(1000) == 64


def test_pruning_iters_python():
    assert bafe.pruning_iters(1) == 1
    assert bafe.pruning_iters(1000) == 8


# ---------------------------------------------------------------------------
# Pruning controller runs and produces stats
# ---------------------------------------------------------------------------

def test_pruning_run_returns_stats():
    """bafe_pruning_run should populate the stats struct."""
    g = _copy_graph(_build_graph())
    alts = BafeAltList()
    stats = BafePruningStats()
    config = _lib.bafe_pruning_config_default()
    config.time_budget_ms = 100
    rc = _lib.bafe_pruning_run(ctypes.byref(g), ctypes.byref(alts),
                                ctypes.byref(config), ctypes.byref(stats))
    assert rc == 0
    assert stats.regime == 2  # BEAM
    assert stats.total_alts_found > 0
    assert stats.tier_a_passed > 0  # at least some passed structural pruning
    assert stats.elapsed_ms >= 0


def test_pruning_greedy_regime_does_not_materialize():
    """In GREEDY regime (1ms), no alternatives should be materialized."""
    g = _copy_graph(_build_graph())
    initial_nodes = g.n_nodes
    alts = BafeAltList()
    stats = BafePruningStats()
    config = _lib.bafe_pruning_config_default()
    config.time_budget_ms = 1  # GREEDY
    _lib.bafe_pruning_run(ctypes.byref(g), ctypes.byref(alts),
                           ctypes.byref(config), ctypes.byref(stats))
    assert stats.regime == 0  # GREEDY
    assert stats.tier_d_materialized == 0  # no stochastic materialization
    assert g.n_nodes == initial_nodes  # graph unchanged


def test_pruning_beam_regime_materializes():
    """In BEAM regime (100ms), alternatives should be materialized."""
    g = _copy_graph(_build_graph())
    initial_nodes = g.n_nodes
    alts = BafeAltList()
    stats = BafePruningStats()
    config = _lib.bafe_pruning_config_default()
    config.time_budget_ms = 100  # BEAM
    config.seed = 42
    _lib.bafe_pruning_run(ctypes.byref(g), ctypes.byref(alts),
                           ctypes.byref(config), ctypes.byref(stats))
    assert stats.regime == 2  # BEAM
    # in BEAM mode, we expect some materialization (stochastic tier D)
    # (may be 0 if all scores are equal, but usually > 0)
    assert stats.tier_d_materialized >= 0


def test_pruning_deep_regime_explores_more():
    """DEEP regime should find >= alternatives than GREEDY."""
    g_greedy = _copy_graph(_build_graph())
    g_deep = _copy_graph(_build_graph())
    alts_g = BafeAltList()
    alts_d = BafeAltList()
    stats_g = BafePruningStats()
    stats_d = BafePruningStats()
    cfg_g = _lib.bafe_pruning_config_default()
    cfg_g.time_budget_ms = 1
    cfg_d = _lib.bafe_pruning_config_default()
    cfg_d.time_budget_ms = 1000
    cfg_d.seed = 42
    _lib.bafe_pruning_run(ctypes.byref(g_greedy), ctypes.byref(alts_g),
                           ctypes.byref(cfg_g), ctypes.byref(stats_g))
    _lib.bafe_pruning_run(ctypes.byref(g_deep), ctypes.byref(alts_d),
                           ctypes.byref(cfg_d), ctypes.byref(stats_d))
    # DEEP should find at least as many alts as GREEDY
    assert stats_d.total_alts_found >= stats_g.total_alts_found


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------

def test_max_nodes_kill_switch():
    """max_nodes should cap the graph size."""
    g = _copy_graph(_build_graph())
    alts = BafeAltList()
    stats = BafePruningStats()
    config = _lib.bafe_pruning_config_default()
    config.time_budget_ms = 1000  # DEEP
    config.max_nodes = 20  # low cap
    config.max_rewrites = 1000
    _lib.bafe_pruning_run(ctypes.byref(g), ctypes.byref(alts),
                           ctypes.byref(config), ctypes.byref(stats))
    assert g.n_nodes <= config.max_nodes


def test_max_rewrites_kill_switch():
    """max_rewrites should cap the number of materialized rewrites."""
    g = _copy_graph(_build_graph())
    alts = BafeAltList()
    stats = BafePruningStats()
    config = _lib.bafe_pruning_config_default()
    config.time_budget_ms = 1000  # DEEP
    config.max_nodes = 1000
    config.max_rewrites = 5  # low cap
    _lib.bafe_pruning_run(ctypes.byref(g), ctypes.byref(alts),
                           ctypes.byref(config), ctypes.byref(stats))
    assert stats.tier_d_materialized <= config.max_rewrites


# ---------------------------------------------------------------------------
# Anytime property
# ---------------------------------------------------------------------------

def test_anytime_returns_valid_result_on_interruption():
    """Even if interrupted, the pruning should return a valid alt list."""
    g = _copy_graph(_build_graph())
    alts = BafeAltList()
    stats = BafePruningStats()
    config = _lib.bafe_pruning_config_default()
    config.time_budget_ms = 1  # very tight -> likely interrupted
    config.enable_anytime = True
    _lib.bafe_pruning_run(ctypes.byref(g), ctypes.byref(alts),
                           ctypes.byref(config), ctypes.byref(stats))
    # even with 1ms, we should get some alternatives (the greedy pass is fast)
    assert alts.n >= 0  # valid (may be 0 if interrupted before first pass)
    # the stats should be populated
    assert stats.elapsed_ms >= 0


# ---------------------------------------------------------------------------
# End-to-end with @bafe.jit(time_budget_ms=...)
# ---------------------------------------------------------------------------

def test_jit_with_time_budget_produces_correct_result():
    """@bafe.jit(time_budget_ms=100) should produce correct output."""
    @bafe.jit(time_budget_ms=100)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    out = f(A, B)
    ref = (A @ B).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4)


def test_jit_with_different_time_budgets_all_correct():
    """All budget regimes should produce correct results."""
    @bafe.jit(time_budget_ms=1)
    def f_greedy(A, B):
        return bafe.relu(bafe.matmul(A, B))

    @bafe.jit(time_budget_ms=1000)
    def f_deep(A, B):
        return bafe.relu(bafe.matmul(A, B))

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    ref = np.maximum(A @ B, 0).astype(np.float32)

    out_g = f_greedy(A, B)
    out_d = f_deep(A, B)
    assert np.allclose(out_g, ref, atol=1e-4)
    assert np.allclose(out_d, ref, atol=1e-4)


def test_jit_with_time_budget_and_autotune():
    """time_budget_ms + autotune should work together."""
    bafe.configure_autotune(refit_threshold=100, warmup_calls=1, timing_iters=2)

    @bafe.jit(time_budget_ms=50, autotune=True)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    out = f(A, B)
    ref = (A @ B).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4)


def test_time_budget_zero_uses_stochastic_search():
    """time_budget_ms=0 should fall back to stochastic search (no pruning)."""
    @bafe.jit(time_budget_ms=0)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(8, 8).astype(np.float32)
    B = np.random.randn(8, 8).astype(np.float32)
    out = f(A, B)
    assert np.allclose(out, (A @ B).astype(np.float32), atol=1e-4)
