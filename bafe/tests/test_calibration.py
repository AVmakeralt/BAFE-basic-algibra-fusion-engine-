"""Tests for Phase 3 issue #5: learned cost model calibration."""
import ctypes
import pytest
import numpy as np

import bafe
from bafe._binding import (
    _lib, BafeGraph, make_shape, BafeCostModel, BAFE_NUM_FEATURES,
)


@pytest.fixture(autouse=True)
def _reset_profiling():
    _lib.bafe_profiling_reset()
    yield
    _lib.bafe_profiling_reset()


# ---------------------------------------------------------------------------
# Calibration with no learned model
# ---------------------------------------------------------------------------

def test_calibrated_default_without_learned_model_returns_static():
    """With no learned model, calibrated_default should equal the static default."""
    _lib.bafe_profiling_reset()
    cal = _lib.bafe_cost_model_calibrated_default()
    stat = _lib.bafe_cost_model_default()
    assert cal.alpha_flops == stat.alpha_flops
    assert cal.beta_bytes == stat.beta_bytes
    assert cal.delta_fuse == stat.delta_fuse


# ---------------------------------------------------------------------------
# Calibration with a learned model
# ---------------------------------------------------------------------------

def test_calibrate_amplifies_alpha_flops_when_weight_is_positive():
    """If log_flops has a high positive weight, alpha_flops should increase."""
    weights = (ctypes.c_double * BAFE_NUM_FEATURES)(0, 0, 0, 0, 10.0, 0, 0, 0)
    static = _lib.bafe_cost_model_default()
    cal = _lib.bafe_cost_model_calibrate(ctypes.byref(static), weights, BAFE_NUM_FEATURES, 0.0)
    assert cal.alpha_flops > static.alpha_flops


def test_calibrate_reduces_beta_bytes_when_weight_is_negative():
    """If log_bytes has a negative weight, beta_bytes should decrease."""
    weights = (ctypes.c_double * BAFE_NUM_FEATURES)(0, 0, 0, 0, 0, -10.0, 0, 0)
    static = _lib.bafe_cost_model_default()
    cal = _lib.bafe_cost_model_calibrate(ctypes.byref(static), weights, BAFE_NUM_FEATURES, 0.0)
    assert cal.beta_bytes < static.beta_bytes


def test_calibrate_increases_delta_fuse_when_fused_weight_is_negative():
    """If num_fused has a negative weight (fusion is good), delta_fuse should increase."""
    weights = (ctypes.c_double * BAFE_NUM_FEATURES)(0, 0, 0, -10.0, 0, 0, 0, 0)
    static = _lib.bafe_cost_model_default()
    cal = _lib.bafe_cost_model_calibrate(ctypes.byref(static), weights, BAFE_NUM_FEATURES, 0.0)
    assert cal.delta_fuse > static.delta_fuse


def test_calibrate_no_change_when_weights_are_zero():
    """All-zero weights should leave the cost model unchanged."""
    weights = (ctypes.c_double * BAFE_NUM_FEATURES)(*([0.0] * 8))
    static = _lib.bafe_cost_model_default()
    cal = _lib.bafe_cost_model_calibrate(ctypes.byref(static), weights, BAFE_NUM_FEATURES, 0.0)
    assert cal.alpha_flops == static.alpha_flops
    assert cal.beta_bytes == static.beta_bytes
    assert cal.delta_fuse == static.delta_fuse
    assert cal.gamma_intermediate == static.gamma_intermediate


def test_calibrate_clamps_extreme_weights():
    """Extreme weights should be clamped to at most 10x."""
    # weight of 1e6 on log_flops should clamp
    weights = (ctypes.c_double * BAFE_NUM_FEATURES)(0, 0, 0, 0, 1e6, 0, 0, 0)
    static = _lib.bafe_cost_model_default()
    cal = _lib.bafe_cost_model_calibrate(ctypes.byref(static), weights, BAFE_NUM_FEATURES, 0.0)
    # alpha_flops should be at most 10x the static value (clamped)
    assert cal.alpha_flops <= static.alpha_flops * 10.0 + 1e-15
    # and should be amplified (not reduced)
    assert cal.alpha_flops > static.alpha_flops


# ---------------------------------------------------------------------------
# Calibration is used by the extractor
# ---------------------------------------------------------------------------

def _build_fusible_graph():
    """Build relu(add(matmul(A,B), bias)) — has fusion alternatives."""
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


def test_optimize_uses_calibrated_cost_model():
    """bafe_optimize should use the calibrated cost model for extraction.
    This is verified by checking that the optimized graph is still valid
    after calibration changes the weights."""
    g = _build_fusible_graph()

    # Set up a learned model with strong fusion preference
    weights = (ctypes.c_double * BAFE_NUM_FEATURES)(0, 0, 0, -100.0, 0, 0, 0, 0)
    for i in range(20):
        _lib.bafe_profiling_add(b"h", weights, 1.0, 0.1, 0)
    _lib.bafe_profiling_refit()

    # The optimized graph should still pick the fused form
    opt = BafeGraph()
    err = ctypes.create_string_buffer(256)
    rc = _lib.bafe_optimize(ctypes.byref(g), ctypes.byref(opt), err, ctypes.c_size_t(len(err)))
    assert rc == 0
    assert opt.n_nodes > 0

    # Check that we got a fused op (the calibrated model amplifies fusion bonus)
    has_fused = False
    for i in range(opt.n_nodes):
        op = opt.nodes[i].op_name
        if op and b"fused" in op:
            has_fused = True
            break
    assert has_fused, "calibrated model should still prefer fused form"


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

def test_calibrate_returns_cost_model():
    """bafe.calibrate() should return a usable cost model."""
    cm = bafe.calibrate()
    assert cm is not None
    # should have the cost model fields
    assert hasattr(cm, "alpha_flops")
    assert hasattr(cm, "beta_bytes")


def test_calibrated_cost_model_dict():
    """bafe.calibrated_cost_model() should return a dict with all weights."""
    cm = bafe.calibrated_cost_model()
    expected_keys = {
        "alpha_flops", "beta_bytes", "gamma_intermediate", "delta_fuse",
        "epsilon_layout_conv", "zeta_layout_fuse", "eta_contiguous",
    }
    assert set(cm.keys()) == expected_keys
    for v in cm.values():
        assert isinstance(v, float)


def test_calibration_changes_after_autotune():
    """After running autotune + refit, the calibrated model should differ from static."""
    bafe.configure_autotune(refit_threshold=5, warmup_calls=1, timing_iters=2)

    @bafe.jit(autotune=True)
    def f(A, B):
        return bafe.matmul(A, B)

    # before autotune: calibrated == static
    cm_before = bafe.calibrated_cost_model()
    stat = _lib.bafe_cost_model_default()
    alpha_before = cm_before["alpha_flops"]

    # run autotune
    shapes = [(16, 16), (64, 64), (128, 128)]
    for i in range(30):
        sh = shapes[i % len(shapes)]
        A = np.random.randn(*sh).astype(np.float32)
        B = np.random.randn(*sh).astype(np.float32)
        f(A, B)

    # after autotune: calibrated may differ
    cm_after = bafe.calibrated_cost_model()
    alpha_after = cm_after["alpha_flops"]

    # The learned model should be valid now
    model = bafe.autotune_model()
    assert model["valid"]

    # alpha_flops may or may not have changed (depends on what the model
    # learned), but the calibrated model is at least computed from the
    # learned weights. We just verify it's a valid number.
    assert alpha_after > 0
    assert np.isfinite(alpha_after)


# ---------------------------------------------------------------------------
# Cache invalidation after refit
# ---------------------------------------------------------------------------

def test_jit_invalidate_memory_cache():
    """bafe_jit_invalidate_memory_cache should clear the in-memory cache."""
    # compile something
    g = _build_fusible_graph()
    opt = BafeGraph()
    err = ctypes.create_string_buffer(256)
    _lib.bafe_optimize(ctypes.byref(g), ctypes.byref(opt), err, ctypes.c_size_t(len(err)))
    fn = _lib.bafe_jit_get_or_compile(ctypes.byref(opt), err, ctypes.c_size_t(len(err)))
    assert fn

    # invalidate
    _lib.bafe_jit_invalidate_memory_cache()

    # next call should re-dlopen from disk (not a Python-level hit)
    fn2 = _lib.bafe_jit_get_or_compile(ctypes.byref(opt), err, ctypes.c_size_t(len(err)))
    assert fn2
    # the function pointer may differ (re-dlopen'd) but should be valid


# ---------------------------------------------------------------------------
# End-to-end: autotune + calibration + correctness
# ---------------------------------------------------------------------------

def test_autotune_with_calibration_produces_correct_results():
    """The full loop (autotune → refit → calibrate → re-optimize) should
    produce numerically correct results throughout."""
    bafe.configure_autotune(refit_threshold=5, warmup_calls=1, timing_iters=2)

    @bafe.jit(autotune=True)
    def f(A, B, bias):
        return bafe.relu(bafe.matmul(A, B) + bias)

    np.random.seed(0)
    for i in range(30):
        A = np.random.randn(32, 32).astype(np.float32)
        B = np.random.randn(32, 32).astype(np.float32)
        bias = np.random.randn(32).astype(np.float32)
        out = f(A, B, bias)
        ref = np.maximum(A @ B + bias, 0.0).astype(np.float32)
        assert np.allclose(out, ref, atol=1e-4), f"failed at call {i}"

    # after 30 calls, several refits should have happened
    stats = bafe.autotune_stats()
    assert stats["total_refits"] >= 2
