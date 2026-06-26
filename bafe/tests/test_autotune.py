"""Tests for Phase 3 issue #6: auto-tuning loop with profiling feedback."""
import ctypes
import pytest
import numpy as np

import bafe
from bafe._binding import (
    _lib, BafeGraph, make_shape,
    BafeAutotuneConfig, BafeAutotuneStats,
    BafeLearnedCostModel, BafeProfilingLog,
    BAFE_NUM_FEATURES,
)


@pytest.fixture(autouse=True)
def _reset_profiling():
    """Reset profiling state before each test."""
    _lib.bafe_profiling_reset()
    yield
    _lib.bafe_profiling_reset()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _build_graph(M=16, N=16, K=16):
    """Build matmul(A, B) -> relu."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([M, K])
    sh2 = make_shape([K, N])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh2), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    r = _lib.bafe_graph_relu(ctypes.byref(g), mm)
    _lib.bafe_graph_set_output(ctypes.byref(g), r)
    return g


def test_feature_extraction_returns_8_features():
    """bafe_profiling_extract_features should produce 8 numeric features."""
    g = _build_graph()
    features = (ctypes.c_double * BAFE_NUM_FEATURES)()
    _lib.bafe_profiling_extract_features(ctypes.byref(g), features)
    vals = [features[i] for i in range(BAFE_NUM_FEATURES)]
    assert len(vals) == 8
    # feature 0 = num_inputs = 2
    assert vals[0] == 2.0
    # feature 1 = num_ops = 2 (matmul + relu)
    assert vals[1] == 2.0
    # feature 2 = num_matmuls = 1
    assert vals[2] == 1.0
    # feature 3 = num_fused = 0
    assert vals[3] == 0.0
    # feature 4 = log(flops+1) > 0 (we have FLOPs)
    assert vals[4] > 0.0
    # feature 5 = log(bytes+1) > 0
    assert vals[5] > 0.0
    # feature 6 = num_intermediates = 2 (matmul + relu both materialize)
    assert vals[6] == 2.0
    # feature 7 = has_col_major = 0 (both inputs are row-major)
    assert vals[7] == 0.0


def test_features_differ_for_different_graphs():
    """Different graphs should produce different feature vectors."""
    g1 = _build_graph(M=16, N=16, K=16)
    g2 = _build_graph(M=64, N=64, K=64)

    f1 = (ctypes.c_double * BAFE_NUM_FEATURES)()
    f2 = (ctypes.c_double * BAFE_NUM_FEATURES)()
    _lib.bafe_profiling_extract_features(ctypes.byref(g1), f1)
    _lib.bafe_profiling_extract_features(ctypes.byref(g2), f2)

    # log(flops) should differ (feature 4)
    assert abs(f1[4] - f2[4]) > 0.1
    # log(bytes) should differ (feature 5)
    assert abs(f1[5] - f2[5]) > 0.1


# ---------------------------------------------------------------------------
# Profiling log
# ---------------------------------------------------------------------------

def test_profiling_log_records():
    """bafe_profiling_add should add records to the log."""
    features = (ctypes.c_double * BAFE_NUM_FEATURES)(*([1.0] * 8))
    _lib.bafe_profiling_add(b"abc123", features, 5.0, 1.5, 0)

    log_ptr = _lib.bafe_profiling_get_log()
    log = log_ptr.contents
    assert log.n == 1
    rec = log.records[0]
    assert rec.graph_hash.startswith(b"abc123")
    assert rec.predicted_cost == 5.0
    assert rec.observed_ms == 1.5
    assert rec.kernel_id == 0


def test_profiling_log_ring_buffer():
    """The log should overwrite old records when full (ring buffer)."""
    features = (ctypes.c_double * BAFE_NUM_FEATURES)(*([0.0] * 8))
    # add many records (more than the log size)
    for i in range(50):
        _lib.bafe_profiling_add(b"hash", features, float(i), float(i * 0.1), i)

    log_ptr = _lib.bafe_profiling_get_log()
    log = log_ptr.contents
    # log should be capped at BAFE_PROFILING_LOG_SIZE (4096)
    assert log.n <= 4096


def test_profiling_dump_jsonl(tmp_path):
    """bafe_profiling_dump_jsonl should write a valid JSONL file."""
    features = (ctypes.c_double * BAFE_NUM_FEATURES)(*([1.5] * 8))
    _lib.bafe_profiling_add(b"deadbeef", features, 3.0, 0.5, 0)
    _lib.bafe_profiling_add(b"cafef00d", features, 4.0, 0.6, 1)

    path = str(tmp_path / "log.jsonl")
    n = _lib.bafe_profiling_dump_jsonl(path.encode("utf-8"))
    assert n == 2

    import json
    with open(path) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert lines[0]["hash"].startswith("deadbeef")
    assert lines[0]["predicted"] == 3.0
    assert lines[0]["observed_ms"] == 0.5
    assert len(lines[0]["features"]) == 8


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------

def test_refit_requires_minimum_samples():
    """Refit should fail with too few samples."""
    features = (ctypes.c_double * BAFE_NUM_FEATURES)(*([1.0] * 8))
    _lib.bafe_profiling_add(b"h", features, 1.0, 1.0, 0)
    rc = _lib.bafe_profiling_refit()
    assert rc != 0  # not enough samples


def test_refit_produces_valid_model():
    """With enough samples, refit should produce a valid learned model."""
    # Use varying features so the model has something to learn
    for i in range(20):
        features = (ctypes.c_double * BAFE_NUM_FEATURES)(*([0.0] * 8))
        features[4] = float(i + 1)  # vary log(flops)
        runtime = 0.01 * (i + 1) * (i + 1)  # quadratic in feature
        _lib.bafe_profiling_add(b"h", features, 1.0, runtime, 0)

    rc = _lib.bafe_profiling_refit()
    assert rc == 0

    model_ptr = _lib.bafe_profiling_get_model()
    model = model_ptr.contents
    assert model.valid
    assert model.n_samples == 20
    # R^2 should be > 0 (we have a clear trend)
    assert model.r_squared > 0.3


def test_refit_with_correlated_features_learns_weights():
    """If runtime correlates with a feature, the learned weight should be non-zero."""
    # feature 4 = log(flops) — vary it across samples, keep runtime proportional
    for i in range(20):
        features = (ctypes.c_double * BAFE_NUM_FEATURES)(*([0.0] * 8))
        features[4] = float(i + 1)  # increasing FLOPs
        runtime = float(i + 1) * 0.01  # proportional runtime
        _lib.bafe_profiling_add(b"h", features, 1.0, runtime, 0)

    rc = _lib.bafe_profiling_refit()
    assert rc == 0

    model_ptr = _lib.bafe_profiling_get_model()
    model = model_ptr.contents
    # the weight on feature 4 should be positive (more FLOPs -> more runtime)
    assert model.weights[4] > 0.0


def test_predict_ms_returns_positive():
    """bafe_profiling_predict_ms should return a positive value after refit."""
    features = (ctypes.c_double * BAFE_NUM_FEATURES)(*([1.0] * 8))
    for i in range(20):
        _lib.bafe_profiling_add(b"h", features, 1.0, 0.5, 0)
    _lib.bafe_profiling_refit()

    pred = _lib.bafe_profiling_predict_ms(features)
    assert pred > 0.0


# ---------------------------------------------------------------------------
# Autotune config
# ---------------------------------------------------------------------------

def test_autotune_config_default():
    """Default config should have sensible values."""
    cfg = _lib.bafe_autotune_config_default()
    assert cfg.refit_threshold == 20
    assert cfg.invalidation_drift == 0.25
    assert cfg.warmup_calls == 2
    assert cfg.timing_iters == 5


def test_autotune_configure_sets_config():
    """bafe_autotune_configure should update the global config."""
    cfg = BafeAutotuneConfig()
    cfg.enabled = True
    cfg.refit_threshold = 99
    cfg.invalidation_drift = 0.5
    cfg.warmup_calls = 3
    cfg.timing_iters = 7
    _lib.bafe_autotune_configure(ctypes.byref(cfg))

    got = _lib.bafe_autotune_get_config()
    assert got.refit_threshold == 99
    assert got.invalidation_drift == 0.5
    assert got.warmup_calls == 3
    assert got.timing_iters == 7


# ---------------------------------------------------------------------------
# End-to-end autotune via @bafe.jit(autotune=True)
# ---------------------------------------------------------------------------

def test_jit_autotune_produces_correct_results():
    """@bafe.jit(autotune=True) should produce numerically correct output."""
    bafe.configure_autotune(refit_threshold=5, warmup_calls=1, timing_iters=2)

    @bafe.jit(autotune=True)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    out = f(A, B)
    ref = (A @ B).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4)


def test_autotune_logs_calls():
    """Autotune should log profiling records as the function is called."""
    bafe.configure_autotune(refit_threshold=100, warmup_calls=1, timing_iters=2)

    @bafe.jit(autotune=True)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    # 1 warmup + 5 timed calls = 5 logged records
    for _ in range(6):
        f(A, B)

    stats = bafe.autotune_stats()
    assert stats["log_size"] >= 5


def test_autotune_refits_after_threshold():
    """Autotune should trigger a refit after collecting enough samples."""
    bafe.configure_autotune(refit_threshold=5, warmup_calls=1, timing_iters=2)

    @bafe.jit(autotune=True)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    # 1 warmup + 10 timed = 10 logs, should trigger 2 refits (at 5 and 10)
    for _ in range(11):
        f(A, B)

    stats = bafe.autotune_stats()
    assert stats["total_refits"] >= 2


def test_autotune_model_becomes_valid():
    """After enough calls, the learned cost model should be valid."""
    bafe.configure_autotune(refit_threshold=5, warmup_calls=1, timing_iters=2)

    @bafe.jit(autotune=True)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    for _ in range(8):
        f(A, B)

    model = bafe.autotune_model()
    assert model["valid"]
    assert model["n_samples"] >= 5


def test_autotune_with_multiple_shapes():
    """Autotune should handle calls with different input shapes."""
    bafe.configure_autotune(refit_threshold=8, warmup_calls=1, timing_iters=2)

    @bafe.jit(autotune=True)
    def f(A, B):
        return bafe.matmul(A, B)

    shapes = [(16, 16), (32, 32), (64, 64)]
    for i in range(30):
        sh = shapes[i % len(shapes)]
        A = np.random.randn(*sh).astype(np.float32)
        B = np.random.randn(*sh).astype(np.float32)
        out = f(A, B)
        ref = (A @ B).astype(np.float32)
        assert np.allclose(out, ref, atol=1e-4)

    model = bafe.autotune_model()
    assert model["valid"]
    # R^2 should be reasonable (different shapes -> different runtimes)
    assert model["r_squared"] > 0.3


def test_autotune_dump_log(tmp_path):
    """bafe.autotune_dump_log should write a JSONL file."""
    bafe.configure_autotune(refit_threshold=100, warmup_calls=1, timing_iters=2)

    @bafe.jit(autotune=True)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(8, 8).astype(np.float32)
    B = np.random.randn(8, 8).astype(np.float32)
    for _ in range(5):
        f(A, B)

    path = str(tmp_path / "log.jsonl")
    n = bafe.autotune_dump_log(path)
    assert n >= 4  # 1 warmup + 4 timed

    import json
    with open(path) as fh:
        lines = [json.loads(line) for line in fh]
    assert len(lines) == n
    assert "features" in lines[0]
    assert "observed_ms" in lines[0]


def test_autotune_reset_clears_state():
    """bafe.autotune_reset should clear the log and model."""
    bafe.configure_autotune(refit_threshold=5, warmup_calls=1, timing_iters=2)

    @bafe.jit(autotune=True)
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    for _ in range(8):
        f(A, B)

    assert bafe.autotune_stats()["log_size"] > 0

    bafe.autotune_reset()
    assert bafe.autotune_stats()["log_size"] == 0
    model = bafe.autotune_model()
    assert not model["valid"]


def test_jit_without_autotune_doesnt_log():
    """@bafe.jit (without autotune) should NOT log profiling records."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    for _ in range(5):
        f(A, B)

    assert bafe.autotune_stats()["log_size"] == 0
