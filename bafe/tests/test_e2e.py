"""End-to-end tests: full pipeline from @bafe.jit to numerical result."""
import numpy as np
import pytest
import bafe


def _check_close(out, ref, atol=1e-4):
    assert out.shape == ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
    assert out.dtype == ref.dtype, f"dtype mismatch: {out.dtype} vs {ref.dtype}"
    assert np.allclose(out, ref, atol=atol), \
        f"values differ: max err = {np.max(np.abs(out - ref))}"


def test_matmul_only():
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    out = f(A, B)
    ref = (A @ B).astype(np.float32)
    _check_close(out, ref)


def test_relu_only():
    @bafe.jit
    def f(X):
        return bafe.relu(X)

    X = np.random.randn(32, 32).astype(np.float32) * 2 - 1
    out = f(X)
    ref = np.maximum(X, 0).astype(np.float32)
    _check_close(out, ref)


def test_add_two_tensors():
    @bafe.jit
    def f(A, B):
        return bafe.add(A, B)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    out = f(A, B)
    ref = (A + B).astype(np.float32)
    _check_close(out, ref)


def test_matmul_plus_relu():
    @bafe.jit
    def f(A, B):
        return bafe.relu(bafe.matmul(A, B))

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    out = f(A, B)
    ref = np.maximum(A @ B, 0).astype(np.float32)
    _check_close(out, ref)


def test_matmul_plus_C_relu_rank2():
    """relu(matmul(A,B) + C) where C is rank-2 (not a bias)."""
    @bafe.jit
    def f(A, B, C):
        return bafe.relu(bafe.matmul(A, B) + C)

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    C = np.random.randn(16, 24).astype(np.float32)
    out = f(A, B, C)
    ref = np.maximum(A @ B + C, 0).astype(np.float32)
    _check_close(out, ref)


def test_matmul_plus_bias_relu_rank1():
    """relu(matmul(A,B) + bias) where bias is rank-1.

    This should trigger the fused_matmul_bias_relu rewrite.
    """
    @bafe.jit
    def f(A, B, bias):
        return bafe.relu(bafe.matmul(A, B) + bias)

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    bias = np.random.randn(24).astype(np.float32)
    out = f(A, B, bias)
    ref = np.maximum(A @ B + bias, 0).astype(np.float32)
    _check_close(out, ref)


def test_matmul_plus_bias_no_relu():
    """matmul(A,B) + bias (no relu). Should trigger fused_matmul_bias."""
    @bafe.jit
    def f(A, B, bias):
        return bafe.matmul(A, B) + bias

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    bias = np.random.randn(24).astype(np.float32)
    out = f(A, B, bias)
    ref = (A @ B + bias).astype(np.float32)
    _check_close(out, ref)


def test_jit_cache_hit_on_second_call():
    """Second call with same shapes should hit the JIT cache.

    The Python binding caches the compiled fn_ptr per-shape; the C-level
    JIT cache is only exercised when we re-optimize. Here we verify the
    Python-level cache by checking that the second call doesn't trigger
    a recompile (which would show up as another 'miss' in JIT stats).
    """
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    out1 = f(A, B)
    out2 = f(A, B)
    _check_close(out1, (A @ B).astype(np.float32))
    _check_close(out2, (A @ B).astype(np.float32))

    # The Python-level cache should have 1 entry; the C JIT cache should
    # show exactly 1 miss (the first call) and 0 hits (the second call
    # short-circuits in Python before reaching the C cache).
    assert len(f._compiled) == 1
    stats = _get_jit_stats()
    assert stats["misses"] == 1


def test_c_jit_cache_hit_on_reoptimize():
    """If we re-optimize the same graph, the C-level JIT cache should hit."""
    import ctypes
    from bafe._binding import _lib, BafeGraph, make_shape

    # Build a simple graph: relu(A)
    def build():
        g = BafeGraph()
        _lib.bafe_graph_init(ctypes.byref(g))
        sh = make_shape([16, 16])
        a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
        r = _lib.bafe_graph_relu(ctypes.byref(g), a)
        _lib.bafe_graph_set_output(ctypes.byref(g), r)
        return g

    err = ctypes.create_string_buffer(256)
    g1 = build()
    fn1 = _lib.bafe_jit_get_or_compile(ctypes.byref(g1), err, ctypes.c_size_t(len(err)))
    assert fn1, f"first compile failed: {err.value}"

    g2 = build()  # identical graph
    fn2 = _lib.bafe_jit_get_or_compile(ctypes.byref(g2), err, ctypes.c_size_t(len(err)))
    assert fn2, f"second compile failed: {err.value}"

    stats = _get_jit_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 1


def test_jit_recompiles_for_different_shapes():
    """Different input shapes should trigger a new compile."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    A1 = np.random.randn(16, 16).astype(np.float32)
    B1 = np.random.randn(16, 16).astype(np.float32)
    out1 = f(A1, B1)

    A2 = np.random.randn(32, 32).astype(np.float32)
    B2 = np.random.randn(32, 32).astype(np.float32)
    out2 = f(A2, B2)

    _check_close(out1, (A1 @ B1).astype(np.float32))
    _check_close(out2, (A2 @ B2).astype(np.float32))

    stats = _get_jit_stats()
    assert stats["misses"] == 2  # two different shapes -> two compiles


def _get_jit_stats():
    from bafe._binding import _lib, BafeJitStats
    s = _lib.bafe_jit_get_stats()
    return {
        "hits": s.hits,
        "misses": s.misses,
        "compiles": s.compiles,
        "compile_failures": s.compile_failures,
    }


def test_chained_relu_sigmoid_tanh():
    @bafe.jit
    def f(X):
        return bafe.tanh(bafe.sigmoid(bafe.relu(X)))

    X = np.random.randn(32, 32).astype(np.float32) * 2 - 1
    out = f(X)
    ref = np.tanh(1.0 / (1.0 + np.exp(-np.maximum(X, 0)))).astype(np.float32)
    _check_close(out, ref, atol=1e-4)


def test_bias_add_op():
    @bafe.jit
    def f(X, bias):
        return bafe.bias_add(X, bias)

    X = np.random.randn(16, 32).astype(np.float32)
    bias = np.random.randn(32).astype(np.float32)
    out = f(X, bias)
    ref = (X + bias).astype(np.float32)
    _check_close(out, ref)
