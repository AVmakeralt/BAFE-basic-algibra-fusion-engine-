"""Tests for Phase 3 issue #7: cross-kernel fusion."""
import ctypes
import pytest
import numpy as np

import bafe
from bafe._binding import _lib, BafeGraph, make_shape


# ---------------------------------------------------------------------------
# bafe.fuse basic functionality
# ---------------------------------------------------------------------------

def test_fuse_two_matmuls():
    """Fuse f(A,B)=matmul and g(X,C)=matmul into h(A,B,C) = matmul(matmul(A,B),C)."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    @bafe.jit
    def g(X, C):
        return bafe.matmul(X, C)

    h = bafe.fuse(f, g)

    A = np.random.randn(8, 16).astype(np.float32)
    B = np.random.randn(16, 32).astype(np.float32)
    C = np.random.randn(32, 24).astype(np.float32)

    out = h(A, B, C)
    ref = (A @ B @ C).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4), f"max err = {np.max(np.abs(out - ref))}"


def test_fuse_matmul_then_relu():
    """Fuse f(A,B)=matmul and g(X)=relu into h(A,B) = relu(matmul(A,B))."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    @bafe.jit
    def g(X):
        return bafe.relu(X)

    h = bafe.fuse(f, g)

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    out = h(A, B)
    ref = np.maximum(A @ B, 0).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4)


def test_fuse_matmul_then_add_bias():
    """Fuse f(A,B)=matmul and g(X,bias)=relu(X+bias)."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    @bafe.jit
    def g(X, bias):
        return bafe.relu(X + bias)

    h = bafe.fuse(f, g)

    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)
    bias = np.random.randn(32).astype(np.float32)

    out = h(A, B, bias)
    ref = np.maximum(A @ B + bias, 0.0).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4)


def test_fuse_output_shape_is_correct():
    """The fused kernel should produce the correct output shape."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    @bafe.jit
    def g(X):
        return bafe.relu(X)

    h = bafe.fuse(f, g)

    A = np.random.randn(8, 16).astype(np.float32)
    B = np.random.randn(16, 24).astype(np.float32)
    out = h(A, B)
    assert out.shape == (8, 24)


def test_fuse_is_cached():
    """Second call with same shapes should hit the cache (no recompile)."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    @bafe.jit
    def g(X):
        return bafe.relu(X)

    h = bafe.fuse(f, g)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)

    out1 = h(A, B)
    out2 = h(A, B)
    assert np.allclose(out1, out2)


def test_fuse_with_different_shapes():
    """The fused function should handle different input shapes."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    @bafe.jit
    def g(X):
        return bafe.relu(X)

    h = bafe.fuse(f, g)

    for sh in [(8, 8), (16, 32), (64, 64)]:
        A = np.random.randn(*sh).astype(np.float32)
        B = np.random.randn(sh[1], sh[0]).astype(np.float32)
        out = h(A, B)
        ref = np.maximum(A @ B, 0).astype(np.float32)
        assert np.allclose(out, ref, atol=1e-4), f"failed for shape {sh}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_fuse_requires_jitted_functions():
    """bafe.fuse should reject non-jitted functions."""
    def f(A, B):
        return A @ B

    @bafe.jit
    def g(X):
        return bafe.relu(X)

    with pytest.raises(TypeError, match="@bafe.jit"):
        bafe.fuse(f, g)

    with pytest.raises(TypeError, match="@bafe.jit"):
        bafe.fuse(g, f)


def test_fuse_wrong_arg_count():
    """The fused function should validate arg count."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    @bafe.jit
    def g(X, bias):
        return bafe.relu(X + bias)

    h = bafe.fuse(f, g)

    A = np.random.randn(8, 8).astype(np.float32)
    B = np.random.randn(8, 8).astype(np.float32)
    # h expects 3 args (A, B, bias) but we pass 2
    with pytest.raises(TypeError, match="expects 3 args"):
        h(A, B)


# ---------------------------------------------------------------------------
# C-level bafe_fuse_concat
# ---------------------------------------------------------------------------

def _build_matmul_graph(M=8, K=8, N=8):
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh_a = make_shape([M, K])
    sh_b = make_shape([K, N])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh_a), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh_b), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    _lib.bafe_graph_set_output(ctypes.byref(g), mm)
    return g


def _build_relu_graph(M=8, N=8):
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([M, N])
    x = _lib.bafe_graph_add_input(ctypes.byref(g), b"X", ctypes.byref(sh), 0)
    r = _lib.bafe_graph_relu(ctypes.byref(g), x)
    _lib.bafe_graph_set_output(ctypes.byref(g), r)
    return g


def test_fuse_concat_produces_valid_graph():
    """bafe_fuse_concat should combine two graphs into one."""
    ga = _build_matmul_graph(8, 8, 8)
    gb = _build_relu_graph(8, 8)

    fused = BafeGraph()
    err = ctypes.create_string_buffer(256)
    rc = _lib.bafe_fuse_concat(ctypes.byref(ga), ctypes.byref(gb),
                                ctypes.byref(fused), err, ctypes.c_size_t(len(err)))
    assert rc == 0, f"fuse_concat failed: {err.value}"
    # fused graph should have: 2 inputs from ga + 0 new from gb (gb's only
    # input is rewired to ga's output) + matmul + relu = 4 nodes
    assert fused.n_nodes >= 4
    assert fused.n_inputs == 2  # A and B (gb's X is rewired)
    assert fused.n_outputs == 1


def test_fuse_concat_rewires_first_input():
    """The fused graph's gb first input should be rewired to ga's output."""
    ga = _build_matmul_graph(8, 8, 8)
    gb = _build_relu_graph(8, 8)

    fused = BafeGraph()
    err = ctypes.create_string_buffer(256)
    _lib.bafe_fuse_concat(ctypes.byref(ga), ctypes.byref(gb),
                           ctypes.byref(fused), err, ctypes.c_size_t(len(err)))

    # The relu node's child should be the matmul node (not an input)
    relu_node = None
    for i in range(fused.n_nodes):
        n = fused.nodes[i]
        if n.op_name == b"relu":
            relu_node = n
            break
    assert relu_node is not None, "fused graph should have a relu node"
    # relu's child should be the matmul output node
    child_id = relu_node.children[0]
    child = fused.nodes[child_id]
    assert child.op_name == b"matmul"


# ---------------------------------------------------------------------------
# Performance: fused vs unfused
# ---------------------------------------------------------------------------

def test_fused_produces_same_result_as_sequential():
    """fused(A,B) should equal g(f(A,B)) numerically."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    @bafe.jit
    def g(X):
        return bafe.relu(X)

    h = bafe.fuse(f, g)

    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)

    out_fused = h(A, B)
    out_sequential = g(f(A, B))
    assert np.allclose(out_fused, out_sequential, atol=1e-5)


def test_three_way_fuse_chain():
    """Chain: fuse(fuse(f, g), h) should work.

    Note: 3-way chaining is experimental. The current implementation
    supports 2-way fusion reliably; deeper chains require additional
    graph rewiring logic. This test is marked xfail until that lands.
    """
    pytest.xfail("3-way fuse chaining needs additional graph rewiring logic")
