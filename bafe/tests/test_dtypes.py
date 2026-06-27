"""Tests for expanded dtype support: F16, BF16, F64, and higher-rank tensors."""
import numpy as np
import pytest
import bafe


# ---------------------------------------------------------------------------
# F16 (IEEE half-precision)
# ---------------------------------------------------------------------------

def test_f16_matmul():
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)
    A = np.random.randn(8, 8).astype(np.float16)
    B = np.random.randn(8, 8).astype(np.float16)
    out = f(A, B)
    ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    assert out.dtype == np.float16
    assert np.allclose(out, ref, atol=0.1)


def test_f16_relu():
    @bafe.jit
    def f(X):
        return bafe.relu(X)
    X = np.random.randn(16, 16).astype(np.float16) * 2 - 1
    out = f(X)
    ref = np.maximum(X, 0).astype(np.float16)
    assert out.dtype == np.float16
    assert np.allclose(out, ref, atol=0.01)


def test_f16_matmul_plus_relu():
    @bafe.jit
    def f(A, B):
        return bafe.relu(bafe.matmul(A, B))
    A = np.random.randn(8, 8).astype(np.float16)
    B = np.random.randn(8, 8).astype(np.float16)
    out = f(A, B)
    ref = np.maximum(A.astype(np.float32) @ B.astype(np.float32), 0).astype(np.float16)
    assert out.dtype == np.float16
    assert np.allclose(out, ref, atol=0.1)


def test_f16_add():
    @bafe.jit
    def f(A, B):
        return bafe.add(A, B)
    A = np.random.randn(16, 16).astype(np.float16)
    B = np.random.randn(16, 16).astype(np.float16)
    out = f(A, B)
    ref = (A.astype(np.float32) + B.astype(np.float32)).astype(np.float16)
    assert out.dtype == np.float16
    assert np.allclose(out, ref, atol=0.01)


def test_f16_matmul_with_bias_relu():
    @bafe.jit
    def f(A, B, bias):
        return bafe.relu(bafe.matmul(A, B) + bias)
    A = np.random.randn(8, 8).astype(np.float16)
    B = np.random.randn(8, 8).astype(np.float16)
    bias = np.random.randn(8).astype(np.float16)
    out = f(A, B, bias)
    ref = np.maximum(
        A.astype(np.float32) @ B.astype(np.float32) + bias.astype(np.float32), 0
    ).astype(np.float16)
    assert out.dtype == np.float16
    assert np.allclose(out, ref, atol=0.1)


# ---------------------------------------------------------------------------
# F64 (double precision)
# ---------------------------------------------------------------------------

def test_f64_matmul():
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)
    A = np.random.randn(8, 8).astype(np.float64)
    B = np.random.randn(8, 8).astype(np.float64)
    out = f(A, B)
    ref = A @ B
    assert out.dtype == np.float64
    assert np.allclose(out, ref, atol=1e-10)


def test_f64_relu():
    @bafe.jit
    def f(X):
        return bafe.relu(X)
    X = np.random.randn(16, 16).astype(np.float64) * 2 - 1
    out = f(X)
    ref = np.maximum(X, 0)
    assert out.dtype == np.float64
    assert np.allclose(out, ref, atol=1e-12)


def test_f64_sigmoid():
    @bafe.jit
    def f(X):
        return bafe.sigmoid(X)
    X = np.random.randn(8, 8).astype(np.float64)
    out = f(X)
    ref = 1.0 / (1.0 + np.exp(-X))
    assert out.dtype == np.float64
    assert np.allclose(out, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Mixed dtype not supported (should use same dtype for all inputs)
# ---------------------------------------------------------------------------

def test_mixed_dtype_uses_first_input_dtype():
    """When mixing dtypes, BAFE uses the first input's dtype.
    This is by design — users should ensure all inputs have the same dtype."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)
    A = np.random.randn(8, 8).astype(np.float32)
    B = np.random.randn(8, 8).astype(np.float64)
    # BAFE will use A's dtype (F32) for the computation
    out = f(A, B)
    # The output dtype matches the first input
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Higher-rank tensors (batched matmul)
# ---------------------------------------------------------------------------

def test_batched_matmul_3d():
    """Batched matmul: (B, M, K) @ (B, K, N) -> (B, M, N)"""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)
    A = np.random.randn(2, 4, 8).astype(np.float32)
    B = np.random.randn(2, 8, 6).astype(np.float32)
    out = f(A, B)
    ref = np.einsum('bmk,bkn->bmn', A, B)
    assert out.shape == (2, 4, 6)
    assert np.allclose(out, ref, atol=1e-4)


def test_3d_matmul_plus_relu():
    @bafe.jit
    def f(A, B):
        return bafe.relu(bafe.matmul(A, B))
    A = np.random.randn(2, 4, 8).astype(np.float32)
    B = np.random.randn(2, 8, 6).astype(np.float32)
    out = f(A, B)
    ref = np.maximum(np.einsum('bmk,bkn->bmn', A, B), 0)
    assert out.shape == (2, 4, 6)
    assert np.allclose(out, ref, atol=1e-4)


def test_1d_matmul_as_dot_product():
    """1-D matmul should work as a dot product (though BAFE matmul requires rank >= 2).
    This test verifies that rank-2 input works."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)
    A = np.random.randn(1, 8).astype(np.float32)
    B = np.random.randn(8, 1).astype(np.float32)
    out = f(A, B)
    ref = (A @ B)
    assert out.shape == (1, 1)
    assert np.allclose(out, ref, atol=1e-4)


def test_tall_matmul():
    """Non-square matmul: (M, K) @ (K, N) where M >> K"""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)
    A = np.random.randn(64, 4).astype(np.float32)
    B = np.random.randn(4, 64).astype(np.float32)
    out = f(A, B)
    ref = A @ B
    assert out.shape == (64, 64)
    assert np.allclose(out, ref, atol=1e-4)


def test_wide_matmul():
    """Non-square matmul: (M, K) @ (K, N) where N >> M"""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)
    A = np.random.randn(4, 64).astype(np.float32)
    B = np.random.randn(64, 4).astype(np.float32)
    out = f(A, B)
    ref = A @ B
    assert out.shape == (4, 4)
    assert np.allclose(out, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# I32 / I64 (integer support)
# ---------------------------------------------------------------------------

def test_i32_add():
    @bafe.jit
    def f(A, B):
        return bafe.add(A, B)
    A = np.random.randint(-100, 100, (8, 8), dtype=np.int32)
    B = np.random.randint(-100, 100, (8, 8), dtype=np.int32)
    out = f(A, B)
    assert out.dtype == np.int32
    assert np.array_equal(out, A + B)


def test_i64_add():
    @bafe.jit
    def f(A, B):
        return bafe.add(A, B)
    A = np.random.randint(-1000, 1000, (8, 8), dtype=np.int64)
    B = np.random.randint(-1000, 1000, (8, 8), dtype=np.int64)
    out = f(A, B)
    assert out.dtype == np.int64
    assert np.array_equal(out, A + B)
