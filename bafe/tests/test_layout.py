"""Tests for Phase 2 layout superoptimizer."""
import ctypes
import pytest
import numpy as np

import bafe
from bafe._binding import (
    _lib, BafeGraph, make_shape, BafeOpAttrs,
    BafeAlternative, BafeAltList, BafeCostModel,
)


# ---------------------------------------------------------------------------
# Layout field and propagation
# ---------------------------------------------------------------------------

def test_input_default_layout_is_row_major():
    """Inputs without explicit layout should default to ROW_MAJOR."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([4, 4])
    nid = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    assert _lib.bafe_graph_get_node_layout(ctypes.byref(g), nid) == 0  # ROW_MAJOR


def test_input_col_major_layout():
    """Inputs can be tagged as COL_MAJOR."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([4, 4])
    nid = _lib.bafe_graph_add_input_with_layout(
        ctypes.byref(g), b"A", ctypes.byref(sh), 0, 1  # COL_MAJOR
    )
    assert _lib.bafe_graph_get_node_layout(ctypes.byref(g), nid) == 1  # COL_MAJOR


def test_set_node_layout():
    """bafe_graph_set_node_layout should change a node's layout."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([4, 4])
    nid = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    assert _lib.bafe_graph_set_node_layout(ctypes.byref(g), nid, 1) == 0
    assert _lib.bafe_graph_get_node_layout(ctypes.byref(g), nid) == 1


def test_layout_propagates_through_relu():
    """relu(x) should inherit x's layout."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([4, 4])
    x = _lib.bafe_graph_add_input_with_layout(
        ctypes.byref(g), b"X", ctypes.byref(sh), 0, 1  # COL_MAJOR
    )
    r = _lib.bafe_graph_relu(ctypes.byref(g), x)
    assert _lib.bafe_graph_get_node_layout(ctypes.byref(g), r) == 1  # inherited COL_MAJOR


def test_layout_propagates_through_add():
    """add(a, b) should inherit a's layout (first child)."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([4, 4])
    a = _lib.bafe_graph_add_input_with_layout(
        ctypes.byref(g), b"A", ctypes.byref(sh), 0, 1  # COL_MAJOR
    )
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh), 0)  # ROW_MAJOR
    s = _lib.bafe_graph_add_op(ctypes.byref(g), a, b)
    assert _lib.bafe_graph_get_node_layout(ctypes.byref(g), s) == 1  # inherited from A


def test_transpose_flips_layout_row_to_col():
    """transpose(x, (1,0)) on a row-major rank-2 tensor -> col-major output."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([4, 6])
    x = _lib.bafe_graph_add_input(ctypes.byref(g), b"X", ctypes.byref(sh), 0)  # ROW_MAJOR
    perm = (ctypes.c_int32 * 2)(1, 0)
    t = _lib.bafe_graph_transpose(ctypes.byref(g), x, perm, 2)
    assert _lib.bafe_graph_get_node_layout(ctypes.byref(g), t) == 1  # COL_MAJOR


def test_transpose_flips_layout_col_to_row():
    """transpose(x, (1,0)) on a col-major rank-2 tensor -> row-major output."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([4, 6])
    x = _lib.bafe_graph_add_input_with_layout(
        ctypes.byref(g), b"X", ctypes.byref(sh), 0, 1  # COL_MAJOR
    )
    perm = (ctypes.c_int32 * 2)(1, 0)
    t = _lib.bafe_graph_transpose(ctypes.byref(g), x, perm, 2)
    assert _lib.bafe_graph_get_node_layout(ctypes.byref(g), t) == 0  # ROW_MAJOR


# ---------------------------------------------------------------------------
# Layout rewrite rules
# ---------------------------------------------------------------------------

from ctypes import POINTER

BAFE_MAX_CHILDREN = 4


def _find_alts(g):
    alts = BafeAltList()
    n = _lib.bafe_rewrite_find(ctypes.byref(g), ctypes.byref(alts))
    return alts, n


def test_layout_rules_registered():
    """Phase 2 should add at least 4 new layout rules (total >= 16)."""
    n = _lib.bafe_rewrite_default_count()
    assert n >= 16, f"expected >= 16 rules after Phase 2, got {n}"


def test_free_transpose_col_to_row_fires():
    """transpose(x, (1,0)) on col-major x should trigger free_transpose_col_to_row."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([4, 4])
    x = _lib.bafe_graph_add_input_with_layout(
        ctypes.byref(g), b"X", ctypes.byref(sh), 0, 1  # COL_MAJOR
    )
    perm = (ctypes.c_int32 * 2)(1, 0)
    t = _lib.bafe_graph_transpose(ctypes.byref(g), x, perm, 2)

    alts, n = _find_alts(g)
    found = False
    for i in range(n):
        if alts.items[i].op_name == b"layout_transform":
            # check the target layout is "row"
            assert alts.items[i].attrs.name == b"row"
            assert alts.items[i].children[0] == x
            found = True
    assert found, "free_transpose_col_to_row should fire for col-major transpose"


def test_free_transpose_row_to_col_fires():
    """transpose(x, (1,0)) on row-major x should trigger free_transpose_row_to_col."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([4, 4])
    x = _lib.bafe_graph_add_input(ctypes.byref(g), b"X", ctypes.byref(sh), 0)  # ROW_MAJOR
    perm = (ctypes.c_int32 * 2)(1, 0)
    t = _lib.bafe_graph_transpose(ctypes.byref(g), x, perm, 2)

    alts, n = _find_alts(g)
    found = False
    for i in range(n):
        if alts.items[i].op_name == b"layout_transform":
            assert alts.items[i].attrs.name == b"col"
            assert alts.items[i].children[0] == x
            found = True
    assert found, "free_transpose_row_to_col should fire for row-major transpose"


def test_free_transpose_does_not_fire_for_non_2d_perm():
    """Only the rank-2 (1,0) transpose should trigger free transpose rules."""
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    sh = make_shape([2, 3, 4])
    x = _lib.bafe_graph_add_input_with_layout(
        ctypes.byref(g), b"X", ctypes.byref(sh), 0, 1  # COL_MAJOR
    )
    perm = (ctypes.c_int32 * 3)(2, 0, 1)  # 3D transpose
    t = _lib.bafe_graph_transpose(ctypes.byref(g), x, perm, 3)

    alts, n = _find_alts(g)
    for i in range(n):
        # the free_transpose rules should NOT fire for 3D
        if alts.items[i].op_name == b"layout_transform":
            pytest.fail("free_transpose rules should not fire for 3D transpose")


# ---------------------------------------------------------------------------
# Cost model with layouts
# ---------------------------------------------------------------------------

def test_cost_model_has_layout_weights():
    """The cost model should give a LOWER cost for row+col matmul (cache-friendly)
    than row+row matmul (strided access for B)."""
    cm = _lib.bafe_cost_model_default()

    # row + row matmul
    g_row = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g_row))
    sh = make_shape([16, 16])
    a = _lib.bafe_graph_add_input(ctypes.byref(g_row), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g_row), b"B", ctypes.byref(sh), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g_row), a, b)
    _lib.bafe_graph_set_output(ctypes.byref(g_row), mm)
    cost_row_row = _lib.bafe_cost_graph(ctypes.byref(cm), ctypes.byref(g_row))

    # row + col matmul
    g_col = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g_col))
    a2 = _lib.bafe_graph_add_input(ctypes.byref(g_col), b"A", ctypes.byref(sh), 0)
    b2 = _lib.bafe_graph_add_input_with_layout(
        ctypes.byref(g_col), b"B", ctypes.byref(sh), 0, 1  # COL_MAJOR
    )
    mm2 = _lib.bafe_graph_matmul(ctypes.byref(g_col), a2, b2)
    _lib.bafe_graph_set_output(ctypes.byref(g_col), mm2)
    cost_row_col = _lib.bafe_cost_graph(ctypes.byref(cm), ctypes.byref(g_col))

    assert cost_row_col < cost_row_row, \
        f"row+col matmul ({cost_row_col}) should be cheaper than row+row ({cost_row_row})"


# ---------------------------------------------------------------------------
# End-to-end with layouts
# ---------------------------------------------------------------------------

def test_matmul_with_col_major_B():
    """matmul(A, B) where B is col-major should give correct results."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    # make B col-major (Fortran order)
    B_col = np.asfortranarray(B)

    out = f(A, B_col)
    ref = (A @ B).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4), \
        f"max err = {np.max(np.abs(out - ref))}"


def test_matmul_with_both_col_major():
    """matmul(A, B) where both inputs are col-major should still work.

    Note: when both inputs are col-major, the codegen emits col-major
    access for A too. This may be slower than row+col but must be correct.
    """
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    A_col = np.asfortranarray(A)
    B_col = np.asfortranarray(B)

    out = f(A_col, B_col)
    ref = (A @ B).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4), \
        f"max err = {np.max(np.abs(out - ref))}"


def test_explicit_layout_param_on_input():
    """bafe.input() should accept a layout= parameter.

    When the user explicitly tags an input as col-major, they MUST pass
    col-major data (numpy Fortran-contiguous). The kernel will be compiled
    to read col-major strides.
    """
    @bafe.jit
    def f(A, B):
        # User can explicitly tag inputs with layouts (overrides auto-detection)
        a = bafe.input(A.shape, dtype="float32", name="A", layout="row")
        b = bafe.input(B.shape, dtype="float32", name="B", layout="col")
        return bafe.matmul(a, b)

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    B_col = np.asfortranarray(B)  # match the explicit "col" layout

    out = f(A, B_col)
    ref = (A @ B).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4)


def test_invalid_layout_raises():
    """Passing an invalid layout string should raise ValueError."""
    @bafe.jit
    def f(A):
        return bafe.input(A.shape, dtype="float32", name="A", layout="invalid")

    A = np.random.randn(4, 4).astype(np.float32)
    with pytest.raises(ValueError, match="unsupported layout"):
        f(A)


def test_layout_autodetect_from_numpy_flags():
    """When the user passes a Fortran-contiguous array, BAFE should auto-tag it as col-major."""
    @bafe.jit
    def f(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 24).astype(np.float32)
    B_f = np.asfortranarray(B)

    # This should not raise, and should produce correct output.
    # The auto-detection happens in _compile().
    out = f(A, B_f)
    ref = (A @ B).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-4)
