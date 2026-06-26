"""Tests for the rewrite engine: rules and alternatives."""
import pytest
import ctypes
from bafe._binding import _lib, BafeGraph, BafeOpAttrs, make_shape

import ctypes.util
from ctypes import Structure, c_int, c_int32, c_char_p, c_size_t, POINTER, byref

BAFE_MAX_CHILDREN = 4
BAFE_MAX_ALTERNATIVES = 512


class BafeAlternative(Structure):
    _fields_ = [
        ("original_node_id", c_int32),
        ("op_name", c_char_p),
        ("attrs", BafeOpAttrs),
        ("n_children", c_int),
        ("children", c_int32 * BAFE_MAX_CHILDREN),
    ]


class BafeAltList(Structure):
    _fields_ = [
        ("items", BafeAlternative * BAFE_MAX_ALTERNATIVES),
        ("n", c_int),
    ]


_lib.bafe_rewrite_find.argtypes = [ctypes.POINTER(BafeGraph), ctypes.POINTER(BafeAltList)]
_lib.bafe_rewrite_find.restype = c_int
_lib.bafe_rewrite_default_count.argtypes = []
_lib.bafe_rewrite_default_count.restype = c_int


def make_graph():
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    return g


def find_alts(g):
    alts = BafeAltList()
    n = _lib.bafe_rewrite_find(ctypes.byref(g), ctypes.byref(alts))
    return alts, n


def test_default_rule_count():
    """We should have at least 12 rules registered."""
    n = _lib.bafe_rewrite_default_count()
    assert n >= 12


def test_no_alts_for_input_only():
    g = make_graph()
    sh = make_shape([4, 4])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    alts, n = find_alts(g)
    assert n == 0


def test_add_commutative_fires():
    g = make_graph()
    sh = make_shape([4, 4])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh), 0)
    ad = _lib.bafe_graph_add_op(ctypes.byref(g), a, b)

    alts, n = find_alts(g)
    assert n >= 1
    # the alt should target the add node
    found = False
    for i in range(n):
        if alts.items[i].original_node_id == ad:
            assert alts.items[i].op_name == b"add"
            # commuted children
            assert alts.items[i].children[0] == b
            assert alts.items[i].children[1] == a
            found = True
    assert found


def test_fuse_matmul_relu_fires():
    g = make_graph()
    sh = make_shape([4, 4])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    r = _lib.bafe_graph_relu(ctypes.byref(g), mm)

    alts, n = find_alts(g)
    # should find at least: add_commutative (no, no add here), fuse_matmul_relu
    found_fuse = False
    for i in range(n):
        if alts.items[i].op_name == b"fused_matmul_relu":
            assert alts.items[i].original_node_id == r
            assert alts.items[i].children[0] == a
            assert alts.items[i].children[1] == b
            found_fuse = True
    assert found_fuse


def test_fuse_matmul_bias_fires_only_for_rank1():
    """fused_matmul_bias should only fire when the added term is rank-1."""
    g = make_graph()
    sh = make_shape([4, 4])
    sh_bias = make_shape([4])  # rank-1
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh), 0)
    bias = _lib.bafe_graph_add_input(ctypes.byref(g), b"bias", ctypes.byref(sh_bias), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    ad = _lib.bafe_graph_add_op(ctypes.byref(g), mm, bias)

    alts, n = find_alts(g)
    found = False
    for i in range(n):
        if alts.items[i].op_name == b"fused_matmul_bias":
            assert alts.items[i].children[0] == a
            assert alts.items[i].children[1] == b
            assert alts.items[i].children[2] == bias
            found = True
    assert found, "fused_matmul_bias should fire for rank-1 bias"


def test_fuse_matmul_bias_does_NOT_fire_for_rank2():
    """When the added term is rank-2 (not a bias vector), fusion should NOT fire."""
    g = make_graph()
    sh = make_shape([4, 4])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh), 0)
    c = _lib.bafe_graph_add_input(ctypes.byref(g), b"C", ctypes.byref(sh), 0)  # rank-2!
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    ad = _lib.bafe_graph_add_op(ctypes.byref(g), mm, c)

    alts, n = find_alts(g)
    for i in range(n):
        assert alts.items[i].op_name != b"fused_matmul_bias", \
            "fused_matmul_bias must not fire for rank-2 addend"


def test_fuse_matmul_bias_relu_fires():
    """relu(add(matmul(A, B), bias)) -> fused_matmul_bias_relu(A, B, bias)."""
    g = make_graph()
    sh = make_shape([4, 4])
    sh_bias = make_shape([4])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh), 0)
    bias = _lib.bafe_graph_add_input(ctypes.byref(g), b"bias", ctypes.byref(sh_bias), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    ad = _lib.bafe_graph_add_op(ctypes.byref(g), mm, bias)
    r = _lib.bafe_graph_relu(ctypes.byref(g), ad)

    alts, n = find_alts(g)
    found = False
    for i in range(n):
        if alts.items[i].op_name == b"fused_matmul_bias_relu":
            assert alts.items[i].original_node_id == r
            assert alts.items[i].children[0] == a
            assert alts.items[i].children[1] == b
            assert alts.items[i].children[2] == bias
            found = True
    assert found
