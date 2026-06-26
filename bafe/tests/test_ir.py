"""Tests for IR Graph construction."""
import pytest
import ctypes
from bafe._binding import _lib, BafeGraph, make_shape


def make_graph():
    g = BafeGraph()
    _lib.bafe_graph_init(ctypes.byref(g))
    return g


def test_add_input():
    g = make_graph()
    sh = make_shape([32, 32])
    nid = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    assert nid == 0
    assert g.n_nodes == 1
    assert g.n_inputs == 1
    node = g.nodes[0]
    assert node.is_input
    assert node.input_name == b"A"
    assert node.shape.dims[0] == 32
    assert node.shape.dims[1] == 32


def test_matmul_shape_inference():
    g = make_graph()
    sh_a = make_shape([4, 8])
    sh_b = make_shape([8, 16])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh_a), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh_b), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    node = g.nodes[mm]
    assert node.shape.rank == 2
    assert node.shape.dims[0] == 4
    assert node.shape.dims[1] == 16
    assert node.op_name == b"matmul"
    assert node.n_children == 2
    assert node.children[0] == a
    assert node.children[1] == b


def test_unary_op():
    g = make_graph()
    sh = make_shape([4, 4])
    x = _lib.bafe_graph_add_input(ctypes.byref(g), b"X", ctypes.byref(sh), 0)
    r = _lib.bafe_graph_relu(ctypes.byref(g), x)
    node = g.nodes[r]
    assert node.op_name == b"relu"
    assert node.shape.dims[0] == 4
    assert node.shape.dims[1] == 4


def test_set_output():
    g = make_graph()
    sh = make_shape([4, 4])
    x = _lib.bafe_graph_add_input(ctypes.byref(g), b"X", ctypes.byref(sh), 0)
    r = _lib.bafe_graph_relu(ctypes.byref(g), x)
    _lib.bafe_graph_set_output(ctypes.byref(g), r)
    assert g.n_outputs == 1
    assert g.outputs[0] == r


def test_topo_order():
    """Topo order must have children before parents."""
    g = make_graph()
    sh = make_shape([4, 4])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    r = _lib.bafe_graph_relu(ctypes.byref(g), mm)
    _lib.bafe_graph_set_output(ctypes.byref(g), r)

    order = (ctypes.c_int32 * 16)()
    n = _lib.bafe_graph_topo_order(ctypes.byref(g), order, 16)
    assert n == 4

    # children must come before parents
    pos = {order[i]: i for i in range(n)}
    assert pos[a] < pos[mm]
    assert pos[b] < pos[mm]
    assert pos[mm] < pos[r]


def test_summary():
    g = make_graph()
    sh = make_shape([4, 4])
    a = _lib.bafe_graph_add_input(ctypes.byref(g), b"A", ctypes.byref(sh), 0)
    b = _lib.bafe_graph_add_input(ctypes.byref(g), b"B", ctypes.byref(sh), 0)
    mm = _lib.bafe_graph_matmul(ctypes.byref(g), a, b)
    _lib.bafe_graph_set_output(ctypes.byref(g), mm)

    buf = ctypes.create_string_buffer(8192)
    _lib.bafe_graph_summary.argtypes = [ctypes.POINTER(BafeGraph), ctypes.c_char_p, ctypes.c_size_t]
    _lib.bafe_graph_summary.restype = ctypes.c_int
    _lib.bafe_graph_summary(ctypes.byref(g), buf, ctypes.c_size_t(len(buf)))
    s = buf.value.decode()
    assert "nodes=3" in s
    assert "inputs=2" in s
    assert "ops=1" in s
    assert "outputs=1" in s
