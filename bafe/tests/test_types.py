"""Tests for IR types: Dtype, Shape, Layout."""
import pytest
from bafe._binding import _lib, BafeShape, make_shape
import ctypes


def test_dtype_c_names():
    assert _lib.bafe_dtype_c_name(0) == b"float"
    assert _lib.bafe_dtype_c_name(1) == b"double"
    assert _lib.bafe_dtype_c_name(2) == b"int32_t"
    assert _lib.bafe_dtype_c_name(3) == b"int64_t"


def test_dtype_numpy_names():
    assert _lib.bafe_dtype_numpy_name(0) == b"float32"
    assert _lib.bafe_dtype_numpy_name(1) == b"float64"
    assert _lib.bafe_dtype_numpy_name(2) == b"int32"
    assert _lib.bafe_dtype_numpy_name(3) == b"int64"


def test_dtype_byte_size():
    assert _lib.bafe_dtype_byte_size(0) == 4
    assert _lib.bafe_dtype_byte_size(1) == 8
    assert _lib.bafe_dtype_byte_size(2) == 4
    assert _lib.bafe_dtype_byte_size(3) == 8


def test_dtype_from_str():
    assert _lib.bafe_dtype_from_str(b"f32") == 0
    assert _lib.bafe_dtype_from_str(b"float64") == 1
    assert _lib.bafe_dtype_from_str(b"i32") == 2
    assert _lib.bafe_dtype_from_str(b"int64") == 3
    assert _lib.bafe_dtype_from_str(b"unknown") == 0  # default


def test_shape_make():
    s = make_shape([3, 4, 5])
    assert s.rank == 3
    assert s.dims[0] == 3
    assert s.dims[1] == 4
    assert s.dims[2] == 5


def test_shape_scalar():
    s = make_shape([])
    assert s.rank == 0
    assert _lib.bafe_shape_is_scalar(ctypes.byref(s))


def test_shape_numel():
    s = make_shape([3, 4])
    assert _lib.bafe_shape_numel(ctypes.byref(s)) == 12
    s = make_shape([2, 3, 4])
    assert _lib.bafe_shape_numel(ctypes.byref(s)) == 24
    s = make_shape([])
    assert _lib.bafe_shape_numel(ctypes.byref(s)) == 1


def test_shape_broadcast():
    a = make_shape([1, 4])
    b = make_shape([3, 1])
    c = _lib.bafe_shape_broadcast(ctypes.byref(a), ctypes.byref(b))
    assert c.rank == 2
    assert c.dims[0] == 3
    assert c.dims[1] == 4


def test_shape_eq():
    a = make_shape([3, 4])
    b = make_shape([3, 4])
    c = make_shape([4, 3])
    assert _lib.bafe_shape_eq(ctypes.byref(a), ctypes.byref(b))
    assert not _lib.bafe_shape_eq(ctypes.byref(a), ctypes.byref(c))


def test_layout_name():
    assert _lib.bafe_layout_name(0) == b"row"
    assert _lib.bafe_layout_name(1) == b"col"
    assert _lib.bafe_layout_name(2) == b"blocked"
    assert _lib.bafe_layout_name(3) == b"tc"
