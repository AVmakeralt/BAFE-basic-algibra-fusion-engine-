"""BAFE - Basic Algebra Fusion Engine.

A domain-specific superoptimizer for tensor math IR.

Public API:
    import bafe
    bafe.matmul, bafe.add, bafe.relu, ...
    bafe.jit  (decorator)
    bafe.Tensor  (symbolic tensor handle)
"""

from bafe.api import (
    Tensor,
    matmul,
    add,
    sub,
    mul,
    relu,
    sigmoid,
    tanh,
    transpose,
    reduce_sum,
    reduce_max,
    reshape,
    broadcast_to,
    scale,
    bias_add,
    jit,
    optimize,  # explicit optimize function for tests
)
from bafe.ir.graph import Graph
from bafe.ir.types import Dtype, Shape, Layout

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "matmul",
    "add",
    "sub",
    "mul",
    "relu",
    "sigmoid",
    "tanh",
    "transpose",
    "reduce_sum",
    "reduce_max",
    "reshape",
    "broadcast_to",
    "scale",
    "bias_add",
    "jit",
    "optimize",
    "Graph",
    "Dtype",
    "Shape",
    "Layout",
]
