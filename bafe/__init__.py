"""BAFE - Basic Algebra Fusion Engine.

Public API is exported here. During early development this file is rebuilt
as new modules land.
"""

from bafe.ir.types import Dtype, Shape, Layout
from bafe.ir.graph import Graph, Node
from bafe.ir.ops import get_op, all_ops, is_fused, DEFAULT_DTYPE

__version__ = "0.1.0"

__all__ = ["Dtype", "Shape", "Layout", "Graph", "Node", "get_op", "all_ops", "is_fused"]
