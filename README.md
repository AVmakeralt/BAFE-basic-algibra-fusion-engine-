# BAFE - Basic Algebra Fusion Engine

A **domain-specific superoptimizer for tensor math IR**. BAFE explores the
space of equivalent tensor programs via rewrite rules + e-graphs, ranks them
with a hardware-aware cost model, and synthesizes optimized C kernels on
demand via a JIT cache.

This is *not* a compiler in the LLVM sense. It is a **search engine over
equivalent tensor programs** that emits the cheapest one under a real cost
model.

## Pipeline

```
Python API
    |
    v
Tensor IR Graph
    |
    v
Rewrite engine (algebraic + fusion rules)
    |
    v
E-graph (compresses equivalence explosion)
    |
    v
Cost model (FLOPs + memory traffic + fusion savings)
    |
    v
Extractor (DP min-cost program selection)
    |
    v
C kernel synthesis
    |
    v
JIT cache (hash -> compiled .so)
    |
    v
Execution
```

## Phase 1 scope (this prototype)

* IR with 16 ops (matmul, add, mul, sub, relu, sigmoid, tanh, broadcast,
  transpose, reduce_sum, reduce_max, reshape, broadcast_to, scale, bias_add,
  fused_matmul_relu, fused_matmul_bias)
* Deterministic rewrite rules (associativity, commutativity, identities,
  distributivity, fusion, transpose-matmul, etc.)
* E-graph with union-find + rebuild
* Cost model: FLOPs, memory traffic, intermediate-tensor cost, fusion bonus
* DP extractor (min-cost subtree)
* C backend: emits real compilable C99 with nested loops and tiled matmul
* JIT cache: SHA-256 keyed, compiles with `cc`, `dlopen`s the result
* Python frontend: `@bafe.jit` decorator + functional ops

## Phase 2 (planned)

* Stochastic search layer (chaos generator on top of e-graph)
* Layout search (row-major / col-major / blocked)
* GPU backend via Vulkan compute / SPIR-V
* Learned cost model

## Phase 3 (planned)

* Auto-tuning loop (profiling feedback)
* SMT sanity proofs (optional)
* Multi-kernel fusion across call boundaries

## Quick start

```python
import numpy as np
import bafe

@bafe.jit
def f(A, B, C):
    return bafe.relu(bafe.matmul(A, B) + C)

A = np.random.randn(64, 64).astype(np.float32)
B = np.random.randn(64, 64).astype(np.float32)
C = np.random.randn(64, 64).astype(np.float32)

out = f(A, B, C)   # triggers full pipeline on first call, cached after
ref = np.maximum(A @ B + C, 0.0).astype(np.float32)
assert np.allclose(out, ref, atol=1e-4)
```

## Tests

```
pip install -e ".[test]"
pytest
```

## Status

Phase 1 prototype. Real IR, real rewrites, real e-graph, real C output, real
JIT cache. Nothing is stubbed.

## License

Apache-2.0
