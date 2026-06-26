# BAFE - Basic Algebra Fusion Engine

A **domain-specific superoptimizer for tensor math IR**, implemented in C
with a thin Python binding for the frontend.

BAFE explores the space of equivalent tensor programs via rewrite rules
and an e-graph, ranks them with a hardware-aware cost model, and
synthesizes optimized C kernels on demand via a JIT cache.

This is *not* a compiler in the LLVM sense. It is a **search engine over
equivalent tensor programs** that emits the cheapest one under a real
cost model.

## Why C?

The optimizer itself is in C. Python is only a thin `ctypes` binding so
users can write:

```python
import bafe

@bafe.jit
def f(A, B, C):
    return bafe.relu(bafe.matmul(A, B) + C)
```

The IR, rewrite engine, e-graph, cost model, extractor, C codegen, and
JIT cache all live in `libbafe.so`. Python is just a steering wheel.

## Pipeline

```
Python API  ─►  libbafe.so  ─►  C kernel (.so)  ─►  execution
                   │
                   ├─ IR Graph
                   ├─ Rewrite engine (algebraic + fusion rules)
                   ├─ E-graph (congruence closure)
                   ├─ Cost model (FLOPs + memory + fusion bonus)
                   ├─ Extractor (DP min-cost)
                   ├─ Codegen (emits C99 source)
                   └─ JIT cache (cc compile, dlopen)
```

## Build

```
make            # builds libbafe.so
make test       # runs pytest against the .so
make clean
```

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

out = f(A, B, C)
ref = np.maximum(A @ B + C, 0.0).astype(np.float32)
assert np.allclose(out, ref, atol=1e-4)
```

## Phase 1 scope (this prototype)

- IR with 16 ops (matmul, add, mul, sub, relu, sigmoid, tanh, neg,
  transpose, reduce_sum, reduce_max, reshape, broadcast_to, scale,
  bias_add, and 4 fused forms)
- 13 deterministic rewrite rules (algebra + fusion)
- E-graph with union-find + congruence closure rebuild
- Cost model: FLOPs, memory traffic, intermediate-tensor cost, fusion bonus
- DP extractor (min-cost subtree)
- C99 codegen: real nested loops, tiled matmul, fused kernels
- JIT cache: SHA-256 keyed, compiled with `cc`, `dlopen`'d at runtime
- Python frontend: `@bafe.jit` decorator + functional ops

## Phase 2 (planned)

- Stochastic search layer
- Layout superoptimizer (row/col/blocked/tensor-core)
- GPU backend via Vulkan compute / SPIR-V
- Learned cost model
- Multi-tier pruning with time budget

## Phase 3 (planned)

- Auto-tuning loop with profiling feedback
- SMT sanity proofs (optional)
- Multi-kernel fusion across call boundaries

## Layout

```
bafe/
├── include/bafe/     public C API
├── src/              C implementation
├── python/bafe/      ctypes binding + public Python API
├── tests/            pytest tests (drive the C library)
├── examples/         standalone demo scripts
├── Makefile
└── pyproject.toml
```

## License

Apache-2.0
