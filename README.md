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

## Phase 2: Layout superoptimizer

BAFE now treats memory layout as a **first-class variable** that the
optimizer explores. Every IR node carries a `bafe_layout` tag
(`ROW_MAJOR`, `COL_MAJOR`, `BLOCKED`, `TENSOR_CORE`), and the optimizer
uses this tag to:

1. **Pick cache-friendly access patterns** in codegen
   (e.g., `B[k + j*K]` for col-major B instead of `B[k*N + j]`)
2. **Reward layout-compatible fusion** in the cost model
3. **Eliminate redundant transposes** via rewrite rules
   (`transpose(col_major_x) === x` — a free metadata flip)

### Layout-aware matmul benchmark

```
matmul(256x256, 256x256) benchmark

  numpy (row+row):       0.11 ms
  BAFE row+row:         10.09 ms
  BAFE row+col:          6.84 ms   <-- 1.47x faster than row+row

  Correctness:
    row+row max err: 4.96e-05
    row+col max err: 4.96e-05
```

The col-major B variant emits `B_ptr[K*j + k]` (k contiguous in the
inner loop) instead of `B_ptr[k*N + j]` (j contiguous, strided). This
gives a measurable 1.47x speedup on 256x256 matmuls.

### Using layouts from Python

```python
import numpy as np
import bafe

@bafe.jit
def f(A, B):
    return bafe.matmul(A, B)

# Auto-detection: BAFE reads numpy's F_CONTIGUOUS flag
A = np.random.randn(256, 256).astype(np.float32)
B = np.random.randn(256, 256).astype(np.float32)
B_col = np.asfortranarray(B)   # col-major

out = f(A, B_col)   # BAFE compiles a col-major variant automatically

# Explicit tagging (overrides auto-detection):
@bafe.jit
def g(A, B):
    a = bafe.input(A.shape, dtype="float32", name="A", layout="row")
    b = bafe.input(B.shape, dtype="float32", name="B", layout="col")
    return bafe.matmul(a, b)
```

## Phase 2: Stochastic search layer

The deterministic rewrite engine does **one pass** — it applies every
rule to every node once. It never re-applies rules to the *new* nodes
created by previous rewrites, so multi-step transformations are missed.

The stochastic search layer fixes this by doing **multiple passes**:
after each pass, materialized alternatives create new nodes, which the
next pass can match rules against — discovering deeper rewrites.

### Stochastic vs deterministic

```
Graph: relu(add(matmul(A, B), mul(C, D)))

  Deterministic single-pass:  2 alternatives found
  Stochastic (5 iters, T=2):  32 alternatives found  (16x more)
                               62 rewrites materialized
                               graph grew from 8 to 70 nodes
```

### Using stochastic search from Python

```python
import bafe

# Default: deterministic (backward compatible)
@bafe.jit
def f(A, B):
    return bafe.matmul(A, B)

# Stochastic: pass iters/temperature/seed
@bafe.jit(iters=8, temperature=2.0, seed=42)
def f(A, B):
    return bafe.matmul(A, B)

# Full custom budget
budget = bafe.make_search_budget(
    max_iters=16, max_nodes=500, max_rewrites=200,
    time_budget_ms=100, temperature=1.5, seed=42,
)
@bafe.jit(budget=budget)
def f(A, B):
    return bafe.matmul(A, B)
```

### Budget parameters

| Parameter         | Default   | Description                                |
|-------------------|-----------|--------------------------------------------|
| `max_iters`       | 4         | Number of stochastic passes                |
| `max_nodes`       | 256       | Hard cap on graph size during search       |
| `max_rewrites`    | 64        | Cap on total rewrites materialized         |
| `time_budget_ms`  | 0         | Wall-clock limit (0 = no limit)            |
| `temperature`     | 1.0       | 0 = greedy, high = explore randomly        |
| `seed`            | 0xBAFE5EED| PRNG seed for reproducibility              |
| `enable_multi_pass` | true    | If false, degrades to deterministic        |

The temperature controls a Boltzmann acceptance criterion:
`P(materialize) = exp(-cost_delta / T)`. At low T, only cost-reducing
rewrites are materialized. At high T, all rewrites are roughly equally
likely (exploration).

## Phase 3: Auto-tuning loop with profiling feedback

BAFE can now **learn** a cost model from observed kernel runtimes. The
auto-tuning loop closes the feedback cycle: compile → measure → refit →
re-optimize.

### How it works

```
user calls f(A, B) repeatedly
   ↓
JIT compiles kernel (first call) or hits cache
   ↓
autotune layer times the kernel + logs:
   (graph_hash, features, predicted_cost, observed_runtime)
   ↓
after N samples, refits the cost model via linear regression:
   log(runtime_ms) = w · features + b
   ↓
the learned model replaces the hand-tuned one for future extractions
```

### Feature vector (8 features per kernel)

| Index | Feature           | Description                              |
|-------|-------------------|------------------------------------------|
| 0     | `num_inputs`      | Number of input tensors                  |
| 1     | `num_ops`         | Number of non-input/constant nodes       |
| 2     | `num_matmuls`     | Number of matmul ops                     |
| 3     | `num_fused`       | Number of fused ops                      |
| 4     | `log_flops`       | log(total FLOPs + 1)                     |
| 5     | `log_bytes`       | log(total memory traffic + 1)            |
| 6     | `num_intermediates` | Number of materialized intermediates   |
| 7     | `has_col_major`   | 1 if any input is col-major, else 0     |

### Using autotune from Python

```python
import bafe

# Configure the autotune loop
bafe.configure_autotune(
    refit_threshold=10,    # refit after 10 new samples
    warmup_calls=1,        # skip timing for the first call
    timing_iters=3,        # average runtime over 3 invocations
)

@bafe.jit(autotune=True)
def f(A, B):
    return bafe.matmul(A, B)

# Just call f normally — autotune logs + refits in the background
for i in range(50):
    A = np.random.randn(64, 64).astype(np.float32)
    B = np.random.randn(64, 64).astype(np.float32)
    out = f(A, B)

# Inspect the learned model
stats = bafe.autotune_stats()
print(f"R^2 = {stats['last_refit_r_squared']:.4f}")

model = bafe.autotune_model()
print(f"Weights: {model['weights']}")
print(f"Bias: {model['bias']}")

# Dump the profiling log
bafe.autotune_dump_log("/tmp/bafe_log.jsonl")
```

### Demo results

After 50 calls across 5 different shapes (16×16 to 128×128):
- **45 profiling records** logged
- **4 cost model refits** triggered (at log sizes 10, 20, 30, 40)
- **R² = 0.94** — the learned model explains 94% of runtime variance
- The model learned that `log_bytes` has a large negative weight (larger
  tensors amortize per-element overhead) and `log_flops` has a large
  positive weight (more compute = slower)

## Phase 1 scope (this prototype)

- IR with 17 ops (matmul, add, mul, sub, relu, sigmoid, tanh, neg,
  transpose, reduce_sum, reduce_max, reshape, broadcast_to, scale,
  bias_add, layout_transform, and 4 fused forms)
- 16 deterministic rewrite rules (algebra + fusion + layout)
- E-graph with union-find + congruence closure rebuild
- Cost model: FLOPs, memory traffic, intermediate-tensor cost, fusion bonus,
  layout conversion cost, layout-compatible fusion bonus, contiguous access bonus
- DP extractor (min-cost subtree)
- C99 codegen: real nested loops, tiled matmul, **layout-aware access patterns**,
  fused kernels
- JIT cache: SHA-256 keyed (includes layout), compiled with `cc`, `dlopen`'d at runtime
- Python frontend: `@bafe.jit` decorator + functional ops + **layout auto-detection**
  from numpy array flags

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
