"""BAFE demo: show the full pipeline in action.

Run with:
    BAFE_LIB=bafe/build/libbafe.so PYTHONPATH=bafe/python python3 bafe/examples/demo.py
"""
import os
import sys
import time
from pathlib import Path

# Make bafe importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("BAFE_LIB", str(ROOT / "bafe" / "build" / "libbafe.so"))

import numpy as np
import bafe


def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    banner("BAFE demo: relu(matmul(A, B) + bias)")

    @bafe.jit
    def fused_matmul_bias_relu(A, B, bias):
        return bafe.relu(bafe.matmul(A, B) + bias)

    A = np.random.randn(128, 256).astype(np.float32)
    B = np.random.randn(256, 512).astype(np.float32)
    bias = np.random.randn(512).astype(np.float32)

    print(f"\nInputs:")
    print(f"  A:    shape={A.shape}, dtype={A.dtype}")
    print(f"  B:    shape={B.shape}, dtype={B.dtype}")
    print(f"  bias: shape={bias.shape}, dtype={bias.dtype}")

    print(f"\nCalling jitted function (first call triggers full pipeline)...")
    t0 = time.perf_counter()
    out = fused_matmul_bias_relu(A, B, bias)
    t1 = time.perf_counter()
    print(f"  First call: {(t1 - t0) * 1000:.1f} ms (includes optimize + compile)")

    ref = np.maximum(A @ B + bias, 0.0).astype(np.float32)
    print(f"\nResult:")
    print(f"  shape:     {out.shape}")
    print(f"  dtype:     {out.dtype}")
    print(f"  max err:   {np.max(np.abs(out - ref)):.2e}")
    print(f"  allclose:  {np.allclose(out, ref, atol=1e-4)}")

    print(f"\nCalling again (should hit Python-level cache)...")
    t0 = time.perf_counter()
    out2 = fused_matmul_bias_relu(A, B, bias)
    t1 = time.perf_counter()
    print(f"  Second call: {(t1 - t0) * 1000:.3f} ms")

    banner("JIT stats")
    from bafe._binding import _lib
    stats = _lib.bafe_jit_get_stats()
    print(f"  hits:             {stats.hits}")
    print(f"  misses:           {stats.misses}")
    print(f"  compiles:         {stats.compiles}")
    print(f"  compile failures: {stats.compile_failures}")

    banner("Cache directory")
    print(f"  {os.environ.get('BAFE_CACHE_DIR', _lib.bafe_jit_cache_dir().decode())}")

    banner("What just happened (under the hood)")
    print("""
1. @bafe.jit wrapped `fused_matmul_bias_relu`.
2. On first call, BAFE traced the function and built an IR graph:
     n3 = matmul(n0, n1)
     n4 = add(n3, n2)        # bias is rank-1 -> matches fuse_matmul_bias
     n5 = relu(n4)           # matches fuse_matmul_bias_relu
3. The rewrite engine found 3 alternatives:
     - add(matmul, bias) -> add(bias, matmul)   (commutative)
     - add(matmul, bias) -> fused_matmul_bias   (fusion)
     - relu(add(matmul, bias)) -> fused_matmul_bias_relu  (deeper fusion)
4. The e-graph merged these into equivalence classes.
5. The cost model + extractor picked `fused_matmul_bias_relu`
   because it has a fusion bonus (no intermediate materialization).
6. The codegen emitted a single C99 loop nest:
     for (i, j) { acc = sum_k A[i,k]*B[k,j] + bias[j]; out[i,j] = max(0, acc); }
7. The JIT compiled it with `cc -O2` and dlopen'd the .so.
8. ctypes called the kernel directly.
9. The second call hit the Python-level cache (no recompile).
""")


if __name__ == "__main__":
    main()
