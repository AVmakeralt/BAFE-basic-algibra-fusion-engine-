"""Benchmark: row-major vs col-major matmul.

Demonstrates that the layout superoptimizer actually produces different
(and faster) code when given col-major inputs.

Run with:
    BAFE_LIB=bafe/build/libbafe.so PYTHONPATH=bafe/python python3 bafe/examples/bench_layout.py
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("BAFE_LIB", str(ROOT / "bafe" / "build" / "libbafe.so"))

import numpy as np
import bafe


def bench(fn, args, n_warmup=2, n_iters=10):
    """Time a function, returning the median time in ms."""
    for _ in range(n_warmup):
        fn(*args)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]  # median


def main():
    # Use large enough matrices that memory access patterns matter
    M, K, N = 256, 256, 256

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    A_col = np.asfortranarray(A)
    B_col = np.asfortranarray(B)

    print(f"matmul({M}x{K}, {K}x{N}) benchmark")
    print(f"  A: {A.nbytes/1024:.0f} KB, B: {B.nbytes/1024:.0f} KB")
    print()

    # numpy reference
    t_numpy = bench(lambda a, b: a @ b, (A, B))
    print(f"  numpy (row+row):       {t_numpy:.2f} ms")

    # BAFE row+row
    @bafe.jit
    def f_row_row(A, B):
        return bafe.matmul(A, B)

    t_row_row = bench(f_row_row, (A, B))
    print(f"  BAFE row+row:          {t_row_row:.2f} ms")

    # BAFE row+col (B stored col-major)
    @bafe.jit
    def f_row_col(A, B):
        return bafe.matmul(A, B)

    t_row_col = bench(f_row_col, (A, B_col))
    print(f"  BAFE row+col:          {t_row_col:.2f} ms")

    # BAFE col+col (both col-major)
    @bafe.jit
    def f_col_col(A, B):
        return bafe.matmul(A, B)

    t_col_col = bench(f_col_col, (A_col, B_col))
    print(f"  BAFE col+col:          {t_col_col:.2f} ms")

    print()
    print(f"  Speedup row+col vs row+row: {t_row_row/t_row_col:.2f}x")
    print(f"  Speedup col+col vs row+row: {t_row_row/t_col_col:.2f}x")

    # correctness check
    ref = (A @ B).astype(np.float32)
    out_rr = f_row_row(A, B)
    out_rc = f_row_col(A, B_col)
    out_cc = f_col_col(A_col, B_col)
    print()
    print(f"  Correctness:")
    print(f"    row+row max err: {np.max(np.abs(out_rr - ref)):.2e}")
    print(f"    row+col max err: {np.max(np.abs(out_rc - ref)):.2e}")
    print(f"    col+col max err: {np.max(np.abs(out_cc - ref)):.2e}")

    # Show the emitted C code for each variant
    print()
    print("=" * 60)
    print("  Emitted C code (matmul inner loop, row+row variant)")
    print("=" * 60)
    # We can't easily get the source back from the .so, but we can show
    # the cache file path so the user can inspect it.
    cache_dir = os.environ.get("BAFE_CACHE_DIR", os.path.expanduser("~/.bafecache"))
    print(f"  Cache dir: {cache_dir}")
    print(f"  Look at the .c files there to see the layout-aware access patterns.")


if __name__ == "__main__":
    main()
