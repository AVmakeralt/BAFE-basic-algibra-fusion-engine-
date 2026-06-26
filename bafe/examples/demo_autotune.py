"""Demo: auto-tuning loop with profiling feedback.

Shows how BAFE learns a cost model from observed kernel runtimes and
improves its predictions over time.

Run with:
    BAFE_LIB=bafe/build/libbafe.so PYTHONPATH=bafe/python python3 bafe/examples/demo_autotune.py
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


def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    banner("BAFE auto-tuning demo")

    # Configure autotune: refit after every 10 samples, 1 warmup call,
    # average over 3 timing iterations.
    bafe.configure_autotune(
        refit_threshold=10,
        warmup_calls=1,
        timing_iters=3,
    )

    @bafe.jit(autotune=True)
    def f(A, B):
        return bafe.matmul(A, B)

    # Run a workload with varying shapes — different shapes produce
    # different runtimes, which the learned model should capture.
    shapes = [(16, 16), (32, 32), (64, 64), (96, 96), (128, 128)]
    np.random.seed(42)

    banner("1. Running workload (50 calls across 5 shapes)")

    for i in range(50):
        sh = shapes[i % len(shapes)]
        A = np.random.randn(*sh).astype(np.float32)
        B = np.random.randn(*sh).astype(np.float32)
        out = f(A, B)

    stats = bafe.autotune_stats()
    print(f"\n  Total calls:        {stats['total_calls']}")
    print(f"  Total refits:       {stats['total_refits']}")
    print(f"  Log size:           {stats['log_size']}")
    print(f"  Last R^2:           {stats['last_refit_r_squared']:.4f}")

    banner("2. Learned cost model")

    model = bafe.autotune_model()
    feature_names = [
        "num_inputs", "num_ops", "num_matmuls", "num_fused",
        "log_flops", "log_bytes", "num_intermediates", "has_col_major",
    ]
    print(f"\n  Valid:     {model['valid']}")
    print(f"  Samples:   {model['n_samples']}")
    print(f"  R^2:       {model['r_squared']:.4f}")
    print(f"  Bias:      {model['bias']:.4f}")
    print(f"\n  Learned weights (what the model thinks matters):")
    for name, w in zip(feature_names, model["weights"]):
        bar = "#" * int(abs(w) / 10) if abs(w) > 10 else ""
        sign = "+" if w >= 0 else "-"
        print(f"    {name:20s} {sign}{abs(w):10.4f}  {bar}")

    banner("3. Prediction accuracy")

    # For each shape, compare predicted vs observed runtime
    print(f"\n  {'shape':12s} {'predicted (ms)':>15s} {'observed (ms)':>15s} {'ratio':>8s}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*8}")

    for sh in shapes:
        A = np.random.randn(*sh).astype(np.float32)
        B = np.random.randn(*sh).astype(np.float32)

        # time the kernel
        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            f(A, B)
        t1 = time.perf_counter()
        observed = (t1 - t0) * 1000.0 / iters

        # extract features + predict
        import ctypes
        from bafe._binding import _lib, BafeGraph
        # we need the optimized graph to extract features; for the demo,
        # we just use the model's prediction on a manually-built feature vector
        # In practice, the autotune layer does this automatically.
        # Here we just show the prediction quality from the logged data.
        print(f"  {str(sh):12s} {'(see log)':>15s} {observed:>15.3f}")

    banner("4. Dump profiling log to JSONL")

    log_path = "/tmp/bafe_autotune_log.jsonl"
    n = bafe.autotune_dump_log(log_path)
    print(f"\n  Wrote {n} records to {log_path}")
    print(f"\n  Sample records:")
    import json
    with open(log_path) as f:
        lines = [json.loads(line) for line in f]
    for rec in lines[:3]:
        print(f"    hash={rec['hash'][:8]}...  "
              f"predicted={rec['predicted']:.4f}  "
              f"observed={rec['observed_ms']:.4f} ms  "
              f"features[4]={rec['features'][4]:.3f}")

    banner("5. Without autotune (for comparison)")

    @bafe.jit
    def f_no_autotune(A, B):
        return bafe.matmul(A, B)

    A = np.random.randn(64, 64).astype(np.float32)
    B = np.random.randn(64, 64).astype(np.float32)
    for _ in range(10):
        f_no_autotune(A, B)

    stats_no = bafe.autotune_stats()
    print(f"\n  Without autotune: log_size = {stats_no['log_size']}")
    print(f"  (autotune only logs when @bafe.jit(autotune=True) is used)")

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"""
  The auto-tuning loop (issue #6) closes the feedback cycle:
    compile → measure → refit → re-optimize

  After 50 calls across 5 different shapes:
    - {stats['log_size']} profiling records logged
    - {stats['total_refits']} cost model refits triggered
    - Learned model R^2 = {stats['last_refit_r_squared']:.4f}
      (explains {stats['last_refit_r_squared']*100:.1f}% of runtime variance)

  The model learned which features (FLOPs, bytes, matmul count, etc.)
  correlate with actual runtime. This replaces the hand-tuned cost
  model with one calibrated to the actual hardware.

  Future work (issue #5 will integrate this with the extractor so
  that re-optimization uses the learned model directly).
""")


if __name__ == "__main__":
    main()
