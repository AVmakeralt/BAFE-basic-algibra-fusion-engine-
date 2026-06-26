"""Shared test fixtures: ensure libbafe.so is built and importable."""
import os
import sys
import subprocess
from pathlib import Path

import pytest

# Make sure the bafe package is importable
BAFE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BAFE_ROOT / "python"))

# Make sure libbafe.so is built
LIB = BAFE_ROOT / "build" / "libbafe.so"
if not LIB.exists():
    subprocess.run(["make"], cwd=BAFE_ROOT, check=True)
os.environ["BAFE_LIB"] = str(LIB)

# Use a per-test-run cache dir to avoid stale cache issues
CACHE_DIR = BAFE_ROOT / ".bafecache_test"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["BAFE_CACHE_DIR"] = str(CACHE_DIR)


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    """Clear the in-memory JIT cache between tests so each test recompiles."""
    import bafe._binding as b
    b._lib.bafe_jit_clear()
    yield


import bafe  # noqa: E402  (import after sys.path setup)
