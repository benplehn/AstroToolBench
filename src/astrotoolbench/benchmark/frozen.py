"""The reviewed v0.1 corpus used for model comparisons."""

import hashlib
import json
from pathlib import Path

from .loader import load_benchmark
from .paths import DEFAULT_TASKS
from .schema import BenchmarkTask

BENCHMARK_TAG = "benchmark-v0.1"
BENCHMARK_COMMIT = "222125788897206010e9f4ba72ddf3945ce18efa"
MANIFEST_PATH = Path("tasks/astrodynamics/benchmark-v0.1.json")
MANIFEST_SHA256 = "b8260ca229238cb624029037c0273f78f96918b32ab1d4326efc350f0f1cf231"


def load_frozen_benchmark(
    root: str | Path = ".", *, split: str | None = None, family: str | None = None,
) -> list[BenchmarkTask]:
    """Check the full snapshot before returning any selected tasks.

    Hashes cover IDs, prompts, families, splits, answers and tolerances, plus the
    source specifications and verification artifacts. Regenerating otherwise
    valid references does not silently replace the experimental baseline.
    """
    root = Path(root)
    manifest_bytes = (root / MANIFEST_PATH).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError(f"{BENCHMARK_TAG}: the freeze manifest has changed.")
    manifest = json.loads(manifest_bytes)
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{BENCHMARK_TAG}: frozen artifact has changed: {name}")
    return load_benchmark(root / DEFAULT_TASKS, split=split, family=family)
