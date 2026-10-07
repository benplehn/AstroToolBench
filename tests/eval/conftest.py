from pathlib import Path

import pytest

from astrotoolbench.benchmark import load_frozen_benchmark

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def corpus():
    return load_frozen_benchmark(ROOT)
