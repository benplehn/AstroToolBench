"""Committed official records shared by the scientific benchmark checks."""

from pathlib import Path

import pytest

from astrotoolbench.benchmark import load_benchmark, load_specifications

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def corpus():
    return load_benchmark(ROOT / "tasks/astrodynamics/tasks.jsonl")


@pytest.fixture
def task_by_id(corpus):
    return {task.id: task for task in corpus}


@pytest.fixture(scope="session")
def specifications():
    return load_specifications(ROOT / "tasks/astrodynamics/specifications.jsonl")
