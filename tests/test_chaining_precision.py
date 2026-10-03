"""Intermediate time rounding must not silently pass a dependent numerical task."""

import json
from pathlib import Path

import pytest

from astrodyn_tools.kepler import propagate_kepler
from astrodyn_tools.maneuvers import hohmann_transfer
from atb.scoring import score
from atb.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def task():
    return next(task for task in load_tasks(ROOT / "benchmark/tasks.jsonl") if task.id == "multi-001")


def prediction(task, dt_s):
    position, _ = propagate_kepler(task.params["r0"], task.params["v0"], dt_s)
    return json.dumps({"answer": {"r_final_x_km": float(position[0])}})


def test_full_precision_dependent_calculation_passes(task):
    duration = hohmann_transfer(task.params["r1"], task.params["r2"])["tof"]
    assert score(task, prediction(task, duration))["correct"]


@pytest.mark.parametrize("round_duration", [int, round], ids=["truncate-to-second", "round-to-second"])
def test_rounded_intermediate_time_fails_current_but_passed_original_tolerance(task, round_duration):
    duration = hohmann_transfer(task.params["r1"], task.params["r2"])["tof"]
    text = prediction(task, float(round_duration(duration)))
    assert score(task, text)["correct"] is False
    old_task = task.model_copy(update={"tolerance": {"r_final_x_km": 5.0}})
    assert score(old_task, text)["correct"] is True


def test_recorded_nvidia_answer_passes_tightened_tolerance(task):
    trace = json.loads((ROOT / "docs/examples/nvidia-nemotron-agent.json").read_text())
    assert score(task, trace["final_answer"])["correct"]
