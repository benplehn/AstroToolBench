"""Scientific references must agree with a separate implementation, not just self-regenerate."""

import math

import numpy as np
import pytest

from astrotoolbench.tools import MU_EARTH, propagate_kepler
from astrotoolbench.benchmark import BenchmarkTask, load_benchmark
from astrotoolbench.benchmark.reference import calculate
from astrotoolbench.benchmark.verification import independent_answer, integrate_cartesian, verify_task
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = load_benchmark(ROOT / "tasks/astrodynamics/tasks.jsonl")


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.id)
def test_every_committed_reference_is_independently_verified(task):
    verify_task(task)


@pytest.mark.parametrize("a, eccentricity, duration", [
    (9000.0, 0.0, 3500.0), (10000.0, 0.2, 4500.0),
    (20000.0, 0.65, 6000.0), (11000.0, 0.25, -1234.5),
])
def test_cartesian_rk4_step_halving_bounds_error(a, eccentricity, duration):
    radius = a * (1-eccentricity)
    position = [radius, 0, 0]
    velocity = [0, math.sqrt(MU_EARTH * (2/radius - 1/a)), 0]
    coarse = integrate_cartesian(position, velocity, duration, MU_EARTH, max_step=2)
    fine = integrate_cartesian(position, velocity, duration, MU_EARTH, max_step=1)
    backend = propagate_kepler(position, velocity, duration)
    np.testing.assert_allclose(coarse[0], fine[0], rtol=0, atol=5e-5)
    np.testing.assert_allclose(coarse[1], fine[1], rtol=0, atol=5e-8)
    np.testing.assert_allclose(fine[0], backend[0], rtol=0, atol=5e-5)
    np.testing.assert_allclose(fine[1], backend[1], rtol=0, atol=5e-8)


def test_elliptic_half_period_has_analytic_apogee_state(task_by_id):
    task = task_by_id["prop-periapsis-apogee"]
    answer = independent_answer(task).values
    radius = 12000.0
    speed = math.sqrt(MU_EARTH * (2/radius - 1/10000.0))
    np.testing.assert_allclose(answer["position"], [-radius, 0, 0], rtol=0, atol=5e-5)
    np.testing.assert_allclose(answer["velocity"], [0, -speed, 0], rtol=0, atol=5e-8)


def test_independent_propagator_does_not_call_backend(monkeypatch, task_by_id):
    import astrotoolbench.benchmark.reference as reference

    def fail_if_called(_):
        raise AssertionError("Backend calculator was called by independent verification")

    monkeypatch.setitem(reference._CALCULATORS, "propagation", fail_if_called)
    task = task_by_id["prop-elliptic-arbitrary"]
    assert independent_answer(task).error is None


def test_independent_computation_does_not_read_stored_reference(task_by_id):
    task = task_by_id["prop-circular-quarter"]
    data = task.model_dump()
    data["expected"]["outputs"]["position"]["value"][0] += 1000
    altered = BenchmarkTask.model_validate(data)
    assert independent_answer(altered).values == independent_answer(task).values
    with pytest.raises(ValueError, match="verification budget"):
        verify_task(altered)


def test_reference_drift_inside_scoring_tolerance_still_fails_verification(task_by_id):
    data = task_by_id["prop-circular-quarter"].model_dump()
    data["expected"]["outputs"]["position"]["value"][0] += 5e-4
    with pytest.raises(ValueError, match="verification budget"):
        verify_task(BenchmarkTask.model_validate(data))


def test_independent_eclipse_certificate_catches_missed_event(task_by_id):
    data = task_by_id["eclipse-one-period"].model_dump()
    data["inputs"]["step"]["value"] = data["inputs"]["duration"]["value"]
    data["outputs"] = ["total_duration"]
    data["expected"]["outputs"] = {"total_duration": data["expected"]["outputs"]["total_duration"]}
    data["expected"]["outputs"]["total_duration"]["value"] = 0.0
    data["verification"]["absolute_tolerances"] = {"total_duration": 5e-6}
    task = BenchmarkTask.model_validate(data)
    assert calculate(task).values["total_duration"] == 0.0
    assert independent_answer(task).values["total_duration"] > 1000
    with pytest.raises(ValueError, match="verification budget"):
        verify_task(task)


@pytest.mark.parametrize("identifier", ["eclipse-one-period", "eclipse-two-periods", "proximity-interior-crossing"])
def test_search_references_stable_under_step_and_tolerance_reduction(task_by_id, identifier):
    original = task_by_id[identifier]
    data = original.model_dump()
    data["inputs"]["step"]["value"] /= 2
    data["inputs"]["tol"]["value"] /= 2
    fine = BenchmarkTask.model_validate(data)
    verify_task(fine)


def test_backend_and_independent_policy_detect_bad_shape(task_by_id):
    data = task_by_id["prop-circular-quarter"].model_dump()
    data["inputs"]["r0"]["value"] = [7000.0, 0.0]
    data["outputs"] = []
    data["expected"] = {"kind": "error", "code": "invalid_input", "reason": "Position is not a Cartesian triple."}
    data["verification"] = {"method": "independent_input_check", "rationale": "Malformed vector shape.", "absolute_tolerances": {}}
    verify_task(BenchmarkTask.model_validate(data))


def test_wrong_error_reference_is_rejected(task_by_id):
    data = task_by_id["trap-zero-gravity"].model_dump()
    data["expected"]["code"] = "physical_impossibility"
    with pytest.raises(ValueError, match="not reproduced independently"):
        verify_task(BenchmarkTask.model_validate(data))
