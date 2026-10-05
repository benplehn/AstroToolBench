"""Scoring honors result types, shapes and explicitly selected tolerances."""

import pytest

from astrotoolbench.benchmark.comparison import matches_answer, matches_numeric
from astrotoolbench.benchmark.schema import NumericResult


@pytest.mark.parametrize("actual", ["10", True, float("nan"), float("inf"), [10], 10**400, {"value": 10}, [[1], []]])
def test_malformed_numeric_predictions(actual):
    expected = NumericResult(kind="numeric", value=10.0, unit="km", absolute_tolerance=0.1)
    assert not matches_numeric(expected, actual)


def test_absolute_and_relative_tolerances():
    absolute = NumericResult(kind="numeric", value=10.0, unit="km", absolute_tolerance=0.1)
    relative = NumericResult(kind="numeric", value=1000.0, unit="km", relative_tolerance=1e-3)
    assert matches_numeric(absolute, 10.05)
    assert not matches_numeric(absolute, 10.2)
    assert matches_numeric(relative, 1000.5)
    assert not matches_numeric(relative, 1002.0)


def test_vectors_are_compared_componentwise_without_broadcasting():
    expected = NumericResult(kind="numeric", value=[1.0, 0.0, 3.0], unit="km", absolute_tolerance=0.01)
    assert matches_numeric(expected, [1.005, 0.001, 3.005])
    assert not matches_numeric(expected, 1.0)
    assert not matches_numeric(expected, [1.0, False, 3.0])
    assert not matches_numeric(expected, [1.0, 0.0, 3.1])


def test_boolean_is_exact_not_numeric(task_by_id):
    task = task_by_id["shadow-day-side"]
    assert matches_answer(task, {"outputs": {"in_shadow": False}})
    assert not matches_answer(task, {"outputs": {"in_shadow": 0}})


def test_trap_difficulty_does_not_automatically_mean_refusal(task_by_id):
    data = task_by_id["prop-backward"].model_dump()
    data["difficulty"] = "trap"
    from astrotoolbench.benchmark import BenchmarkTask
    task = BenchmarkTask.model_validate(data)
    answer = {"outputs": {name: result.value for name, result in task.expected.outputs.items()}}
    assert matches_answer(task, answer)
    assert not matches_answer(task, {"error": "invalid_input", "reason": "negative time"})


def test_error_code_is_required_and_reason_must_exist(task_by_id):
    task = task_by_id["trap-zero-gravity"]
    assert matches_answer(task, {"error": "invalid_input", "reason": "mu must be positive"})
    assert not matches_answer(task, {"error": "physical_impossibility", "reason": "wrong cause"})
    assert not matches_answer(task, {"error": "invalid_input", "reason": " "})
    assert not matches_answer(task, {"outputs": {"dv_total": 0.0}})


def test_every_requested_multistep_output_is_required(task_by_id):
    task = task_by_id["multi-transfer-propagate-circle"]
    answer = {"outputs": {name: result.value for name, result in task.expected.outputs.items()}}
    assert matches_answer(task, answer)
    del answer["outputs"]["tof"]
    assert not matches_answer(task, answer)
