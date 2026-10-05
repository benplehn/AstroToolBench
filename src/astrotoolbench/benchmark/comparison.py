"""Exact categorical comparison and explicit componentwise numeric tolerances."""

from typing import Any

import numpy as np

from .schema import BenchmarkTask, BooleanResult, Failure, NumericResult


def matches_numeric(expected: NumericResult, actual: Any) -> bool:
    """Use max(abs_tol, rel_tol*|reference|), with no implicit default tolerance."""
    try:
        array = np.asarray(actual)
        components = np.asarray(actual, dtype=object).flat
    except (TypeError, ValueError, OverflowError):
        return False
    if any(isinstance(value, (bool, np.bool_)) for value in components):
        return False
    target = np.asarray(expected.value)
    if array.shape != target.shape or array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        return False
    limit = np.maximum(expected.absolute_tolerance or 0.0,
                       (expected.relative_tolerance or 0.0) * np.abs(target))
    return bool(np.all(np.abs(array - target) <= limit))


def matches_answer(task: BenchmarkTask, answer: dict[str, Any]) -> bool:
    """Check the official answer object; difficulty never implies refusal.

    Success: {"outputs": {name: value, ...}}. Error: {"error": code, "reason":
    nonempty explanation}. Error reasons are retained for review, not evaluated
    for semantic correctness by this deterministic comparison.
    """
    if not isinstance(answer, dict):
        return False
    if isinstance(task.expected, Failure):
        reason = answer.get("reason")
        return (set(answer) == {"error", "reason"} and answer["error"] == task.expected.code
                and isinstance(reason, str) and bool(reason.strip()))
    if set(answer) != {"outputs"} or not isinstance(answer["outputs"], dict):
        return False
    if set(answer["outputs"]) != set(task.expected.outputs):
        return False
    for name, expected in task.expected.outputs.items():
        actual = answer["outputs"][name]
        if isinstance(expected, BooleanResult):
            if actual is not expected.value:
                return False
        elif not matches_numeric(expected, actual):
            return False
    return True
