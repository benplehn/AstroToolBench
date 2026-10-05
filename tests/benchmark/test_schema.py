"""Reject malformed records while preserving intentionally invalid science inputs."""

import json

import pytest
from pydantic import ValidationError

from astrotoolbench.benchmark import BenchmarkTask, load_benchmark
from astrotoolbench.benchmark.schema import NumericResult


def set_path(data, path, value):
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value


@pytest.mark.parametrize("path, value", [
    (("schema_version",), "9.0"), (("difficulty",), "easy"), (("split",), "dev"),
    (("family",), ""), (("id",), "bad id"), (("domain",), "chemistry"),
    (("prompt",), "   "),
    (("inputs", "r0", "unit"), "parsec"), (("inputs", "r0", "value"), [7000, True, 0]),
    (("inputs", "mu", "value"), "398600.4418"), (("inputs", "mu", "value"), float("nan")),
    (("expected", "outputs", "position", "value"), [float("inf"), 0, 0]),
    (("expected", "outputs", "position", "unit"), "m"),
    (("expected", "outputs", "position", "absolute_tolerance"), -1),
    (("expected", "outputs", "position", "absolute_tolerance"), None),
    (("required_tools",), ["teleport"]), (("outputs",), ["position", "position"]),
    (("outputs",), ["nonexistent"]), (("verification", "absolute_tolerances"), {}),
    (("verification", "absolute_tolerances", "position"), 1.0),
    (("verification", "method"), "independent_input_check"),
])
def test_invalid_schema_records(task_by_id, path, value):
    data = task_by_id["prop-circular-quarter"].model_dump()
    set_path(data, path, value)
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(data)


@pytest.mark.parametrize("path", [("inputs", "mu", "unit"), ("family",), ("context",), ("verification",)])
def test_required_metadata_cannot_be_omitted(task_by_id, path):
    data = task_by_id["prop-circular-quarter"].model_dump()
    node = data
    for name in path[:-1]:
        node = node[name]
    del node[path[-1]]
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(data)


def test_unknown_fields_are_rejected(task_by_id):
    data = task_by_id["prop-circular-quarter"].model_dump()
    data["tollerance"] = 1e-3
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(data)


def test_boolean_results_are_not_numbers(task_by_id):
    data = task_by_id["shadow-day-side"].model_dump()
    data["expected"]["outputs"]["in_shadow"] = {
        "kind": "numeric", "value": 0.0, "unit": "1", "absolute_tolerance": 0.1,
    }
    data["verification"]["absolute_tolerances"] = {"in_shadow": 0.01}
    with pytest.raises(ValidationError, match="result type"):
        BenchmarkTask.model_validate(data)


def test_relative_tolerance_needs_absolute_budget_at_zero():
    with pytest.raises(ValidationError):
        NumericResult(kind="numeric", value=[1.0, 0.0], unit="km", relative_tolerance=1e-5)
    result = NumericResult(kind="numeric", value=10.0, unit="km", relative_tolerance=1e-5)
    assert result.absolute_tolerance is None


def test_intentionally_bad_scientific_values_are_schema_valid(task_by_id):
    for identifier in ("trap-zero-gravity", "proximity-negative-window", "dv-mixed-units", "diagnostic-missing-state"):
        task = task_by_id[identifier]
        assert BenchmarkTask.model_validate(task.model_dump()) == task


def test_loader_rejects_duplicates_before_filtering(tmp_path, task_by_id):
    data = task_by_id["prop-circular-quarter"].model_dump()
    path = tmp_path / "duplicate.jsonl"
    path.write_text(json.dumps(data) + "\n" + json.dumps(data))
    with pytest.raises(ValueError, match=r"duplicate.jsonl:2:.*Duplicate task ID"):
        load_benchmark(path, split="test")


def test_duplicate_json_keys_and_line_context(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('\n{"id":"one", "id":"two"}')
    with pytest.raises(ValueError, match=r"broken.jsonl:2:.*Duplicate JSON key"):
        load_benchmark(path)


@pytest.mark.parametrize("contents", ["", "\n\n", "null", "[]", "{not json}"])
def test_empty_or_malformed_files(tmp_path, contents):
    path = tmp_path / "broken.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_benchmark(path)


def test_official_loader_rejects_legacy_records(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(dict(id="old", family="old", level="simple", prompt="legacy")))
    with pytest.raises(ValueError):
        load_benchmark(path)


def test_loader_filters_are_explicit(corpus):
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    assert len(load_benchmark(ROOT / "tasks/astrodynamics/tasks.jsonl", split="train")) == 24
    family = corpus[0].family
    selected = load_benchmark(ROOT / "tasks/astrodynamics/tasks.jsonl", family=family)
    assert selected and all(task.family == family for task in selected)


def test_verification_method_must_describe_actual_certificate(task_by_id):
    data = task_by_id["hohmann-raising"].model_dump()
    data["verification"]["method"] = "cartesian_rk4"
    with pytest.raises(ValidationError, match="not implemented"):
        BenchmarkTask.model_validate(data)
