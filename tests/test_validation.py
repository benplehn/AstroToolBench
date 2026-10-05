"""Publication validation, scientific drift and actionable whole-file diagnostics."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from astrotoolbench import tools
from astrotoolbench.benchmark import load_benchmark
from astrotoolbench.benchmark.artifacts import serialize_records
from astrotoolbench.benchmark.paths import DEFAULT_REPORT, DEFAULT_SPECIFICATIONS, DEFAULT_SPLITS, DEFAULT_TASKS
from astrotoolbench.benchmark.validator import validate_benchmark
from astrotoolbench.validate import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    return load_benchmark(ROOT / DEFAULT_TASKS)


@pytest.fixture
def task(corpus):
    return corpus[0].model_dump(mode="json")


def write_tasks(tmp_path, *records):
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records), encoding="utf-8")
    return path


def test_official_corpus_and_companions_are_valid():
    result = validate_benchmark(ROOT / DEFAULT_TASKS, specifications=ROOT / DEFAULT_SPECIFICATIONS,
                                splits=ROOT / DEFAULT_SPLITS, report=ROOT / DEFAULT_REPORT)
    assert result.valid and not result.issues
    assert (result.task_count, result.valid_count, result.invalid_count) == (50, 50, 0)
    assert result.difficulties == {"simple": 20, "multistep": 12, "diagnostic": 8, "trap": 10}
    assert result.splits == {"train": 24, "validation": 6, "test": 20}
    assert result.family_count == 22 and len(result.tools) == 6
    assert result.outcomes == {"success": 38, "error": 12}


@pytest.mark.parametrize("field", ["id", "domain", "category", "difficulty", "family", "required_tools"])
def test_required_metadata_cannot_be_omitted(tmp_path, task, field):
    del task[field]
    result = validate_benchmark(write_tasks(tmp_path, task), publication=False)
    assert not result.valid
    assert result.invalid_count == 1
    assert any(issue.code == "schema" and field in issue.message for issue in result.issues)


@pytest.mark.parametrize("mutation", ["unit", "tolerance", "unknown_tool", "nonfinite"])
def test_units_tolerances_and_tool_names_are_strict(tmp_path, task, mutation):
    if mutation == "unit":
        del task["inputs"]["mu"]["unit"]
    elif mutation == "tolerance":
        output = task["expected"]["outputs"]["position"]
        output["absolute_tolerance"] = output["relative_tolerance"] = 0.0
    elif mutation == "unknown_tool":
        task["required_tools"] = ["imaginary_solver"]
    else:
        task["expected"]["outputs"]["position"]["value"][0] = "NaN"
    result = validate_benchmark(write_tasks(tmp_path, task), publication=False)
    assert not result.valid and result.invalid_count == 1
    assert result.issues[0].code == "schema"


def test_multiple_broken_lines_are_all_reported_and_good_lines_still_checked(tmp_path, task):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "unfinished"\n\n{"id": "second", "id": "duplicate-key"}\n'
                    + json.dumps(task) + "\n", encoding="utf-8")
    result = validate_benchmark(path, publication=False)
    assert (result.task_count, result.valid_count, result.invalid_count) == (3, 1, 2)
    assert {issue.line for issue in result.issues if issue.code == "schema"} == {1, 3}
    assert all(issue.path == str(path) for issue in result.issues)


def test_duplicate_ids_invalidate_both_records(tmp_path, task):
    result = validate_benchmark(write_tasks(tmp_path, task, task), publication=False)
    assert not result.valid and result.invalid_count == 2
    assert {issue.line for issue in result.issues if issue.code == "duplicate_id"} == {1, 2}


def test_modified_reference_is_recomputed_and_rejected(tmp_path, task):
    task["expected"]["outputs"]["position"]["value"][0] += 1.0
    result = validate_benchmark(write_tasks(tmp_path, task), publication=False)
    assert not result.valid and result.invalid_count == 1
    issue = next(issue for issue in result.issues if issue.code == "reference")
    assert issue.task_id == task["id"] and issue.line == 1
    assert "verification budget" in issue.message


def test_solver_failure_is_an_issue_not_a_valid_diagnostic(tmp_path, task, monkeypatch):
    from astrotoolbench.benchmark import reference

    def failed_solver(_):
        raise RuntimeError("forced nonconvergence")

    monkeypatch.setitem(reference._CALCULATORS, "propagation", failed_solver)
    result = validate_benchmark(write_tasks(tmp_path, task), publication=False)
    assert result.invalid_count == 1 and not result.valid
    assert any("forced nonconvergence" in issue.message for issue in result.issues)


def test_declared_tools_must_actually_exist(tmp_path, task, monkeypatch):
    monkeypatch.setattr(tools, "propagate_kepler", None)
    result = validate_benchmark(write_tasks(tmp_path, task), publication=False)
    assert result.invalid_count == 1
    assert any(issue.code == "tool" for issue in result.issues)


def test_expected_scientific_errors_are_valid_records(tmp_path, corpus):
    errors = [task for task in corpus if task.expected.kind == "error"]
    path = tmp_path / "errors.jsonl"
    path.write_text(serialize_records(errors), encoding="utf-8")
    result = validate_benchmark(path, publication=False)
    assert result.valid and result.valid_count == 12


def test_a_wrong_expected_error_is_not_accepted(tmp_path, corpus):
    task = next(task for task in corpus if task.id == "trap-zero-gravity").model_dump(mode="json")
    task["expected"]["code"] = "physical_impossibility"
    result = validate_benchmark(write_tasks(tmp_path, task), publication=False)
    assert not result.valid and result.invalid_count == 1
    assert any(issue.code == "reference" for issue in result.issues)


def test_split_drift_is_rejected(tmp_path, task):
    task["split"] = "test"
    result = validate_benchmark(write_tasks(tmp_path, task), publication=False)
    assert result.invalid_count == 1 and not result.valid
    assert any(issue.code == "family_split" for issue in result.issues)


def test_cross_split_equivalent_problem_is_a_corpus_error(tmp_path, corpus):
    original = next(task for task in corpus if task.id == "eclipse-zero-sun-direction").model_dump(mode="json")
    clone = {**original, "id": "disguised-duplicate", "family": "invalid_search", "split": "validation"}
    result = validate_benchmark(write_tasks(tmp_path, original, clone), publication=False)
    # Individually sound records can still fail the corpus-wide contamination audit.
    assert not result.valid and result.valid_count == 2
    assert any(issue.code == "split_policy" and "numerical duplicate" in issue.message for issue in result.issues)


def test_subset_is_not_certified_as_a_publication(tmp_path, task):
    path = write_tasks(tmp_path, task)
    assert validate_benchmark(path, publication=False).valid
    result = validate_benchmark(path)
    assert not result.valid and any(issue.code == "coverage" for issue in result.issues)


@pytest.mark.parametrize("contents", ["", "\n  \n"])
def test_empty_corpus_is_rejected_even_in_subset_mode(tmp_path, contents):
    path = tmp_path / "empty.jsonl"
    path.write_text(contents)
    result = validate_benchmark(path, publication=False)
    assert not result.valid and result.task_count == 0
    assert any(issue.code == "empty" for issue in result.issues)


def test_missing_and_unreadable_files_return_diagnostics(tmp_path):
    for path in (tmp_path / "missing.jsonl", tmp_path):
        result = validate_benchmark(path)
        assert not result.valid and result.issues[0].code == "file"


def test_source_contract_mismatch_is_rejected(tmp_path, task):
    task["notes"] = "Changed metadata with an unchanged scientific answer."
    result = validate_benchmark(write_tasks(tmp_path, task), specifications=ROOT / DEFAULT_SPECIFICATIONS,
                                publication=False)
    assert not result.valid and result.invalid_count == 1
    assert any(issue.code == "source_contract" for issue in result.issues)


@pytest.mark.parametrize("artifact", ["splits", "report", "specifications"])
def test_companion_artifact_failure_fails_validation(tmp_path, task, artifact):
    broken = tmp_path / "broken.json"
    broken.write_text("{}")
    result = validate_benchmark(write_tasks(tmp_path, task), **{artifact: broken}, publication=False)
    assert not result.valid and result.issues


@pytest.mark.parametrize("change", ["hash", "certificate_ids", "budget", "observed_error"])
def test_accuracy_certificate_tampering_is_rejected(tmp_path, change):
    report = json.loads((ROOT / DEFAULT_REPORT).read_text())
    if change == "hash":
        report["dataset_sha256"] = "0" * 64
    elif change == "certificate_ids":
        report["tasks"][1]["id"] = report["tasks"][0]["id"]
    elif change == "budget":
        report["tasks"][0]["absolute_error_budgets"]["position"] = 1.0
    else:
        report["tasks"][0]["observed_absolute_errors"]["position"] = 1.0
    path = tmp_path / "accuracy.json"
    path.write_text(json.dumps(report))
    result = validate_benchmark(ROOT / DEFAULT_TASKS, report=path)
    assert not result.valid and any(issue.code == "accuracy_report" for issue in result.issues)


def test_validation_never_changes_inputs_or_artifacts(tmp_path, task):
    path = write_tasks(tmp_path, task)
    manifest = tmp_path / "splits.json"
    manifest.write_text("{}")
    before = {file: file.read_bytes() for file in (path, manifest)}
    validate_benchmark(path, splits=manifest, publication=False)
    assert {file: file.read_bytes() for file in before} == before
    assert set(tmp_path.iterdir()) == set(before)


def test_cli_json_is_machine_readable_and_uses_failure_exit_code(tmp_path, task, capsys):
    path = write_tasks(tmp_path, task)
    assert main(["--tasks", str(path), "--allow-subset", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert main(["--tasks", str(path), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is False and result["issues"][0]["code"] == "coverage"


def test_cli_text_reports_locations_and_summary(tmp_path, task, capsys):
    task["expected"]["outputs"]["position"]["value"][0] += 1.0
    path = write_tasks(tmp_path, task)
    assert main(["--tasks", str(path), "--allow-subset"]) == 1
    output = capsys.readouterr()
    assert "AstroToolBench v0.1" in output.out and "Benchmark invalid" in output.out
    assert f"{path}:1" in output.err and task["id"] in output.err


def test_public_validator_imports_no_llm_dependencies(tmp_path, task):
    path = write_tasks(tmp_path, task)
    code = '''
import builtins
import runpy
original_import = builtins.__import__
def offline_import(name, *args, **kwargs):
    if name.split(".")[0] in {"openai", "dotenv", "transformers"}:
        raise AssertionError("Optional LLM dependency imported: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = offline_import
runpy.run_module("astrotoolbench.validate", run_name="__main__")
'''
    completed = subprocess.run([sys.executable, "-c", code, "--tasks", str(path), "--allow-subset", "--json"],
                               env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=tmp_path,
                               capture_output=True, text=True, check=False, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["valid"] is True
