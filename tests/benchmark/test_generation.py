"""Answer-free sources, independently verified regeneration and failure-safe publishing."""

import json

import pytest
from pydantic import ValidationError

from astrotoolbench.benchmark import TaskSpecification, load_benchmark
from astrotoolbench.benchmark.artifacts import serialize_records
from astrotoolbench.benchmark.generation import GenerationError, generate_records, main


@pytest.fixture
def cli_paths(tmp_path):
    return {name: tmp_path / filename for name, filename in (
        ("tasks", "source.jsonl"), ("output", "references.jsonl"),
        ("report", "report.json"), ("splits", "splits.json"),
    )}


def arguments(paths):
    return [item for name, path in paths.items() for item in ("--" + name, str(path))]


def test_specifications_contain_no_numerical_or_boolean_answers(specifications):
    for specification in specifications:
        data = specification.model_dump()
        assert "expected" not in data
        if data["answer_contract"]["kind"] == "success":
            for contract in data["answer_contract"]["outputs"].values():
                assert "value" not in contract


@pytest.mark.parametrize("identifier, output, value", [
    ("prop-circular-quarter", "position", [0.0, 7000.0, 0.0]),
    ("shadow-day-side", "in_shadow", False),
])
def test_reference_values_cannot_be_injected_into_source(specifications, identifier, output, value):
    data = next(spec for spec in specifications if spec.id == identifier).model_dump()
    data["answer_contract"]["outputs"][output]["value"] = value
    with pytest.raises(ValidationError, match="Extra inputs"):
        TaskSpecification.model_validate(data)


def test_generating_from_loaded_sources_does_not_use_authored_templates(monkeypatch, specifications, corpus):
    import astrotoolbench.benchmark.families as families

    def forbidden():
        raise AssertionError("Reference generation must load inputs, not rebuild authoring templates")

    monkeypatch.setattr(families, "build_specifications", forbidden)
    generated = generate_records(specifications)
    assert [task.id for task in generated.tasks] == [task.id for task in corpus]
    assert set(generated.absolute_errors) == {task.id for task in corpus}


def test_cli_regenerates_new_inputs_without_any_previous_answers(cli_paths, specifications):
    source = specifications[0].model_dump()
    source["id"] = "new-input-propagation"
    source["inputs"]["dt"]["value"] = 321.123
    source["prompt"] = "Propager l'état fourni pendant 321.123 s dans le référentiel ECI."
    specification = TaskSpecification.model_validate(source)
    cli_paths["tasks"].write_text(serialize_records([specification]))
    assert main(arguments(cli_paths)) == 0
    result = load_benchmark(cli_paths["output"])[0]
    assert result.id == specification.id
    assert result.inputs["dt"].value == 321.123
    assert isinstance(result.expected.outputs["position"].value, list)
    report = json.loads(cli_paths["report"].read_text())
    assert report["task_count"] == 1
    assert report["source_sha256"]
    assert main(arguments(cli_paths) + ["--check"]) == 0


def test_cli_check_detects_reference_drift_and_never_changes_files(cli_paths, specifications):
    cli_paths["tasks"].write_text(serialize_records([specifications[0]]))
    assert main(arguments(cli_paths)) == 0
    data = json.loads(cli_paths["output"].read_text())
    data["expected"]["outputs"]["position"]["value"][0] += 1
    tampered = json.dumps(data) + "\n"
    cli_paths["output"].write_text(tampered)
    assert main(arguments(cli_paths) + ["--check"]) == 1
    assert cli_paths["output"].read_text() == tampered


def test_all_scientific_errors_are_reported_before_any_publication(cli_paths, specifications, capsys):
    invalid = []
    for specification in specifications:
        data = specification.model_dump()
        if specification.id == "prop-circular-quarter":
            data["inputs"]["mu"]["value"] = 0.0
        elif specification.id == "eclipse-one-period":
            data["inputs"]["sun_dir"]["value"] = [0.0, 0.0, 0.0]
        invalid.append(TaskSpecification.model_validate(data))
    cli_paths["tasks"].write_text(serialize_records(invalid))
    for name in ("output", "report", "splits"):
        cli_paths[name].write_text("existing artifact")
    assert main(arguments(cli_paths)) == 1
    error = capsys.readouterr().err
    assert "prop-circular-quarter" in error and "eclipse-one-period" in error
    for name in ("output", "report", "splits"):
        assert cli_paths[name].read_text() == "existing artifact"


def test_cli_check_compares_exact_artifact_bytes(cli_paths, specifications):
    cli_paths["tasks"].write_text(serialize_records([specifications[0]]))
    assert main(arguments(cli_paths)) == 0
    changed = cli_paths["output"].read_bytes().replace(b"\n", b"\r\n")
    cli_paths["output"].write_bytes(changed)
    assert main(arguments(cli_paths) + ["--check"]) == 1
    assert cli_paths["output"].read_bytes() == changed


def test_unexpected_solver_failures_are_not_stored_as_expected_errors(monkeypatch, specifications):
    import astrotoolbench.benchmark.reference as reference

    def failed_solver(_):
        raise RuntimeError("forced solver nonconvergence")

    monkeypatch.setitem(reference._CALCULATORS, "propagation", failed_solver)
    with pytest.raises(GenerationError, match="forced solver nonconvergence"):
        generate_records([specifications[0]])


def test_source_cannot_be_overwritten_by_output_paths(cli_paths, specifications):
    source = serialize_records([specifications[0]])
    cli_paths["tasks"].write_text(source)
    cli_paths["output"] = cli_paths["tasks"]
    assert main(arguments(cli_paths)) == 1
    assert cli_paths["tasks"].read_text() == source


def test_check_only_missing_artifacts_does_not_create_files(cli_paths, specifications):
    cli_paths["tasks"].write_text(serialize_records([specifications[0]]))
    assert main(arguments(cli_paths) + ["--check"]) == 1
    assert not cli_paths["output"].exists()
    assert not cli_paths["report"].exists()
    assert not cli_paths["splits"].exists()


def test_public_module_command_runs_without_llm_dependencies(cli_paths, specifications):
    import subprocess
    import sys

    cli_paths["tasks"].write_text(serialize_records([specifications[0]]))
    # A fresh process isolates this check from optional client imports in other tests.
    code = '''
import builtins
import runpy
original_import = builtins.__import__
def offline_import(name, *args, **kwargs):
    if name.split(".")[0] in {"openai", "dotenv", "transformers"}:
        raise AssertionError("Optional LLM dependency imported: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = offline_import
runpy.run_module("astrotoolbench.generate_references", run_name="__main__")
'''
    completed = subprocess.run([sys.executable, "-c", code, *arguments(cli_paths)],
                               capture_output=True, text=True, check=False, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "Verified successes: 1" in completed.stdout
