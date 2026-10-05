"""Corpus coverage, versioned schema and intentionally frozen reference records."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from astrotoolbench.benchmark.families import build_tasks
from astrotoolbench.benchmark.schema import BenchmarkTask, TaskSpecification

ROOT = Path(__file__).resolve().parents[2]


def test_family_coverage_and_difficulty_balance(corpus):
    assert len(corpus) == 50
    assert Counter(task.category for task in corpus) == {
        "propagation": 8, "maneuvers": 9, "eclipse": 8, "proximity": 8, "multistep": 8, "diagnostic": 9,
    }
    assert Counter(task.difficulty for task in corpus) == {
        "simple": 20, "multistep": 12, "diagnostic": 8, "trap": 10,
    }
    assert len({task.family for task in corpus}) >= 6
    assert Counter(task.split for task in corpus) == {"train": 24, "validation": 6, "test": 20}
    assert len({task.id for task in corpus}) == len(corpus)


def test_committed_schema_matches_models():
    schema = BenchmarkTask.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:astrotoolbench:task:0.1"
    assert json.loads((ROOT / "schemas/task-v0.1.schema.json").read_text()) == schema


def test_committed_specification_schema_matches_models():
    schema = TaskSpecification.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:astrotoolbench:task-spec:0.1"
    assert json.loads((ROOT / "schemas/task-spec-v0.1.schema.json").read_text()) == schema


def test_authored_problems_reproduce_committed_records(corpus):
    # JSON structure and metadata must reproduce exactly; numerical comparison
    # uses the stricter scientific certificates elsewhere, without platform ULPs.
    rebuilt = build_tasks()
    for stored, generated in zip(corpus, rebuilt):
        old, new = stored.model_dump(), generated.model_dump()
        if old["expected"]["kind"] == "success":
            for result in old["expected"]["outputs"].values():
                result.pop("value")
            for result in new["expected"]["outputs"].values():
                result.pop("value")
        assert old == new
    assert len(rebuilt) == len(corpus)


def test_reference_snapshot_requires_explicit_review(corpus):
    references = {task.id: task.expected.model_dump() for task in corpus}
    digest = hashlib.sha256(json.dumps(references, sort_keys=True, allow_nan=False).encode()).hexdigest()
    # Updating this snapshot is a deliberate review step after scientific verification.
    assert digest == "05d5268afc34523ea0596f01f7b74e1a58f3684f9a2b83e9bcd190223cbfabe0"


def test_original_48_references_are_preserved(corpus):
    added = {"hohmann-equal-radii", "diagnostic-zero-impulse"}
    references = {task.id: task.expected.model_dump() for task in corpus if task.id not in added}
    digest = hashlib.sha256(json.dumps(references, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert digest == "6bfc7b1f8ecfef154baff0761cb3dce1b68c166d2ac4346df1752018bd9e9595"


def test_accuracy_report_identifies_reviewed_corpus(corpus):
    report = json.loads((ROOT / "docs/reference-verification.json").read_text())
    assert report["dataset_sha256"] == hashlib.sha256((ROOT / "tasks/astrodynamics/tasks.jsonl").read_bytes()).hexdigest()
    assert report["source_sha256"] == hashlib.sha256((ROOT / "tasks/astrodynamics/specifications.jsonl").read_bytes()).hexdigest()
    assert report["task_count"] == len(corpus)
    assert {record["id"] for record in report["tasks"]} == {task.id for task in corpus}
    for record in report["tasks"]:
        assert set(record["observed_absolute_errors"]) == set(record["absolute_error_budgets"])
        for name, error in record["observed_absolute_errors"].items():
            assert error <= record["absolute_error_budgets"][name]
