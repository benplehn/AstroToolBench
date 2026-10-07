"""Experiments must reject drift, even in an unselected split."""

import json
from pathlib import Path
import shutil

import pytest

from astrotoolbench.benchmark import BENCHMARK_COMMIT, BENCHMARK_TAG, load_frozen_benchmark
from astrotoolbench.benchmark.frozen import MANIFEST_PATH

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def snapshot(tmp_path):
    manifest = json.loads((ROOT / MANIFEST_PATH).read_text())
    for name in [str(MANIFEST_PATH), *manifest["files"]]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    return tmp_path


def test_frozen_corpus_and_selection(corpus):
    assert load_frozen_benchmark(ROOT) == corpus
    held_out = load_frozen_benchmark(ROOT, split="test")
    assert len(held_out) == 20
    family = held_out[0].family
    assert load_frozen_benchmark(ROOT, split="test", family=family) == [
        task for task in held_out if task.family == family
    ]
    manifest = json.loads((ROOT / MANIFEST_PATH).read_text())
    assert manifest["tag"] == BENCHMARK_TAG
    assert manifest["source_commit"] == BENCHMARK_COMMIT
    assert manifest["task_count"] == len(corpus)


@pytest.mark.parametrize("field,value", [
    ("id", "another-id"), ("family", "another-family"), ("prompt", "Reworded task"),
])
def test_metadata_drift_in_train_rejected_before_test_filter(snapshot, field, value):
    path = snapshot / "tasks/astrodynamics/tasks.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    next(task for task in records if task["split"] == "train")[field] = value
    path.write_text("\n".join(json.dumps(task) for task in records) + "\n")
    with pytest.raises(ValueError, match="frozen artifact has changed: tasks/astrodynamics/tasks.jsonl"):
        load_frozen_benchmark(snapshot, split="test")


@pytest.mark.parametrize("field", ["value", "absolute_tolerance"])
def test_reference_and_tolerance_drift_rejected(snapshot, field):
    path = snapshot / "tasks/astrodynamics/tasks.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    result = next(
        output for task in records if task["expected"]["kind"] == "success"
        for output in task["expected"]["outputs"].values() if output["kind"] == "numeric"
    )
    result[field] = 123.0
    path.write_text("\n".join(json.dumps(task) for task in records) + "\n")
    with pytest.raises(ValueError, match="frozen artifact has changed"):
        load_frozen_benchmark(snapshot)


@pytest.mark.parametrize("artifact", [
    "tasks/astrodynamics/specifications.jsonl", "tasks/astrodynamics/splits.json",
    "schemas/task-v0.1.schema.json", "schemas/task-spec-v0.1.schema.json",
    "docs/reference-verification.json",
])
def test_companion_drift_rejected(snapshot, artifact):
    path = snapshot / artifact
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=f"frozen artifact has changed: {artifact}"):
        load_frozen_benchmark(snapshot)


def test_editing_manifest_cannot_approve_new_hashes(snapshot):
    path = snapshot / MANIFEST_PATH
    manifest = json.loads(path.read_text())
    manifest["files"]["tasks/astrodynamics/tasks.jsonl"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="freeze manifest has changed"):
        load_frozen_benchmark(snapshot)


def test_missing_artifact_rejected(snapshot):
    (snapshot / "tasks/astrodynamics/splits.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_frozen_benchmark(snapshot)
