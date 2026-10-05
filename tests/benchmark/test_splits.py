"""Family and full-computation isolation, including deliberate leakage attempts."""

import json
from pathlib import Path

import pytest

from astrotoolbench.benchmark import BenchmarkTask, load_benchmark
from astrotoolbench.benchmark.coverage import audit_coverage
from astrotoolbench.benchmark.splits import GROUPS, FAMILY_CONTRACTS, audit_splits, group_for_family, problem_fingerprint

ROOT = Path(__file__).resolve().parents[2]


def test_reviewed_registry_covers_each_family_exactly_once(corpus):
    declared = [family for group in GROUPS.values() for family in group.families]
    assert len(declared) == len(set(declared))
    assert set(declared) == set(FAMILY_CONTRACTS) == {task.family for task in corpus}


def test_manifest_matches_actual_partitions(corpus):
    manifest = audit_splits(corpus)
    assert json.loads((ROOT / "tasks/astrodynamics/splits.json").read_text()) == manifest
    assert {name: len(ids) for name, ids in manifest["splits"].items()} == {"train": 24, "validation": 6, "test": 20}
    for task in corpus:
        assert manifest["groups"][group_for_family(task.family)]["split"] == task.split


def test_mathematical_variants_are_grouped_together():
    assert group_for_family("kepler_circular") == group_for_family("kepler_elliptic")
    assert group_for_family("hohmann_transfer") == group_for_family("transfer_comparison")
    assert group_for_family("kepler_comparison") == group_for_family("separation_at_epoch")


def test_family_move_is_rejected_even_with_new_radius(task_by_id):
    data = task_by_id["hohmann-raising"].model_dump()
    data["id"] = "new-radius-test"
    data["split"] = "test"
    data["inputs"]["r1"]["value"] = 7200.0
    with pytest.raises(ValueError, match="belongs to"):
        audit_splits([BenchmarkTask.model_validate(data)])


def test_new_label_requires_explicit_policy_review(task_by_id):
    data = task_by_id["hohmann-raising"].model_dump()
    data["family"] = "hohmann_new_radius"
    with pytest.raises(ValueError, match="reviewed split"):
        audit_splits([BenchmarkTask.model_validate(data)])


def test_known_label_cannot_disguise_a_seen_recipe(task_by_id):
    data = task_by_id["hohmann-raising"].model_dump()
    data["family"] = "closest_approach"
    data["split"] = "test"
    with pytest.raises(ValueError, match="does not match reviewed family"):
        audit_splits([BenchmarkTask.model_validate(data)])


def test_cross_split_duplicate_ignores_id_wording_and_output_subset(task_by_id):
    original = task_by_id["eclipse-zero-sun-direction"]
    data = original.model_dump()
    data["id"] = "disguised-validation-error"
    data["prompt"] = "Different wording of exactly the same numerical problem."
    data["family"] = "invalid_search"
    data["split"] = "validation"
    clone = BenchmarkTask.model_validate(data)
    with pytest.raises(ValueError, match="numerical duplicate"):
        audit_splits([original, clone])


def test_problem_fingerprint_ignores_scan_and_serialization_details(task_by_id):
    original = task_by_id["proximity-interior-crossing"]
    data = original.model_dump()
    data["id"] = "same-problem-finer-scan"
    data["inputs"]["step"]["value"] = 4.0
    data["inputs"]["tol"]["value"] = 5e-7
    data["inputs"]["r1"]["value"][2] = -0.0
    assert problem_fingerprint(original) == problem_fingerprint(BenchmarkTask.model_validate(data))


def test_loader_checks_split_drift_before_filtering(tmp_path, task_by_id):
    data = task_by_id["hohmann-raising"].model_dump()
    data["split"] = "test"
    path = tmp_path / "leak.jsonl"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="belongs to"):
        load_benchmark(path, split="train")


def test_complete_successful_recipes_are_disjoint_but_primitives_are_reused(corpus):
    recipes = {split: {task.recipe for task in corpus if task.split == split and task.expected.kind == "success"}
               for split in ("train", "validation", "test")}
    assert not recipes["train"] & recipes["test"]
    assert not recipes["train"] & recipes["validation"]
    assert not recipes["validation"] & recipes["test"]
    train_tools = {tool for task in corpus if task.split == "train" for tool in task.required_tools}
    test_tools = {tool for task in corpus if task.split == "test" for tool in task.required_tools}
    assert "hohmann_transfer" in train_tools & test_tools
    assert "propagate_kepler" in train_tools & test_tools


def test_publication_coverage_and_ratio_targets(corpus):
    report = audit_coverage(corpus)
    assert report["task_count"] == report["desired_task_count"] == 50
    assert report["difficulty_fractions"] == {"simple": 0.40, "multistep": 0.24, "diagnostic": 0.16, "trap": 0.20}


def test_undersized_corpus_is_rejected(corpus):
    with pytest.raises(ValueError, match="at least 40"):
        audit_coverage(corpus[:39])


def test_easy_only_corpus_is_rejected(corpus):
    tasks = [task.model_copy(update={"difficulty": "simple"}) for task in corpus]
    with pytest.raises(ValueError, match="all four difficulty"):
        audit_coverage(tasks)


def test_unbalanced_corpus_is_rejected(corpus):
    tasks = list(corpus)
    changed = 0
    for index, task in enumerate(tasks):
        if task.difficulty == "multistep" and changed < 5:
            tasks[index] = task.model_copy(update={"difficulty": "simple"})
            changed += 1
    with pytest.raises(ValueError, match="target"):
        audit_coverage(tasks)
