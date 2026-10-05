"""Strict JSONL loading, whole-file IDs and reviewed split enforcement."""

import json
from dataclasses import dataclass
from pathlib import Path

from .schema import BenchmarkTask, Record, TaskSpecification
from .splits import audit_splits


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass
class RecordFile:
    records: list[tuple[int, Record]]
    errors: list[tuple[int, str]]
    task_count: int


def read_record_file(path: str | Path, model: type[Record]) -> RecordFile:
    """Scan every nonblank line, preserving all schema failures for validation.

    Duplicate IDs and scientific policy are checked by the caller. Keeping JSON
    parsing here gives the loader and validator identical strict semantics.
    """
    path = Path(path)
    records, errors, count = [], [], 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        count += 1
        try:
            data = json.loads(line, object_pairs_hook=_unique_object)
            records.append((line_number, model.model_validate(data)))
        except (ValueError, TypeError) as error:
            errors.append((line_number, str(error)))
    return RecordFile(records, errors, count)


def _load_records(path, model, split=None, family=None):
    path = Path(path)
    scanned = read_record_file(path, model)
    if scanned.errors:
        line_number, message = scanned.errors[0]
        raise ValueError(f"{path}:{line_number}: {message}")
    tasks, seen = [], set()
    for line_number, task in scanned.records:
        try:
            if task.id in seen:
                raise ValueError(f"Duplicate task ID: {task.id}")
            seen.add(task.id)
            tasks.append(task)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if not tasks:
        raise ValueError(f"{path}: the benchmark must contain at least one task.")
    try:
        audit_splits(tasks)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    # All records, including filtered-out partitions, have already been checked.
    return [task for task in tasks if (split is None or task.split == split)
            and (family is None or task.family == family)]


def load_benchmark(path: str | Path, *, split: str | None = None, family: str | None = None) -> list[BenchmarkTask]:
    """Read official records; reject legacy formats and unreviewed partitions."""
    return _load_records(path, BenchmarkTask, split, family)


def load_specifications(path: str | Path) -> list[TaskSpecification]:
    """Read answer-free task sources, enforcing their declared family splits."""
    return _load_records(path, TaskSpecification)
