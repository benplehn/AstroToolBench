"""Compute, independently verify and publish references from answer-free sources."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

from .artifacts import pretty_json, serialize_records, write_artifacts
from .loader import load_specifications
from .paths import DEFAULT_REPORT, DEFAULT_SPECIFICATIONS, DEFAULT_SPLITS, DEFAULT_TASKS
from .reference import calculate
from .schema import BenchmarkTask, TaskSpecification
from .splits import audit_splits
from .verification import verify_task


@dataclass(frozen=True)
class GenerationIssue:
    task_id: str
    message: str


class GenerationError(ValueError):
    """All per-task failures; no partial dataset is published."""
    def __init__(self, issues: list[GenerationIssue]):
        self.issues = issues
        super().__init__("\n".join(f"{issue.task_id}: {issue.message}" for issue in issues))


@dataclass(frozen=True)
class GeneratedReferences:
    tasks: list[BenchmarkTask]
    absolute_errors: dict[str, dict[str, float]]


def generate_records(specifications: list[TaskSpecification]) -> GeneratedReferences:
    """Never load existing answers; verify every success and declared error."""
    if not specifications:
        raise ValueError("At least one task specification is required.")
    audit_splits(specifications)
    tasks, observed, issues = [], {}, []
    for specification in specifications:
        try:
            data = specification.model_dump(mode="json")
            contract = data.pop("answer_contract")
            if contract["kind"] == "success":
                computed = calculate(specification)
                if computed.error:
                    raise ValueError(f"Expected a computation, backend returned {computed.error}.")
                for name, value in computed.values.items():
                    contract["outputs"][name]["value"] = value
            data["expected"] = contract
            task = BenchmarkTask.model_validate(data)
            observed[task.id] = verify_task(task)
            tasks.append(task)
        except (ValueError, RuntimeError, TypeError, ArithmeticError) as error:
            issues.append(GenerationIssue(specification.id, str(error)))
    if issues:
        raise GenerationError(issues)
    audit_splits(tasks)
    return GeneratedReferences(tasks, observed)


def accuracy_report(generated: GeneratedReferences, source_contents: str | bytes) -> dict:
    tasks = generated.tasks
    manifest = audit_splits(tasks)
    source_bytes = source_contents.encode("utf-8") if isinstance(source_contents, str) else source_contents
    return {
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "dataset_sha256": hashlib.sha256(serialize_records(tasks).encode("utf-8")).hexdigest(),
        "task_count": len(tasks),
        "categories": dict(Counter(task.category for task in tasks)),
        "difficulties": dict(Counter(task.difficulty for task in tasks)),
        "outcomes": dict(Counter(task.expected.kind for task in tasks)),
        "split_counts": {split: len(ids) for split, ids in manifest["splits"].items()},
        "split_audit": manifest,
        "tasks": [{"id": task.id, "method": task.verification.method,
                   "observed_absolute_errors": generated.absolute_errors[task.id],
                   "absolute_error_budgets": task.verification.absolute_tolerances}
                  for task in tasks],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_SPECIFICATIONS, help="Answer-free task specifications.")
    parser.add_argument("--output", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--check", action="store_true", help="Verify existing artifacts without modifying them.")
    arguments = parser.parse_args(argv)
    try:
        paths = [path.resolve() for path in (arguments.tasks, arguments.output, arguments.report, arguments.splits)]
        if len(set(paths)) != len(paths):
            raise ValueError("Source, dataset, report and split-manifest paths must be distinct.")
        specifications = load_specifications(arguments.tasks)
        generated = generate_records(specifications)
        report = accuracy_report(generated, arguments.tasks.read_bytes())
        write_artifacts({arguments.output: serialize_records(generated.tasks), arguments.report: pretty_json(report),
                         arguments.splits: pretty_json(report["split_audit"])},
                        check=arguments.check)
    except (ValueError, OSError) as error:
        print(f"Reference generation failed:\n{error}", file=sys.stderr)
        return 1
    print(f"Tasks: {len(generated.tasks)} | Verified successes: {report['outcomes'].get('success', 0)} | "
          f"Verified expected errors: {report['outcomes'].get('error', 0)} | Invalid: 0")
    print("Splits: " + ", ".join(f"{name}={count}" for name, count in report["split_counts"].items()))
    print("Artifacts unchanged." if arguments.check else "References, split manifest and verification report written.")
    return 0
