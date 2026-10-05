"""Read-only corpus validation with independent scientific checks and diagnostics."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from .. import tools
from .coverage import audit_coverage
from .loader import _unique_object, load_specifications, read_record_file
from .schema import BenchmarkTask
from .splits import audit_splits
from .verification import verify_task


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str
    line: int | None = None
    task_id: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    task_count: int
    valid_count: int
    issues: list[ValidationIssue]
    categories: dict[str, int]
    difficulties: dict[str, int]
    splits: dict[str, int]
    family_count: int
    tools: list[str]
    outcomes: dict[str, int]
    checked_artifacts: list[str]

    @property
    def invalid_count(self) -> int:
        return self.task_count - self.valid_count

    @property
    def valid(self) -> bool:
        return self.task_count > 0 and not self.issues

    def to_dict(self) -> dict:
        return {"benchmark": "AstroToolBench", "schema_version": "0.1",
                **asdict(self), "invalid_count": self.invalid_count, "valid": self.valid}


def _source_record(task: BenchmarkTask) -> dict:
    """Remove computed values to compare all authoring metadata and contracts."""
    data = task.model_dump(mode="json")
    contract = data.pop("expected")
    if contract["kind"] == "success":
        for result in contract["outputs"].values():
            result.pop("value")
    data["answer_contract"] = contract
    return data


def _read_json(path: Path):
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=nonfinite)


def _check_accuracy_report(path, dataset, specifications, tasks, manifest):
    """Check provenance and certificate budgets without requiring identical ULPs.

    Current calculations are independently checked by verify_task. Historical
    observed errors can differ across numerical platforms within their budgets.
    """
    report = _read_json(path)
    if not isinstance(report, dict):
        raise ValueError("The accuracy report must be a JSON object.")
    expected = {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "task_count": len(tasks),
        "categories": dict(Counter(task.category for task in tasks)),
        "difficulties": dict(Counter(task.difficulty for task in tasks)),
        "outcomes": dict(Counter(task.expected.kind for task in tasks)),
        "split_counts": {split: len(ids) for split, ids in manifest["splits"].items()},
        "split_audit": manifest,
    }
    if specifications is not None:
        expected["source_sha256"] = hashlib.sha256(specifications.read_bytes()).hexdigest()
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Accuracy report field {key!r} does not match the corpus/source.")
    certificates = report.get("tasks")
    if not isinstance(certificates, list) or len(certificates) != len(tasks):
        raise ValueError("Accuracy report must contain one certificate per task.")
    by_id = {}
    for certificate in certificates:
        if not isinstance(certificate, dict) or not isinstance(certificate.get("id"), str):
            raise ValueError("Invalid accuracy certificate record.")
        if certificate["id"] in by_id:
            raise ValueError(f"Duplicate certificate ID: {certificate['id']}")
        by_id[certificate["id"]] = certificate
    if set(by_id) != {task.id for task in tasks}:
        raise ValueError("Accuracy certificate IDs do not match the corpus.")
    for task in tasks:
        certificate = by_id[task.id]
        budgets = task.verification.absolute_tolerances
        if certificate.get("method") != task.verification.method or certificate.get("absolute_error_budgets") != budgets:
            raise ValueError(f"{task.id}: inaccurate certificate method/error budgets.")
        errors = certificate.get("observed_absolute_errors")
        if not isinstance(errors, dict) or set(errors) != set(budgets):
            raise ValueError(f"{task.id}: incomplete observed-error certificate.")
        for name, error in errors.items():
            if (isinstance(error, bool) or not isinstance(error, (int, float))
                    or not 0 <= error <= budgets[name] or not math.isfinite(error)):
                raise ValueError(f"{task.id}/{name}: reported error is outside its verification budget.")


def validate_benchmark(
    path: str | Path, *, specifications: str | Path | None = None,
    splits: str | Path | None = None, report: str | Path | None = None,
    publication: bool = True,
) -> ValidationReport:
    """Validate every record and optional companions, without changing any file.

    Invalid records contribute to invalid_count. Corpus-wide issues (coverage,
    split audit or artifact provenance) are reported separately and always fail
    validation, even if every individual record is sound. Expected scientific
    refusals are valid only when both implementations reproduce their error code.
    """
    path = Path(path)
    issues, invalid_lines, checked = [], set(), [str(path)]

    def issue(code, message, file=path, line=None, task_id=None):
        issues.append(ValidationIssue(code, str(message), str(file), line, task_id))
        if line is not None and file == path:
            invalid_lines.add(line)

    try:
        scanned = read_record_file(path, BenchmarkTask)
    except (OSError, UnicodeError) as error:
        issue("file", error)
        return ValidationReport(0, 0, issues, {}, {}, {}, 0, [], {}, checked)
    for line, message in scanned.errors:
        issue("schema", message, line=line)
    records = scanned.records
    tasks = [task for _, task in records]
    if scanned.task_count == 0:
        issue("empty", "The benchmark must contain at least one task.")

    id_lines = defaultdict(list)
    for line, task in records:
        id_lines[task.id].append(line)
    for task_id, lines in id_lines.items():
        if len(lines) > 1:
            for line in lines:
                issue("duplicate_id", f"Duplicate task ID at lines {lines}.", line=line, task_id=task_id)

    for line, task in records:
        try:
            audit_splits([task])
        except ValueError as error:
            issue("family_split", error, line=line, task_id=task.id)
        missing = [name for name in task.required_tools if not callable(getattr(tools, name, None))]
        if missing:
            issue("tool", f"Unavailable scientific tools: {', '.join(missing)}.", line=line, task_id=task.id)
            continue
        try:
            verify_task(task)
        except (ValueError, RuntimeError, TypeError, ArithmeticError, KeyError) as error:
            issue("reference", error, line=line, task_id=task.id)

    manifest = None
    try:
        manifest = audit_splits(tasks)
    except ValueError as error:
        issue("split_policy", error)
    if publication:
        try:
            audit_coverage(tasks)
        except ValueError as error:
            issue("coverage", error)

    if specifications is not None:
        specifications = Path(specifications)
        checked.append(str(specifications))
        try:
            sources = load_specifications(specifications)
            by_id = {source.id: source for source in sources}
            if set(by_id) != set(id_lines):
                issue("source_ids", "Specification IDs do not match dataset IDs.", file=specifications)
            for line, task in records:
                if task.id in by_id and _source_record(task) != by_id[task.id].model_dump(mode="json"):
                    issue("source_contract", "Task inputs, metadata or answer contract differ from the specification.",
                          line=line, task_id=task.id)
        except (OSError, ValueError, TypeError) as error:
            issue("specifications", error, file=specifications)

    if splits is not None:
        splits = Path(splits)
        checked.append(str(splits))
        try:
            if manifest is None or _read_json(splits) != manifest:
                raise ValueError("Split manifest does not match the reviewed corpus partitions.")
        except (OSError, ValueError, TypeError) as error:
            issue("manifest", error, file=splits)

    if report is not None:
        report = Path(report)
        checked.append(str(report))
        try:
            if manifest is None:
                raise ValueError("Cannot certify an accuracy report with invalid corpus partitions.")
            _check_accuracy_report(report, path, specifications, tasks, manifest)
        except (OSError, ValueError, TypeError, KeyError) as error:
            issue("accuracy_report", error, file=report)

    return ValidationReport(
        scanned.task_count, sum(line not in invalid_lines for line, _ in records), issues,
        dict(Counter(task.category for task in tasks)), dict(Counter(task.difficulty for task in tasks)),
        dict(Counter(task.split for task in tasks)), len({task.family for task in tasks}),
        sorted({name for task in tasks for name in task.required_tools}),
        dict(Counter(task.expected.kind for task in tasks)), checked,
    )
