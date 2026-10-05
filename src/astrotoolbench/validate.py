"""Validate the official scientific corpus and its artifacts without an LLM."""

import argparse
from pathlib import Path
import sys

from .benchmark.artifacts import pretty_json
from .benchmark.paths import DEFAULT_REPORT, DEFAULT_SPECIFICATIONS, DEFAULT_SPLITS, DEFAULT_TASKS
from .benchmark.validator import validate_benchmark


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--specifications", type=Path, help="Check authoring metadata/contracts against these sources.")
    parser.add_argument("--splits", type=Path, help="Check this split manifest.")
    parser.add_argument("--report", type=Path, help="Check this scientific accuracy report and provenance.")
    parser.add_argument("--allow-subset", action="store_true", help="Skip publication size/balance checks only.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report to stdout.")
    args = parser.parse_args(argv)
    # Official validation includes all companion artifacts. Custom datasets can
    # select their own companions explicitly, without accidental cross-comparison.
    official = args.tasks.resolve() == DEFAULT_TASKS.resolve()
    result = validate_benchmark(
        args.tasks, specifications=args.specifications or (DEFAULT_SPECIFICATIONS if official else None),
        splits=args.splits or (DEFAULT_SPLITS if official else None),
        report=args.report or (DEFAULT_REPORT if official else None), publication=not args.allow_subset,
    )
    if args.json:
        print(pretty_json(result.to_dict()), end="")
    else:
        print("AstroToolBench v0.1\n")
        for label, value in (("Tasks", result.task_count), ("Valid", result.valid_count),
                             ("Invalid", result.invalid_count), ("Issues", len(result.issues))):
            print(f"{label + ':':<20}{value}")
        print()
        for label, key in (("Simple", "simple"), ("Multi-step", "multistep"),
                           ("Diagnostic", "diagnostic"), ("Trap", "trap")):
            print(f"{label + ':':<20}{result.difficulties.get(key, 0)}")
        print(f"\n{'Families:':<20}{result.family_count}\n{'Tools:':<20}{len(result.tools)}")
        print("Splits: " + ", ".join(f"{name}={result.splits.get(name, 0)}" for name in ("train", "validation", "test")))
        print("\n✓ Benchmark valid" if result.valid else "\n✗ Benchmark invalid")
        for issue in result.issues:
            location = issue.path + (f":{issue.line}" if issue.line is not None else "")
            identity = f" ({issue.task_id})" if issue.task_id else ""
            print(f"{location}{identity}: [{issue.code}] {issue.message}", file=sys.stderr)
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
