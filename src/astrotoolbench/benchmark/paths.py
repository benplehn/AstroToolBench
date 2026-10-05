"""Repository-relative defaults for the official scientific corpus."""

from pathlib import Path

DOMAIN_DIRECTORY = Path("tasks/astrodynamics")
DEFAULT_TASKS = DOMAIN_DIRECTORY / "tasks.jsonl"
DEFAULT_SPECIFICATIONS = DOMAIN_DIRECTORY / "specifications.jsonl"
DEFAULT_SPLITS = DOMAIN_DIRECTORY / "splits.json"
DEFAULT_REPORT = Path("docs/reference-verification.json")
