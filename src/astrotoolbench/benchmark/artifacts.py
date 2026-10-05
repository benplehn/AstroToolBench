"""Deterministic scientific artifacts with check-only and atomic file replacement."""

import json
import os
from pathlib import Path
import tempfile

from .schema import Record


def serialize_records(records: list[Record]) -> str:
    return "".join(json.dumps(record.model_dump(mode="json"), ensure_ascii=False, allow_nan=False,
                             sort_keys=True) + "\n" for record in records)


def pretty_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"


def write_artifacts(artifacts: dict[Path, str], *, check: bool = False) -> None:
    """Prepare every file before replacing any target; check mode never creates files.

    Each replacement is atomic. This is not a filesystem transaction across all
    files; the report hashes identify an incomplete multi-file replacement.
    """
    resolved = [path.resolve() for path in artifacts]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Artifact paths must be distinct.")
    if check:
        differences = [str(path) for path, contents in artifacts.items()
                       if not path.exists() or path.read_bytes() != contents.encode("utf-8")]
        if differences:
            raise ValueError("Artifacts differ from verified regeneration: " + ", ".join(differences))
        return
    temporary = []
    try:
        for path, contents in artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix="." + path.name + ".", delete=False) as file:
                temporary.append((Path(file.name), path))
                file.write(contents)
        for source, target in temporary:
            os.replace(source, target)
    finally:
        for source, _ in temporary:
            source.unlink(missing_ok=True)
