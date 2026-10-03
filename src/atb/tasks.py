"""Benchmark task records and JSONL loading."""

import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str
    family: str
    level: str  # "simple", "multi_step", "trap"
    split: str | None = "dev"
    prompt: str
    expected_tools: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    reference: dict[str, Any] = Field(default_factory=dict)
    tolerance: dict[str, float] = Field(default_factory=dict)
    notes: str | None = None


def load_tasks(
    path: str | Path,
    split: str | None = None,
    family: str | None = None,
) -> list[Task]:
    """Load tasks, reject duplicate IDs and optionally filter by split or family."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier de tâches introuvable : {file_path}")

    tasks: list[Task] = []
    seen_ids = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            task = Task(**data)

            if task.id in seen_ids:
                raise ValueError(f"ID en double détecté : {task.id} (ligne {line_num})")
            seen_ids.add(task.id)

            if split is not None and task.split != split:
                continue
            if family is not None and task.family != family:
                continue

            tasks.append(task)

    return tasks
