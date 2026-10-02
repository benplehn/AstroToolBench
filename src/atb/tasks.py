import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str
    family: str
    level: str  # "simple", "multi_step", "trap"
    split: Optional[str] = "dev"
    prompt: str
    expected_tools: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    reference: Dict[str, Any] = Field(default_factory=dict)
    tolerance: Dict[str, float] = Field(default_factory=dict)
    notes: Optional[str] = None


def load_tasks(
    path: str | Path,
    split: Optional[str] = None,
    family: Optional[str] = None,
) -> List[Task]:
    """Charge les tâches JSONL avec filtrage optionnel par split et famille."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier de tâches introuvable : {file_path}")

    tasks: List[Task] = []
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