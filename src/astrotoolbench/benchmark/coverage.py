"""Publication-level size and difficulty coverage for the v0.1 corpus."""

from collections import Counter

from .schema import BenchmarkTask

MINIMUM_TASKS = 40
DESIRED_TASKS = 50
DIFFICULTY_TARGETS = {"simple": 0.40, "multistep": 0.25, "diagnostic": 0.15, "trap": 0.20}
CATEGORIES = {"propagation", "maneuvers", "eclipse", "proximity", "multistep", "diagnostic"}


def audit_coverage(tasks: list[BenchmarkTask]) -> dict:
    """Require all six groups and four types, allowing five percentage points of slack."""
    count = len(tasks)
    if count < MINIMUM_TASKS:
        raise ValueError(f"v0.1 requires at least {MINIMUM_TASKS} tasks, received {count}.")
    categories = Counter(task.category for task in tasks)
    difficulties = Counter(task.difficulty for task in tasks)
    families = {task.family for task in tasks}
    if set(categories) != CATEGORIES or len(families) < 5:
        raise ValueError("v0.1 requires all six categories and at least five problem families.")
    if set(difficulties) != set(DIFFICULTY_TARGETS):
        raise ValueError("v0.1 requires all four difficulty types.")
    for name, target in DIFFICULTY_TARGETS.items():
        fraction = difficulties[name] / count
        if abs(fraction - target) > 0.05 + 1e-12:
            raise ValueError(f"Difficulty {name} is {fraction:.1%}; target {target:.0%} ± 5 percentage points.")
    return {"task_count": count, "desired_task_count": DESIRED_TASKS, "family_count": len(families),
            "categories": dict(categories), "difficulties": dict(difficulties),
            "difficulty_fractions": {name: difficulties[name] / count for name in DIFFICULTY_TARGETS}}
