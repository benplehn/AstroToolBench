"""Shared message construction for inspection and A/B/C agent runs."""

from atb.tasks import Task

SYSTEM_PROMPT = (
    "You are a spacecraft flight-dynamics assistant. Use the provided numerical "
    "tools when available. When no tools are provided, solve directly without "
    "claiming that a tool was executed. Do not invent tool results. "
    "For Earth calculations use an equatorial radius of 6378.137 km and a "
    "gravitational parameter of 398600.4418 km^3/s^2. "
    "Your final response must be exactly one JSON object, without prose or "
    "Markdown fences. For a valid task, return "
    '{"answer": {...}} with numeric values and explicit units in the keys. '
    "For a physically invalid task or missing essential information, return "
    '{"refuse": true, "reason": "..."}, explaining the problem in reason. '
    "Preserve the precision of intermediate tool results when chaining calls."
)


def build_messages(task: Task) -> list[dict[str, str]]:
    """Expose the prompt only; reference data stays on the evaluation side."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
