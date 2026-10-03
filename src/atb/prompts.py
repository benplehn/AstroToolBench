"""Shared message construction for inspection and A/B/C agent runs."""

from atb.tasks import Task

SYSTEM_PROMPT = (
    "You are a spacecraft flight-dynamics assistant. Use the provided numerical "
    "tools when available. When no tools are provided, solve directly without "
    "claiming that a tool was executed. Do not invent tool results. "
    "For Earth calculations use an equatorial radius of 6378.137 km and a "
    "gravitational parameter of 398600.4418 km^3/s^2. "
    "If the problem is physically invalid or missing essential information, "
    "explain why. When a final answer is possible, return "
    '{"answer": {...}} with explicit units in the keys, or '
    '{"refuse": true, "reason": "..."} for an invalid request.'
)


def build_messages(task: Task) -> list[dict[str, str]]:
    """Expose the prompt only; reference data stays on the evaluation side."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
