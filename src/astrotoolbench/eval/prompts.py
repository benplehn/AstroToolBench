"""Public task data shared by model conditions, without reference answers."""

import json

from ..benchmark.schema import BenchmarkTask
from ..models import Message

SYSTEM_PROMPT = (
    "You are solving an orbital-mechanics benchmark. Solve the problem directly "
    "when no tools are available; do not claim to have executed a tool. "
    "The scientific contract supports two-body elliptic orbits outside Earth, "
    "a cylindrical Earth shadow and a fixed Sun direction. Respect the supplied "
    "units, frame and numerical search settings. Return exactly one JSON object "
    "without prose or Markdown fences. For a valid problem, return "
    '{"outputs": {"requested_output_name": value}} with exactly the outputs '
    "requested in the problem, in their stated units. Values may be numbers, "
    "vectors or booleans. For an invalid or unsupported problem, return "
    '{"error": "code", "reason": "explanation"}. Error codes are invalid_input, '
    "inconsistent_units, missing_input, physical_impossibility, unsupported_model "
    "and unachievable_precision. Do not invent missing data."
)


def build_messages(task: BenchmarkTask) -> list[Message]:
    data = {
        "context": task.context.model_dump(mode="json"),
        "inputs": {name: quantity.model_dump(mode="json") for name, quantity in task.inputs.items()},
    }
    # Do not serialize the whole task: notes and verification can reveal answers.
    content = task.prompt + "\n\nProblem data:\n" + json.dumps(data, ensure_ascii=False, allow_nan=False)
    return [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=content)]
