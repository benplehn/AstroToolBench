"""Result metadata and final-answer checks shared by both conditions."""

import json

from ..benchmark import BENCHMARK_COMMIT, BENCHMARK_TAG
from ..benchmark.comparison import matches_answer
from ..benchmark.loader import _unique_object
from ..benchmark.schema import BenchmarkTask
from ..models import Message, ModelResponse


def new_result(task: BenchmarkTask, model: str, condition: str, messages: list[Message]) -> dict:
    return {
        "task_id": task.id, "family": task.family, "difficulty": task.difficulty,
        "category": task.category, "split": task.split, "model": model,
        "condition": condition, "benchmark": BENCHMARK_TAG, "benchmark_commit": BENCHMARK_COMMIT,
        "answer": None, "correct": False, "format_ok": False,
        "input_tokens": None, "output_tokens": None, "usage_complete": False,
        "response": None, "error": None,
        "messages": [message.model_dump(mode="json") for message in messages],
    }


def _nonfinite(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _parse_answer(text: str | None) -> dict | None:
    try:
        parsed = json.loads(text or "", object_pairs_hook=_unique_object, parse_constant=_nonfinite)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _extract_answer(text: str) -> dict | None:
    """Last JSON object with "outputs" or "error", wherever it is in the text.

    Models often wrap the answer in ```json fences or add a sentence around it.
    That is a format problem, not a wrong value, so it is scored separately.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    found, start = None, text.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict) and ("outputs" in value or "error" in value):
            found = value
        start = text.find("{", end)
    return found


def grade_response(result: dict, task: BenchmarkTask, response: ModelResponse) -> None:
    result["answer"] = response.content
    if response.tool_calls:
        status = "unexpected_tool_call"
    elif response.refusal is not None or response.finish_reason == "content_filter":
        status = "refused"
    elif response.finish_reason == "length":
        status = "truncated"
    elif response.finish_reason == "invalid_tool_call":
        status = "invalid_tool_call"
    elif response.finish_reason != "stop":
        status = "incomplete_response"
    elif not response.content or not response.content.strip():
        status = "empty_response"
    else:
        strict = _parse_answer(response.content)
        answer = strict if strict is not None else _extract_answer(response.content)
        status = "completed" if answer is not None else "invalid_answer"
        result["format_ok"] = strict is not None
        result["correct"] = answer is not None and matches_answer(task, answer)
    result["status"] = status
