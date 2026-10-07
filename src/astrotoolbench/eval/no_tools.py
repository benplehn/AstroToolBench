"""Single-turn baseline A and incremental JSONL results."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter_ns

from ..benchmark import BENCHMARK_COMMIT, BENCHMARK_TAG, load_frozen_benchmark
from ..benchmark.comparison import matches_answer
from ..benchmark.loader import _unique_object
from ..benchmark.schema import BenchmarkTask
from ..models import BackendError, ModelBackend
from ..models.catalog import ModelProfile
from .prompts import build_messages


def _parse_answer(text: str | None) -> dict | None:
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    try:
        parsed = json.loads(text or "", object_pairs_hook=_unique_object, parse_constant=nonfinite)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def evaluate_task(task: BenchmarkTask, backend: ModelBackend, *, model: str) -> dict:
    messages = build_messages(task)
    result = {
        "task_id": task.id, "family": task.family, "difficulty": task.difficulty,
        "category": task.category, "split": task.split, "model": model,
        "condition": "no_tools", "benchmark": BENCHMARK_TAG, "benchmark_commit": BENCHMARK_COMMIT,
        "answer": None, "correct": False, "input_tokens": None, "output_tokens": None,
        "response": None, "error": None,
        "messages": [message.model_dump(mode="json") for message in messages],
    }
    started = perf_counter_ns()
    try:
        response = backend.generate(messages, tools=None)
    except BackendError as error:
        result.update(status="backend_error", error={"type": type(error).__name__,
                      "message": str(error), "status_code": error.status_code})
        result["latency_ms"] = (perf_counter_ns() - started) / 1_000_000
        return result
    result["latency_ms"] = (perf_counter_ns() - started) / 1_000_000
    result.update(
        answer=response.content, response=response.model_dump(mode="json"),
        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
    )
    if response.tool_calls:
        status = "unexpected_tool_call"
    elif response.refusal is not None or response.finish_reason == "content_filter":
        status = "refused"
    elif response.finish_reason == "length":
        status = "truncated"
    elif response.finish_reason != "stop":
        status = "incomplete_response"
    elif not response.content or not response.content.strip():
        status = "empty_response"
    else:
        answer = _parse_answer(response.content)
        status = "completed" if answer is not None else "invalid_answer"
        result["correct"] = answer is not None and matches_answer(task, answer)
    result["status"] = status
    return result


def select_tasks(
    root: str | Path = ".", *, split: str | None = None, family: str | None = None,
    task_id: str | None = None,
) -> list[BenchmarkTask]:
    tasks = load_frozen_benchmark(root, split=split, family=family)
    selected = [task for task in tasks if task_id is None or task.id == task_id]
    if not selected:
        raise ValueError("No benchmark tasks match the requested selection.")
    return selected


def run_no_tools(
    backend: ModelBackend, profile: ModelProfile, output: str | Path, *,
    root: str | Path = ".", split: str | None = None, family: str | None = None,
    task_id: str | None = None, generation: dict | None = None,
) -> list[dict]:
    tasks = select_tasks(root, split=split, family=family, task_id=task_id)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    # An existing run is evidence, including its failures; never overwrite it.
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for task in tasks:
            record = evaluate_task(task, backend, model=profile.model)
            record.update(model_profile=asdict(profile), generation=dict(generation or {}))
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            records.append(record)
    return records
