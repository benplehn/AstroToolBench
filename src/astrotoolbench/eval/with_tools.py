"""Bounded model/tool exchanges with a checkpoint after every event."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter_ns

from ..benchmark.artifacts import pretty_json, write_artifacts
from ..benchmark.schema import BenchmarkTask
from ..models import BackendError, ModelBackend
from ..models.catalog import ModelProfile
from .common import grade_response, new_result
from .no_tools import select_tasks
from .prompts import build_messages
from .raw_tools import execute_call, requested_call, result_message, tool_definitions


def _check_limits(max_steps, max_tool_calls):
    for name, value in (("max_steps", max_steps), ("max_tool_calls", max_tool_calls)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")


def evaluate_raw_task(
    task: BenchmarkTask, backend: ModelBackend, *, model: str,
    max_steps: int = 8, max_tool_calls: int = 32,
    on_update: Callable[[dict], None] | None = None,
) -> dict:
    _check_limits(max_steps, max_tool_calls)
    messages = build_messages(task)
    definitions = tool_definitions()
    result = new_result(task, model, "raw_tools", messages)
    result.update(status="running", steps=[], tool_calls=[],
                  tools=[tool.model_dump(mode="json") for tool in definitions],
                  max_steps=max_steps, max_tool_calls=max_tool_calls)
    seen_ids = set()
    started = perf_counter_ns()

    def checkpoint():
        result["messages"] = [message.model_dump(mode="json") for message in messages]
        result["latency_ms"] = (perf_counter_ns() - started) / 1_000_000
        complete = bool(result["steps"])
        for field in ("input_tokens", "output_tokens"):
            counts = [step["response"]["usage"][field] if step["response"] else None for step in result["steps"]]
            known = [count for count in counts if count is not None]
            # Keep the turns we know about; usage_complete says whether the total is exact.
            result[field] = sum(known) if known else None
            complete = complete and len(known) == len(counts)
        result["usage_complete"] = complete
        if on_update is not None:
            on_update(deepcopy(result))

    checkpoint()
    for index in range(1, max_steps + 1):
        step = {"index": index, "messages": [message.model_dump(mode="json") for message in messages],
                "response": None, "error": None, "latency_ms": None}
        result["steps"].append(step)
        checkpoint()
        turn_started = perf_counter_ns()
        try:
            response = backend.generate(deepcopy(messages), tools=deepcopy(definitions))
        except BackendError as error:
            step["latency_ms"] = (perf_counter_ns() - turn_started) / 1_000_000
            step["error"] = {"type": type(error).__name__, "message": str(error),
                             "status_code": error.status_code, "raw_response": error.raw_response}
            result.update(status="backend_error", error=step["error"])
            checkpoint()
            break
        step["latency_ms"] = (perf_counter_ns() - turn_started) / 1_000_000
        step["response"] = response.model_dump(mode="json")
        result["response"] = step["response"]
        result["answer"] = response.content
        messages.append(response.as_message())
        pending = [requested_call(call, index) for call in response.tool_calls]
        offset = len(result["tool_calls"])
        result["tool_calls"].extend(pending)
        if response.refusal is not None or response.finish_reason in {"length", "content_filter"}:
            grade_response(result, task, response.model_copy(update={"tool_calls": ()}))
        elif response.tool_calls:
            if response.finish_reason != "tool_calls":
                # Some gateways report "stop" with tool calls; run them anyway, but keep a note.
                step["warnings"] = [f"tool calls with finish_reason={response.finish_reason!r}"]
            if any(call.id in seen_ids for call in response.tool_calls):
                result["status"] = "protocol_error"
            elif len(result["tool_calls"]) > max_tool_calls:
                result["status"] = "max_tool_calls_reached"
            else:
                seen_ids.update(call.id for call in response.tool_calls)
                checkpoint()
                for position, call in enumerate(response.tool_calls, offset):
                    record = execute_call(call, step=index)
                    result["tool_calls"][position] = record
                    messages.append(result_message(record))
                    checkpoint()
                if index == max_steps:
                    result["status"] = "max_steps_reached"
        else:
            grade_response(result, task, response)
        if result["status"] != "running":
            for record in result["tool_calls"][offset:]:
                if record["status"] == "pending":
                    record.update(status="skipped", success=False,
                                  error={"type": result["status"], "message": "Tool call was not executed."})
            checkpoint()
            break
        checkpoint()
    return result


def run_raw_tools(
    backend: ModelBackend, profile: ModelProfile, output: str | Path, *,
    root: str | Path = ".", split: str | None = None, family: str | None = None,
    task_id: str | None = None, generation: dict | None = None,
    max_steps: int = 8, max_tool_calls: int = 32,
) -> list[dict]:
    _check_limits(max_steps, max_tool_calls)
    tasks = select_tasks(root, split=split, family=family, task_id=task_id)
    output = Path(output)
    directory = output.with_suffix(".traces")
    paths = {task.id: directory / (task.id + ".json") for task in tasks}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Task traces already exist in {directory}. Choose another output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        directory.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            path = paths[task.id]
            # Own this path before making any requests; updates replace it atomically.
            with path.open("x", encoding="utf-8") as trace:
                trace.write("{}\n")

            def save(record):
                record.update(model_profile=asdict(profile), generation=dict(generation or {}), trace_path=str(path))
                write_artifacts({path: pretty_json(record)})

            result = evaluate_raw_task(task, backend, model=profile.model, max_steps=max_steps,
                                       max_tool_calls=max_tool_calls, on_update=save)
            result.update(model_profile=asdict(profile), generation=dict(generation or {}), trace_path=str(path))
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            records.append(result)
    return records
