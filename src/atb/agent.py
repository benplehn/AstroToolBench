"""Bounded tool-use orchestration with persisted, replayable conversations."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from time import perf_counter
from typing import Any
from uuid import uuid4

from openai import APIError

from atb.client import ClientSettings, api_error_message, build_chat_request, chat
from atb.executor import execute_tool
from atb.prompts import build_messages
from atb.tasks import Task
from atb.tools import TOOLS_B, TOOLS_C


class ProtocolError(ValueError):
    """A response cannot safely continue the tool-result protocol."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _execute_call(api_version: str, call: dict[str, Any]) -> tuple[str, str | None]:
    """Return a tool-result string and a diagnostic error category.

    Invalid requests become tool messages so the model can correct them. Provider
    protocol defects such as absent call IDs are handled before dispatch.
    """
    name = call["function"]["name"]
    arguments = call["function"]["arguments"]
    try:
        parsed = json.loads(arguments, parse_constant=_reject_constant)
        if not isinstance(parsed, dict):
            raise ValueError("Arguments must be an object.")
        json.dumps(parsed, allow_nan=False)
    except (ValueError, TypeError):
        return json.dumps({"error": "Arguments must be a valid JSON object with finite numbers."}), "invalid_arguments"

    known_names = {tool["function"]["name"] for tool in TOOLS_B}
    try:
        result = execute_tool(api_version, name, arguments)
        data = json.loads(result, parse_constant=_reject_constant)
        if not isinstance(data, dict):
            raise ValueError("Tool result must be an object.")
        # Reject overflow to inf as well as literal NaN/Infinity in the result.
        json.dumps(data, allow_nan=False)
    except Exception as error:
        return json.dumps({"error": f"Tool execution failed ({type(error).__name__})."}), "execution_error"
    if "error" in data:
        return result, "unknown_tool" if name not in known_names else "execution_error"
    return result, None


def _validate_calls(calls: list[dict[str, Any]]) -> None:
    """Validate an entire batch before executing any of its requests."""
    seen = set()
    for call in calls:
        call_id = call.get("id")
        function = call.get("function") or {}
        if not isinstance(call_id, str) or not call_id or call_id in seen:
            raise ProtocolError("Tool call IDs must be non-empty and unique within a response.")
        seen.add(call_id)
        if (
            call.get("type") != "function"
            or not isinstance(function.get("name"), str)
            or not function["name"]
            or not isinstance(function.get("arguments"), str)
        ):
            raise ProtocolError("Expected a function call with a name and JSON argument string.")


def _save_trace(path: Path, trace: dict[str, Any]) -> None:
    """Atomically replace the file reserved by this run, including after failures."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(trace, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _token_totals(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Unknown usage remains unknown instead of being reported as zero."""
    totals: dict[str, Any] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(step.get("usage") or {}).get(field) for step in steps]
        totals[field] = sum(values) if all(isinstance(value, int) for value in values) else None
    totals["usage_complete"] = all(totals[field] is not None for field in totals)
    return totals


def run_agent(
    task: Task,
    model: str,
    api_version: str,
    max_steps: int = 8,
    *,
    settings: ClientSettings | None = None,
    results_dir: str | Path = "results/traces",
    run_id: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Run A (no tools), B or C; save and return the complete run trace.

    ``max_steps`` bounds attempted model requests, including the final-answer
    request. Every function request gets one correlated tool-result message.
    Configuration errors raise before network access; API/protocol failures and
    budget exhaustion return persisted traces with explicit terminal statuses.
    Default configuration reads .env from the current working directory.
    """
    api = api_version.upper()
    if api not in {"A", "B", "C"}:
        raise ValueError("api_version must be A, B or C.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    for label, value in (("run_id", run_id), ("task.id", task.id)):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(f"{label} must be a safe filename component.")
    settings = settings or ClientSettings.from_env(Path.cwd() / ".env", model=model)
    if not settings.api_key:
        raise ValueError("An API key is required to run the agent.")
    messages: list[dict[str, Any]] = build_messages(task)
    tools = deepcopy({"A": None, "B": TOOLS_B, "C": TOOLS_C}[api])
    # Validate generation settings before reserving an output or making a request.
    build_chat_request(model, messages, tools, temperature=temperature, max_tokens=max_tokens)
    path = Path(results_dir) / run_id / f"{task.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    trace: dict[str, Any] = {
        "schema_version": 1,
        "trace_type": "agent_run",
        "metadata": {
            "run_id": run_id,
            "task_id": task.id,
            "family": task.family,
            "split": task.split,
            "model": model,
            "api_version": api,
            "base_url": settings.base_url,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "max_steps": max_steps,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout_s": settings.timeout_s,
            "max_retries": settings.max_retries,
            "python_version": platform.python_version(),
            "openai_version": version("openai"),
            "task_sha256": hashlib.sha256(task.model_dump_json().encode()).hexdigest(),
        },
        "status": "running",
        "messages": messages,
        "tools": tools,
        "steps": [],
        "tool_calls": [],
        "final_answer": None,
        "tokens": _token_totals([]),
        "latency_s": 0.0,
        "step_count": 0,
        "tool_call_count": 0,
        "error": None,
        "trace_path": str(path),
    }
    # Exclusive reservation prevents a repeated run/task ID from overwriting data.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    start = perf_counter()
    for index in range(1, max_steps + 1):
        request = build_chat_request(model, deepcopy(messages), tools, temperature=temperature, max_tokens=max_tokens)
        step: dict[str, Any] = {"index": index, "request": request, "response": None, "usage": None}
        trace["steps"].append(step)
        step_start = perf_counter()
        try:
            response = chat(
                messages=deepcopy(messages), tools=tools, model=model,
                settings=settings, temperature=temperature, max_tokens=max_tokens,
            )
            raw = response.model_dump(mode="json")
            step["response"] = raw
            step["usage"] = raw.get("usage")
            if len(raw.get("choices") or []) != 1:
                raise ProtocolError("Expected exactly one completion choice.")
            choice = raw["choices"][0]
            message = choice.get("message") or {}
            if message.get("role") != "assistant":
                raise ProtocolError("Expected an assistant response.")
            # Replay only Chat Completions message fields. Provider extras are
            # preserved in step.response, not sent as unsupported request fields.
            assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
            calls = message.get("tool_calls") or []
            if calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.get("id"),
                        "type": call.get("type"),
                        "function": {
                            "name": (call.get("function") or {}).get("name"),
                            "arguments": (call.get("function") or {}).get("arguments"),
                        },
                    }
                    for call in calls
                ]
            messages.append(assistant)
            finish = choice.get("finish_reason")
            if finish in {"length", "content_filter"}:
                trace["status"] = "truncated" if finish == "length" else "content_filtered"
            elif calls:
                if tools is None:
                    raise ProtocolError("Provider returned tool calls for condition A; no tools were executed.")
                _validate_calls(calls)
                for call in calls:
                    tool_start = perf_counter()
                    result, error_type = _execute_call(api, call)
                    trace["tool_calls"].append({
                        "step": index,
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                        "result": result,
                        "error_type": error_type,
                        "latency_s": perf_counter() - tool_start,
                    })
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            elif finish == "stop" and isinstance(message.get("content"), str) and message["content"].strip():
                trace["final_answer"] = message["content"]
                trace["status"] = "completed"
            elif message.get("refusal"):
                trace["status"] = "provider_refusal"
                trace["error"] = {"type": "provider_refusal", "message": "The provider declined to return content."}
            else:
                raise ProtocolError("Response has neither function calls nor a non-empty final answer.")
        except APIError as error:
            trace["status"] = "api_error"
            trace["error"] = {"type": type(error).__name__, "message": api_error_message(error)}
        except ProtocolError as error:
            trace["status"] = "protocol_error"
            trace["error"] = {"type": "ProtocolError", "message": str(error)}
        finally:
            step["latency_s"] = perf_counter() - step_start
            trace["step_count"] = len(trace["steps"])
            trace["tool_call_count"] = len(trace["tool_calls"])
            trace["tokens"] = _token_totals(trace["steps"])
            trace["latency_s"] = perf_counter() - start
            if trace["status"] == "running" and index == max_steps:
                trace["status"] = "max_steps_reached"
            _save_trace(path, trace)
        if trace["status"] != "running":
            break
    return trace
