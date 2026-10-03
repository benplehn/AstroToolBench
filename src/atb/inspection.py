"""Inspect a benchmark prompt, tool requests and the raw provider response."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any
from uuid import uuid4

from openai import APIError

from atb.client import ClientSettings, api_error_message, build_chat_request, chat
from atb.prompts import build_messages
from atb.tasks import load_tasks
from atb.tools import TOOLS_C


def inspect_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Describe the wire-level fields without executing or altering tool calls."""
    choices = []
    for choice in raw.get("choices") or []:
        calls = []
        for call in choice.get("message", {}).get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            parsed = None
            error = None
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    if not isinstance(parsed, dict):
                        error = "Arguments JSON must be an object."
                except json.JSONDecodeError:
                    error = "Arguments are not valid JSON."
            else:
                error = "Expected arguments to be a JSON string."
            calls.append({
                "id": call.get("id"),
                "type": call.get("type"),
                "name": function.get("name"),
                "arguments_type": type(arguments).__name__,
                "arguments_raw": arguments,
                "arguments_parsed": parsed,
                "arguments_error": error,
            })
        choices.append({
            "index": choice.get("index"),
            "finish_reason": choice.get("finish_reason"),
            "tool_call_count": len(calls),
            "tool_calls": calls,
        })
    return {
        "choices": choices,
        "usage": raw.get("usage"),
        "tools_executed": False,
    }


def main(argv: list[str] | None = None, *, project_root: Path | None = None) -> int:
    project_root = project_root or Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="period-001", help="Benchmark task ID")
    parser.add_argument("--tasks", type=Path, default=project_root / "benchmark/tasks.jsonl")
    parser.add_argument("--model", help="Override LLM_MODEL")
    parser.add_argument("--env-file", type=Path, default=project_root / ".env")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tool-choice", choices=("auto", "required", "none"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Print the request without credentials or network")
    parser.add_argument("--require-tool-call", action="store_true", help="Exit 1 if no function call with valid JSON object arguments is returned")
    parser.add_argument("--output", type=Path, help="Save trace here (must not already exist)")
    args = parser.parse_args(argv)

    try:
        tasks = load_tasks(args.tasks)
        task = next((item for item in tasks if item.id == args.task), None)
        if task is None:
            raise ValueError(f"Unknown task ID: {args.task}")
        settings = ClientSettings.from_env(
            args.env_file, model=args.model, timeout_s=args.timeout,
            require_api_key=not args.dry_run,
        )
        messages = build_messages(task)
        request = build_chat_request(
            settings.model, messages, TOOLS_C,
            max_tokens=args.max_tokens, tool_choice=args.tool_choice,
        )
        output = args.output or (
            project_root / "results/completions" /
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.json"
        )
        if not args.dry_run:
            if output.exists():
                raise ValueError(f"Output already exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
        print("=== Complete request body (credentials excluded) ===", flush=True)
        print(json.dumps({"base_url": settings.base_url, "request": request}, indent=2, ensure_ascii=False), flush=True)
        if args.dry_run:
            return 0

        started_at = datetime.now(timezone.utc).isoformat()
        start = perf_counter()
        response = chat(
            messages=messages, tools=TOOLS_C, settings=settings,
            max_tokens=args.max_tokens, tool_choice=args.tool_choice,
        )
        elapsed = perf_counter() - start
        raw = response.model_dump(mode="json")
        observations = inspect_response(raw)
        trace = {
            "schema_version": 1,
            "metadata": {
                "task_id": task.id,
                "family": task.family,
                "split": task.split,
                "api_condition": "C",
                "base_url": settings.base_url,
                "started_at_utc": started_at,
                "latency_s": elapsed,
                "timeout_s": settings.timeout_s,
                "max_retries": settings.max_retries,
                "python_version": platform.python_version(),
                "openai_version": version("openai"),
                "benchmark_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
            },
            "request": request,
            "response": raw,
            "observations": observations,
        }
        with output.open("x", encoding="utf-8") as handle:
            json.dump(trace, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        print("\n=== Raw response (response.model_dump) ===")
        print(json.dumps(raw, indent=2, ensure_ascii=False))
        print("\n=== Fields to observe (no tool has been executed) ===")
        print(json.dumps(observations, indent=2, ensure_ascii=False))
        print(f"\nTrace saved: {output}")
        calls = [call for choice in observations["choices"] for call in choice["tool_calls"]]
        if not calls:
            print("No tool call returned. Inspect content and finish_reason; auto allows a text answer.")
        if args.require_tool_call and (
            not calls or any(
                call["arguments_error"] or not call["id"]
                or call["type"] != "function" or not call["name"]
                for call in calls
            )
        ):
            print("Required tool-call check failed; raw trace preserved.", file=sys.stderr)
            return 1
        return 0
    except APIError as error:
        print(f"Error: {api_error_message(error)}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
