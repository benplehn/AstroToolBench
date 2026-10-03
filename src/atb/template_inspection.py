"""Render benchmark prompts and tool definitions with an open chat tokenizer."""

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

from atb.prompts import build_messages
from atb.tasks import load_tasks
from atb.tools import TOOLS_B, TOOLS_C


DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


def load_tokenizer(model: str = DEFAULT_TOKENIZER, *, revision: str = "main", local_files_only: bool = False):
    """Load tokenizer assets and require an attached conversation template."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if not tokenizer.chat_template:
        raise ValueError(f"Tokenizer {model!r} has no chat template.")
    return tokenizer


def prepare_messages(messages: list[dict]) -> list[dict]:
    """Adapt API argument strings to Transformers objects without changing a trace."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a non-empty message list.")
    prepared = deepcopy(messages)
    for message in prepared:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Expected messages with system, user, assistant or tool roles.")
        if not isinstance(message.get("content"), str):
            if message.get("content") is None and message.get("tool_calls"):
                message["content"] = ""
            else:
                raise ValueError("Expected textual message content, or null content with tool calls.")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("Expected a tool-call list.")
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise ValueError("Expected named function tool calls.")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError as error:
                    raise ValueError("Cannot render tool arguments that are invalid JSON.") from error
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must decode to a JSON object.")
            json.dumps(arguments, allow_nan=False)
            function["arguments"] = arguments
    return prepared


def render_chat(tokenizer, messages: list[dict], *, tools=TOOLS_C, add_generation_prompt: bool = True) -> str:
    """Serialize tools and messages; append an assistant prefix only when requested."""
    return tokenizer.apply_chat_template(
        prepare_messages(messages),
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def count_tokens(tokenizer, text: str) -> int:
    """Templates already contain special tokens; avoid adding them twice."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _without_descriptions(value):
    if isinstance(value, dict):
        return {key: _without_descriptions(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [_without_descriptions(item) for item in value]
    return value


def compare_tools(tokenizer, messages: list[dict]) -> dict:
    """Measure B/C on identical initial messages, before any interface-specific calls."""
    initial = []
    for message in messages:
        if message.get("role") not in {"system", "user"}:
            break
        initial.append(message)
    if not initial or not any(message["role"] == "user" for message in initial):
        raise ValueError("Tool comparison requires initial instructions and a user prompt.")
    baseline = count_tokens(tokenizer, render_chat(tokenizer, initial, tools=None))
    conditions = {}
    for condition, tools in (("B", TOOLS_B), ("C", TOOLS_C)):
        schema_json = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        prompt_tokens = count_tokens(tokenizer, render_chat(tokenizer, initial, tools=tools))
        stripped_tokens = count_tokens(tokenizer, render_chat(tokenizer, initial, tools=_without_descriptions(tools)))
        conditions[condition] = {
            "schema_json_tokens": count_tokens(tokenizer, schema_json),
            "prompt_tokens": prompt_tokens,
            "tool_overhead_tokens": prompt_tokens - baseline,
            "prompt_tokens_without_descriptions": stripped_tokens,
            "description_fields_cost_tokens": prompt_tokens - stripped_tokens,
            "schema_sha256": hashlib.sha256(schema_json.encode()).hexdigest(),
        }
    return {
        "messages": initial,
        "baseline_prompt_tokens": baseline,
        "conditions": conditions,
        "C_minus_B_schema_json_tokens": conditions["C"]["schema_json_tokens"] - conditions["B"]["schema_json_tokens"],
        "C_minus_B_prompt_tokens": conditions["C"]["prompt_tokens"] - conditions["B"]["prompt_tokens"],
        "method": {
            "schema_json": "Compact UTF-8 JSON list, ensure_ascii=False; no extra special tokens.",
            "prompt": "Same initial messages and generation prefix, varying only tool definitions.",
            "tool_overhead": "Prompt with tools minus prompt without tools; includes template tool instructions.",
            "descriptions": "Prompt with full schemas minus prompt with all description fields removed; includes description keys and punctuation.",
        },
    }


def _save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def main(argv: list[str] | None = None, *, project_root: Path | None = None) -> int:
    root = project_root or Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_TOKENIZER, help="Hugging Face tokenizer repository or local directory")
    parser.add_argument("--revision", default="main", help="Repository branch, tag or commit hash")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached assets without accessing the network")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--task", help="Benchmark task ID (default: period-001)")
    source.add_argument("--trace", type=Path, help="Recorded agent-run JSON; render its messages and recorded tools")
    parser.add_argument("--tasks", type=Path, default=root / "benchmark/tasks.jsonl", help="Benchmark task file")
    parser.add_argument("--output", type=Path, help="Save only the rendered text; the file must not exist")
    parser.add_argument("--compare-tools", action="store_true", help="Compare B/C schema and prompt tokens and isolate description costs")
    parser.add_argument("--report", type=Path, help="Save inspection metadata and token measurements as JSON; the file must not exist")
    args = parser.parse_args(argv)
    try:
        for output in (args.output, args.report):
            if output and output.exists():
                raise ValueError(f"Output already exists: {output}")
        if args.output and args.report and args.output.resolve() == args.report.resolve():
            raise ValueError("Rendered text and report must use different paths.")
        if args.trace:
            raw = args.trace.read_bytes()
            trace = json.loads(raw)
            if not isinstance(trace, dict) or trace.get("trace_type") != "agent_run":
                raise ValueError("Expected an agent-run trace JSON object.")
            messages = prepare_messages(trace.get("messages"))
            tools = trace.get("tools")
            if tools is not None and not isinstance(tools, list):
                raise ValueError("Trace tools must be a schema list or null.")
            label = (trace.get("metadata") or {}).get("task_id", args.trace.stem)
            condition = (trace.get("metadata") or {}).get("api_version", "recorded")
            # A completed conversation is inspected as-is, without starting another answer.
            generation_prompt = messages[-1]["role"] != "assistant" or bool(messages[-1].get("tool_calls"))
            provenance = {"trace": str(args.trace), "trace_sha256": hashlib.sha256(raw).hexdigest()}
        else:
            task_id = args.task or "period-001"
            task = next((task for task in load_tasks(args.tasks) if task.id == task_id), None)
            if task is None:
                raise ValueError(f"Unknown task ID: {task_id}")
            messages, tools, label, condition = build_messages(task), TOOLS_C, task.id, "C"
            generation_prompt = True
            provenance = {"task_id": task.id, "task_sha256": hashlib.sha256(task.model_dump_json().encode()).hexdigest()}
        tokenizer = load_tokenizer(args.model, revision=args.revision, local_files_only=args.local_files_only)
        rendered = render_chat(tokenizer, messages, tools=tools, add_generation_prompt=generation_prompt)
        report = {
            "schema_version": 1,
            "inspection_type": "chat_template",
            "tokenizer": args.model,
            "revision": args.revision,
            "transformers_version": version("transformers"),
            "tokenizers_version": version("tokenizers"),
            "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template(tools=tools).encode()).hexdigest(),
            "source": provenance,
            "condition": condition,
            "add_generation_prompt": generation_prompt,
            "message_count": len(messages),
            "tool_call_count": sum(len(message.get("tool_calls") or []) for message in messages),
            "rendered_tokens": count_tokens(tokenizer, rendered),
            "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
        if args.compare_tools:
            report["comparison"] = compare_tools(tokenizer, messages)
        if args.output:
            _save_text(args.output, rendered)
        if args.report:
            _save_text(args.report, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    except ImportError:
        print('Error: install the tokenizer extra with: python -m pip install ".[tokenizer]"', file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Tokenizer: {args.model}")
    print(f"Revision: {args.revision}")
    print(f"Class: {type(tokenizer).__name__}")
    print(f"Vocabulary entries (including added tokens): {len(tokenizer)}")
    print(f"Fast tokenizer: {tokenizer.is_fast}")
    print("Chat template: available")
    print("Model weights: not loaded")
    print(f"Task: {label}")
    print(f"Condition: {condition}")
    print(f"Messages: {report['message_count']} · Tool calls: {report['tool_call_count']}")
    print(f"Rendered tokens: {report['rendered_tokens']}")
    print(f"Generation prompt: {generation_prompt}")
    print("\n=== Rendered chat ===")
    print(rendered)
    if args.output:
        print(f"\nRendered text saved to: {args.output}")
    if args.compare_tools:
        print("\n=== B/C token comparison (same initial messages) ===")
        print("Condition | Schema JSON | Prompt | Tool overhead | Description fields")
        for condition, counts in report["comparison"]["conditions"].items():
            print(f"{condition} | {counts['schema_json_tokens']} | {counts['prompt_tokens']} | "
                  f"{counts['tool_overhead_tokens']} | {counts['description_fields_cost_tokens']}")
        print(f"C minus B prompt tokens: {report['comparison']['C_minus_B_prompt_tokens']}")
        print("Description cost removes all description fields, including their keys and punctuation.")
    if args.report:
        print(f"Report saved to: {args.report}")
    return 0
