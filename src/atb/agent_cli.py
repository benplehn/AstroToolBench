"""Run one benchmark task through the bounded agent and report its score."""

import argparse
import json
from pathlib import Path
import sys

from atb.agent import run_agent
from atb.client import ClientSettings
from atb.scoring import score
from atb.tasks import load_tasks


def _display_content(content: object) -> str:
    """Indent JSON for reading while preserving malformed arguments as text."""
    if isinstance(content, str):
        try:
            content = json.dumps(json.loads(content), indent=2, ensure_ascii=False)
        except ValueError:
            pass
    else:
        content = json.dumps(content, indent=2, ensure_ascii=False)
    return "\n".join(f"    {line}" for line in content.splitlines())


def format_run_report(task_id: str, model: str, api: str, trace: dict, grading: dict | None, *, verbose: bool) -> str:
    """Render the persisted conversation followed by execution metrics and grade."""
    lines = [f"Task: {task_id}", f"Model: {model}", f"Condition: {api}", "", "Conversation"]
    turn = 0
    for message in trace["messages"]:
        role = message["role"]
        if role in {"system", "user"} and not verbose:
            continue
        if role == "assistant":
            turn += 1
            lines.append(f"  Turn {turn} · Assistant")
        elif role == "tool":
            suffix = f" [{message['tool_call_id']}]" if verbose else ""
            lines.append(f"  Tool result{suffix}")
        else:
            lines.append(f"  {role.capitalize()}")
        if message.get("content"):
            lines.append(_display_content(message["content"]))
        for call in message.get("tool_calls") or []:
            suffix = f" [{call['id']}]" if verbose else ""
            lines.append(f"    Call: {call['function']['name']}{suffix}")
            lines.append("    Arguments:")
            lines.append(_display_content(call["function"]["arguments"]))

    if verbose:
        lines.extend(["", "Model requests"])
        for step in trace["steps"]:
            response = step.get("response") or {}
            choices = response.get("choices") or []
            finish = choices[0].get("finish_reason") if choices else None
            usage = step.get("usage") or {}
            tokens = usage.get("total_tokens")
            lines.append(
                f"  Turn {step['index']}: {step['latency_s']:.3f} s; "
                f"tokens: {tokens if tokens is not None else 'unknown'}; "
                f"finish reason: {finish or 'unavailable'}"
            )
        for call in trace["tool_calls"]:
            if call.get("error_type"):
                lines.append(f"  Tool error [{call['id']}]: {call['error_type']}")

    tokens = trace["tokens"]
    usage = "; ".join(
        f"{label}: {tokens.get(key) if tokens.get(key) is not None else 'unknown'}"
        for label, key in (("prompt", "prompt_tokens"), ("completion", "completion_tokens"), ("total", "total_tokens"))
    )
    lines.extend([
        "", f"Status: {trace['status']}",
        f"Model requests: {trace['step_count']} · Tool calls: {trace['tool_call_count']}",
        f"Tokens: {usage}", f"Latency: {trace['latency_s']:.3f} s",
        f"Trace: {trace['trace_path']}",
    ])
    if trace["final_answer"] is None:
        lines.append("Final answer: unavailable")
    if trace.get("error"):
        lines.append(f"Error: {trace['error']['message']}")
    if grading is None:
        lines.append("Score: NOT EVALUATED (no completed final answer)")
    else:
        lines.append(f"Score: {'PASS (1/1)' if grading['correct'] else 'FAIL (0/1)'}")
        lines.append(grading["message"])
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, project_root: Path | None = None) -> int:
    root = project_root or Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--tasks", type=Path, default=root / "benchmark/tasks.jsonl")
    parser.add_argument("--model", help="Override LLM_MODEL")
    parser.add_argument("--api", choices=("A", "B", "C"), default="C")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--verbose", action="store_true", help="Include system/user messages, call IDs and per-turn metrics")
    output.add_argument("--json", action="store_true", help="Print a machine-readable run summary instead of the conversation")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-budget", type=int, help="NVIDIA reasoning-token limit; omitted by default (-1 means no enforcement)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--env-file", type=Path, default=root / ".env")
    parser.add_argument("--results-dir", type=Path, default=root / "results/traces")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        task = next((task for task in load_tasks(args.tasks) if task.id == args.task), None)
        if task is None:
            raise ValueError(f"Unknown task ID: {args.task}")
        settings = ClientSettings.from_env(args.env_file, model=args.model, timeout_s=args.timeout)
        trace = run_agent(
            task, settings.model, args.api, args.max_steps, settings=settings,
            results_dir=args.results_dir, run_id=args.run_id, max_tokens=args.max_tokens,
            reasoning_budget=args.reasoning_budget,
        )
        grading = score(task, trace["final_answer"]) if trace["status"] == "completed" else None
        summary = {
            "task_id": task.id,
            "status": trace["status"],
            "step_count": trace["step_count"],
            "tool_call_count": trace["tool_call_count"],
            "tokens": trace["tokens"],
            "latency_s": trace["latency_s"],
            "final_answer": trace["final_answer"],
            "score": grading,
            "error": trace["error"],
            "trace_path": trace["trace_path"],
        }
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print(format_run_report(task.id, settings.model, args.api, trace, grading, verbose=args.verbose))
        return 0 if grading and grading["correct"] else 1
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
