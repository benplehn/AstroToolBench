"""Run one benchmark task through the bounded agent and report its score."""

import argparse
import json
from pathlib import Path
import sys

from atb.agent import run_agent
from atb.client import ClientSettings
from atb.scoring import score
from atb.tasks import load_tasks


def main(argv: list[str] | None = None, *, project_root: Path | None = None) -> int:
    root = project_root or Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--tasks", type=Path, default=root / "benchmark/tasks.jsonl")
    parser.add_argument("--model", help="Override LLM_MODEL")
    parser.add_argument("--api", choices=("A", "B", "C"), default="C")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
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
        )
        grading = score(task, trace["final_answer"]) if trace["status"] == "completed" else None
        print(json.dumps({
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
        }, indent=2, ensure_ascii=False))
        return 0 if grading and grading["correct"] else 1
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
