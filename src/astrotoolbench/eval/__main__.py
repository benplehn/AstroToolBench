"""Run a baseline on selected frozen benchmark tasks."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

from ..models import BackendError
from ..models.catalog import MODEL_PROFILES, create_backend, get_profile
from .no_tools import run_no_tools, select_tasks
from .prompts import build_messages
from .with_tools import run_raw_tools
from .raw_tools import tool_definitions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[profile.name for profile in MODEL_PROFILES])
    parser.add_argument("--condition", choices=["no_tools", "raw_tools"], default="no_tools")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show requests without loading or calling a model")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--family")
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum model turns, including the final answer")
    parser.add_argument("--max-tool-calls", type=int, default=32)
    args = parser.parse_args(argv)
    if args.list_models:
        print(json.dumps([asdict(profile) for profile in MODEL_PROFILES], indent=2))
        return 0
    if args.model is None:
        parser.error("--model is required unless --list-models is used")
    if args.max_tokens <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--max-tokens and --timeout must be positive and finite")
    if args.max_steps <= 0 or args.max_tool_calls <= 0:
        parser.error("--max-steps and --max-tool-calls must be positive")
    try:
        profile = get_profile(args.model)
        tasks = select_tasks(args.root, split=args.split, family=args.family, task_id=args.task)
        if profile.backend == "local":
            generation = {
                "max_tokens": args.max_tokens, "do_sample": False,
                "device": args.device, "local_files_only": args.local_files_only,
            }
        else:
            generation = {
                "max_tokens": args.max_tokens, "temperature": 0.0,
                "timeout_s": args.timeout, "max_retries": 0,
            }
        if args.dry_run:
            preview = {
                "model_profile": asdict(profile), "condition": args.condition,
                "generation": generation,
                "requests": [
                    {"task_id": task.id, "messages": [m.model_dump(mode="json") for m in build_messages(task)]}
                    for task in tasks
                ],
            }
            if args.condition == "raw_tools":
                preview.update(tools=[tool.model_dump(mode="json") for tool in tool_definitions()],
                               max_steps=args.max_steps, max_tool_calls=args.max_tool_calls)
            print(json.dumps(preview, ensure_ascii=False, indent=2))
            return 0
        output = args.output or args.root / "results" / profile.name / (args.condition + ".jsonl")
        if output.exists():
            raise FileExistsError(f"Results already exist: {output}. Choose another --output.")
        backend = create_backend(
            profile, env_file=args.env_file or args.root / ".env", max_tokens=args.max_tokens,
            timeout_s=args.timeout, device=args.device, local_files_only=args.local_files_only,
        )
        options = {"root": args.root, "split": args.split, "family": args.family,
                   "task_id": args.task, "generation": generation}
        if args.condition == "raw_tools":
            records = run_raw_tools(backend, profile, output, max_steps=args.max_steps,
                                    max_tool_calls=args.max_tool_calls, **options)
        else:
            records = run_no_tools(backend, profile, output, **options)
        print(f"Saved {len(records)} tasks to {output}")
        return 1 if any(record["status"] == "backend_error" for record in records) else 0
    except (BackendError, ImportError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
