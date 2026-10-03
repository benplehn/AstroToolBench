"""Run five representative tasks and retain every outcome, including failures."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from atb.agent import run_agent
from atb.client import ClientSettings
from atb.scoring import score
from atb.tasks import load_tasks


TASK_IDS = ("period-001", "hoh-001", "ecl-001", "multi-001", "trap-001")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--api", choices=("A", "B", "C"), default="C")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--reasoning-budget", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = ClientSettings.from_env(root / ".env", model=args.model, timeout_s=args.timeout)
    tasks = {task.id: task for task in load_tasks(root / "benchmark/tasks.jsonl")}
    selected = [tasks[task_id] for task_id in TASK_IDS]
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-smoke-{uuid4().hex[:6]}"
    path = root / "results/smoke" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "model": settings.model,
        "api_version": args.api,
        "temperature": 0.0,
        "max_steps": 8,
        "max_tokens": 4096,
        "reasoning_budget": args.reasoning_budget,
        "timeout_s": settings.timeout_s,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": [],
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Run: {run_id}", flush=True)
    for task in selected:
        trace = run_agent(
            task, settings.model, args.api, settings=settings, run_id=run_id,
            results_dir=root / "results/traces", reasoning_budget=args.reasoning_budget,
        )
        grade = score(task, trace["final_answer"]) if trace["final_answer"] is not None else None
        try:
            parsed = json.loads(trace["final_answer"] or "")
            final_json_valid = isinstance(parsed, dict) and ("answer" in parsed or "refuse" in parsed)
        except ValueError:
            final_json_valid = False
        report["tasks"].append({
            "task_id": task.id, "status": trace["status"], "score": grade,
            "final_json_valid": final_json_valid,
            "step_count": trace["step_count"], "tool_call_count": trace["tool_call_count"],
            "tokens": trace["tokens"], "latency_s": trace["latency_s"],
            "final_answer": trace["final_answer"], "error": trace["error"],
            "trace_path": str(Path(trace["trace_path"]).relative_to(root)),
            "task_sha256": trace["metadata"]["task_sha256"],
        })
        report["passed"] = sum(bool(row["score"] and row["score"]["correct"]) for row in report["tasks"])
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        outcome = "PASS" if grade and grade["correct"] else "FAIL"
        print(f"{task.id}: {outcome} ({trace['status']}, {trace['tool_call_count']} tools)", flush=True)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary: {path.relative_to(root)}", flush=True)
    return 0 if report["passed"] == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
