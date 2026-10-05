# Running the agent

The runner connects a model to the numerical tools: it keeps the conversation,
executes the tool calls and stops when there's a final answer (or when
something goes wrong).

## Usage

Install `.[dev,llm]` and set up your provider (see
[completion inspection](completion-inspection.md)). Then from the repo root:

```bash
atb-run --task multi-001 --api C --reasoning-budget 256 --timeout 120
```

It prints the conversation (model answers, tool calls with arguments and
results), then status, tokens, latency, trace path and `PASS` / `FAIL`. If the
run didn't finish, it shows `NOT EVALUATED`. Exit code is 0 only for a finished
run with a correct answer.

`--verbose` adds the system/user messages, call IDs, per-request latency, token
usage, finish reasons and tool error types. `--json` prints a summary for
scripts instead (can't be combined with `--verbose`). The full trace is saved
either way. Missing token usage is shown as unknown.

```bash
atb-run --task multi-001 --model nvidia/nemotron-3.5-lightning-30b-a3b --api C --verbose
atb-run --task multi-001 --api C --json
python scripts/run_agent.py --task multi-001 --api C --max-steps 8 --verbose
```

From Python:

```python
from atb.agent import run_agent
from atb.client import ClientSettings
from atb.tasks import load_tasks

settings = ClientSettings.from_env(".env")
task = next(t for t in load_tasks("benchmark/tasks.jsonl") if t.id == "multi-001")
trace = run_agent(task, settings.model, "C", max_steps=8, settings=settings)
print(trace["status"], trace["final_answer"], trace["trace_path"])
```

`run_agent()` doesn't score; the CLI calls `atb.scoring.score` after the run.

`--reasoning-budget` limits reasoning tokens per request on NVIDIA models that
support it ([API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-5-lightning-30b-a3b-infer)).
By default the field isn't sent; `-1` disables the limit. It's different from
`--max-tokens` (output limit) and `--timeout`. Other endpoints may ignore it.
Use the same value when comparing B and C.

## Message flow

For a two-step task:

```text
system: instructions + final JSON format
user: task prompt
assistant: tool_calls=[hohmann_transfer, id h]
tool: tool_call_id=h, transfer result
assistant: tool_calls=[propagate_orbit with the returned duration, id p]
tool: tool_call_id=p, final state
assistant: final JSON answer
```

Assistant messages keep their tool calls and raw argument strings, and each call
gets exactly one `tool` reply. Arguments must be JSON objects with finite
numbers. Only the existing tool functions can be called.

The runner checks the JSON; numerical/domain checks happen in the executor.
There's no full JSON Schema validation of the arguments yet.

A, B and C all get the same system prompt (including Earth radius and μ). The
model never sees the reference answer, tolerance or expected tools.

## Conditions and step budget

| Condition | Tools sent |
| --- | --- |
| A | none (`tools` and `tool_choice` are omitted) |
| B | `TOOLS_B` |
| C | `TOOLS_C` |

`max_steps` counts model requests, including the last one. Two dependent tool
calls usually take three requests. If the model asks for tools on the last
allowed request, they're executed and saved, and the run stops without a final
answer.

Options: `--model`, `--api`, `--max-steps`, `--max-tokens`, `--reasoning-budget`,
`--timeout`, `--env-file`, `--results-dir`, `--run-id`, `--verbose`, `--json`.
Several tasks can share a run ID, but a task that already has a trace in that run
is refused before any request.

## Statuses

| Status | Meaning |
| --- | --- |
| `completed` | there's a final answer (scored separately) |
| `max_steps_reached` | out of requests, no final answer |
| `api_error` | network/provider error; everything before is kept |
| `protocol_error` | missing/duplicate call IDs, tool calls in condition A, empty choices… |
| `truncated` | hit the output token limit; partial tool calls aren't run |
| `content_filtered` | stopped by the provider's content filter |
| `provider_refusal` | refusal without a usable answer |

Bad JSON, unknown tools and tool failures don't stop the run: they're sent back
as error results (and logged in `tool_calls[].error_type`) so the model can try
again.

## Trace file

Saved at `results/traces/<run_id>/<task_id>.json`, created before the first
request and rewritten after each turn.

| Field | Contents |
| --- | --- |
| `metadata` | run/task, family, split, task hash, model, condition, provider, limits, versions |
| `messages` | the conversation |
| `tools` | schemas sent (null for A) |
| `steps` | each request body, raw response, usage, latency |
| `tool_calls` | IDs, names, arguments, results, error types, turn, latency |
| `final_answer` | final text or null |
| `tokens` | totals + `usage_complete`; missing usage is null, not 0 |
| `step_count`, `tool_call_count` | requests made, tool calls handled |
| `latency_s` | total time, including tools and trace writes |
| `status`, `error` | final status and cleaned-up error message |

No API keys or auth headers in traces. `results/` is gitignored.

## Tests

```bash
python -m pytest -q
```

The tests use fake model responses with the real tools: dependent calls,
parallel calls, error recovery, A/B/C, step limit, protocol errors, missing
usage, trace writing and error redaction.

## A real run

On 2 October 2026, Nemotron 3 Super solved `multi-001` with condition C:

| | |
| --- | --- |
| Requests | 3 |
| Tool calls | 2, one per response |
| Sequence | `hohmann_transfer` → `propagate_orbit` |
| Final x | −5368.7591375471 km |
| Reference / tolerance at the time | −5368.76 km / 5 km |
| Score | correct |
| Tokens | 6110 |
| Latency | ~15.09 s |

The model passed the exact `transfer_time_s` from the first tool to the second.
The [recorded conversation](examples/nvidia-nemotron-agent.json) is the real
one; I only made the trace path relative.

I had to clarify the prompt for this task: it now gives the ECI initial state
and says explicitly to propagate the original orbit without the transfer burns.
The reference didn't change. One task, so not a benchmark result.

## Tolerance and rounding

`multi-001` now accepts an x error of at most **0.1 km** (reference stored to
0.01 km). This is about consistency with the two-body model, not real-world
orbit prediction.

Why I tightened it: if the transfer time is truncated from 3560.540788789012 s to
3560 s, x becomes about −5371.376840 km, i.e. **2.62 km** off (about 4.08 km in
full position, but only x is scored). The old 5 km tolerance would have accepted
that. With 0.1 km it fails, which is what I want. Tests cover truncated,
rounded and exact durations, plus the recorded answer.

The recorded NVIDIA answer passes the new tolerance too (just rescored, no new
request). The table above shows the tolerance in place at the time.

In condition C the model also has to convert radii to altitudes (7000 and
9000 km → 621.863 and 2621.863 km), which is another source of error worth
tracking when comparing A/B/C.
