# Agent execution

The bounded runner connects a model to AstroToolBench's numerical tools. It
preserves the conversation, executes function requests and continues until a
final response or an explicit stopping condition.

## Usage

Install `.[dev,llm]` and configure the provider as described in
[completion inspection](completion-inspection.md). From the repository root:

```bash
atb-run --task multi-001 --api C --reasoning-budget 256 --timeout 120
```

The default report prints the conversation in order: assistant responses, tool
names, arguments and results, followed by status, tokens, latency, trace path and
the numerical grade (`PASS` or `FAIL`). An incomplete run is explicitly marked
`NOT EVALUATED`. Exit code 0 requires a completed conversation and a passing score.
An incorrect final answer or any other terminal status produces exit code 1.

Use `--verbose` to include the initial system/user messages, call IDs that match
each tool result to its request, per-request latency, token usage and finish
reasons, and tool error categories. These reports are printed after execution;
the trace file is saved after every turn by the runner.

```bash
atb-run --task multi-001 --model nvidia/nemotron-3.5-lightning-30b-a3b --api C --verbose
atb-run --task multi-001 --api C --json
```

`--json` retains the machine-readable summary for scripts. It is mutually
exclusive with `--verbose`; the complete trace remains in the recorded JSON file
regardless of the output mode. Missing provider token usage is shown as unknown.

For the repository-local wrapper:

```bash
python scripts/run_agent.py --task multi-001 --api C --max-steps 8 --verbose
```

The Python API returns a JSON-serializable trace:

```python
from atb.agent import run_agent
from atb.client import ClientSettings
from atb.tasks import load_tasks

settings = ClientSettings.from_env(".env")
task = next(t for t in load_tasks("benchmark/tasks.jsonl") if t.id == "multi-001")
trace = run_agent(task, settings.model, "C", max_steps=8, settings=settings)
print(trace["status"], trace["final_answer"], trace["trace_path"])
```

`run_agent()` itself does not grade answers. Reference scoring belongs to the
caller; the CLI uses `atb.scoring.score` after a completed run.

For NVIDIA models that support it, `--reasoning-budget` limits reasoning tokens
per request. The value is recorded in metadata and every request body. The
default omits the field, leaving the provider's setting in effect; `-1` disables
budget enforcement. This is separate from `--max-tokens`, which limits output,
and `--timeout`, which limits how long the client waits.

The [Lightning API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-5-lightning-30b-a3b-infer)
documents this provider extension. Other compatible endpoints may not support it.
Keep the same budget when comparing interface conditions.

## Message protocol

A dependent two-tool conversation contains:

```text
system: shared scientific instructions and final JSON format
user: benchmark prompt
assistant: tool_calls=[Hohmann request, ID h]
tool: tool_call_id=h, content=serialized transfer result
assistant: tool_calls=[propagation request using the returned duration, ID p]
tool: tool_call_id=p, content=serialized final state
assistant: final JSON answer
```

The runner keeps the assistant's calls and raw JSON argument strings, then
appends one `role: "tool"` result per call. Multiple calls in one assistant
response each receive a result before the next model request. Tool arguments
must be JSON objects with finite numbers. Function dispatch is restricted to the
existing numerical executor; arbitrary code is never evaluated.

The runner validates JSON transport constraints. Numerical/domain checks remain
in the executor; complete JSON Schema validation is not implemented here.

Outbound messages use standard Chat Completions fields. Provider extras are
preserved in the raw response records. The model is given the same system
instructions in A, B and C, including the Earth radius and gravitational
parameter. Ground-truth answers, tolerances and expected tools are excluded.

## Conditions and budgets

| Condition | Tool definitions sent to the model |
| --- | --- |
| A | `None`: both `tools` and `tool_choice` are omitted. |
| B | `TOOLS_B`, dispatched through the B executor interface. |
| C | `TOOLS_C`, dispatched through the C executor interface. |

`max_steps` counts model requests, including the request that returns the final
answer. Two dependent tool requests normally need three model turns. The runner
executes calls returned on the last allowed turn, preserves their results, and
then stops without fabricating a final answer.

CLI controls include `--model`, `--api`, `--max-steps`, `--max-tokens`, `--reasoning-budget`, `--timeout`,
`--env-file`, `--results-dir`, `--run-id`, `--verbose` and `--json`. A caller can
share a run ID across different task IDs. Existing task traces under that run ID
are rejected before a new request is sent.

## Errors and terminal statuses

| Status | Meaning |
| --- | --- |
| `completed` | A non-empty final assistant response is available; score it separately. |
| `max_steps_reached` | Request budget exhausted; `final_answer` remains null. |
| `api_error` | Provider/network failure; prior messages and successful requests are retained. |
| `protocol_error` | Missing/duplicate call IDs, calls in condition A, an empty choice list or another unusable response. |
| `truncated` | Provider stopped at its output-token limit; partial tool requests are not executed. |
| `content_filtered` | Provider stopped generation through its content filter. |
| `provider_refusal` | Provider returned a refusal field without a usable final answer. |

Malformed JSON, unknown tools and execution failures produce tool-result error
messages. These errors do not terminate the conversation by themselves. They are
recorded in `tool_calls[].error_type`; the model can respond with a corrected
request. Entire batches are checked for usable, unique IDs before dispatch.

## Agent trace

Each run is saved at `results/traces/<run_id>/<task_id>.json`. A trace is reserved
before the first request and saved atomically after every turn. Configuration
errors raise before execution. API failures and budget exhaustion return traces
with explicit terminal statuses.

| Field | Contents |
| --- | --- |
| `metadata` | Run/task identity, family, split, task hash, model, condition, provider, generation limits and runtime versions. |
| `messages` | Ordered model/tool conversation, including the final response when available. |
| `tools` | The selected schemas, or null for condition A. |
| `steps` | Exact request bodies, raw SDK responses, usage and per-turn latency. |
| `tool_calls` | IDs, names, raw arguments, result strings, error categories, turn indices and execution latency. |
| `final_answer` | Final assistant text, or null when execution did not produce one. |
| `tokens` | Aggregate prompt/completion/total tokens and `usage_complete`. Missing usage is reported as null, not zero. |
| `step_count`, `tool_call_count` | Attempted model requests and handled tool requests, including error results. |
| `latency_s` | Observed elapsed time for the loop, including tool execution and intermediate trace writes. |
| `status`, `error` | Terminal state and sanitized diagnostics. |

Credentials and authorization headers are excluded. Local traces are ignored by
Git. Raw run artifacts include failures and development tasks; selecting training
data requires separate quality checks and family-based separation from evaluation.

## Verification

```bash
python -m pytest -q
```

Offline tests use fabricated model responses and the real numerical executor.
They verify dependent calls, parallel requests, error recovery, conditions A/B/C,
budget exhaustion, protocol failures, missing usage, trace persistence and API
error redaction. A hosted-model integration check is documented separately from
these deterministic tests.

## Recorded hosted-model run

On 2 October 2026, Nemotron 3 Super completed `multi-001` with condition C:

| Measurement | Observed value |
| --- | --- |
| Model requests | 3 |
| Tool calls | 2, in successive assistant responses |
| Tool sequence | `hohmann_transfer` → `propagate_orbit` |
| Final x coordinate | −5368.7591375471 km |
| Reference / original absolute tolerance | −5368.76 km / 5 km |
| Numerical score | Correct |
| Total reported tokens | 6110 |
| Observed loop latency | Approximately 15.09 s |

The propagation call used the exact `transfer_time_s` returned by the first
tool. The [recorded conversation](examples/nvidia-nemotron-agent.json) preserves
the real requests, responses and numerical results; its local trace path has
been normalized to a repository-relative path for publication.

The prompt specifies the ECI initial state and explicitly propagates the original
orbit without applying transfer burns. This removes a prior ambiguity about
which state to propagate. The reference remains unchanged. This is a single
development-task integration check, not a comparative benchmark result.

## Numerical tolerance and intermediate precision

The current `multi-001` acceptance criterion is an absolute x-coordinate error
of at most **0.1 km**. Its reference is stored to 0.01 km precision. This criterion
checks consistency within the project's two-body model; it is not a claim about
real-world orbit prediction accuracy.

Truncating the transfer duration from 3560.540788789012 s to 3560 s changes the
final x coordinate to approximately −5371.376840 km: an error of **2.616840 km**
against the reference. The full position difference is approximately 4.080821 km,
but this task scores x only. The original 5 km criterion would accept this error;
the tightened 0.1 km criterion rejects it. Tests cover both truncated and
nearest-second durations, as well as the exact chain and the recorded answer.

The recorded NVIDIA answer also passes the tightened criterion. The record keeps
its original task fingerprint and the table above reports the original tolerance;
this is a re-evaluation of an existing answer, without another model request.

The model's radius-to-altitude conversion in condition C is another potential
error source: 7000 km and 9000 km become 621.863 km and 2621.863 km using the stated
Earth radius. Keep such tasks when classifying unit and parameter-convention
errors in future A/B/C comparisons.
