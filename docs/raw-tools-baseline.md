# Baseline B: raw tools

Same system prompt and task data as baseline A, but the model also gets the
scientific functions, as they are. Every task gets the same ten tools; the
runner doesn't tell the model which ones the task needs.

The idea is to give the model a library written for humans, not for agents.
The tool schema just uses the Python parameter names, defaults and types, and
the description is the first line of the docstring. No unit conversion, no
task-specific shortcut. Making the API agent-friendly is the next step (B → C).

| Function | Result (as JSON) |
| --- | --- |
| `orbital_period` | period |
| `solve_kepler` | eccentric anomaly |
| `coe_to_rv` | `[position, velocity]` |
| `rv_to_coe` | `[a, e, i, raan, argp, nu]` |
| `propagate_kepler` | `[position, velocity]` |
| `delta_v` | magnitude |
| `hohmann_transfer` | `dv1`, `dv2`, `dv_total`, `tof` |
| `in_cylindrical_shadow` | boolean |
| `eclipse_windows` | list of `[start, end]` |
| `closest_approach` | `tca`, `miss_distance` |

NumPy arrays and tuples become JSON lists, at full precision. The model still
has to figure out units, pick the right function and arguments, combine results,
check the physics and format the answer. Some functions don't check that the
orbit stays outside the Earth, so a successful call doesn't mean the final
answer is right.

## Running it

```bash
python -m astrotoolbench.eval --model nemotron --condition raw_tools \
  --task prop-circular-quarter --dry-run

python -m astrotoolbench.eval --model nemotron --condition raw_tools --split test
```

`--dry-run` prints the messages and the tool definitions without loading a model
or calling an API. Hosted profiles need `.[llm]` and their key; Qwen needs
`.[local]`. `--device` and `--local-files-only` work in both conditions.

For Qwen, the local backend uses the checkpoint's own
[chat template](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer_config.json)
and parses the `<tool_call>` blocks it generates, giving each call an ID. The raw
generated text is kept in `response.raw_content`. If a `<tool_call>` block is
broken, the task ends with status `invalid_tool_call`: that's counted as a model
mistake, not a backend failure. Truncated calls are never executed. Other local
models would need their own parser.

## The tool loop

A response can contain several calls. They're all recorded, executed in order,
and each gets a `tool` message with its `tool_call_id` before the next model
turn. Results keep full precision, so later calls can reuse them.

Bad JSON, unknown functions, missing/extra parameters, wrong types and
numerical errors are sent back to the model as error results, so it can try
again. An unknown function name is always reported as `unknown_tool`, even if
its arguments are also broken, so hallucinated tools are counted correctly.
Nothing is ever run as Python code.

A call ID can't be reused across turns (`protocol_error`). Refusals and
truncated responses end the task; their calls show up as `skipped`. Some
providers return tool calls with `finish_reason` set to `stop` (or nothing)
instead of `tool_calls`: the calls are executed anyway and the step gets a
`warnings` entry.

Limits: 8 model turns and 32 tool calls per task by default (`--max-steps`,
`--max-tool-calls`). The final answer uses a turn. If a batch would go over the
call limit, none of it runs. Hitting a limit ends the task without grading
whatever text was there.

There are also safety caps: 10,000 samples for the searches, 10,000 Kepler
iterations, 64,000 bytes of arguments. They stop runaway calls but don't fix
the model's inputs. NaN and infinity can't get into tool results.

## What gets saved

```text
results/nemotron/raw_tools.jsonl
results/nemotron/raw_tools.traces/<task_id>.json
```

`--output` changes the JSONL path; the `.traces` folder goes next to it. Nothing
is overwritten. Every task gets a JSONL row, including failures.

For each tool call:

- step and call ID;
- tool name and the raw argument JSON;
- parsed arguments and bound parameters (with defaults);
- whether it ran and whether it succeeded;
- the result, or the error type and message;
- execution time in ms (including argument checks and serialization).

The trace also has the tool definitions, the messages sent at each turn, the
responses, per-turn latency and usage, the limits and the final status. If the
provider sends something malformed, it's kept in `raw_response` when possible.

The trace file is rewritten (atomically) before each request, after each
response and after each tool call. So if the run crashes, the calls already
made are still there, and anything unfinished is marked `pending`.

Tokens are summed over all turns. If a turn has no usage (e.g. an API error),
the total only covers the known turns and `usage_complete` is `false`. Total
latency covers the whole loop, including checkpoint writes; per-turn and
per-call times are stored separately. Final answers are scored exactly like
baseline A (including `format_ok`).

The tests use fake model responses with the real tools: chaining, parallel
calls, error recovery, crashes. They check the plumbing, not any model.
