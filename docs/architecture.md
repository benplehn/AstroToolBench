# Architecture

Two parts:

- `astrotoolbench`: the benchmark itself (tools, tasks, reference generation,
  validation) and the common model interface. The scientific code and model
  contract run offline; the optional API adapter needs the `llm` extra.
- `atb`: my first agent prototype, which runs a model against the tools. It
  imports the same scientific functions, so the two never disagree on the math.

## Layout

```text
src/
  astrotoolbench/
    tools/
      propagation.py   Kepler solver, orbital elements, propagation
      maneuvers.py     ΔV and Hohmann transfer
      eclipse.py       shadow test and eclipse intervals
      proximity.py     closest approach over a time window
      constants.py     Earth constants
      _validation.py   input checks shared by the tools
    benchmark/
      schema.py        task and specification models (Pydantic)
      catalog.py       recipes: inputs, outputs, tools used
      loader.py        loads a whole file, filters by split
      frozen.py        checks the benchmark-v0.1 snapshot before loading
      reference.py     runs the recipe for a task
      verification.py  independent checks (analytic, RK4, inputs)
      validator.py     read-only validation and reports
      comparison.py    compares an answer to the reference
      families.py      the problem definitions
      coverage.py      minimum size/difficulty requirements
      splits.py        family registry and leak checks
      generation.py    specifications -> references + report
      artifacts.py     deterministic JSON output, safe file writes
      paths.py         default data paths
      build.py         regenerates all committed artifacts
    models/
      base.py          provider-independent messages, calls and responses
      api.py           OpenAI-compatible adapter using the existing transport
    generate_references.py
    validate.py
  astrodyn_tools/       old import path, re-exports the same functions
  atb/
    tools.py           B/C tool schemas
    executor.py        dispatches a tool call, returns JSON
    tasks.py           dev task model and loader
    scoring.py         parses the final answer and scores it
    client.py          provider config and Chat Completions calls
    prompts.py         system prompt and messages
    agent.py           the tool-calling loop and traces
    agent_cli.py       atb-run
    inspection.py      atb-inspect (single completion)
    template_inspection.py  chat template rendering and token counts
tasks/astrodynamics/
  specifications.jsonl  tasks without answers
  tasks.jsonl          tasks with verified answers
  splits.json          split manifest
  benchmark-v0.1.json   hashes of the frozen corpus and companion files
schemas/                exported JSON Schemas
examples/scientific_workflow.py
benchmark/tasks.jsonl   the older dev tasks used by atb
scripts/
  inspect_completion.py, run_agent.py   wrappers around the CLIs
  inspect_template.py   template inspection / B vs C tokens
  list_models.py        list provider models
  make_references.py    builds the dev tasks
  smoke_test.py         five-task integration test
results/                local outputs (completions, traces, templates, smoke), gitignored
```

`atb-inspect` → `atb.inspection.main`, `atb-run` → `atb.agent_cli.main`. The
scripts in `scripts/` call the same code but resolve paths from the repo root.

CI is in `.github/workflows/tests.yml`.

## Tools

`astrotoolbench.tools` has period, propagation, ΔV, Hohmann, eclipse and closest
approach. Units and assumptions are handled there and tested there.
`astrodyn_tools` just re-exports them (moving the package didn't change any
value).

`atb.tools` exposes a subset to the model: period, Hohmann, propagation and
eclipses. B and C call the same functions; C uses altitudes instead of radii for
period and transfer, and puts units in the parameter names.

`atb.executor.execute_tool(api_version, name, arguments_json)` looks up a known
function, converts the arguments and returns JSON with a result or an error. The
model never gets to run arbitrary code.

## Benchmark data

`astrotoolbench.benchmark` defines the task format. Loader, generator and
validator all use the same strict parsing and split checks. Generation starts
from specifications without answers; validation recomputes every reference and
checks it against the independent method. See [validation](validation.md) and
[reference generation](reference-generation.md).

The scientific modules used to live in `atb.benchmark` and the data files at the
root; I moved them to `astrotoolbench.benchmark` and `tasks/astrodynamics/`. The
old `atb` commands and `benchmark/tasks.jsonl` still work as before.

On the `atb` side, `atb.tasks.Task` holds a prompt, family, split, expected
tools, parameters, reference and tolerance. `atb.scoring.score(task, final_text)`
checks the final JSON answer: missing fields, wrong refusals, values out of
tolerance. It doesn't check whether the right tools were called.

Reference answers never go to the model: requests only contain the system prompt,
the task prompt and the tool schemas. Traces store task ID, family and split, but
not the reference.

## Model calls and inspection

New experimental code will use `astrotoolbench.models.ModelBackend`, returning
common response types rather than SDK objects. `OpenAIBackend` adapts
`atb.client`; another provider or a local model can implement the same
`generate` method. See [model backends](model-backends.md) and
[the benchmark snapshot](benchmark-freeze.md).

`atb.client` builds requests for the configured endpoint and picks the right API
key. Keys never show up in reprs or logs, and loading `.env` doesn't modify
`os.environ`. Each call returns the raw SDK response and closes the client.

`atb.inspection` sends a single request and saves the raw response, then prints
the tool calls (IDs, names, arguments, finish reason, usage). It doesn't execute
the tools. `--dry-run` shows the request without network or key.

An inspection record has four parts:

| Section | Contents |
| --- | --- |
| `metadata` | task, split, interface, endpoint, time, latency, timeout, retries, versions, dataset SHA-256 |
| `request` | full request body (messages, schemas, settings) |
| `response` | raw SDK response, including provider-specific fields |
| `observations` | what tool calls came back and usage; notes that nothing was executed |

Temperature 0 helps, but a hosted model can still answer differently on the
same input.

## Agent loop

`atb.agent.run_agent(task, model, api_version, max_steps=8)` picks the A/B/C
schemas, builds the messages and loops: call the model, run any requested
tools, send back one tool message per call (with the matching `tool_call_id`),
repeat. Several calls in one response are all run before the next request.

Bad JSON, unknown tools or a failing tool become error messages the model can
react to. Missing or duplicate call IDs stop the run before anything is
executed. Truncation, provider refusal, API errors and running out of steps each
have their own status.

The trace saves settings, schemas, messages, tool results, every request and raw
response, tokens and latency. It's written atomically after every turn and never
overwrites an existing trace.

`run_agent` doesn't score anything; `atb.agent_cli` calls `score()` afterwards
if there is a final answer. Details in [agent execution](agent-execution.md).

`atb.template_inspection` renders a conversation with a local tokenizer's chat
template and counts tokens, without calling any API.

## Evaluation vs training data

All conditions get the same system prompt (with Earth constants and the expected
answer format) and the same task prompt. Condition A sends no tools at all.

Traces include failures and dev runs. They aren't training data as-is: a future
training export will have to pick good traces and respect the family split.
Evaluation and training are future work.
