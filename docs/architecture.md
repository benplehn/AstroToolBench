# Architecture

The official scientific benchmark lives in `astrotoolbench` and runs offline.
Its numerical tools, authoring, reference generation and validation have no model
client dependency. The existing `atb` package contains the legacy interface and
model experiments; it uses the same scientific functions through compatibility
imports. This keeps the scientific corpus independent of those integrations.

## Repository layout

```text
src/
  astrotoolbench/
    tools/
      propagation.py   Kepler solver, orbital elements and propagation
      maneuvers.py     Vector delta-v and Hohmann transfers
      eclipse.py       Shadow geometry and sampled eclipse intervals
      proximity.py     Minimum separation over an observation window
      constants.py     Earth constants
      _validation.py   Shared scalar/vector and sampling checks
    benchmark/
      schema.py        Strict task and answer-free specification models
      catalog.py       Scientific recipe inputs, outputs and tool dependencies
      loader.py        Whole-file parsing and partition-aware loading
      reference.py     Scientific recipe execution
      verification.py Independent analytic, Cartesian RK4 and input checks
      validator.py     Read-only corpus validation and diagnostic reports
      comparison.py    Strict answer comparison with per-output tolerances
      families.py      Authored physical and diagnostic situations
      coverage.py      Publication size and difficulty requirements
      splits.py        Reviewed family registry and contamination audit
      generation.py    Source-to-reference generation and verification report
      artifacts.py     Deterministic serialization and safe file replacement
      paths.py         Shared repository-relative data defaults
      build.py         Authoring regeneration of all reviewed artifacts
    generate_references.py  Public offline reference command
    validate.py        Public offline validation command
  astrodyn_tools/       Compatibility imports; no duplicated calculations
  atb/
    tools.py           B/C function schemas
    executor.py        Tool dispatch, conversion and result serialization
    tasks.py           Legacy task model and JSONL loader
    scoring.py         Final-response parsing and reference comparison
    client.py          Provider configuration and Chat Completions transport
    prompts.py         Shared system instructions and prompt-only messages
    agent.py           Bounded model/tool orchestration and conversation traces
    agent_cli.py       Single-task execution and scoring command
    inspection.py      Completion inspection, trace persistence and CLI
    template_inspection.py  Local chat rendering and token-cost comparison
tasks/
  astrodynamics/
    specifications.jsonl  Authored inputs/contracts without numeric answers
    tasks.jsonl        Official independently verified references
    splits.json        Reviewed family groups and partition task IDs
schemas/
  task-v0.1.schema.json       Exported official task schema
  task-spec-v0.1.schema.json  Exported authoring schema
examples/
  scientific_workflow.py  Offline transfer, propagation and arrival shadow
benchmark/
  tasks.jsonl          Legacy development data for existing agent experiments
scripts/
  inspect_completion.py  Repository-local wrapper around the inspection CLI
  inspect_template.py    Chat-template inspection and B/C token counts
  run_agent.py           Repository-local wrapper around the agent CLI
  list_models.py        Provider model discovery
  make_references.py    Legacy development task/reference generation
  smoke_test.py         Five-task integration check with a saved outcome summary
results/
  completions/         Raw request/response traces, excluded from Git
  traces/<run_id>/      Agent conversations, excluded from Git
  templates/           Rendered text and token reports, excluded from Git
  smoke/               Integration summaries, including failures, excluded from Git
pyproject.toml         Package metadata, dependencies and optional integration extras
MANIFEST.in            Reviewed assets included in source distributions
LICENSE                MIT license
```

Documentation, tests and CI live in `docs/`, `tests/` and `.github/workflows/`.
`tests/tools/` tests science, `tests/benchmark/` tests the official corpus, and
`tests/test_validation.py` tests the validator. `MANIFEST.in` includes reviewed
data, exported schemas, documentation and examples in source distributions.
The `src/` layout keeps the package importable through a normal installation.
License metadata and the full `LICENSE` text are included in built distributions.
The installed `atb-inspect` command calls
`atb.inspection.main`; `atb-run` calls `atb.agent_cli.main`. Repository wrappers
call the same implementations with paths anchored to the checkout.

## Numerical backend

`astrotoolbench.tools` exposes period, propagation, delta-v, Hohmann transfer, eclipse and closest
approach routines. It has no dependency on a model provider. Physical assumptions
and units belong to the numerical routines and their tests. Existing public
`astrodyn_tools` imports forward to exactly the same functions. No numerical
algorithm or reference value changes as part of the package move.

`atb.tools` describes the subset available to models: period, Hohmann, propagation
and eclipses. The B and C interfaces wrap the same calculations. C changes the
input conventions for period and transfer to altitudes and makes units explicit.

`atb.executor.execute_tool(api_version, name, arguments_json)` receives the tool
name and JSON argument string. It dispatches to a known function, performs the
interface-specific conversions and returns JSON containing a result or error.
There is no execution of arbitrary model-generated Python.

## Benchmark data and scoring

`astrotoolbench.benchmark` owns the official versioned scientific task format.
Its loader, generator and validator share strict parsing and the reviewed split
audit. Generation takes answer-free specifications; validation recomputes stored
references and checks independent scientific budgets. The exported schemas stay
in `schemas/`, while their Python source models stay beside the loader.
See [validation](validation.md) and [reference generation](reference-generation.md).

The milestone 10 reorganization moves scientific modules formerly in
`atb.benchmark` to `astrotoolbench.benchmark` and moves the official flat data files
into `tasks/astrodynamics/`. Imports and documented commands use these canonical
paths. Existing legacy model commands and `benchmark/tasks.jsonl` retain their
own integration contract.

`atb.tasks.Task` defines a prompt, family, split, expected tools, numerical
parameters, reference and tolerances. `load_tasks()` reads JSONL, rejects duplicate
IDs and supports family/split filtering.

`atb.scoring.score(task, final_text)` evaluates an explicit final-response JSON
object against the reference. It reports missing fields, incorrect refusals and
values outside tolerance. Its current scope is final-answer correctness;
tool-selection and argument correctness need separate evaluation.

References and expected calls remain on the evaluation side. Completion requests
include only the task prompt, system message and tool schemas. Trace metadata
records the task ID, family and split for analysis, while reference values and
tolerances are excluded from the trace.

## Model transport and inspection

`atb.client` manages endpoint configuration and request construction. Credentials
are selected for the configured provider and excluded from representations.
Configuration loading does not mutate the process environment. The transport
returns the unmodified SDK response and closes its client after each request.

`atb.inspection` selects a task, constructs messages, sends one completion request
and records the raw response. Its response inspector reports IDs, function names,
JSON argument strings, parsed arguments, finish reasons and usage. The CLI can
preview a request without network access or credentials.

Inspection finishes at the model response. The requested function is not executed
in this path. A valid JSON argument object is a transport check; domain validity
and numerical correctness are responsibilities of execution and evaluation.

## Inspection trace contract

Each completion-inspection record has a schema version and four sections:

| Section | Contents |
| --- | --- |
| `metadata` | Task identity, split, interface condition, endpoint, timestamp, latency, timeout, retries, runtime/SDK versions and benchmark SHA-256. |
| `request` | Complete Chat Completions request body, including messages, schemas and generation settings. |
| `response` | Unmodified JSON-compatible SDK dump, including provider-specific fields. |
| `observations` | Inspection of tool requests and usage; explicitly records that tools were not executed. |

The benchmark hash identifies the data snapshot used by the request. Reproducible
inputs do not guarantee identical responses from a hosted model. Temperature zero
and recorded settings support comparison but do not establish determinism.

## Agent orchestration

`atb.agent.run_agent(task, model, api_version, max_steps=8)` selects the A/B/C
schemas, constructs shared system/user messages and executes bounded model turns.
Every assistant message retains its function calls. One tool-result message per
call carries the original `tool_call_id`. Multiple calls in a response are
handled before requesting another completion.

Invalid JSON, unknown tools and numerical execution failures become correlated
error results that the model can correct. Malformed protocol IDs stop the run
before any call in that batch is dispatched. Truncation, provider refusal, API
errors and exhausted request budgets have distinct terminal statuses.

Outbound assistant messages contain standard conversation fields. Raw provider
responses, including extra fields, remain in the per-request records. Generation
settings, schemas, messages, tool results, request/response pairs, token usage and
latency are persisted. Writes are atomic and an existing run/task trace is never
overwritten by a new run.

The agent returns its trace independently of reference scoring. `atb.agent_cli`
loads the task, runs the agent and invokes `score()` only if a final answer is
available. A completed conversation can therefore still have an incorrect score.
See [agent execution](agent-execution.md) for the detailed contract.

`atb.template_inspection` adapts recorded argument strings to dictionaries in a
copy of the messages, renders a local tokenizer's chat template and counts tokens.
It uses recorded schemas for trace inspection and compares current B/C schemas
on the same initial messages. It never calls a model API or executes a tool.

## Evaluation and training boundaries

All conditions receive the same system instructions and task prompt. Shared
instructions state Earth constants and the final-answer JSON contract. Reference
values, tolerances, expected calls and private parameters remain outside model
requests. Condition A omits both tools and tool choice from the request.

Traces record the task family, split and a task fingerprint. They are raw run
artifacts, including failed and development runs. Training exports must separately
select eligible, successful traces and enforce family-based separation from the
evaluation set. Current development traces are not automatically training data.
Comparative evaluation, training and inference experiments remain future work.
