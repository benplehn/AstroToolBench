# Task format (schema 0.1)

The dataset is **`tasks/astrodynamics/tasks.jsonl`**, one JSON object per line
(UTF-8). Its JSON Schema (draft 2020-12) is `schemas/task-v0.1.schema.json`,
generated from `astrotoolbench.benchmark.schema.BenchmarkTask`, which also does
the cross-field checks (recipes, units, outputs).

`benchmark/tasks.jsonl` and `atb.tasks.Task` are the older format used by the
agent prototype. They're kept for the recorded traces and their tests, but the
official loader refuses them.

## Fields

| Field | |
| --- | --- |
| `schema_version` | `"0.1"` |
| `id` | unique slug across the whole file |
| `domain` | `astrodynamics` |
| `category` | `propagation`, `maneuvers`, `eclipse`, `proximity`, `multistep`, `diagnostic` |
| `difficulty` | `simple`, `multistep`, `diagnostic`, `trap` (doesn't decide success vs error) |
| `split` | `train`, `validation`, `test`, set by family (see [data splits](data-splits.md)) |
| `family` | problem structure, independent of the numbers or wording |
| `prompt` | the question, with all inputs, units, frame/epoch and what to return; no answers |
| `context` | ECI frame, epoch (s), body and radius, dynamics, shadow model, optional time accuracy |
| `inputs` | named values `{ "value": number / vector / null, "unit": ... }`; `null` = deliberately missing |
| `recipe` | a computation from `astrotoolbench.benchmark.catalog.RECIPES` (no code or expressions) |
| `outputs` | which results are requested; empty for error tasks |
| `required_tools` | tools used, in order; for errors, the tool that would have been used |
| `expected` | `success` or `error`, see below |
| `verification` | independent method, explanation and budgets (at least as strict as scoring) |
| `notes` | conventions that matter for the task |

Units: `km`, `km/s`, `s`, `km^3/s^2`, `rad`, `1`, `m`, `m/s`, `min`. The
non-standard ones only show up in the mixed-unit tasks, which forbid silent
conversions. Context fields are `epoch_s`, `body_radius_km`,
`requested_time_accuracy_s`.

Numbers must be finite; booleans and numeric strings aren't numbers. Unknown
fields, unknown enum values, empty text and duplicate JSON keys are rejected.
Physically wrong data is still allowed (negative window, 2D state, μ = 0) since
that's what error tasks are made of. Passing the schema doesn't mean the answer
is right; that's what the verification is for.

## Success

ΔV from `[3,0,0]` to `[0,4,0]` km/s:

```json
{
  "kind": "success",
  "outputs": {
    "delta_v": {
      "kind": "numeric",
      "value": 5.0,
      "unit": "km/s",
      "absolute_tolerance": 0.000001,
      "relative_tolerance": null
    }
  }
}
```

Values are scalars or vectors with an exact shape. At least one tolerance must be
positive. A component passes if
`abs(prediction-reference) <= max(abs_tol, rel_tol*abs(reference))`. A zero
reference needs an absolute tolerance. No default tolerance, no broadcasting,
and scalars/vectors/booleans aren't interchangeable.

Booleans look like `{"kind":"boolean","value":false,"unit":"1"}` and are
compared exactly.

## Error

```json
{
  "kind": "error",
  "code": "invalid_input",
  "reason": "An observation window must be nonnegative."
}
```

Codes: `invalid_input`, `inconsistent_units`, `missing_input`,
`physical_impossibility`, `unsupported_model`, `unachievable_precision`. No
outputs or tolerances. The code has to be reproduced by both the tools and a
separate input checker; the `reason` is documentation, not an input.

Answers are compared as `{"outputs":{"name":value,...}}` for success and
`{"error":"code","reason":"..."}` for errors. All requested outputs are
required. For errors, only the code and the presence of a reason are checked;
judging the explanation itself isn't implemented.

## Specifications (no answers)

`tasks/astrodynamics/specifications.jsonl` has the same fields but
`answer_contract` instead of `expected`: kind, unit and tolerances, no `value`.
Validated by `TaskSpecification` and `schemas/task-spec-v0.1.schema.json`. Error
contracts keep their code and reason.

`load_specifications` reads specs, `load_benchmark` reads the dataset; neither
accepts the other format. See [reference generation](reference-generation.md).

## Loading and checking

```python
from astrotoolbench.benchmark import load_benchmark
from astrotoolbench.benchmark.reference import calculate
from astrotoolbench.benchmark.verification import verify_task

for task in load_benchmark("tasks/astrodynamics/tasks.jsonl", split="validation"):
    outcome = calculate(task)  # doesn't read expected values
    observed_errors = verify_task(task)  # independent check
```

The loader validates the whole file before filtering, refuses duplicate IDs and
empty files, and reports file and line on errors.

```bash
python -m astrotoolbench.generate_references --check
python -m astrotoolbench.benchmark.build --check
python -m astrotoolbench.validate
python -m pytest tests/benchmark -q
```

Both `--check` commands compare against an exact regeneration on the current
machine. The tests use scientific tolerances across platforms, plus a pinned
digest of the reviewed references.
