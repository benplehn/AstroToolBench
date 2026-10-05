# Official task format — schema 0.1

The official corpus is **`tasks/astrodynamics/tasks.jsonl`**: one UTF-8 JSON object
per line. `schemas/task-v0.1.schema.json` is its generated JSON Schema (draft
2020-12). `astrotoolbench.benchmark.schema.BenchmarkTask` is the source of that schema and
adds semantic checks between fields, recipes, units and output contracts.

`benchmark/tasks.jsonl`, `atb.tasks.Task` and the existing agent commands remain
the **legacy integration dataset and interface**. They are retained for the
recorded model traces and their regression tests. They do not define schema 0.1;
the strict official loader rejects legacy records. The new format does not change
the B/C model interfaces or start any model experiments.

## Record fields

| Field | Contract |
| --- | --- |
| `schema_version` | Exactly `"0.1"`; incompatible format changes need another version. |
| `id` | Unique stable slug across the entire dataset, including filtered-out records. |
| `domain` | `astrodynamics`. |
| `category` | `propagation`, `maneuvers`, `eclipse`, `proximity`, `multistep`, `diagnostic`. These are the six authoring groups, not split identifiers. |
| `difficulty` | `simple`, `multistep`, `diagnostic`, `trap`. Difficulty does **not** determine success versus error. |
| `split` | `train`, `validation`, `test`. Reviewed family/group assignment: 24 train, 6 validation, 20 test; see `docs/data-splits.md`. |
| `family` | Structural problem-family name, independent of the particular radius or wording. |
| `prompt` | Self-contained question, explicit numerical inputs and units, common frame/epoch and requested outputs. No expected numerical values are inserted. |
| `context` | ECI frame, common epoch in seconds, central body and radius, requested dynamics, fixed cylindrical shadow model, optional requested time accuracy. |
| `inputs` | Named quantities: `{ "value": number or vector or null, "unit": unit }`. `null` means information explicitly absent in a diagnostic problem. |
| `recipe` | A finite scientific computation from `astrotoolbench.benchmark.catalog.RECIPES`; no arbitrary code or expressions. |
| `outputs` | Unique selected result names supported by the recipe. Empty for error outcomes. |
| `required_tools` | Scientific functions used by the recipe, in dependency order. For error problems these identify the relevant attempted computation; a valid diagnosis may stop before invoking any tool. Helpers and ordinary arithmetic are not separate tools. |
| `expected` | Discriminated `success` or `error` object, described below. |
| `verification` | Independent method, explanatory rationale and absolute verification budgets for every numeric output. Verification budgets must be at least as strict as scoring. |
| `notes` | Authoring/model conventions that matter to the task. |

Supported quantity units are `km`, `km/s`, `s`, `km^3/s^2`, `rad`, `1` (dimensionless),
`m`, `m/s`, `min`. Noncanonical units deliberately appear in inconsistent-unit
problems; these tasks explicitly prohibit implicit conversions. Physical tools
otherwise receive the units declared by their recipe. Context uses the explicit
`epoch_s`, `body_radius_km`, `requested_time_accuracy_s` suffixes.

Numbers are finite and real; Boolean values and numeric strings are not numbers.
Unknown fields, unknown enum values, blank text and duplicate JSON keys are
rejected. Scientific invalidity must remain representable: a negative observation
window, two-component state or zero mu is valid **task data** whose expected
outcome can be a scientific error. Schema validation is not a substitute for
executing and independently verifying the scientific reference.

## Expected success

For illustration, the impulse from `[3,0,0]` to `[0,4,0]` km/s has this result:

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

Numeric values are scalars or nonempty vectors, with exact output shape. Every
numeric result declares at least one positive tolerance. Each component passes
when `abs(prediction-reference) <= max(abs_tol, rel_tol*abs(reference))`; omitted
terms contribute zero. A zero reference component needs a positive absolute
tolerance. There is no implicit fallback tolerance and no broadcasting. Scalars,
vectors and Booleans are not interchangeable.

Categorical geometry uses `{"kind":"boolean","value":false,"unit":"1"}`.
Booleans are compared exactly and do not receive an artificial numeric tolerance.

## Expected error

```json
{
  "kind": "error",
  "code": "invalid_input",
  "reason": "An observation window must be nonnegative."
}
```

Stable error codes are `invalid_input`, `inconsistent_units`, `missing_input`,
`physical_impossibility`, `unsupported_model`, `unachievable_precision`. Error
outcomes have no numeric outputs or tolerances. Their code is verified exactly by
the backend/policy and a separate input checker. The supplied reason documents
why the error is correct; it is not used to tell the calculator which error to
produce.

Official answer comparison accepts `{"outputs":{"name":value,...}}` for
success and `{"error":"code","reason":"explanation"}` for errors. It requires
all requested outputs. The deterministic error comparison checks the code and
presence of an explanation; judging the explanation's meaning would require a
separate evaluation and is not implemented here.

## Answer-free task sources

`tasks/astrodynamics/specifications.jsonl` uses the same metadata, inputs, recipes and selected
outputs as official records, with `answer_contract` in place of `expected`.
Successful output contracts contain kind, unit and tolerances, but no `value`.
This is validated by `TaskSpecification` and the generated
`schemas/task-spec-v0.1.schema.json`. Error contracts retain their explicit code
and rationale and must still be reproduced independently.

Use `load_specifications` for authoring sources and `load_benchmark` for generated
records. Neither loader silently accepts the other format. See
[reference generation](reference-generation.md) for the load/compute/verify/publish
workflow.

## Reading and checking

From the repository root after installing `.[dev]`:

```python
from astrotoolbench.benchmark import load_benchmark
from astrotoolbench.benchmark.reference import calculate
from astrotoolbench.benchmark.verification import verify_task

for task in load_benchmark("tasks/astrodynamics/tasks.jsonl", split="validation"):
    outcome = calculate(task)  # Does not read expected values.
    observed_errors = verify_task(task)  # Separate scientific certificate.
```

The loader validates the entire file before filtering, rejects duplicate IDs and
reports the file and line for a malformed record. Empty datasets are rejected.

```bash
python -m astrotoolbench.generate_references --check
python -m astrotoolbench.benchmark.build --check
python -m astrotoolbench.validate
python -m pytest tests/benchmark -q
```

Both commands verify all references before writing any artifact. The authoring
command regenerates specifications, schemas, partitions, dataset and accuracy report.
The reference generator loads the specification file. With `--check`, either
command compares its artifacts with exact deterministic regeneration in the
current environment.
The tests use scientific tolerances for recomputed numerical values across
platforms and a committed digest for reviewed references. A changed reference
must be verified and explicitly reviewed before changing that digest.
