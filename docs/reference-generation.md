# Reproducible references — milestone 7

From the repository root, with base dependencies installed:

```bash
python -m astrotoolbench.generate_references
python -m astrotoolbench.generate_references --check
```

The public command calls `astrotoolbench.benchmark.generation`. It imports no
model client and needs no API keys or optional LLM
packages.

## Source and outputs

| Artifact | Purpose |
| --- | --- |
| `tasks/astrodynamics/specifications.jsonl` | Input task records with scientific parameters, units, metadata, splits, selected outputs and answer contracts. Successful contracts contain **no numerical or Boolean reference values**. |
| `schemas/task-spec-v0.1.schema.json` | Machine-readable authoring format; injected result values are rejected. |
| `tasks/astrodynamics/tasks.jsonl` | Official task records with generated `expected` values, full precision and individual tolerances. |
| `tasks/astrodynamics/splits.json` | Reviewed family/group policy and task IDs in each partition. |
| `docs/reference-verification.json` | Source/dataset SHA-256, counts, split audit and per-task independent numerical error budgets/observations. |

Expected error codes and their rationale are authored contracts, not numerical
references. The generator independently reproduces those errors too. A solver
crash or failure to converge cannot silently become an expected diagnostic result.

The command:

1. Loads and validates the entire specification file, including IDs and family splits.
2. Calls the deterministic scientific recipe for each successful computation.
3. Materializes a strict official record with the calculated outputs and their contracts.
4. Checks the record against an independent analytic, Cartesian RK4 or input-policy
   certificate; reports all per-task computational failures.
5. Rechecks split integrity and builds the accuracy report/manifest.
6. Publishes outputs only after every task succeeds in verification.

It never reads existing `expected` numbers as its computational input, and never
rebuilds task templates behind the user's back. A changed specification is actually
recomputed. Each output file is replaced atomically after all temporary files are
prepared. Multi-file replacement is not a filesystem transaction: the recorded
hashes detect an incomplete update. A scientific failure leaves existing outputs
untouched.

## Controls

```bash
python -m astrotoolbench.generate_references \
  --tasks tasks/astrodynamics/specifications.jsonl \
  --output tasks/astrodynamics/tasks.jsonl \
  --report docs/reference-verification.json \
  --splits tasks/astrodynamics/splits.json
```

All four paths must be distinct. Custom source files may contain a subset of
reviewed families for development; generating that subset does not certify it as
a complete v0.1 corpus. The official authoring command enforces the publication
size, coverage and difficulty targets.

`--check` compares exact deterministic regeneration with the existing dataset,
manifest and report and writes nothing. It exits with code 1 for missing/different
artifacts, malformed sources, unreviewed partitions or failed scientific checks.
The normal command also exits with code 1 on failure and prints task/file diagnostics
to stderr. Declared and independently reproduced scientific errors count as valid
benchmark records.

Generated artifacts have stable ordering and no timestamps, random seeds, machine
paths or credentials. Exact byte comparison is intended for regeneration in the
same numerical environment. Cross-platform numerical checks use the independent
scientific error budgets, not assumed bit-identical floating-point arithmetic.

## Authoring and review

`astrotoolbench.benchmark.families` contains the authored problem definitions. After an
intentional input/contract/family change:

```bash
python -m astrotoolbench.benchmark.build
python -m astrotoolbench.generate_references --check
python -m astrotoolbench.validate
python -m pytest -q
```

The authoring command regenerates the answer-free specifications, both schemas,
manifest and independently verified references. It requires at least 40 tasks,
all six categories, at least five families and all four difficulties. Difficulty
shares may deviate by at most five percentage points from 40/25/15/20.

The reference snapshot test requires an explicit review when expected values
change. Milestone 5 adds equal-radius Hohmann and zero-impulse cases; the numerical
reference digest of the preceding 48 tasks is retained as a separate regression
check. The current complete corpus has 50 tasks, 38 numerical/categorical successes
and 12 verified errors.
