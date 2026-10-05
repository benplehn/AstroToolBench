# Validation

From the repo root:

```bash
python -m astrotoolbench.validate
python -m astrotoolbench.validate --json
python -m pytest -q
```

The validator reads every line of the dataset and checks: schema (with file and
line on errors), duplicate IDs, family registry, that the tools exist, units,
outputs, finite numbers, positive tolerances, and it recomputes every reference
and compares it to the independent check. Error tasks only count as valid if
both the tools and the input checker produce the declared code. A solver failure
is reported as an issue.

By default it also checks the dataset as a whole: at least 40 tasks, 6
categories, 5 families, 4 difficulties around 40/25/15/20 %. And it compares
the dataset with `specifications.jsonl`, `splits.json` and
`docs/reference-verification.json` (hashes, counts, methods, budgets). The stored
errors only need to be within budget, not bit-identical on another machine.

Current output:

```text
AstroToolBench v0.1

Tasks:              50
Valid:              50
Invalid:            0
Issues:             0

Simple:             20
Multi-step:         12
Diagnostic:         8
Trap:               10

Families:           22
Tools:              6
Splits: train=24, validation=6, test=20

✓ Benchmark valid
```

`Invalid` counts broken tasks; `Issues` also counts dataset-level problems
(coverage, outdated report…). Those fail validation even if every task is fine.

## Custom files and exit codes

```bash
python -m astrotoolbench.validate \
  --tasks /path/to/tasks.jsonl \
  --specifications /path/to/specifications.jsonl \
  --splits /path/to/splits.json \
  --report /path/to/reference-verification.json \
  --json
```

With the default dataset, the companion files are checked automatically. With a
custom `--tasks`, only the ones you pass are. `--allow-subset` skips the size and
coverage checks (everything else still runs) for working on part of the
dataset. An empty file is always invalid.

Exit **0** = all good, **1** = at least one issue. Errors go to stderr with file,
line and task ID; `--json` prints one JSON report to stdout. The validator never
modifies files and never calls a model.

## Tests and CI

- `tests/tools/`: normal, edge and invalid cases for each tool, plus the old
  import path.
- `tests/benchmark/`: schema, coverage, split leaks, comparisons, generation
  failures, independent checks.
- `tests/test_validation.py`: broken lines, duplicates, missing tools, drifting
  numbers, wrong error codes, spec/dataset mismatch, CLI output, read-only.

The reference digest (and the older 48-task one) are pinned in the tests, so
regenerating alone can't change an expected value.

CI runs on Python 3.10 and 3.12. One job installs only the base package +
pytest, validates the committed data, regenerates everything in a temp folder
and validates that too (no LLM packages, no keys, no tokenizer). A second job
runs the full suite with the `atb` extras.
