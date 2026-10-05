# Corpus validation — milestones 8–9

From the repository root after installing the base package:

```bash
python -m astrotoolbench.validate
python -m astrotoolbench.validate --json
python -m pytest -q
```

The validator reads every nonblank JSONL record, gathers schema failures with
file/line context, checks duplicate IDs and the reviewed family policy, verifies
that required tools exist, and recomputes every scientific reference against its
independent certificate. It checks units, supported outputs, finite numbers and
positive numerical tolerances through the same strict schema used by the loader.
Expected errors count as valid tasks only if the backend and independent input
checks reproduce their declared code. A solver failure is a validation issue.

Default validation also requires publication coverage: at least 40 tasks, six
categories, five families and four difficulties, with the reviewed approximate
40/25/15/20 mix. It compares the dataset's metadata, inputs and answer contracts
with `tasks/astrodynamics/specifications.jsonl`, checks `tasks/astrodynamics/splits.json`,
and verifies hashes, counts, methods and budgets in `docs/reference-verification.json`.
Historical observed errors are checked against their budgets; they need not be
bit-identical to calculations on another numerical platform.

The current human-readable summary is:

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

The six functions span five scientific areas: propagation, ΔV, transfer, eclipse
geometry/duration, and proximity. `Invalid` counts broken records; `Issues` also
includes corpus-wide failures such as inadequate coverage or an outdated artifact.
A corpus-wide issue fails validation even when its records are individually sound.

## Custom inputs and exit codes

```bash
python -m astrotoolbench.validate \
  --tasks /path/to/tasks.jsonl \
  --specifications /path/to/specifications.jsonl \
  --splits /path/to/splits.json \
  --report /path/to/reference-verification.json \
  --json
```

Companion files are included automatically for the default official dataset.
For custom `--tasks` paths, only explicitly supplied companions are compared.
Use `--allow-subset` during development to skip publication coverage, while keeping
schema, uniqueness, tool, scientific and split checks. An empty file is always
invalid; a development subset is not a certified complete v0.1 release.

Exit code **0** means every requested check passed. Exit code **1** means one or
more issues; malformed/missing files also produce reports. Human-readable issues
go to stderr with file, line and task ID when available. `--json` prints a single
JSON report to stdout containing counts, partitions, artifact paths and issues.
Validation never modifies data, repairs references or calls a model provider.

## Regression protection and CI

- `tests/tools/` checks nominal, boundary and invalid inputs for each scientific
  area, plus compatibility imports after the package reorganization.
- `tests/benchmark/` checks the schema, corpus coverage, split leakage, numerical
  comparisons, generation failures and independent scientific certificates.
- `tests/test_validation.py` checks broken lines, duplicates, unavailable tools,
  numerical drift, wrong error codes, source/artifact mismatch, CLI output and
  read-only behavior.

The reviewed reference digest and the original 48-task digest remain frozen.
Changing an expected value requires an explicit review after independent
verification; regeneration alone does not update those test snapshots.

CI runs a scientific job on Python 3.10 and 3.12 with only base dependencies and
pytest. It validates committed data, regenerates into a temporary directory and
validates that output, without LLM packages, credentials or a tokenizer download.
The existing integration job separately covers the legacy model clients.
