# Generating references

From the repo root (base install is enough):

```bash
python -m astrotoolbench.generate_references
python -m astrotoolbench.generate_references --check
```

No LLM client, no API key.

## Files

| File | What it is |
| --- | --- |
| `tasks/astrodynamics/specifications.jsonl` | input: tasks with parameters, units, split, outputs and answer contract, **no answers** |
| `schemas/task-spec-v0.1.schema.json` | schema for the above; rejects any answer value |
| `tasks/astrodynamics/tasks.jsonl` | output: tasks with computed `expected` values and tolerances |
| `tasks/astrodynamics/splits.json` | split manifest |
| `docs/reference-verification.json` | hashes, counts, split checks and per-task verification errors |

For error tasks, the expected error code and its reason are written by hand in
the spec, but the generator still has to reproduce that error independently. A
crash or a solver that doesn't converge can't silently turn into an "expected
error".

What it does:

1. Load and validate the whole spec file (IDs, families, splits).
2. Run the recipe for each task.
3. Build the full task record with the computed outputs.
4. Check it against the independent method (analytic, RK4 or input checks).
5. Check the splits again and build the report and manifest.
6. Write files only if every task passed.

It never uses existing `expected` values as input. Each file is written to a temp
file and swapped in atomically. Writing several files isn't a real transaction,
but the hashes in the report would show a half-finished update. If any task
fails, nothing is overwritten.

## Options

```bash
python -m astrotoolbench.generate_references \
  --tasks tasks/astrodynamics/specifications.jsonl \
  --output tasks/astrodynamics/tasks.jsonl \
  --report docs/reference-verification.json \
  --splits tasks/astrodynamics/splits.json
```

The four paths must be different. You can run it on a subset of families while
developing, but that doesn't make it a complete v0.1 dataset (the build command
checks size and coverage).

`--check` regenerates in memory, compares with the files on disk and writes
nothing. Exit 1 if anything differs or fails; errors go to stderr. Expected
errors that are correctly reproduced count as valid tasks.

Output is stable: sorted, no timestamps, no random seeds, no local paths. Byte
comparison only makes sense on the same numerical setup; across machines, use
the validator's error budgets.

## Changing tasks

Task definitions are in `astrotoolbench.benchmark.families`. After changing one:

```bash
python -m astrotoolbench.benchmark.build
python -m astrotoolbench.generate_references --check
python -m astrotoolbench.validate
python -m pytest -q
```

`build` regenerates specs, both schemas, the manifest and the references. It
requires at least 40 tasks, all 6 categories, 5 families and 4 difficulties,
with each difficulty within 5 points of 40/25/15/20 %.

A snapshot test fails if any expected value changes, so you have to update it
deliberately. When I added the equal-radius Hohmann and zero-ΔV tasks, I kept
the digest of the 48 earlier tasks as a separate check. Current total: 50 tasks,
38 answers, 12 expected errors.
