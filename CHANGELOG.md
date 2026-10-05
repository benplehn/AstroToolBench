# Changelog

## 0.1.0 — 2026-10-05

First real release of the benchmark.

- Cleaned up the propagation, ΔV, Hohmann, eclipse and closest-approach tools,
  with input validation and documented units/assumptions.
- 50 tasks (6 categories, 22 families), 20/12/8/10 difficulty mix,
  24/6/20 train/validation/test split by family.
- Task sources without answers, JSON Schemas, and a split registry that refuses
  duplicates or near-duplicates across splits.
- Every reference answer is checked by a second method (analytic, RK4 or input
  checks), including the 12 tasks where the correct answer is an error.
- New commands: `python -m astrotoolbench.generate_references` and
  `python -m astrotoolbench.validate`.
- Tests for tools, dataset and validator; CI on Python 3.10 and 3.12 without
  the LLM dependencies.
- pytest only collects `tests/`, so `.[dev]` alone is enough.
- Moved the scientific code to `astrotoolbench` (old `astrodyn_tools` imports
  still work). Data, schemas, examples and license are included in the sdist.

The first agent prototype (`atb`) and its traces are kept as they were. No model
comparison, fine-tuning or GPU results in this release.
