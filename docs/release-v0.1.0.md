# v0.1.0 release notes

Goal for this version: a benchmark that anyone can rebuild and check from a
clean clone, before running any model on it.

## What I wanted before tagging

| Goal | Where it stands |
| --- | --- |
| At least 40 tasks | 50 in `tasks/astrodynamics/tasks.jsonl` |
| At least 5 problem families | 22 families, grouped into 11 split groups |
| At least 4 task types | 20 simple, 12 multistep, 8 diagnostic, 10 trap |
| Every reference verified | 33 analytic, 5 RK4, 12 input checks |
| A tolerance on every number | Per-output tolerances; booleans and error codes compared exactly |
| Every task has a family | Enforced by the schema and the family registry |
| Tests pass | 422 passed locally on Python 3.10 and 3.12. Clean install with `.[dev]` only: 352 passed, 8 skipped (optional LLM/tokenizer tests) |
| Validator is happy | 50 valid, 0 invalid, 0 issues |
| README readable | What / why / quick start come first |

Tasks where the right answer is an error (e.g. an orbit that goes through the
Earth) count as valid tasks, as long as both the tool and the independent
checker produce the same error code.

## Reproducing

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m astrotoolbench.generate_references
python -m astrotoolbench.validate
python examples/scientific_workflow.py
```

`pip install -e .` without `[dev]` is enough for the scientific commands (no
pytest). The LLM client tests need `llm`, the tokenizer tests need `tokenizer`
and a cached tokenizer; they're skipped otherwise.

The `--check` modes compare files byte for byte, which only makes sense on the
same machine/numerical setup. Across platforms, the validator uses the
scientific error budgets instead.

## Before pushing the tag

- full test suite + `validate`
- both `--check` commands leave the files unchanged
- fresh clone + fresh venv, without the LLM extras
- build sdist and wheel, check data/schemas/docs/license are inside
- check doc links, make sure `.env` and generated files aren't committed
- CI green on 3.10 and 3.12 for that commit

On release day GitHub Actions had a
[runner incident](https://www.githubstatus.com/incidents/3q1yb5m7ltvb): only the
Linux 3.12 scientific job ran, the others stayed queued. The full suite passed
locally on 3.10 and 3.12, so I released anyway and noted it on the release page.

## Limits

Two-body dynamics and a cylindrical shadow with a fixed Sun only. Eclipse and
closest-approach searches are sampled, so they can miss very short events; the
committed tasks are chosen so they don't. No J2, no Lambert solver yet.

The family split prevents obvious leaks between train and test, but tools are
shared across splits on purpose, and of course I can't know what's in a model's
pretraining data.

[Repository](https://github.com/benplehn/AstroToolBench) ·
[Changelog](../CHANGELOG.md) · [Validation](validation.md) ·
[Ground truth](ground-truth.md) · [Splits](data-splits.md)
