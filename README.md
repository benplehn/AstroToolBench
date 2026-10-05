# AstroToolBench

AstroToolBench is an open-source benchmark for evaluating reliable scientific
**tool use by language models**. Version **0.1.0** provides deterministic orbital
mechanics tools, 50 independently verified tasks, reproducible references and a
validator that runs without a model, API key or GPU.

## Why?

A model using numerical tools must choose a suitable calculation, supply valid
arguments and units, preserve intermediate precision, and recognize physical or
model limits. Plausible prose alone cannot establish correctness. AstroToolBench
makes those behaviors testable with explicit inputs, answer contracts and
scientific error budgets.

## Why astrodynamics first?

Orbital problems produce numerical answers that can be checked with analytic
relations or independent integration. They combine propagation, maneuvers,
eclipses and proximity searches into dependent calculations, with clear physical
assumptions and useful boundary cases. This makes astrodynamics a practical first
domain for testing scientific tool use.

## Where is this going?

```text
Scientific benchmark → API design → LLM evaluation → post-training → GPU inference
```

The v0.1 release establishes the scientific dataset and its reproducibility.
Comparative model evaluation, fine-tuning, MCP, vLLM and CUDA experiments remain
future work. Existing model integration experiments are documented separately
below; their development tasks are excluded from official training partitions.

## Quick start

Python **3.10 or later**, from the repository root:

```bash
git clone https://github.com/benplehn/AstroToolBench.git
cd AstroToolBench
git checkout v0.1.0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m astrotoolbench.generate_references
python -m astrotoolbench.validate
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.
The base package needs only NumPy and Pydantic; the `dev` extra installs pytest.
Optional model-client and tokenizer tests are skipped when their dependencies
or cached tokenizer are absent. Scientific tests and commands never call an LLM.

To verify existing artifacts without rewriting them:

```bash
python -m astrotoolbench.generate_references --check
python -m astrotoolbench.benchmark.build --check
python -m astrotoolbench.validate --json
```

The validator exits **0** when all checks pass and **1** for dataset or artifact
issues. Default validation covers the complete official corpus and its sources,
partitions and scientific evidence. See the [validation guide](docs/validation.md)
for custom paths, subset checks and machine-readable diagnostics.

## What v0.1 contains

| Item | Release contents |
| --- | --- |
| Scientific areas | Propagation, ΔV, Hohmann transfer, eclipse geometry/duration, closest approach. |
| Tasks | 50 across six categories and 22 descriptive families. |
| Difficulty | 20 simple, 12 multistep, 8 diagnostic, 10 trap. |
| Outcomes | 38 successful computations and 12 independently reproduced errors. |
| Partitions | 24 train, 6 validation, 20 test; 11 reviewed family/composition groups. |
| Ground truth | 33 analytic, 5 Cartesian RK4 and 12 independent input certificates. |
| Contracts | Explicit units, frame, model assumptions, tolerances and error codes. |

A diagnostic or trap can have a valid answer; its label does not automatically
imply refusal. Numeric outputs have individual scoring tolerances and tighter
verification budgets. Boolean results and error codes use exact comparison.
The split policy keeps numerical variants together and permits primitive tool
reuse in held-out compositions. It does not establish absence from external
model pretraining data. See [the split policy](docs/data-splits.md).

## Data and tools

```text
src/astrotoolbench/tools/       Deterministic scientific implementations
src/astrotoolbench/benchmark/   Schema, loader, recipes and independent verification
src/astrotoolbench/validate.py  Public validation command
tasks/astrodynamics/            Answer-free sources, references and split manifest
schemas/                       Exported JSON Schemas
tests/tools/                   Scientific tool tests
tests/benchmark/               Corpus, generation and reference regression tests
tests/test_validation.py       Validator and CLI tests
examples/                      Offline scientific workflows
docs/                          Scientific contracts, evidence and release checklist
```

The generator reads `tasks/astrodynamics/specifications.jsonl`, calculates the
answers and verifies them independently before publishing any output. It writes
`tasks/astrodynamics/tasks.jsonl`, the split manifest and the
[accuracy report](docs/reference-verification.json). It does not use existing
expected numbers as computational input. The reviewed reference snapshots remain
protected by regression tests.

The production model is two-body orbital dynamics with a fixed-Sun cylindrical
shadow. Distances are km, velocities km/s, time s and angles rad in an ECI frame
for Earth tasks. Sampled event searches have explicit resolution limits; new
families must provide suitable independent certificates.

```bash
python examples/scientific_workflow.py
```

[Tool contracts](docs/scientific-tools.md) · [Task format](docs/task-format.md) ·
[Ground-truth evidence](docs/ground-truth.md) · [Reference generation](docs/reference-generation.md) ·
[Problem families](docs/problem-families.md) · [Architecture](docs/architecture.md) ·
[Release checklist](docs/release-v0.1.0.md)

## Optional model experiments

The existing `atb` package supports legacy A/B/C interface experiments and saved
traces. Install `.[dev,llm]` for model-client tests and commands, or add `tokenizer`
for local template inspection. `astrodyn_tools` imports remain compatible with
the canonical scientific functions. The distribution name remains
`astrodyn-tools`; the public scientific namespace is `astrotoolbench`.

See [model execution](docs/agent-execution.md), [completion inspection](docs/completion-inspection.md),
[tokenizer inspection](docs/template-inspection.md) and the
[historical integration report](docs/release-validation.md). The recorded five-task
Lightning smoke test passed 4/5 numerical checks; it is a development integration
check, not a result on the official held-out corpus.

## License

[MIT](LICENSE).
