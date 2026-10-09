# AstroToolBench

A small benchmark for checking whether language models use **scientific tools**
correctly, starting with orbital mechanics.

v0.1.0 ships deterministic orbital-mechanics tools, 50 tasks with verified
reference answers, and a validator that runs offline (no model, no API key, no GPU).

## Why?

When a model calls a numerical tool, a lot can go wrong: wrong function, wrong
units, rounding an intermediate result, or happily computing something that is
physically impossible. A nice-sounding answer doesn't tell you any of that. I
wanted tasks where the answer is a number you can check, with a clear tolerance.

## Why astrodynamics?

Orbital problems are a good fit: answers can be checked analytically or by
integrating the equations of motion, the physics assumptions are well defined,
and the tools chain together naturally (transfer → propagate → eclipse check).
There are also plenty of edge cases to build trap questions from.

## Roadmap

```text
scientific benchmark → tool API design → LLM evaluation → post-training → GPU inference
```

v0.1 covers the first step: the dataset and making it reproducible. Model
comparisons, fine-tuning, MCP, vLLM and CUDA work come later. The early model
experiments I ran are described [further down](#model-experiments).

## Quick start

Needs Python 3.10+.

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

(Windows: `.venv\Scripts\Activate.ps1`.)

The base package only depends on NumPy and Pydantic. Tests that need the LLM
client or a tokenizer are skipped if those extras aren't installed; the
scientific tests never call a model.

To check the committed files without rewriting anything:

```bash
python -m astrotoolbench.generate_references --check
python -m astrotoolbench.benchmark.build --check
python -m astrotoolbench.validate --json
```

`validate` exits with 0 if everything passes, 1 otherwise. More options in the
[validation guide](docs/validation.md).

## What's in v0.1

| | |
| --- | --- |
| Topics | Propagation, ΔV, Hohmann transfers, eclipses, closest approach |
| Tasks | 50, in 6 categories and 22 families |
| Difficulty | 20 simple, 12 multistep, 8 diagnostic, 10 trap |
| Outcomes | 38 numerical answers, 12 expected errors |
| Splits | 24 train / 6 validation / 20 test, split by family |
| Reference checks | 33 analytic, 5 RK4 integration, 12 input checks |

Trap and diagnostic tasks don't always mean "refuse": some have a valid answer.
Each numerical output has its own tolerance; booleans and error codes must match
exactly. Splits are done by problem family so that changing a radius doesn't
leak a task into the test set (details in [data splits](docs/data-splits.md)).
This obviously says nothing about what's in a model's pretraining data.

## Layout

```text
src/astrotoolbench/tools/       scientific tools
src/astrotoolbench/benchmark/   schema, loader, task definitions, verification
src/astrotoolbench/models/      common model interface, optional API adapter
src/astrotoolbench/eval/        no-tools / raw-tools baselines and traces
src/astrotoolbench/validate.py  validation command
tasks/astrodynamics/            task sources, references, split manifest
schemas/                        JSON Schemas
tests/                          tests (tools/, benchmark/, models/, eval/, validator)
examples/                       offline examples
docs/                           documentation
```

References are generated from `tasks/astrodynamics/specifications.jsonl`, which
contains no answers. The generator computes each answer, checks it with an
independent method (closed-form solution, RK4 or input checks) and only then
writes `tasks/astrodynamics/tasks.jsonl`, the split manifest and the
[verification report](docs/reference-verification.json).

The physics model is two-body motion with a cylindrical Earth shadow and a
fixed Sun direction. Units: km, km/s, s, rad, in an ECI frame.

```bash
python examples/scientific_workflow.py
```

More docs: [tools](docs/scientific-tools.md) · [task format](docs/task-format.md) ·
[ground truth](docs/ground-truth.md) · [reference generation](docs/reference-generation.md) ·
[problem families](docs/problem-families.md) · [architecture](docs/architecture.md) ·
[release notes](docs/release-v0.1.0.md)

## Model experiments

Model runs use the frozen `benchmark-v0.1` corpus: the
[frozen loader](docs/benchmark-freeze.md) checks file hashes before returning
any task, so every result is computed on exactly the same tasks.

I picked four [models](docs/baseline-models.md): Claude Sonnet 4.6, Gemini 2.5
Pro, Nemotron 3.5 Lightning and Qwen3 4B (local, pinned). They all go through the
same small [backend interface](docs/model-backends.md).

Two conditions so far:

- [**no_tools**](docs/no-tools-baseline.md): one turn, the model answers alone.
- [**raw_tools**](docs/raw-tools-baseline.md): the model can call the scientific
  functions as they are, with their Python names and one-line docstrings.

```bash
# preview a request, no API call
python -m astrotoolbench.eval --model nemotron --task prop-circular-quarter --dry-run

# run the test split (needs .[llm] and NVIDIA_API_KEY)
python -m astrotoolbench.eval --model nemotron --condition no_tools --split test
python -m astrotoolbench.eval --model nemotron --condition raw_tools --split test
```

Results go to `results/<profile>/<condition>.jsonl` and are never overwritten.
With tools, each task also gets a trace file with every call, its arguments,
result or error and timing. Everything is tested offline with fake models; real
numbers are coming.

Before building the official dataset, I wrote a first prototype (`atb` package)
that runs a model in a tool-calling loop against these tools, with three
interface variants (A: no tools, B and C: two different tool schemas). It's
still there and still tested. Install `.[dev,llm]` to use it, plus `tokenizer`
for the chat-template experiments.

The `astrodyn_tools` imports still work and point to the same functions. The
package on disk is still called `astrodyn-tools`; the import name is
`astrotoolbench`.

See [running the agent](docs/agent-execution.md), [inspecting completions](docs/completion-inspection.md),
[tokenizer inspection](docs/template-inspection.md) and the
[first smoke test](docs/release-validation.md) (Nemotron 3.5 Lightning, 4/5 tasks
passed). That was an integration test on a few dev tasks, not a benchmark result.

## License

[MIT](LICENSE).
