# Baseline A: no tools

One model turn per task, no tools. This measures what the model can do on its
own.

The request has the problem text, input values with units and the physical
context. No reference, tolerance, recipe, verification details, notes or
difficulty label. The model has to answer with one JSON object:

```json
{"outputs": {"delta_v": 1.25}}
```

or, for an invalid request:

```json
{"error": "missing_input", "reason": "The initial velocity is missing."}
```

The reason just has to be non-empty; I don't grade the explanation. Numbers use
each output's tolerance, vectors are compared component by component, booleans
and error codes must match exactly. A trap task doesn't automatically mean
"refuse".

## Running it

```bash
python -m pip install -e ".[dev,llm]"
python -m astrotoolbench.eval --model nemotron --task prop-circular-quarter --dry-run
python -m astrotoolbench.eval --model nemotron --condition no_tools --split test
```

Without `--split`, all 50 tasks run. `--task` and `--family` narrow it down. The
whole frozen corpus is hash-checked before any call, even for a single task.
`--dry-run` needs no key, loads no weights and writes nothing.

Results go to `results/<profile>/no_tools.jsonl` (or `--output`). Existing files
are never overwritten. Other options: `--max-tokens`, `--timeout`, `--env-file`.
Keys and local setup: see [models](baseline-models.md).

Each line has: task ID, family, difficulty, category, split, benchmark tag and
commit, model profile, generation settings, the exact messages, the raw answer,
the parsed response, `correct`, `format_ok`, tokens and latency (ms). The model ID
actually returned by the API is in `response.model`. No keys, no references.

## Scoring

Only a finished response (`finish_reason` = `stop`) is scored. Two separate
things are recorded:

- `format_ok`: the whole answer is a single valid JSON object, as asked.
- `correct`: the answer has the right values. If the JSON is wrapped in
  ```` ```json ```` fences or some text, the last object with `outputs` or `error`
  is used. So a right value with bad formatting is `correct: true,
  format_ok: false`.

Duplicate keys and NaN/Infinity are rejected either way. Truncated, refused,
empty answers and unexpected tool calls are kept with `correct: false` and their
own `status`. An API refusal is not the same as a scientific error returned in
the JSON. Missing token counts stay `null`.

If the provider fails, it's recorded and the run moves on. A bug in my code stops
the run, but earlier lines are kept (each line is flushed before the next call).
Exit code 0 means the run finished, even with wrong answers; 1 means backend or
setup errors.

Latency is one `generate` call, network included. Loading a local model happens
before the timer. Temperature is 0 for hosted models and `do_sample=false`
locally, which still doesn't guarantee identical answers over time or across
hardware.

So far this has only been tested offline with fake responses on all 50 tasks.
No real A vs B numbers yet.

[Baseline B](raw-tools-baseline.md) uses the same messages and scoring, plus the
scientific functions and a tool loop.
