# Baseline A: no tools

Each task gets one model turn with the same system prompt and public task data.
No tool definitions are sent and no scientific function is executed on behalf
of the model.

The request contains the original problem text, input values and units, and
physical context. It excludes references, numerical tolerances, recipes,
verification details, notes and difficulty labels. The prompt asks for exactly
one JSON object, matching the existing scientific comparator:

```json
{"outputs": {"delta_v": 1.25}}
```

For an invalid request:

```json
{"error": "missing_input", "reason": "The initial velocity is missing."}
```

The error reason must be nonempty; its semantic correctness is not evaluated by
the deterministic comparator. Numerical outputs use the committed per-output
tolerances, vectors are compared component by component, and booleans and error
codes must match exactly. Difficulty does not decide whether a task should be
refused.

## Run or inspect a request

From the repository root, after installing the package:

```bash
python -m pip install -e ".[dev,llm]"
python -m astrotoolbench.eval --model nemotron --task prop-circular-quarter --dry-run
python -m astrotoolbench.eval --model nemotron --condition no_tools --split test
```

Without a split filter, the command runs all 50 frozen tasks. `--task` and
`--family` can narrow the selection. The full corpus and companion hashes are
checked before any model call, including when running a single task. A dry run
needs no API key and does not load weights or create result files.

The default result path is `results/<profile>/no_tools.jsonl`. Use `--output` for
a different run; existing files are never overwritten. `--max-tokens`,
`--timeout` and `--env-file` are explicit request options. See
[model selection](baseline-models.md) for keys and local model setup.

Each JSONL line stores the task ID, family, difficulty, category, split, frozen
benchmark tag and commit, requested model profile, generation settings, exact
messages, raw answer text, normalized response, correctness, tokens and latency
in milliseconds. The returned model ID is retained in `response.model`.
Credentials and references are not written.

Only a complete response with finish reason `stop` and a valid JSON object is
compared. Extra prose, Markdown fences, duplicate keys and nonfinite JSON
constants are rejected. Truncation, refusal, empty responses and unexpected tool
calls stay in the results with `correct=false` and a distinct `status`. An API
refusal differs from a scientific diagnostic returned in the required JSON.
Unknown token counts stay `null`.

Provider errors are recorded and the run continues to the next task. Programming
errors stop the run, preserving previous lines. Each line is flushed to disk
before the next call. A finished experiment exits with 0 even if answers are
wrong; backend or setup failures exit with 1.

Latency covers one `generate` call. API connection setup and network time are
included. Local model loading happens before the task timer. Temperature is 0
for hosted profiles; local decoding uses `do_sample=false`. Neither setting
guarantees identical answers across hosted model updates or different hardware.

The implementation has been tested offline with fake responses covering all 50
tasks. Those checks are not model baselines. Real runs need the API keys or the
local dependencies and weights; no measured A-vs-B table exists yet.
