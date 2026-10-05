# Integration validation

This is the historical week-end 1 development integration report. The official
50-task scientific release is described in [the v0.1.0 checklist](release-v0.1.0.md).
The task counts and live model results below refer to the legacy development data.

Validated on 3 October 2026 with Python 3.12.13. The local suite passed **116
tests**, including numerical checks, tool-call protocol failures, CLI output and
cached-tokenizer rendering. Dependency consistency, source compilation and the
offline request preview also passed.

## Five-task live check

The task selection covers period, transfer, eclipse, a dependent calculation and
a physically invalid request. It was fixed before execution. All five tasks use
the same model, condition and generation settings:

| Setting | Value |
| --- | --- |
| Provider | NVIDIA API Catalog, `https://integrate.api.nvidia.com/v1` |
| Model | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| Interface | C |
| Temperature | 0 |
| Maximum model requests per task | 8 |
| Output-token limit per request | 4096 |
| Reasoning budget per request | 256 |
| Client timeout per request | 120 s |
| Automatic SDK retries | 0 |
| Run | `20261003T125148Z-smoke-aefe03` |

Reproduce from the checkout with a configured `.env`:

```bash
python scripts/smoke_test.py --model nvidia/nemotron-3.5-lightning-30b-a3b \
  --api C --reasoning-budget 256 --timeout 120
```

Hosted responses and latency can vary even at temperature zero. The
[summary](examples/nvidia-lightning-smoke.json) links all five full traces and
records their SHA-256 hashes. Only local trace paths were normalized for publication.

| Task | Numerical score | Requests / tools | Reported total tokens | Latency |
| --- | --- | ---: | ---: | ---: |
| `period-001` | Pass: 5553.624271252228 s | 2 / 1 | 3443 | 9.89 s |
| `hoh-001` | Fail: no final `answer` object | 2 / 1 | 3684 | 4.42 s |
| `ecl-001` | Pass: 2297.9669189453125 s | 2 / 1 | 3605 | 6.70 s |
| `multi-001` | Pass: −5368.7591375471 km | 3 / 2 | 6103 | 62.82 s |
| `trap-001` | Pass: physically invalid altitude refused | 1 / 0 | 1601 | 9.85 s |

**Result: 4/5 numerical checks passed; 3/5 responses were standalone final JSON
objects.** Every request received a response in this final run, and no numerical
tool reported an execution error. The script correctly returned exit code 1
because Hohmann failed.

The Hohmann tool returned the correct delta-v, but the model's final response
repeated input arguments in a code block instead of answering the task. This is
a model output failure; the runner retained it and the scorer rejected it.
The eclipse response contained an extra closing brace after its answer object.
It passed the numerical scorer, which extracts a JSON answer from surrounding
text, but failed the separate standalone-JSON check. Numerical correctness and
format compliance should remain separate metrics in future evaluations.

The dependent task called `hohmann_transfer`, then passed its full-precision
duration, 3560.540788789012 s, to `propagate_orbit`. Its x-coordinate error against
the rounded reference is approximately 0.000862 km, within the **0.1 km** tolerance.
Rounding the intermediate duration to whole seconds is rejected by regression tests.

## Earlier attempts and fixes

Earlier outcomes are retained in the [attempt ledger](examples/smoke-attempts.json).
They are not pooled with the final five-task result:

- The former default, Nemotron 3 Super, returned HTTP 410. A Nano ID from provider
  discovery returned HTTP 404. Lightning was then verified and selected explicitly;
  there is no automatic model fallback within a run. Historical Super examples
  keep their original model identity and requests.
- Lightning's first five-task run passed 3/5, with eclipse and dependent-task
  timeouts at 60 s. An explicit eclipse retry passed; the dependent task still
  timed out at 120 s. Missing usage is recorded as unknown.
- A complete run with a 256-token reasoning budget originally scored 1/5.
  Reviewing its dependent-task answer exposed a parser bug: an earlier fenced
  tool-arguments object took precedence over a later valid answer. The corrected
  parser selects the last answer/refusal object regardless of Markdown wrapping.
  Re-scoring that existing run gives 2/5; this does not represent a new request.
- The shared A/B/C system instruction now explicitly requires a standalone JSON
  final response and preservation of intermediate precision. The final complete
  run above uses this instruction, so it is a separate validation configuration.
  Hohmann's remaining failure is reported rather than converted into a pass.

The optional reasoning setting is sent and recorded at every turn. See the
[NVIDIA API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-5-lightning-30b-a3b-infer)
for its provider-specific semantics. An explicit small budget can affect answer
quality; these runs do not establish its optimal value.

## Experimental limits and observations

These five development tasks validate integration, not comparative model quality.
All 24 tasks in this historical dataset belong to the development split. They
remain excluded from official training exports. The official scientific corpus
now has a [reviewed family/composition split](data-splits.md); comparative held-out
model evaluation and external data provenance review remain future work.

Condition C requires the model to convert radii into altitudes for some tasks.
That conversion is a potential failure source and part of the interface treatment.
B/C differences include input conventions and schema constraints as well as
descriptions; their future comparison must hold model and settings constant.

The tokenizer experiment uses Qwen, while the recorded API conversation uses
Nemotron. Its **375-token C-versus-B prompt difference** and **407-token cost of
C description fields** describe the pinned Qwen template and historical initial
messages. They are not NVIDIA billing estimates. Schemas are resent at every
request, so their token overhead recurs throughout a tool-use conversation.
See [tokenizer inspection](template-inspection.md) for the measurement method.

Credentials remain local; the published records contain no API keys or personal
filesystem paths.
