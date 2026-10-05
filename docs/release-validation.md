# First smoke test (Nemotron 3.5 Lightning)

This is the integration test I ran on the `atb` agent prototype on 3 October
2026, before building the official 50-task dataset. The tasks here come from the
older development set (`benchmark/tasks.jsonl`), not from v0.1.0.

At that point the test suite had 116 tests (Python 3.12.13).

## Setup

Five tasks picked in advance: an orbital period, a Hohmann transfer, an
eclipse, a two-step calculation and a physically impossible request.

| Setting | Value |
| --- | --- |
| Provider | NVIDIA API Catalog (`https://integrate.api.nvidia.com/v1`) |
| Model | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| Interface | C |
| Temperature | 0 |
| Max requests per task | 8 |
| Max output tokens | 4096 |
| Reasoning budget | 256 |
| Timeout | 120 s |
| SDK retries | 0 |
| Run ID | `20261003T125148Z-smoke-aefe03` |

To rerun it (needs a `.env` with an API key):

```bash
python scripts/smoke_test.py --model nvidia/nemotron-3.5-lightning-30b-a3b \
  --api C --reasoning-budget 256 --timeout 120
```

Results can change between runs even at temperature 0. The
[summary](examples/nvidia-lightning-smoke.json) links the five full traces with
their SHA-256 (I only replaced local paths with relative ones).

## Results

| Task | Result | Requests / tool calls | Tokens | Latency |
| --- | --- | ---: | ---: | ---: |
| `period-001` | Pass: 5553.624271252228 s | 2 / 1 | 3443 | 9.89 s |
| `hoh-001` | Fail: no final `answer` object | 2 / 1 | 3684 | 4.42 s |
| `ecl-001` | Pass: 2297.9669189453125 s | 2 / 1 | 3605 | 6.70 s |
| `multi-001` | Pass: −5368.7591375471 km | 3 / 2 | 6103 | 62.82 s |
| `trap-001` | Pass: refused the impossible altitude | 1 / 0 | 1601 | 9.85 s |

**4/5 correct numbers, but only 3/5 answers were clean standalone JSON.** No
tool errors. The script exits with 1 because of the Hohmann failure.

- **Hohmann:** the tool returned the right ΔV, but the model then just repeated
  the tool arguments in a code block instead of answering. Counted as a fail.
- **Eclipse:** correct value, but with an extra `}` after the JSON. The scorer
  still finds the answer, so it passes numerically but not on format. I think
  format and correctness should be two separate metrics.
- **Two-step task:** the model passed the full-precision transfer time
  (3560.540788789012 s) from `hohmann_transfer` into `propagate_orbit`. The x error
  is about 0.000862 km, well inside the 0.1 km tolerance. There's a regression
  test showing that rounding that time to whole seconds would fail.

## What went wrong before that

All earlier attempts are in the [attempt log](examples/smoke-attempts.json). I
don't merge them with the result above.

- My default model at first, Nemotron 3 Super, started returning HTTP 410. A Nano
  model ID from the `/models` list gave 404. I switched to Lightning by hand; the
  runner never switches models on its own. Old Super traces keep their model name.
- First Lightning run: 3/5, with timeouts at 60 s on the eclipse and two-step
  tasks. Retrying eclipse worked; the two-step task still timed out at 120 s.
- With the 256-token reasoning budget, a full run first scored 1/5. Looking at
  the two-step answer I found a bug in my parser: it picked up a JSON block with
  tool arguments instead of the final answer that came after it. Fixed it to take
  the last answer object. Rescoring the same run gives 2/5 (no new API call).
- I then made the system prompt clearer: final answer must be a standalone JSON
  object, and keep full precision between steps. The final run above uses this
  prompt, so it's a different setup from the earlier ones.

The reasoning budget is sent with every request. See the
[NVIDIA API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-5-lightning-30b-a3b-infer).
I haven't tried to find the best value; a small budget may hurt answers.

## Notes

Five dev tasks is an integration check, not a model evaluation. All 24 tasks of
this old dataset are dev-only and are never used as training data.

In interface C, the model has to convert radii to altitudes for some tasks,
which is one more place to make mistakes. B and C differ in more than just the
descriptions (input conventions, schema constraints), so a proper comparison
will need the same model and settings.

The token numbers I measured (C costs 375 more prompt tokens than B; C's
descriptions alone cost 407) come from the Qwen tokenizer, not Nemotron, so they
are not NVIDIA billing numbers. Since tool schemas are resent on each request,
that overhead adds up over a conversation. Details in
[tokenizer inspection](template-inspection.md).

No API keys or personal paths are in the published files.
