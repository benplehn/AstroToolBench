# Inspecting a completion

`atb-inspect` sends one task to a model through an OpenAI-compatible Chat
Completions API and shows how it asks for tools. It prints the request, keeps the
raw response and saves everything to a JSON file. It doesn't execute any tool.

## Usage

Install `.[dev,llm]`, then from the repo root:

```bash
atb-inspect --dry-run
atb-inspect --task period-001 --require-tool-call
```

`period-001` asks for the period of a circular orbit at 400 km altitude. The
request contains the four C tool schemas, and of course not the answer.

The installed command resolves paths from the current directory. To resolve
them from the repo instead:

```bash
python scripts/inspect_completion.py --task period-001
```

Output goes to `results/completions/` (or `--output`; an existing file is
refused before calling the API). Record format: see [architecture](architecture.md).

## Configuration

Copy `.env.example` to `.env` and add your key. Defaults: endpoint
`https://integrate.api.nvidia.com/v1`, model `nvidia/nemotron-3.5-lightning-30b-a3b`.

| Host | Key used |
| --- | --- |
| `integrate.api.nvidia.com` | `NVIDIA_API_KEY` |
| `openrouter.ai` | `OPENROUTER_API_KEY` |
| `api.openai.com` | `OPENAI_API_KEY` |
| anything else | `LLM_API_KEY` |

`LLM_API_KEY` always wins if set. Environment variables override `.env`, and
`--model` overrides `LLM_MODEL`. To switch provider, set `LLM_BASE_URL` and
`LLM_MODEL` (the model is required for non-NVIDIA endpoints).

List models (when the provider has `/models`):

```bash
python scripts/list_models.py --filter nemotron
```

Being in the list doesn't mean the model supports tool calls.

I started with Nemotron 3 Super, but its endpoint returned HTTP 410 on 3 October
2026, so the default is now Nemotron 3.5 Lightning. The old Super traces keep
their model name. See the [smoke test](release-validation.md).

## Reading the response

| Field | Meaning |
| --- | --- |
| `message.tool_calls` | the tool calls the model asked for |
| `finish_reason` | `tool_calls` = tool request, `stop` = usually a text answer, `length` = cut off |
| `function.name` | requested function |
| `function.arguments` | a JSON **string**, to parse and validate |
| `tool_calls[].id` | ID to use in the tool reply |
| `usage` | token counts, may be missing |

It handles multiple calls, broken JSON, empty choices and missing usage.
`--require-tool-call` fails unless there's at least one call and every call has
a name, an ID and JSON-object arguments. It doesn't check the schema, the
physics or whether it's the right function.

## Example

On 2 October 2026, Nemotron 3 Super with `tool_choice: "auto"` asked for
`orbital_period` with `{"altitude_km":400}` and `finish_reason: "tool_calls"`.
1,331 prompt + 61 completion = 1,392 tokens, about 11.95 s.

The [full trace](examples/nvidia-nemotron-orbital-period.json) has the real
request and response. It only shows that the model produces a sensible tool call
on one task.

## Options

| Option | |
| --- | --- |
| `--dry-run` | print the request, no network or key needed |
| `--tool-choice auto` | model decides (default) |
| `--tool-choice required` | force a tool call if the provider supports it (changes the experiment) |
| `--tool-choice none` | schemas sent but calls disabled (condition A omits the schemas instead) |
| `--timeout 60` | request timeout in s |
| `--max-tokens 4096` | output tokens, may include reasoning |
| `--env-file PATH` | another dotenv file |

Temperature 0, no automatic retries, client closed after the call. API errors
give exit code 1 with a short message (the provider's error body isn't printed)
and no trace. If `--require-tool-call` fails, the response is still saved, then
exit 1.

On HTTP 410 it suggests passing `--model`. It never switches models silently.

## Tests

```bash
python -m pytest -q
```

Covers config, key selection, the exact request sent, client cleanup, answers
never leaking into the request, broken/multiple calls, trace writing, dry runs
and errors. Tests use fake responses, so CI needs no key.

macOS gotcha: if `atb` can't be imported after `pip install -e`, check whether
the editable install's `.pth` file is hidden (Python skips hidden `.pth` files).
A normal `python -m pip install ".[dev,llm]"` avoids it, but then reinstall after
each change.

References: [NVIDIA function calling](https://docs.nvidia.com/nim/large-language-models/latest/function-calling.html),
[OpenAI Python SDK](https://github.com/openai/openai-python).
