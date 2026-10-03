# Completion inspection

Inspect how a model represents scientific tool requests through an
OpenAI-compatible Chat Completions API. The command prints the request, preserves
the raw response and records observations in a JSON trace.

## Usage

Install `.[dev,llm]`, then run from the repository root:

```bash
atb-inspect --dry-run
atb-inspect --task period-001 --require-tool-call
```

`period-001` asks for the period of a circular Earth orbit at 400 km altitude. The
request includes all four C tool schemas. The model receives the prompt without
the reference, tolerance, numerical parameters or expected tool names.

The installed command resolves default paths relative to the current directory.
For a repository-local entry point that resolves defaults relative to the checkout:

```bash
python scripts/inspect_completion.py --task period-001
```

Traces are saved under `results/completions/`. `--output` selects another path;
existing files are rejected before contacting the provider. See the
[architecture](architecture.md) for the trace format and component boundaries.

## Configuration

Copy `.env.example` to `.env` and set the credential locally. The default endpoint
is `https://integrate.api.nvidia.com/v1` and the default model is
`nvidia/nemotron-3.5-lightning-30b-a3b`.

| Endpoint host | Credential fallback |
| --- | --- |
| `integrate.api.nvidia.com` | `NVIDIA_API_KEY` |
| `openrouter.ai` | `OPENROUTER_API_KEY` |
| `api.openai.com` | `OPENAI_API_KEY` |
| Other compatible endpoint | Set `LLM_API_KEY` explicitly |

`LLM_API_KEY` overrides the matching provider credential. Process variables take
precedence over `.env`; `--model` overrides `LLM_MODEL`. To change providers, set
both `LLM_BASE_URL` and `LLM_MODEL`, then provide that provider's key. Non-NVIDIA
endpoints require an explicit model ID.

Model IDs can be listed where the provider supports `/models`:

```bash
python scripts/list_models.py --filter nemotron
```

Being listed does not establish support for function calling.
The first integration records used Nemotron 3 Super. Its hosted endpoint returned
HTTP 410 on 3 October 2026, so the default now uses Nemotron 3.5 Lightning. Saved
Super traces keep their original model identity. See the
[release validation](release-validation.md) for the current five-task check.

## Reading a response

| Field | Interpretation |
| --- | --- |
| `message.tool_calls` | Structured function requests produced by the model. |
| `finish_reason` | `tool_calls` indicates a tool request; `stop` usually indicates a text answer; `length` indicates truncation. |
| `function.name` | Requested function name. |
| `function.arguments` | A string containing JSON that the application must parse and validate. |
| `tool_calls[].id` | Correlation ID for a later tool-result message. |
| `usage` | Provider-reported tokens and optional breakdowns; may be absent. |

The inspector handles multiple calls, malformed argument JSON, empty choices and
absent usage. It preserves the raw fields while displaying parsed arguments.
`--require-tool-call` checks that all returned calls have a function name, ID and
JSON-object arguments, and that at least one call exists. Schema compliance,
physical validity and correct function selection need additional validation.

## Recorded integration check

A call on 2 October 2026 with Nemotron 3 Super and `tool_choice: "auto"` requested
`orbital_period` with the JSON string `{"altitude_km":400}` and returned
`finish_reason: "tool_calls"`. The API reported 1,331 prompt tokens, 61 completion
tokens and 1,392 total tokens. Observed completion latency was approximately
11.95 seconds.

The [complete recorded trace](examples/nvidia-nemotron-orbital-period.json)
contains the actual request and SDK response. This verifies tool-request generation
for one development task. It does not establish numerical answer accuracy or a
success rate across the benchmark.

## Controls and failure handling

| Option | Behavior |
| --- | --- |
| `--dry-run` | Prints the request without network access or a credential. |
| `--tool-choice auto` | Lets the model decide whether to request tools; the default. |
| `--tool-choice required` | Forces a tool request if supported by the provider; changes the experimental condition. |
| `--tool-choice none` | Exposes the schemas but disables calls; condition A must instead omit schemas. |
| `--timeout 60` | SDK request timeout in seconds. |
| `--max-tokens 4096` | Output token budget, potentially including reasoning. |
| `--env-file PATH` | Explicit dotenv configuration file. |

Requests use temperature zero. Automatic retries are disabled. The client is
closed after the request. API failures return exit code 1 with concise diagnostic
messages and no provider error body. No response trace is written for a failed API
request. Responses that fail `--require-tool-call` are preserved before exiting
with code 1.

HTTP 410 reports that the model is unavailable and suggests an explicit
`--model` override. The runner never silently switches models within an experiment.

## Verification

```bash
python -m pytest -q
```

Tests cover configuration, credential routing, request fidelity, client cleanup,
exclusion of reference answers, malformed/multiple calls, trace persistence,
offline previews and errors. CI requires no API credential. Fabricated responses
are used in tests; the documentation example is a recorded API response.

If an editable installation fails to import `atb` on macOS, check whether the
editable-install `.pth` file is marked hidden: Python skips hidden `.pth` files.
A regular installation, `python -m pip install ".[dev,llm]"`, avoids this mechanism;
reinstall after changing package source when using that installation mode.

Protocol references: [NVIDIA function calling](https://docs.nvidia.com/nim/large-language-models/latest/function-calling.html)
and the [official Python SDK](https://github.com/openai/openai-python).
