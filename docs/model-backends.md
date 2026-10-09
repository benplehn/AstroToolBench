# Model backends

The evaluation code only talks to `ModelBackend.generate(messages, tools=None)`.
The types live in `astrotoolbench.models.base`, with no provider SDK objects, so
the benchmark doesn't depend on any particular API.

| Type | Contents |
| --- | --- |
| `Message` | role, text, assistant tool calls or tool result ID |
| `ToolDefinition` | name, description, JSON Schema of the parameters |
| `ToolCall` | call ID, function name, arguments as a JSON string |
| `ModelResponse` | model ID, text, calls, finish reason, refusal, usage, raw text |
| `TokenUsage` | input/output tokens, `None` if not reported |

One call = one assistant turn. It never runs a tool. Arguments stay as strings,
even broken JSON, so the execution layer can record the mistake. Empty or
duplicate call IDs are rejected, since the results couldn't be matched to the
right call.

To continue a conversation: add `response.as_message()`, then one
`Message(role="tool", content=..., tool_call_id=...)` per executed call, then
call `generate` again.

## OpenAI-compatible APIs

`OpenAIBackend` reuses the `atb.client` transport (endpoint, key selection,
timeouts, cleanup), so it works with NVIDIA, OpenRouter and anything else that
speaks Chat Completions.

Install `.[llm]` and configure the endpoint, model and key. This makes a real
API call:

```python
from atb.client import ClientSettings
from astrotoolbench.models import Message, OpenAIBackend

backend = OpenAIBackend(ClientSettings.from_env(".env"))
response = backend.generate([Message(role="user", content="Hello.")])
print(response.content)
```

Defaults: temperature 0, 4096 output tokens, no automatic retries. NVIDIA's
`reasoning_budget` stays in this adapter, not in the common interface. With no
tools, `tools` and `tool_choice` aren't sent at all.

SDK errors become `BackendError` (with `status_code`), malformed responses
become `BackendProtocolError`. The message never includes the provider's error
body; the original exception is kept as the cause, and for protocol errors the
raw JSON can be kept in `raw_response`. Truncation and refusals come back as
normal response fields.

## Adding a backend

Subclass `ModelBackend`, convert the messages and tools into what the model
expects, and return a `ModelResponse`. SDK imports and provider options stay in
the adapter.

Importing `astrotoolbench.models` doesn't import OpenAI, Torch or Transformers,
read keys or call anything.

`HuggingFaceBackend` runs a pinned local checkpoint with greedy decoding
(`.[local]`). Weights are loaded once when the backend is created, before any
task timer. For Qwen, tools and tool results go through its own chat template,
and the `<tool_call>` blocks it generates are parsed into `ToolCall`s. The raw
text stays in `raw_content`. If a block can't be parsed, the response comes
back with `finish_reason="invalid_tool_call"` instead of raising, since that's
the model's mistake and should be counted as such.

Models: [baseline models](baseline-models.md). Experiments:
[baseline A](no-tools-baseline.md), [baseline B](raw-tools-baseline.md).
