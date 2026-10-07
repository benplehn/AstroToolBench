# Model backends

The experimental code talks to `ModelBackend.generate(messages, tools=None)`.
Messages, tool definitions and responses live in `astrotoolbench.models.base`;
they contain no provider SDK objects.

| Type | Contents |
| --- | --- |
| `Message` | Role, text, optional assistant tool calls or tool result ID |
| `ToolDefinition` | Name, description, JSON Schema for parameters |
| `ToolCall` | Call ID, function name, arguments as a JSON string |
| `ModelResponse` | Returned model ID, text, calls, finish reason, refusal, usage |
| `TokenUsage` | Input and output tokens; `None` when not reported |

One call produces one assistant turn. It does not run any scientific function.
Tool arguments remain strings, including malformed JSON: the execution layer
will need to record and handle that error. Empty or duplicate call IDs are
rejected because a tool result could not be matched reliably to its request.

`response.as_message()` carries the assistant turn back into the conversation.
For each executed call, add a `Message(role="tool", content=..., tool_call_id=...)`
with the matching ID before asking for the next turn.

## OpenAI-compatible APIs

`OpenAIBackend` adapts the existing `atb.client` transport, so endpoint/key
selection, timeouts and client cleanup stay in one place. It works with NVIDIA
and other endpoints implementing the same Chat Completions protocol.

Install `.[llm]` and supply the endpoint, exact model ID and matching key through
the existing configuration. This example makes a real API call when run:

```python
from atb.client import ClientSettings
from astrotoolbench.models import Message, OpenAIBackend

backend = OpenAIBackend(ClientSettings.from_env(".env"))
response = backend.generate([Message(role="user", content="Hello.")])
print(response.content)
```

The adapter defaults to temperature 0, 4096 output tokens and the transport's
default of no automatic retries. These can be configured explicitly. NVIDIA's
optional `reasoning_budget` stays in this adapter rather than the common
interface. With `tools=None` or an empty sequence, the request omits both
`tools` and `tool_choice`.

SDK failures become `BackendError`; HTTP status codes are available through
`status_code`. Malformed responses become `BackendProtocolError`. The exception
message omits provider bodies; the original exception is retained as its cause.
Truncation and refusal are returned as response fields, not successful answers.

## Another provider or a local model

Implement `generate` in a subclass of `ModelBackend`, translate the common
messages and tool definitions into that model's input, then return a
`ModelResponse`. A local backend follows the same contract. SDK imports and
provider-specific options belong in the adapter.

Importing `astrotoolbench.models` does not import OpenAI, load credentials or
contact a provider. Only the API adapter needs the `llm` extra. The tests use
fake responses; no model evaluation has been run through this new interface yet.
