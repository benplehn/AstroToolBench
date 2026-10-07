"""Adapter for OpenAI-compatible Chat Completions, including NVIDIA endpoints."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .base import (
    BackendError, BackendProtocolError, Message, ModelBackend, ModelResponse,
    TokenUsage, ToolCall, ToolDefinition,
)

if TYPE_CHECKING:
    from atb.client import ClientSettings


def _transport():
    try:
        from atb import client
    except ImportError as error:
        raise ImportError('Install the API dependencies with pip install ".[llm]".') from error
    return client


def _message_payload(message: Message) -> dict:
    payload = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {"id": call.id, "type": "function",
             "function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    return payload


def _normalize(response) -> ModelResponse:
    try:
        if len(response.choices) != 1:
            raise ValueError("Expected one completion choice.")
        choice = response.choices[0]
        message = choice.message
        if message.role != "assistant":
            raise ValueError("Expected an assistant response.")
        calls = []
        for call in message.tool_calls or ():
            if call.type != "function":
                raise ValueError("Unsupported tool call type.")
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments))
        usage = response.usage
        return ModelResponse(
            model=response.model, content=message.content, tool_calls=tuple(calls),
            finish_reason=choice.finish_reason, refusal=message.refusal,
            usage=TokenUsage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens)
            if usage is not None else TokenUsage(),
        )
    except (AttributeError, TypeError, ValueError) as error:
        # Validation details may contain provider text; keep them out of the error.
        raise BackendProtocolError("The provider returned an invalid completion response.") from error


class OpenAIBackend(ModelBackend):
    """Use the existing transport, returning only the common model types."""

    def __init__(
        self, settings: "ClientSettings", *, temperature: float = 0.0,
        max_tokens: int = 4096, reasoning_budget: int | None = None,
    ):
        transport = _transport()
        transport.build_chat_request(
            settings.model, [{"role": "user", "content": ""}],
            temperature=temperature, max_tokens=max_tokens, reasoning_budget=reasoning_budget,
        )
        self.settings = settings
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_budget = reasoning_budget

    def generate(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] | None = None,
    ) -> ModelResponse:
        transport = _transport()
        payload = [_message_payload(message) for message in messages]
        definitions = [
            {"type": "function", "function": tool.model_dump(mode="json")}
            for tool in tools
        ] if tools else None
        try:
            response = transport.chat(
                payload, definitions, settings=self.settings,
                temperature=self.temperature, max_tokens=self.max_tokens,
                reasoning_budget=self.reasoning_budget,
            )
        except transport.APIError as error:
            raise BackendError(
                transport.api_error_message(error), status_code=getattr(error, "status_code", None),
            ) from error
        return _normalize(response)
