"""Provider-independent conversation and completion types."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"\S")]
TokenCount = Annotated[int, Field(ge=0)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ToolDefinition(_Model):
    name: Identifier
    description: str
    parameters: dict[str, JsonValue]


class ToolCall(_Model):
    id: Identifier
    name: Identifier
    # Keep malformed JSON so the execution layer can report the model's mistake.
    arguments: str


def _check_call_ids(calls: tuple[ToolCall, ...]) -> None:
    if len({call.id for call in calls}) != len(calls):
        raise ValueError("Tool call IDs must be unique within a response.")


class Message(_Model):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: Identifier | None = None

    @model_validator(mode="after")
    def check_role(self) -> "Message":
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("Only assistant messages can request tools.")
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("A tool_call_id belongs to a tool result and is required there.")
        if self.role != "assistant" and self.content is None:
            raise ValueError("System, user and tool messages need content.")
        _check_call_ids(self.tool_calls)
        return self


class TokenUsage(_Model):
    # None means the provider did not report the count; zero is an actual count.
    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None


class ModelResponse(_Model):
    model: Identifier
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    refusal: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)

    @model_validator(mode="after")
    def check_calls(self) -> "ModelResponse":
        _check_call_ids(self.tool_calls)
        return self

    def as_message(self) -> Message:
        return Message(role="assistant", content=self.content, tool_calls=self.tool_calls)


class BackendError(RuntimeError):
    """A generation failure, independent of a provider's exception classes."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class BackendProtocolError(BackendError):
    """The provider returned a response that cannot represent a valid turn."""


class ModelBackend(ABC):
    @abstractmethod
    def generate(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate one assistant turn without executing any scientific tool."""
