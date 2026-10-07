"""Model interfaces; importing them does not load an API SDK or credentials."""

from .api import OpenAIBackend
from .local import HuggingFaceBackend
from .base import (
    BackendError, BackendProtocolError, Message, ModelBackend, ModelResponse,
    TokenUsage, ToolCall, ToolDefinition,
)

__all__ = [
    "ModelBackend", "Message", "ToolDefinition", "ToolCall", "ModelResponse",
    "TokenUsage", "BackendError", "BackendProtocolError", "OpenAIBackend", "HuggingFaceBackend",
]
