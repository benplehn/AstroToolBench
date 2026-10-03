"""Synchronous transport for OpenAI-compatible Chat Completions APIs.

Importing this module neither loads credentials nor contacts a provider. Numerical
tools are executed separately by ``atb.executor``, never by this transport.
"""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from dotenv import dotenv_values
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletion

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
ToolChoice = Literal["auto", "required", "none"]


@dataclass(frozen=True)
class ClientSettings:
    """Connection settings; the credential is excluded from representations."""

    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_s: float = 60.0
    max_retries: int = 0

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"https", "http"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
        ):
            raise ValueError("LLM_BASE_URL must be an HTTP(S) URL without credentials, query or fragment.")
        if not self.model.strip():
            raise ValueError("Set LLM_MODEL to the provider's exact model ID.")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("Timeout must be a finite, positive number of seconds.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        *,
        model: str | None = None,
        timeout_s: float = 60.0,
        require_api_key: bool = True,
    ) -> "ClientSettings":
        """Read an explicit .env file, with process variables taking precedence.

        No process environment is mutated. Provider credentials are used only for
        their matching endpoint; custom endpoints require LLM_API_KEY.
        ``require_api_key=False`` is reserved for inspecting a request offline.
        """
        values = dict(dotenv_values(env_file)) if env_file is not None else {}
        values.update(os.environ)
        base_url = (values.get("LLM_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
        host = urlsplit(base_url).hostname
        provider_key = {
            "integrate.api.nvidia.com": "NVIDIA_API_KEY",
            "openrouter.ai": "OPENROUTER_API_KEY",
            "api.openai.com": "OPENAI_API_KEY",
        }.get(host)
        api_key = (values.get("LLM_API_KEY") or "").strip()
        if not api_key and provider_key:
            api_key = (values.get(provider_key) or "").strip()
        target_model = model or values.get("LLM_MODEL") or (
            DEFAULT_MODEL if host == "integrate.api.nvidia.com" else ""
        )
        settings = cls(base_url, target_model.strip(), api_key, timeout_s)
        if require_api_key and not api_key:
            hint = f"LLM_API_KEY or {provider_key}" if provider_key else "LLM_API_KEY"
            raise ValueError(f"Missing API key. Set {hint} for this endpoint.")
        return settings


def build_chat_request(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    tool_choice: ToolChoice = "auto",
    reasoning_budget: int | None = None,
) -> dict[str, Any]:
    """Build the exact request body used for inspection and transmission."""
    if not model.strip() or not messages:
        raise ValueError("A model and at least one message are required.")
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")
    if tool_choice not in {"auto", "required", "none"}:
        raise ValueError("tool_choice must be auto, required or none.")
    if not tools and tool_choice == "required":
        raise ValueError("tool_choice='required' needs at least one tool.")
    if reasoning_budget is not None and (
        isinstance(reasoning_budget, bool)
        or not isinstance(reasoning_budget, int)
        or not -1 <= reasoning_budget <= 32768
    ):
        raise ValueError("reasoning_budget must be an integer from -1 to 32768.")
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        request.update(tools=tools, tool_choice=tool_choice)
    if reasoning_budget is not None:
        request["reasoning_budget"] = reasoning_budget
    return request


def get_client(settings: ClientSettings | None = None) -> tuple[OpenAI, str]:
    """Create a client. The caller owns it and must close it after use."""
    settings = settings or ClientSettings.from_env()
    if not settings.api_key:
        raise ValueError("An API key is required to contact the provider.")
    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
    ), settings.model


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    *,
    settings: ClientSettings | None = None,
    max_tokens: int = 4096,
    tool_choice: ToolChoice = "auto",
    reasoning_budget: int | None = None,
) -> ChatCompletion:
    """Send one completion request and return the unmodified SDK response.

    Automatic retries are disabled by default so one observed call corresponds
    to one attempt. Supply settings explicitly when reading a project .env file.
    """
    settings = settings or ClientSettings.from_env(model=model)
    request = build_chat_request(
        model or settings.model, messages, tools,
        temperature=temperature, max_tokens=max_tokens, tool_choice=tool_choice,
        reasoning_budget=reasoning_budget,
    )
    client, _ = get_client(settings)
    with client:
        if reasoning_budget is not None:
            # NVIDIA accepts this extension in the JSON body; the SDK uses extra_body.
            extra_body = {"reasoning_budget": request.pop("reasoning_budget")}
            return client.chat.completions.create(**request, extra_body=extra_body)
        return client.chat.completions.create(**request)


def api_error_message(error: APIError) -> str:
    """Keep provider bodies and credentials out of terminal error messages."""
    if isinstance(error, APITimeoutError):
        return "Request timed out. Increase --timeout or try again later. No automatic retry was made."
    if isinstance(error, APIConnectionError):
        return "Connection failed. Check the endpoint and network connection."
    if isinstance(error, APIStatusError):
        hints = {
            400: "Check the model's support for tools and the request parameters.",
            401: "Check the API key for the configured endpoint.",
            403: "Check account access to this model.",
            404: "Check LLM_BASE_URL and the exact model ID; use scripts/list_models.py.",
            410: "This model is no longer available. Choose another model with --model; use scripts/list_models.py.",
            429: "Rate limit or quota reached. Try again later or check the account quota.",
        }
        return f"Provider returned HTTP {error.status_code}. " + hints.get(
            error.status_code, "Check provider availability."
        )
    return "The provider returned an API error."
