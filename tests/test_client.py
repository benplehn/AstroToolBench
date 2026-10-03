"""Offline checks for provider routing, request fidelity and client ownership."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("openai", reason="Install the llm extra to test the API client")
pytest.importorskip("dotenv", reason="Install the llm extra to test configuration")

import atb.client as client_module
from openai import APIStatusError
from atb.client import ClientSettings, build_chat_request, chat


@pytest.fixture(autouse=True)
def clean_llm_environment(monkeypatch):
    for key in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("url,key", [
    ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    ("https://api.openai.com/v1", "OPENAI_API_KEY"),
])
def test_key_routing(url, key, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", url)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    for name in ("NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, f"test-{name}")
    settings = ClientSettings.from_env()
    assert settings.api_key == f"test-{key}"
    assert settings.api_key not in repr(settings)


def test_explicit_file_and_environment_precedence_without_mutation(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=file-model\nNVIDIA_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    monkeypatch.setenv("LLM_API_KEY", "override-key")
    before = dict(os.environ)
    settings = ClientSettings.from_env(env)
    assert settings.model == "environment-model"
    assert settings.api_key == "override-key"
    assert ClientSettings.from_env(env, model="cli-model").model == "cli-model"
    assert dict(os.environ) == before


def test_custom_endpoint_never_borrows_provider_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "local-model")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
    with pytest.raises(ValueError, match="Missing API key"):
        ClientSettings.from_env()
    assert ClientSettings.from_env(require_api_key=False).api_key == ""


@pytest.mark.parametrize("url", ["not-a-url", "https://user:secret@example.org/v1", "https://example.org/v1?key=secret"])
def test_invalid_endpoint_rejected(url):
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        ClientSettings(url, "model", "key")


def test_chat_sends_exact_body_and_closes_client(monkeypatch):
    messages = [{"role": "user", "content": "Compute a period."}]
    tools = [{"type": "function", "function": {"name": "period"}}]
    response = SimpleNamespace(choices=[])
    fake = MagicMock()
    fake.chat.completions.create.return_value = response
    constructor = MagicMock(return_value=fake)
    monkeypatch.setattr(client_module, "OpenAI", constructor)
    settings = ClientSettings("https://example.org/v1", "default-model", "test-key", timeout_s=12)
    actual = chat(messages, tools, model="override-model", settings=settings, max_tokens=123, tool_choice="required")
    assert actual is response
    fake.chat.completions.create.assert_called_once_with(**build_chat_request(
        "override-model", messages, tools, max_tokens=123, tool_choice="required",
    ))
    constructor.assert_called_once_with(base_url=settings.base_url, api_key="test-key", timeout=12, max_retries=0)
    fake.__exit__.assert_called_once()


def test_without_tools_omits_both_tool_fields():
    request = build_chat_request("model", [{"role": "user", "content": "hello"}])
    assert "tools" not in request and "tool_choice" not in request
    with pytest.raises(ValueError, match="needs at least one tool"):
        build_chat_request("model", request["messages"], tool_choice="required")


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_tokens": True}, {"temperature": float("nan")}, {"tool_choice": "invalid"},
                                   {"reasoning_budget": True}, {"reasoning_budget": -2}, {"reasoning_budget": 32769}])
def test_invalid_request_rejected_before_network(kwargs):
    with pytest.raises(ValueError):
        build_chat_request("model", [{"role": "user", "content": "hello"}], **kwargs)


def test_retired_model_error_suggests_explicit_replacement_without_exposing_body():
    response = SimpleNamespace(status_code=410, request=None, headers={})
    error = APIStatusError("provider-private-detail", response=response, body={"detail": "secret-account-data"})
    message = client_module.api_error_message(error)
    assert "410" in message
    assert "--model" in message
    assert "no longer available" in message
    assert "secret-account-data" not in message
    assert "provider-private-detail" not in message


def test_reasoning_budget_is_recorded_as_wire_field_and_sent_through_sdk_extension(monkeypatch):
    messages = [{"role": "user", "content": "Compute a period."}]
    fake = MagicMock()
    monkeypatch.setattr(client_module, "OpenAI", MagicMock(return_value=fake))
    settings = ClientSettings("https://example.org/v1", "test-model", "test-key")
    wire = build_chat_request(settings.model, messages, reasoning_budget=256)
    assert wire["reasoning_budget"] == 256
    chat(messages, settings=settings, reasoning_budget=256)
    sdk_request = {key: value for key, value in wire.items() if key != "reasoning_budget"}
    fake.chat.completions.create.assert_called_once_with(**sdk_request, extra_body={"reasoning_budget": 256})
    fake.__exit__.assert_called_once()
