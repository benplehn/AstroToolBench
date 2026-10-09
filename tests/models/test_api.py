"""Offline adapter checks using real SDK response types and a fake transport."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

pytest.importorskip("openai", reason="Install the llm extra to test the API adapter")
pytest.importorskip("dotenv", reason="Install the llm extra to test the API adapter")

import atb.client as transport
from atb.client import ClientSettings
from openai import APIStatusError, APITimeoutError
from openai.types.chat import ChatCompletion

from astrotoolbench.models import (
    BackendError, BackendProtocolError, Message, ModelResponse, OpenAIBackend, ToolDefinition,
)


@pytest.fixture
def completion():
    return {
        "id": "completion-1", "object": "chat.completion", "created": 0, "model": "returned-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "5828.5", "refusal": None,
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
    }


@pytest.fixture
def fake_client(monkeypatch, completion):
    fake = MagicMock()
    fake.chat.completions.create.return_value = ChatCompletion.model_validate(completion)
    monkeypatch.setattr(transport, "OpenAI", MagicMock(return_value=fake))
    return fake


def backend(endpoint="https://example.org/v1", **options):
    return OpenAIBackend(ClientSettings(endpoint, "requested-model", "test-key"), **options)


@pytest.mark.parametrize("tools", [None, []])
def test_text_response_no_tools_and_client_closed(fake_client, tools):
    response = backend().generate([Message(role="user", content="Compute a period.")], tools)
    assert isinstance(response, ModelResponse)
    assert response.content == "5828.5"
    assert response.model == "returned-model"
    assert response.usage.input_tokens == 20
    assert response.usage.output_tokens == 4
    assert response.finish_reason == "stop"
    fake_client.chat.completions.create.assert_called_once_with(
        model="requested-model", messages=[{"role": "user", "content": "Compute a period."}],
        temperature=0.0, max_tokens=4096, stream=False,
    )
    fake_client.__exit__.assert_called_once()


def test_parallel_calls_round_trip_and_malformed_json_preserved(fake_client, completion):
    raw = deepcopy(completion)
    raw["choices"][0].update(finish_reason="tool_calls")
    raw["choices"][0]["message"].update(content=None, tool_calls=[
        {"id": "a", "type": "function", "function": {"name": "period", "arguments": '{"radius":7000}'}},
        {"id": "b", "type": "function", "function": {"name": "period", "arguments": "{bad JSON"}},
    ])
    fake_client.chat.completions.create.return_value = ChatCompletion.model_validate(raw)
    tools = [ToolDefinition(name="period", description="Orbital period.", parameters={"type": "object"})]
    history = [Message(role="user", content="Compute two periods.")]
    model = backend()
    response = model.generate(history, tools)
    assert [call.id for call in response.tool_calls] == ["a", "b"]
    assert response.tool_calls[1].arguments == "{bad JSON"
    history.extend([response.as_message(), Message(role="tool", content="5828.5", tool_call_id="a"),
                    Message(role="tool", content="Invalid JSON", tool_call_id="b")])
    model.generate(history, tools)
    request = fake_client.chat.completions.create.call_args.kwargs
    assert request["tools"] == [{"type": "function", "function": tools[0].model_dump()}]
    assert request["messages"][1] == {"role": "assistant", "content": None,
                                      "tool_calls": raw["choices"][0]["message"]["tool_calls"]}
    assert request["messages"][-1] == {"role": "tool", "content": "Invalid JSON", "tool_call_id": "b"}


def test_missing_usage_and_refusal_preserved(fake_client, completion):
    completion.pop("usage")
    completion["choices"][0]["message"].update(content=None, refusal="Cannot answer.")
    fake_client.chat.completions.create.return_value = ChatCompletion.model_validate(completion)
    response = backend().generate([Message(role="user", content="hello")])
    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.refusal == "Cannot answer."


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_non_success_finish_reason_preserved(fake_client, completion, reason):
    completion["choices"][0]["finish_reason"] = reason
    fake_client.chat.completions.create.return_value = ChatCompletion.model_validate(completion)
    assert backend().generate([Message(role="user", content="hello")]).finish_reason == reason


@pytest.mark.parametrize("failure", ["no-choice", "multiple-choices", "duplicate-id", "missing-id", "negative-usage"])
def test_invalid_response_raises_common_protocol_error(fake_client, completion, failure):
    if failure == "no-choice":
        completion["choices"] = []
    elif failure == "multiple-choices":
        completion["choices"] *= 2
    elif failure in {"duplicate-id", "missing-id"}:
        call = {"id": "same" if failure == "duplicate-id" else "", "type": "function",
                "function": {"name": "period", "arguments": "{}"}}
        completion["choices"][0]["message"]["tool_calls"] = [call, call] if failure == "duplicate-id" else [call]
    else:
        completion["usage"]["prompt_tokens"] = -1
    fake_client.chat.completions.create.return_value = ChatCompletion.model_validate(completion)
    with pytest.raises(BackendProtocolError):
        backend().generate([Message(role="user", content="hello")])


def test_protocol_error_keeps_original_calls_for_trace_audit(fake_client, completion):
    completion["choices"][0]["message"]["tool_calls"] = [
        {"id": "", "type": "function", "function": {"name": "period", "arguments": "{bad JSON"}},
    ]
    fake_client.chat.completions.create.return_value = ChatCompletion.model_validate(completion)
    with pytest.raises(BackendProtocolError) as caught:
        backend().generate([Message(role="user", content="hello")])
    raw = caught.value.raw_response["choices"][0]["message"]["tool_calls"][0]
    assert raw["id"] == ""
    assert raw["function"]["arguments"] == "{bad JSON"


def test_status_error_is_normalized_without_provider_body(fake_client):
    response = MagicMock(status_code=429)
    error = APIStatusError("private-provider-detail", response=response, body={"secret": "private"})
    fake_client.chat.completions.create.side_effect = error
    with pytest.raises(BackendError) as caught:
        backend().generate([Message(role="user", content="hello")])
    assert caught.value.status_code == 429
    assert "429" in str(caught.value)
    assert "private" not in str(caught.value)
    fake_client.__exit__.assert_called_once()


def test_timeout_is_normalized_without_retry(fake_client):
    fake_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
    with pytest.raises(BackendError, match="timed out"):
        backend().generate([Message(role="user", content="hello")])
    fake_client.chat.completions.create.assert_called_once()
    assert transport.OpenAI.call_args.kwargs["max_retries"] == 0


def test_nvidia_reasoning_option_stays_in_adapter(fake_client):
    backend("https://integrate.api.nvidia.com/v1", reasoning_budget=256).generate(
        [Message(role="user", content="hello")],
    )
    assert fake_client.chat.completions.create.call_args.kwargs["extra_body"] == {"reasoning_budget": 256}
    assert transport.OpenAI.call_args.kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"


def test_empty_history_rejected_before_network(fake_client):
    with pytest.raises(ValueError, match="at least one message"):
        backend().generate([])
    transport.OpenAI.assert_not_called()


@pytest.mark.parametrize("options", [{"temperature": float("nan")}, {"max_tokens": 0}, {"reasoning_budget": True}])
def test_invalid_settings_rejected_before_network(fake_client, options):
    with pytest.raises(ValueError):
        backend(**options)
    transport.OpenAI.assert_not_called()
