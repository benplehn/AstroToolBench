"""Offline CLI and trace tests; no provider credential or network is needed."""

import json
from pathlib import Path

import pytest

pytest.importorskip("openai", reason="Install the llm extra to test completion inspection")
pytest.importorskip("dotenv", reason="Install the llm extra to test configuration")

from openai import APIConnectionError
from openai.types.chat import ChatCompletion

from atb import inspection
from atb.tasks import Task

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inspector(monkeypatch):
    for key in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(ROOT)
    return inspection


@pytest.fixture
def response():
    return ChatCompletion.model_validate({
        "id": "test-completion", "created": 1, "model": "test-model", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "orbital_period", "arguments": '{"altitude_km":400}'}},
                {"id": "call_2", "type": "function", "function": {"name": "orbital_period", "arguments": '{"altitude_km":800}'}},
            ],
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    })


def test_prompt_contains_no_ground_truth(inspector):
    task = Task(id="secret-task", family="period", level="simple", prompt="A public question.",
                reference={"period_s": 987654321}, params={"secret": 12345}, tolerance={"period_s": 45678})
    messages = inspector.build_messages(task)
    assert messages[1]["content"] == task.prompt
    assert all(secret not in json.dumps(messages) for secret in ("987654321", "12345", "45678", "secret-task"))


def test_multiple_calls_and_usage_are_preserved(inspector, response):
    raw = response.model_dump(mode="json")
    observations = inspector.inspect_response(raw)
    choice = observations["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["tool_call_count"] == 2
    assert choice["tool_calls"][0]["arguments_type"] == "str"
    assert choice["tool_calls"][0]["arguments_parsed"] == {"altitude_km": 400}
    assert observations["usage"]["total_tokens"] == 30
    assert observations["tools_executed"] is False
    assert raw == response.model_dump(mode="json")


@pytest.mark.parametrize("arguments", ['{"broken":', "[]", {"altitude_km": 400}])
def test_malformed_arguments_still_inspectable(inspector, response, arguments):
    raw = response.model_dump(mode="json")
    raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    call = inspector.inspect_response(raw)["choices"][0]["tool_calls"][0]
    assert call["arguments_error"]
    assert call["arguments_raw"] == arguments


def test_missing_usage_empty_choices_and_text_answer(inspector):
    assert inspector.inspect_response({"choices": []})["usage"] is None
    result = inspector.inspect_response({"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "hello"}}]})
    assert result["choices"][0]["tool_call_count"] == 0


def test_dry_run_needs_no_key_or_network(inspector, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(inspector, "chat", lambda **kwargs: pytest.fail("Dry run contacted the provider"))
    output = tmp_path / "trace.json"
    assert inspector.main(["--dry-run", "--env-file", str(tmp_path / "absent.env"), "--output", str(output)]) == 0
    text = capsys.readouterr().out
    assert '"tools"' in text and '"altitude_km"' in text
    assert "5553.62" not in text
    assert not output.exists()


def test_live_path_with_fake_response_saves_exact_trace(inspector, monkeypatch, tmp_path, capsys, response):
    monkeypatch.setenv("LLM_API_KEY", "test-secret-never-print")
    requests = []
    def fake_chat(**kwargs):
        requests.append(kwargs)
        return response
    monkeypatch.setattr(inspector, "chat", fake_chat)
    output = tmp_path / "trace.json"
    assert inspector.main(["--env-file", str(tmp_path / "absent.env"), "--output", str(output), "--require-tool-call"]) == 0
    trace = json.loads(output.read_text())
    assert len(requests) == 1
    assert trace["request"]["messages"] == requests[0]["messages"]
    assert trace["request"]["tools"] == requests[0]["tools"]
    assert trace["response"] == response.model_dump(mode="json")
    assert trace["metadata"]["task_id"] == "period-001"
    assert trace["metadata"]["max_retries"] == 0
    assert len(trace["metadata"]["benchmark_sha256"]) == 64
    assert "test-secret-never-print" not in output.read_text() + capsys.readouterr().out
    assert "reference" not in trace
    assert "tolerance" not in trace


def test_existing_output_rejected_before_api_call(inspector, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setattr(inspector, "chat", lambda **kwargs: pytest.fail("Should fail before a call"))
    output = tmp_path / "trace.json"
    output.write_text("keep me")
    assert inspector.main(["--env-file", str(tmp_path / "absent.env"), "--output", str(output)]) == 1
    assert output.read_text() == "keep me"
    assert "Output already exists" in capsys.readouterr().err


def test_no_tool_call_fails_required_check_but_keeps_raw_trace(inspector, monkeypatch, tmp_path, response):
    monkeypatch.setenv("LLM_API_KEY", "key")
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    monkeypatch.setattr(inspector, "chat", lambda **kwargs: response)
    output = tmp_path / "trace.json"
    assert inspector.main(["--env-file", str(tmp_path / "absent.env"), "--output", str(output), "--require-tool-call"]) == 1
    assert json.loads(output.read_text())["response"]["choices"][0]["finish_reason"] == "stop"


def test_connection_error_is_concise_and_redacted(inspector, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    def fail(**kwargs):
        raise APIConnectionError(message="test-secret", request=None)
    monkeypatch.setattr(inspector, "chat", fail)
    assert inspector.main(["--env-file", str(tmp_path / "absent.env"), "--output", str(tmp_path / "trace.json")]) == 1
    captured = capsys.readouterr()
    assert "Connection failed" in captured.err
    assert "test-secret" not in captured.err + captured.out
