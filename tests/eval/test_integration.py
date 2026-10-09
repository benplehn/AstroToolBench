"""The real request adapter and baseline runner agree on condition A."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("openai")
pytest.importorskip("dotenv")

import atb.client as transport
from openai.types.chat import ChatCompletion

from astrotoolbench.eval.__main__ import main

ROOT = Path(__file__).resolve().parents[2]


def test_baseline_cli_through_the_api_adapter(corpus, monkeypatch, tmp_path):
    task = next(task for task in corpus if task.expected.kind == "success")
    answer = {"outputs": {name: output.value for name, output in task.expected.outputs.items()}}
    fake = MagicMock()
    fake.chat.completions.create.return_value = ChatCompletion.model_validate({
        "id": "completion-1", "object": "chat.completion", "created": 0,
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(answer),
        }}],
        "usage": {"prompt_tokens": 230, "completion_tokens": 45, "total_tokens": 275},
    })
    monkeypatch.setattr(transport, "OpenAI", MagicMock(return_value=fake))
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret-key")
    file = tmp_path / "no_tools.jsonl"
    assert main(["--model", "nemotron", "--root", str(ROOT), "--task", task.id,
                 "--output", str(file), "--env-file", str(tmp_path / ".env")]) == 0
    record = json.loads(file.read_text())
    request = fake.chat.completions.create.call_args.kwargs
    assert "tools" not in request
    assert "tool_choice" not in request
    assert request["temperature"] == 0.0
    assert record["correct"] is True
    assert record["input_tokens"] == 230
    assert record["output_tokens"] == 45
    assert "test-secret-key" not in file.read_text()
    fake.__exit__.assert_called_once()


def test_raw_baseline_executes_a_real_scientific_function_between_sdk_turns(corpus, monkeypatch, tmp_path):
    task = next(task for task in corpus if task.id == "dv-vector-turn")
    call = {"id": "dv-1", "type": "function", "function": {"name": "delta_v", "arguments": json.dumps({
        "v_initial": task.inputs["v_initial"].value, "v_final": task.inputs["v_final"].value,
    })}}
    fake = MagicMock()

    def completion(**request):
        assert "tools" in request and len(request["tools"]) == 10
        if request["messages"][-1]["role"] == "user":
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
            finish = "tool_calls"
        else:
            assert request["messages"][-1]["tool_call_id"] == "dv-1"
            value = json.loads(request["messages"][-1]["content"])["result"]
            message = {"role": "assistant", "content": json.dumps({"outputs": {"delta_v": value}})}
            finish = "stop"
        return ChatCompletion.model_validate({
            "id": "completion", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        })

    fake.chat.completions.create.side_effect = completion
    monkeypatch.setattr(transport, "OpenAI", MagicMock(return_value=fake))
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret-key")
    output = tmp_path / "raw_tools.jsonl"
    assert main(["--model", "nemotron", "--condition", "raw_tools", "--root", str(ROOT),
                 "--task", task.id, "--output", str(output)]) == 0
    record = json.loads(output.read_text())
    assert record["correct"] is True
    assert record["tool_calls"][0]["success"] is True
    assert record["tool_calls"][0]["arguments_json"] == call["function"]["arguments"]
    assert len(record["steps"]) == 2
    assert "test-secret-key" not in output.read_text()
    assert fake.__exit__.call_count == 2
