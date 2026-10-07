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
