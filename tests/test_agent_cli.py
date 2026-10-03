"""The command grades only completed runs and reports incomplete outcomes."""

import json
from pathlib import Path

import pytest

pytest.importorskip("openai", reason="Install the llm extra to test the agent command")
pytest.importorskip("dotenv", reason="Install the llm extra to test configuration")

from atb import agent_cli
from atb.client import ClientSettings


@pytest.mark.parametrize("status,expected_exit", [("completed", 0), ("max_steps_reached", 1)])
def test_command_reports_run_and_scores_completed_answer(monkeypatch, tmp_path, capsys, status, expected_exit):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(agent_cli.ClientSettings, "from_env", lambda *args, **kwargs: ClientSettings(
        "https://example.org/v1", "test-model", "test-key",
    ))
    requests = []
    def fake_agent(task, model, api, max_steps, **kwargs):
        requests.append((task, model, api, max_steps, kwargs))
        return {
            "status": status, "step_count": 2, "tool_call_count": 1,
            "tokens": {"total_tokens": 30}, "latency_s": 1.0,
            "final_answer": '{"answer":{"period_s":5553.62}}' if status == "completed" else None,
            "error": None, "trace_path": str(tmp_path / "trace.json"),
        }
    monkeypatch.setattr(agent_cli, "run_agent", fake_agent)
    result = agent_cli.main(["--task", "period-001", "--api", "B", "--max-steps", "4"], project_root=root)
    assert result == expected_exit
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == status
    assert output["score"] == (None if status != "completed" else {
        "correct": True, "error_type": None, "message": "Réponse conforme à la vérité terrain.",
    })
    assert requests[0][1:4] == ("test-model", "B", 4)
    assert "test-key" not in json.dumps(output)
