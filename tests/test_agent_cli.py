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
    result = agent_cli.main(["--task", "period-001", "--api", "B", "--max-steps", "4", "--json"], project_root=root)
    assert result == expected_exit
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == status
    assert output["score"] == (None if status != "completed" else {
        "correct": True, "error_type": None, "message": "Réponse conforme à la vérité terrain.",
    })
    assert requests[0][1:4] == ("test-model", "B", 4)
    assert "test-key" not in json.dumps(output)


@pytest.fixture
def recorded_run(monkeypatch):
    """Exercise the command with a real, stored multi-turn conversation offline."""
    root = Path(__file__).resolve().parents[1]
    trace = json.loads((root / "docs/examples/nvidia-nemotron-agent.json").read_text())
    monkeypatch.setattr(agent_cli.ClientSettings, "from_env", lambda *args, **kwargs: ClientSettings(
        "https://example.org/v1", "test-model", "test-key",
    ))
    monkeypatch.setattr(agent_cli, "run_agent", lambda *args, **kwargs: trace)
    return root, trace


def test_readable_conversation_and_grade(recorded_run, capsys):
    root, trace = recorded_run
    assert agent_cli.main(["--task", "multi-001", "--api", "C"], project_root=root) == 0
    output = capsys.readouterr().out
    ordered = ("Turn 1", "Call: hohmann_transfer", '"transfer_time_s": 3560.540788789012',
               "Turn 2", "Call: propagate_orbit", '"dt_s": 3560.540788789012', "Turn 3", "Score: PASS (1/1)")
    positions = [output.index(part) for part in ordered]
    assert positions == sorted(positions)
    assert "Tokens: prompt: 5056; completion: 1054; total: 6110" in output
    assert trace["trace_path"] in output
    assert "System" not in output
    assert trace["tool_calls"][0]["id"] not in output
    assert "test-key" not in output


def test_verbose_explains_call_correlation_and_request_usage(recorded_run, capsys):
    root, trace = recorded_run
    assert agent_cli.main(["--task", "multi-001", "--verbose"], project_root=root) == 0
    output = capsys.readouterr().out
    assert trace["messages"][0]["content"] in output
    assert trace["messages"][1]["content"] in output
    for call in trace["tool_calls"]:
        assert output.count(f"[{call['id']}]") == 2
    assert "Model requests\n" in output
    assert "finish reason: tool_calls" in output
    assert "finish reason: stop" in output
    assert "test-key" not in output


@pytest.mark.parametrize("status", ["max_steps_reached", "api_error", "protocol_error"])
def test_incomplete_trace_is_readable_and_ungraded(recorded_run, capsys, status):
    root, trace = recorded_run
    trace.update(status=status, final_answer=None)
    trace["messages"] = trace["messages"][:-1]
    trace["tokens"] = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    if status != "max_steps_reached":
        trace["error"] = {"type": status, "message": "Sanitized failure"}
    if status == "protocol_error":
        trace["messages"].append({"role": "assistant", "content": None, "tool_calls": [
            {"id": None, "type": "function", "function": {"name": "orbital_period", "arguments": None}},
        ]})
    assert agent_cli.main(["--task", "multi-001", "--verbose"], project_root=root) == 1
    output = capsys.readouterr().out
    assert f"Status: {status}" in output
    assert "Score: NOT EVALUATED" in output
    assert "Final answer: unavailable" in output
    assert "total: unknown" in output
    if trace["error"]:
        assert "Error: Sanitized failure" in output


def test_wrong_completed_answer_has_failing_grade(recorded_run, capsys):
    root, trace = recorded_run
    trace["final_answer"] = '{"answer":{"r_final_x_km":0}}'
    trace["messages"][-1]["content"] = trace["final_answer"]
    assert agent_cli.main(["--task", "multi-001"], project_root=root) == 1
    output = capsys.readouterr().out
    assert "Status: completed" in output
    assert "Score: FAIL (0/1)" in output
    assert "hors tolérance" in output


def test_tool_error_and_invalid_json_remain_visible(recorded_run, capsys):
    root, trace = recorded_run
    trace["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{invalid json"
    trace["messages"][3]["content"] = '{"error":"Invalid JSON"}'
    trace["tool_calls"][0]["error_type"] = "invalid_arguments"
    assert agent_cli.main(["--task", "multi-001", "--verbose"], project_root=root) == 0
    output = capsys.readouterr().out
    assert "{invalid json" in output
    assert '"error": "Invalid JSON"' in output
    assert f"Tool error [{trace['tool_calls'][0]['id']}]: invalid_arguments" in output


def test_parallel_calls_to_same_tool_have_distinct_result_ids(recorded_run, capsys):
    root, trace = recorded_run
    trace["messages"] = trace["messages"][:2] + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "function": {"name": "orbital_period", "arguments": json.dumps({"altitude_km": altitude})}}
            for call_id, altitude in (("call-low", 400), ("call-high", 800))
        ]},
        {"role": "tool", "tool_call_id": "call-low", "content": '{"period_s":5553.62}'},
        {"role": "tool", "tool_call_id": "call-high", "content": '{"period_s":6052.41}'},
        {"role": "assistant", "content": trace["final_answer"]},
    ]
    trace["tool_calls"] = []
    assert agent_cli.main(["--task", "multi-001", "--verbose"], project_root=root) == 0
    output = capsys.readouterr().out
    for call_id in ("call-low", "call-high"):
        assert f"Call: orbital_period [{call_id}]" in output
        assert f"Tool result [{call_id}]" in output
    assert "Turn 2 · Assistant" in output
    assert "Turn 3 · Assistant" not in output


def test_condition_a_displays_answer_without_tool_calls(recorded_run, capsys):
    root, trace = recorded_run
    trace["messages"] = trace["messages"][:2] + [trace["messages"][-1]]
    trace["tool_calls"] = []
    trace["tool_call_count"] = 0
    assert agent_cli.main(["--task", "multi-001", "--api", "A"], project_root=root) == 0
    output = capsys.readouterr().out
    assert "Condition: A" in output
    assert "Call:" not in output
    assert "Tool calls: 0" in output
    assert "Score: PASS" in output


def test_verbose_and_json_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as caught:
        agent_cli.main(["--task", "period-001", "--json", "--verbose"])
    assert caught.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
