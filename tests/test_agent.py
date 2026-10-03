"""Agent protocol checks with synthetic model responses and real numerical tools."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("openai", reason="Install the llm extra to test agent orchestration")
pytest.importorskip("dotenv", reason="Install the llm extra to test configuration")

from openai import APIConnectionError
from openai.types.chat import ChatCompletion

from atb import agent
from atb.client import ClientSettings
from atb.scoring import score
from atb.tasks import Task, load_tasks


@pytest.fixture
def settings():
    return ClientSettings("https://example.org/v1", "test-model", "test-secret")


@pytest.fixture
def task():
    return next(task for task in load_tasks(Path(__file__).resolve().parents[1] / "benchmark/tasks.jsonl") if task.id == "multi-001")


def call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def completion(calls=None, content=None, finish=None, usage=True):
    return ChatCompletion.model_validate({
        "id": "test-completion", "created": 1, "model": "test-model", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish or ("tool_calls" if calls else "stop"),
                     "message": {"role": "assistant", "content": content, "tool_calls": calls}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15} if usage else None,
    })


def run(task, settings, tmp_path, **kwargs):
    return agent.run_agent(task, "test-model", "C", settings=settings, results_dir=tmp_path, run_id="test-run", **kwargs)


def test_dependent_calls_replay_protocol_and_real_oracles(monkeypatch, task, settings, tmp_path):
    requests = []
    def fake_chat(**kwargs):
        messages = kwargs["messages"]
        requests.append(deepcopy(kwargs))
        if len(requests) == 1:
            return completion([call("hohmann_transfer", '{"initial_altitude_km":621.863,"final_altitude_km":2621.863}', "transfer")])
        if len(requests) == 2:
            assert messages[-2]["tool_calls"][0]["id"] == "transfer"
            assert messages[-1]["role"] == "tool"
            assert messages[-1]["tool_call_id"] == "transfer"
            transfer = json.loads(messages[-1]["content"])
            return completion([call("propagate_orbit", json.dumps({
                "r0_km": task.params["r0"], "v0_km_s": task.params["v0"], "dt_s": transfer["transfer_time_s"],
            }), "propagation")])
        assert messages[-2]["tool_calls"][0]["id"] == "propagation"
        assert messages[-1]["tool_call_id"] == "propagation"
        position = json.loads(messages[-1]["content"])["r_final_km"]
        return completion(content=json.dumps({"answer": {"r_final_x_km": position[0]}}))
    monkeypatch.setattr(agent, "chat", fake_chat)
    trace = run(task, settings, tmp_path)
    assert trace["status"] == "completed"
    assert trace["step_count"] == 3 and trace["tool_call_count"] == 2
    assert [item["step"] for item in trace["tool_calls"]] == [1, 2]
    assert score(task, trace["final_answer"])["correct"]
    assert trace["tokens"] == {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45, "usage_complete": True}
    assert json.loads(Path(trace["trace_path"]).read_text()) == trace
    for step, request in zip(trace["steps"], requests):
        assert step["request"]["messages"] == request["messages"]
        assert step["request"]["tools"] == request["tools"]
    assert trace["latency_s"] >= 0
    assert "test-secret" not in json.dumps(trace)


def test_reasoning_budget_is_repeated_and_persisted_for_every_turn(monkeypatch, task, settings, tmp_path):
    requests = []
    def fake_chat(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return completion([call("orbital_period", '{"altitude_km":400}')])
        return completion(content='{"answer":{"period_s":5553.624}}')
    monkeypatch.setattr(agent, "chat", fake_chat)
    trace = run(task, settings, tmp_path, reasoning_budget=256)
    assert trace["status"] == "completed"
    assert trace["metadata"]["reasoning_budget"] == 256
    assert all(request["reasoning_budget"] == 256 for request in requests)
    assert all(step["request"]["reasoning_budget"] == 256 for step in trace["steps"])
    assert json.loads(Path(trace["trace_path"]).read_text()) == trace


def test_multiple_calls_include_error_results_and_allow_recovery(monkeypatch, task, settings, tmp_path):
    attempts = []
    def fake_chat(**kwargs):
        attempts.append(deepcopy(kwargs))
        if len(attempts) == 1:
            return completion([
                call("orbital_period", '{"altitude_km":400}', "valid"),
                call("orbital_period", '{bad json', "invalid"),
                call("unknown_tool", '{}', "unknown"),
            ])
        if len(attempts) == 2:
            messages = kwargs["messages"]
            assert [item["tool_call_id"] for item in messages[-3:]] == ["valid", "invalid", "unknown"]
            assert all(item["role"] == "tool" for item in messages[-3:])
            assert "error" in json.loads(messages[-1]["content"])
            assert "error" in json.loads(messages[-2]["content"])
            assert messages[-4]["role"] == "assistant" and len(messages[-4]["tool_calls"]) == 3
            return completion([call("orbital_period", '{"altitude_km":800}', "corrected")])
        return completion(content='{"answer":{"period_s":6052.41}}')
    monkeypatch.setattr(agent, "chat", fake_chat)
    trace = run(task, settings, tmp_path)
    assert trace["status"] == "completed"
    assert trace["tool_call_count"] == 4
    assert [item["error_type"] for item in trace["tool_calls"]] == [None, "invalid_arguments", "unknown_tool", None]


@pytest.mark.parametrize("arguments", ["[]", "null", '{"altitude_km":NaN}', '{"altitude_km":1e999}'])
def test_invalid_arguments_return_tool_error_without_dispatch(monkeypatch, task, settings, tmp_path, arguments):
    responses = iter([completion([call("orbital_period", arguments)]), completion(content='{"refuse":true,"reason":"Invalid input"}')])
    monkeypatch.setattr(agent, "chat", lambda **kwargs: next(responses))
    monkeypatch.setattr(agent, "execute_tool", lambda *args: pytest.fail("Invalid input reached numerical execution"))
    trace = run(task, settings, tmp_path)
    assert trace["status"] == "completed"
    assert trace["tool_calls"][0]["error_type"] == "invalid_arguments"
    assert trace["messages"][3]["tool_call_id"] == "call_1"


def test_condition_a_has_no_tools_and_no_solution_leak(monkeypatch, settings, tmp_path):
    task = Task(id="no-tools", family="period", level="simple", prompt="Calculate the period at 400 km altitude.",
                reference={"sentinel": 999999991}, tolerance={"sentinel": 888888881}, params={"hidden": 777777771})
    def fake_chat(**kwargs):
        assert kwargs["tools"] is None
        assert kwargs["messages"][1]["content"] == task.prompt
        return completion(content='{"answer":{"period_s":5553.62}}')
    monkeypatch.setattr(agent, "chat", fake_chat)
    monkeypatch.setattr(agent, "execute_tool", lambda *args: pytest.fail("Condition A executed a tool"))
    trace = agent.run_agent(task, "test-model", "A", settings=settings, results_dir=tmp_path)
    assert trace["tools"] is None and trace["tool_call_count"] == 0
    assert "tools" not in trace["steps"][0]["request"]
    assert "tool_choice" not in trace["steps"][0]["request"]
    assert all(value not in json.dumps(trace) for value in ("999999991", "888888881", "777777771"))


def test_condition_b_dispatches_raw_arguments(monkeypatch, task, settings, tmp_path):
    responses = iter([completion([call("orbital_period", '{"a":7000}')]), completion(content='{"answer":{"period_s":5828.52}}')])
    monkeypatch.setattr(agent, "chat", lambda **kwargs: next(responses))
    trace = agent.run_agent(task, "test-model", "B", settings=settings, results_dir=tmp_path)
    assert trace["tools"] == agent.TOOLS_B
    assert json.loads(trace["tool_calls"][0]["result"])["period"] == pytest.approx(5828.52, abs=.01)


def test_step_budget_retains_tool_result_without_inventing_final(monkeypatch, task, settings, tmp_path):
    attempts = []
    def fake_chat(**kwargs):
        attempts.append(kwargs)
        return completion([call("orbital_period", '{"altitude_km":400}', f"call_{len(attempts)}")])
    monkeypatch.setattr(agent, "chat", fake_chat)
    trace = run(task, settings, tmp_path, max_steps=2)
    assert len(attempts) == 2
    assert trace["status"] == "max_steps_reached"
    assert trace["final_answer"] is None
    assert trace["messages"][-1]["role"] == "tool"
    assert trace["step_count"] == 2
    assert json.loads(Path(trace["trace_path"]).read_text())["status"] == "max_steps_reached"


@pytest.mark.parametrize("bad_calls", [
    [call("orbital_period", '{}', "")],
    [call("orbital_period", '{}', "same"), call("orbital_period", '{}', "same")],
])
def test_bad_call_ids_stop_before_any_execution(monkeypatch, task, settings, tmp_path, bad_calls):
    monkeypatch.setattr(agent, "chat", lambda **kwargs: completion(bad_calls))
    monkeypatch.setattr(agent, "execute_tool", lambda *args: pytest.fail("Malformed batch reached dispatch"))
    trace = run(task, settings, tmp_path)
    assert trace["status"] == "protocol_error"
    assert trace["tool_call_count"] == 0
    assert trace["step_count"] == 1


def test_unexpected_tool_call_in_a_is_never_executed(monkeypatch, task, settings, tmp_path):
    monkeypatch.setattr(agent, "chat", lambda **kwargs: completion([call("orbital_period", '{"altitude_km":400}')]))
    monkeypatch.setattr(agent, "execute_tool", lambda *args: pytest.fail("Unexpected tool in A executed"))
    trace = agent.run_agent(task, "test-model", "A", settings=settings, results_dir=tmp_path)
    assert trace["status"] == "protocol_error" and not trace["tool_calls"]


def test_missing_usage_is_unknown(monkeypatch, task, settings, tmp_path):
    monkeypatch.setattr(agent, "chat", lambda **kwargs: completion(content='{"answer":{}}', usage=False))
    trace = run(task, settings, tmp_path)
    assert trace["tokens"]["total_tokens"] is None
    assert trace["tokens"]["usage_complete"] is False


def test_api_failure_preserves_prior_messages_and_redacts_error(monkeypatch, task, settings, tmp_path):
    attempts = []
    def fake_chat(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return completion([call("orbital_period", '{"altitude_km":400}')])
        raise APIConnectionError(message="test-secret", request=None)
    monkeypatch.setattr(agent, "chat", fake_chat)
    trace = run(task, settings, tmp_path)
    assert trace["status"] == "api_error"
    assert trace["step_count"] == 2 and trace["tool_call_count"] == 1
    assert trace["tokens"]["total_tokens"] is None
    assert trace["steps"][1]["response"] is None
    assert "test-secret" not in json.dumps(trace)
    assert Path(trace["trace_path"]).is_file()


def test_truncated_tool_request_is_not_executed(monkeypatch, task, settings, tmp_path):
    monkeypatch.setattr(agent, "chat", lambda **kwargs: completion([call("orbital_period", '{"altitude_km":')], finish="length"))
    monkeypatch.setattr(agent, "execute_tool", lambda *args: pytest.fail("Truncated tool request executed"))
    trace = run(task, settings, tmp_path)
    assert trace["status"] == "truncated" and not trace["tool_calls"]
    assert trace["final_answer"] is None


def test_empty_choice_list_returns_protocol_error(monkeypatch, task, settings, tmp_path):
    response = completion(content="unused")
    response.choices = []
    monkeypatch.setattr(agent, "chat", lambda **kwargs: response)
    assert run(task, settings, tmp_path)["status"] == "protocol_error"


def test_nonfinite_tool_result_becomes_error(monkeypatch, task, settings, tmp_path):
    responses = iter([completion([call("orbital_period", '{"altitude_km":400}')]), completion(content='{"answer":{}}')])
    monkeypatch.setattr(agent, "chat", lambda **kwargs: next(responses))
    monkeypatch.setattr(agent, "execute_tool", lambda *args: '{"period_s":NaN}')
    trace = run(task, settings, tmp_path)
    assert trace["tool_calls"][0]["error_type"] == "execution_error"
    assert "NaN" not in trace["tool_calls"][0]["result"]


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"max_steps": True}, {"max_steps": 1.5}, {"run_id": "../escape"}])
def test_invalid_settings_fail_before_network(monkeypatch, task, settings, tmp_path, kwargs):
    monkeypatch.setattr(agent, "chat", lambda **kwargs: pytest.fail("Invalid settings reached API"))
    with pytest.raises(ValueError):
        agent.run_agent(task, "test-model", "C", settings=settings, results_dir=tmp_path, **kwargs)


def test_existing_trace_is_never_overwritten(monkeypatch, task, settings, tmp_path):
    monkeypatch.setattr(agent, "chat", lambda **kwargs: completion(content='{"answer":{}}'))
    trace = run(task, settings, tmp_path)
    original = Path(trace["trace_path"]).read_text()
    monkeypatch.setattr(agent, "chat", lambda **kwargs: pytest.fail("Existing trace reached API"))
    with pytest.raises(FileExistsError):
        run(task, settings, tmp_path)
    assert Path(trace["trace_path"]).read_text() == original


def test_provider_extras_stay_in_raw_response_not_outbound_messages(monkeypatch, task, settings, tmp_path):
    tool = call("orbital_period", '{"altitude_km":400}')
    tool["vendor_index"] = 7
    tool["function"]["vendor_hint"] = "diagnostic"
    first = completion([tool])
    first.choices[0].message.reasoning_content = "Provider-specific analysis."
    attempts = []
    def fake_chat(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return first
        message = kwargs["messages"][-2]
        assert set(message) == {"role", "content", "tool_calls"}
        assert set(message["tool_calls"][0]) == {"id", "type", "function"}
        assert set(message["tool_calls"][0]["function"]) == {"name", "arguments"}
        return completion(content='{"answer":{}}')
    monkeypatch.setattr(agent, "chat", fake_chat)
    trace = run(task, settings, tmp_path)
    raw_message = trace["steps"][0]["response"]["choices"][0]["message"]
    assert raw_message["reasoning_content"] == "Provider-specific analysis."
    assert raw_message["tool_calls"][0]["vendor_index"] == 7
    assert trace["status"] == "completed"
