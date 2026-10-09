"""Correlated multi-turn tool execution and evidence retained on failure."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from astrotoolbench.eval import evaluate_raw_task, run_raw_tools
from astrotoolbench.models import BackendError, BackendProtocolError, ModelResponse, TokenUsage, ToolCall
from astrotoolbench.models.catalog import get_profile

ROOT = Path(__file__).resolve().parents[2]


def turn(*calls, **kwargs):
    return ModelResponse(model="fake", tool_calls=tuple(calls), finish_reason="tool_calls", **kwargs)


def dv_call(id="call-1", arguments=None):
    return ToolCall(id=id, name="delta_v", arguments=arguments or json.dumps(
        {"v_initial": [0, 7.5, 0], "v_final": [0, 8, 0]},
    ))


def test_parallel_calls_then_final_answer_and_complete_history(corpus):
    task = next(task for task in corpus if task.id == "dv-vector-turn")
    backend = MagicMock()
    calls = [ToolCall(id="a", name="delta_v", arguments=json.dumps({
        "v_initial": task.inputs["v_initial"].value, "v_final": task.inputs["v_final"].value,
    })), dv_call("b")]

    def respond(messages, tools):
        assert {tool.name for tool in tools} >= {"delta_v", "propagate_kepler", "hohmann_transfer"}
        if len(messages) == 2:
            return turn(*calls, usage=TokenUsage(input_tokens=20, output_tokens=5))
        assert messages[-3].role == "assistant"
        assert [message.tool_call_id for message in messages[-2:]] == ["a", "b"]
        value = json.loads(messages[-2].content)["result"]
        return ModelResponse(model="fake", content=json.dumps({"outputs": {"delta_v": value}}),
                             finish_reason="stop", usage=TokenUsage(input_tokens=40, output_tokens=7))

    backend.generate.side_effect = respond
    result = evaluate_raw_task(task, backend, model="requested-model")
    assert result["correct"] is True
    assert result["status"] == "completed"
    assert [call["id"] for call in result["tool_calls"]] == ["a", "b"]
    assert all(call["success"] for call in result["tool_calls"])
    assert (result["input_tokens"], result["output_tokens"]) == (60, 12)
    assert result["usage_complete"] is True
    assert len(result["steps"]) == 2


def test_chaining_uses_a_previous_tool_result_not_a_reference(corpus):
    backend = MagicMock()
    responses = [turn(ToolCall(id="transfer", name="hohmann_transfer", arguments='{"r1":7000,"r2":9000}'))]

    def respond(messages, tools):
        if len(messages) == 2:
            return responses[0]
        if len(messages) == 4:
            tof = json.loads(messages[-1].content)["result"]["tof"]
            return turn(ToolCall(id="propagate", name="propagate_kepler", arguments=json.dumps({
                "r0": [7000, 0, 0], "v0": [0, 7.546053290107541, 0], "dt": tof,
            })))
        return ModelResponse(model="fake", content="{}", finish_reason="stop")

    backend.generate.side_effect = respond
    result = evaluate_raw_task(corpus[0], backend, model="fake")
    first, second = result["tool_calls"]
    assert first["result"]["tof"] == second["arguments"]["dt"]
    assert second["success"] is True
    assert len(result["steps"]) == 3


def test_invalid_json_and_unknown_function_are_sent_back_for_recovery(corpus):
    backend = MagicMock()
    backend.generate.side_effect = [
        turn(dv_call(arguments="{bad"), ToolCall(id="unknown", name="invented", arguments="{}")),
        turn(dv_call("fixed")), ModelResponse(model="fake", content="{}", finish_reason="stop"),
    ]
    result = evaluate_raw_task(corpus[0], backend, model="fake")
    assert [call["success"] for call in result["tool_calls"]] == [False, False, True]
    history = backend.generate.call_args_list[1].args[0]
    assert all(json.loads(message.content)["success"] is False for message in history[-2:])
    assert result["status"] == "completed"


@pytest.mark.parametrize("finish", ["stop", None])
def test_tool_calls_with_unusual_finish_reason_are_still_executed(corpus, finish):
    backend = MagicMock()
    backend.generate.side_effect = [ModelResponse(model="fake", tool_calls=(dv_call(),), finish_reason=finish),
                                   ModelResponse(model="fake", content="{}", finish_reason="stop")]
    result = evaluate_raw_task(corpus[0], backend, model="fake")
    assert result["tool_calls"][0]["success"] is True
    assert result["steps"][0]["warnings"] == [f"tool calls with finish_reason={finish!r}"]
    assert result["status"] == "completed"


def test_invalid_local_tool_call_is_a_model_error_not_a_backend_error(corpus):
    backend = MagicMock()
    backend.generate.return_value = ModelResponse(model="fake", content="<tool_call>bad</tool_call>",
                                                  finish_reason="invalid_tool_call")
    result = evaluate_raw_task(corpus[0], backend, model="fake")
    assert result["status"] == "invalid_tool_call"
    assert result["error"] is None
    assert not result["correct"]


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_bad_or_truncated_tool_turn_is_recorded_but_not_executed(corpus, finish):
    backend = MagicMock()
    backend.generate.return_value = ModelResponse(model="fake", tool_calls=(dv_call(),), finish_reason=finish)
    result = evaluate_raw_task(corpus[0], backend, model="fake")
    assert not result["correct"]
    assert result["tool_calls"][0]["status"] == "skipped"
    assert result["tool_calls"][0]["executed"] is False
    backend.generate.assert_called_once()


def test_reused_call_id_rejects_whole_batch_before_execution(corpus):
    backend = MagicMock()
    backend.generate.side_effect = [turn(dv_call()), turn(dv_call("new"), dv_call())]
    result = evaluate_raw_task(corpus[0], backend, model="fake")
    assert result["status"] == "protocol_error"
    assert [call["status"] for call in result["tool_calls"]] == ["completed", "skipped", "skipped"]


def test_turn_and_call_limits_include_the_final_answer_request(corpus):
    backend = MagicMock()
    backend.generate.return_value = turn(dv_call())
    result = evaluate_raw_task(corpus[0], backend, model="fake", max_steps=1)
    assert result["status"] == "max_steps_reached"
    assert result["tool_calls"][0]["success"] is True
    assert not result["correct"]
    backend.generate.return_value = turn(dv_call("a"), dv_call("b"))
    result = evaluate_raw_task(corpus[0], backend, model="fake", max_tool_calls=1)
    assert result["status"] == "max_tool_calls_reached"
    assert not any(call["executed"] for call in result["tool_calls"])


def test_api_failure_after_success_keeps_calls_and_missing_usage(corpus):
    backend = MagicMock()
    backend.generate.side_effect = [turn(dv_call(), usage=TokenUsage(input_tokens=12, output_tokens=5)),
                                   BackendError("Timed out.")]
    result = evaluate_raw_task(corpus[0], backend, model="fake")
    assert result["status"] == "backend_error"
    assert result["tool_calls"][0]["success"] is True
    assert (result["input_tokens"], result["output_tokens"]) == (12, 5)
    assert result["usage_complete"] is False
    assert result["steps"][0]["response"]["usage"]["input_tokens"] == 12


def test_malformed_provider_response_keeps_raw_evidence(corpus):
    backend = MagicMock()
    backend.generate.side_effect = BackendProtocolError("Bad call ID.", raw_response={"tool_calls": [{"id": ""}]})
    result = evaluate_raw_task(corpus[0], backend, model="fake")
    assert result["error"]["raw_response"] == {"tool_calls": [{"id": ""}]}


def test_checkpoint_records_request_before_execution_and_success_after(corpus):
    snapshots = []
    backend = MagicMock()
    backend.generate.side_effect = [turn(dv_call()), ModelResponse(model="fake", content="{}", finish_reason="stop")]
    result = evaluate_raw_task(corpus[0], backend, model="fake", on_update=snapshots.append)
    assert any(snapshot["tool_calls"] and snapshot["tool_calls"][0]["status"] == "pending" for snapshot in snapshots)
    assert any(snapshot["tool_calls"] and snapshot["tool_calls"][0]["status"] == "completed" for snapshot in snapshots)
    assert snapshots[-1] == result
    assert snapshots[0]["steps"] == []


def test_jsonl_and_last_trace_match_and_existing_results_are_preserved(corpus, tmp_path):
    backend = MagicMock()
    backend.generate.side_effect = [turn(dv_call()), ModelResponse(model="fake", content="{}", finish_reason="stop")]
    output = tmp_path / "raw_tools.jsonl"
    results = run_raw_tools(backend, get_profile("nemotron"), output, root=ROOT, task_id=corpus[0].id)
    assert json.loads(output.read_text()) == results[0]
    assert json.loads(Path(results[0]["trace_path"]).read_text()) == results[0]
    with pytest.raises(FileExistsError):
        run_raw_tools(backend, get_profile("nemotron"), output, root=ROOT, task_id=corpus[0].id)


def test_interruption_keeps_executed_calls_even_without_a_final_jsonl_row(corpus, tmp_path):
    backend = MagicMock()
    backend.generate.side_effect = [turn(dv_call()), TypeError("bug")]
    output = tmp_path / "partial.jsonl"
    with pytest.raises(TypeError, match="bug"):
        run_raw_tools(backend, get_profile("nemotron"), output, root=ROOT, task_id=corpus[0].id)
    trace = json.loads((output.with_suffix(".traces") / (corpus[0].id + ".json")).read_text())
    assert output.read_text() == ""
    assert trace["tool_calls"][0]["success"] is True
    assert len(trace["steps"]) == 2
    assert trace["steps"][-1]["response"] is None


@pytest.mark.parametrize("limit", [0, True, 1.5])
def test_invalid_limits_fail_before_any_request(corpus, limit):
    backend = MagicMock()
    with pytest.raises(ValueError):
        evaluate_raw_task(corpus[0], backend, model="fake", max_steps=limit)
    backend.generate.assert_not_called()
