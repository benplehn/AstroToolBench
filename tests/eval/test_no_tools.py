"""Baseline A never exposes references or executes requested tools."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from astrotoolbench.eval import evaluate_task, run_no_tools
from astrotoolbench.eval.no_tools import select_tasks
from astrotoolbench.eval.prompts import build_messages
from astrotoolbench.models import BackendError, ModelResponse, TokenUsage, ToolCall
from astrotoolbench.models.catalog import get_profile

ROOT = Path(__file__).resolve().parents[2]


def correct_answer(task):
    if task.expected.kind == "error":
        return {"error": task.expected.code, "reason": "Invalid problem."}
    return {"outputs": {name: output.value for name, output in task.expected.outputs.items()}}


def test_prompt_uses_public_data_only_and_does_not_change_with_reference(corpus):
    for task in corpus:
        messages = build_messages(task)
        assert len(messages) == 2
        assert messages[1].content.startswith(task.prompt)
        data = json.loads(messages[1].content.split("\n\nProblem data:\n", 1)[1])
        assert set(data) == {"context", "inputs"}
        assert data["context"] == task.context.model_dump(mode="json")
        altered = deepcopy(task)
        altered.notes = "PRIVATE ANSWER LEAK"
        altered.verification.rationale = "PRIVATE VERIFICATION LEAK"
        assert build_messages(altered) == messages
        if task.expected.kind == "success":
            for output in altered.expected.outputs.values():
                output.value = not output.value if output.kind == "boolean" else 12345.678
            assert build_messages(altered) == messages
        else:
            altered.expected.reason = "PRIVATE ERROR LEAK"
            assert build_messages(altered) == messages


def test_every_corpus_outcome_uses_existing_scientific_comparison(corpus):
    for task in corpus:
        backend = MagicMock()
        backend.generate.return_value = ModelResponse(
            model="fake", content=json.dumps(correct_answer(task)), finish_reason="stop",
            usage=TokenUsage(input_tokens=20, output_tokens=5),
        )
        result = evaluate_task(task, backend, model="requested-model")
        assert result["correct"] is True
        assert result["condition"] == "no_tools"
        assert result["status"] == "completed"
        assert result["latency_ms"] >= 0
        assert (result["input_tokens"], result["output_tokens"]) == (20, 5)
        backend.generate.assert_called_once_with(build_messages(task), tools=None)


@pytest.mark.parametrize("content", ["No idea.", "[]", "null", 'Result: {"x": 1}',
                                   '{"outputs":{"x":NaN}}', '{"outputs":{},"outputs":{}}'])
def test_invalid_final_json_is_saved_as_failure(corpus, content):
    backend = MagicMock()
    backend.generate.return_value = ModelResponse(model="fake", content=content, finish_reason="stop")
    result = evaluate_task(corpus[0], backend, model="fake")
    assert result["status"] == "invalid_answer"
    assert result["answer"] == content
    assert result["correct"] is False
    assert result["format_ok"] is False


@pytest.mark.parametrize("wrap", ["```json\n{}\n```", "The answer is {} as requested.",
                                  'Draft: {"x": 1}\nFinal:\n{}'])
def test_wrapped_answer_is_scored_but_flagged_as_bad_format(corpus, wrap):
    task = corpus[0]
    backend = MagicMock()
    content = wrap.replace("{}", json.dumps(correct_answer(task)))
    backend.generate.return_value = ModelResponse(model="fake", content=content, finish_reason="stop")
    result = evaluate_task(task, backend, model="fake")
    assert result["status"] == "completed"
    assert result["correct"] is True
    assert result["format_ok"] is False


def test_last_answer_object_wins(corpus):
    task = corpus[0]
    backend = MagicMock()
    wrong = {"error": "invalid_input", "reason": "first guess"}
    content = json.dumps(wrong) + "\nActually:\n" + json.dumps(correct_answer(task))
    backend.generate.return_value = ModelResponse(model="fake", content=content, finish_reason="stop")
    assert evaluate_task(task, backend, model="fake")["correct"] is True


@pytest.mark.parametrize("options,status", [
    ({"content": "", "finish_reason": "stop"}, "empty_response"),
    ({"content": "{}", "finish_reason": "length"}, "truncated"),
    ({"refusal": "Cannot answer.", "finish_reason": "stop"}, "refused"),
    ({"content": "{}", "finish_reason": "content_filter"}, "refused"),
    ({"content": "{}", "finish_reason": None}, "incomplete_response"),
    ({"tool_calls": (ToolCall(id="1", name="period", arguments="{}"),), "finish_reason": "tool_calls"},
     "unexpected_tool_call"),
])
def test_incomplete_or_unexpected_turns_are_not_scored(corpus, options, status):
    backend = MagicMock()
    backend.generate.return_value = ModelResponse(model="fake", **options)
    result = evaluate_task(corpus[0], backend, model="fake")
    assert result["status"] == status
    assert result["correct"] is False
    assert result["response"] is not None


def test_plausible_json_with_wrong_answer_is_a_completed_failure(corpus):
    backend = MagicMock()
    backend.generate.return_value = ModelResponse(model="fake", content='{"outputs":{}}', finish_reason="stop")
    result = evaluate_task(corpus[0], backend, model="fake")
    assert result["status"] == "completed"
    assert result["correct"] is False


def test_incremental_results_include_api_failure_and_continue(corpus, tmp_path):
    backend = MagicMock()
    backend.generate.side_effect = [BackendError("Timed out."), *[
        ModelResponse(model="fake", content=json.dumps(correct_answer(task)), finish_reason="stop")
        for task in corpus[1:]
    ]]
    path = tmp_path / "baseline.jsonl"
    results = run_no_tools(backend, get_profile("nemotron"), path, root=ROOT)
    stored = [json.loads(line) for line in path.read_text().splitlines()]
    assert stored == results
    assert len(stored) == 50
    assert stored[0]["status"] == "backend_error"
    assert stored[0]["correct"] is False
    assert stored[0]["input_tokens"] is None
    assert all(record["correct"] for record in stored[1:])
    assert stored[0]["model_profile"]["name"] == "nemotron"


def test_existing_results_are_not_overwritten_or_called(corpus, tmp_path):
    backend = MagicMock()
    path = tmp_path / "existing.jsonl"
    path.write_text("original evidence\n")
    with pytest.raises(FileExistsError):
        run_no_tools(backend, get_profile("nemotron"), path, root=ROOT)
    assert path.read_text() == "original evidence\n"
    backend.generate.assert_not_called()


def test_unexpected_programming_errors_do_not_become_model_failures(corpus, tmp_path):
    backend = MagicMock()
    backend.generate.side_effect = [ModelResponse(model="fake", content="{}", finish_reason="stop"), TypeError("bug")]
    output = tmp_path / "partial.jsonl"
    with pytest.raises(TypeError, match="bug"):
        run_no_tools(backend, get_profile("nemotron"), output, root=ROOT)
    assert len(output.read_text().splitlines()) == 1


def test_selection_is_nonempty_and_uses_frozen_loader(corpus):
    assert select_tasks(ROOT, split="test") == [task for task in corpus if task.split == "test"]
    assert select_tasks(ROOT, task_id=corpus[0].id) == [corpus[0]]
    with pytest.raises(ValueError, match="No benchmark tasks"):
        select_tasks(ROOT, task_id="unknown")
