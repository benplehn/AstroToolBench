"""The baseline command has an offline preview and explicit result paths."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from astrotoolbench.eval import __main__ as command
from astrotoolbench.models import BackendError, ModelResponse

ROOT = Path(__file__).resolve().parents[2]


def test_list_models_needs_no_configuration(monkeypatch, capsys):
    constructor = MagicMock()
    monkeypatch.setattr(command, "create_backend", constructor)
    assert command.main(["--list-models"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 4
    constructor.assert_not_called()


def test_dry_run_contains_requests_but_no_model_load_or_results(monkeypatch, capsys, tmp_path):
    constructor = MagicMock()
    monkeypatch.setattr(command, "create_backend", constructor)
    output = tmp_path / "results.jsonl"
    assert command.main(["--model", "qwen", "--root", str(ROOT), "--split", "test",
                         "--dry-run", "--output", str(output)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["requests"]) == 20
    assert data["generation"]["do_sample"] is False
    assert data["model_profile"]["revision"]
    assert not output.exists()
    constructor.assert_not_called()


def test_one_task_runs_and_saves_even_a_wrong_answer(monkeypatch, capsys, tmp_path, corpus):
    backend = MagicMock()
    backend.generate.return_value = ModelResponse(model="actual-model", content="{}", finish_reason="stop")
    monkeypatch.setattr(command, "create_backend", MagicMock(return_value=backend))
    output = tmp_path / "result.jsonl"
    assert command.main(["--model", "nemotron", "--root", str(ROOT), "--task", corpus[0].id,
                         "--output", str(output)]) == 0
    record = json.loads(output.read_text())
    assert record["correct"] is False
    assert record["response"]["model"] == "actual-model"
    assert record["generation"]["temperature"] == 0
    assert "Saved 1 tasks" in capsys.readouterr().out


@pytest.mark.parametrize("options", [["--max-tokens", "0"], ["--timeout", "nan"], ["--condition", "raw_tools"]])
def test_invalid_settings_or_unimplemented_condition_rejected_before_backend(monkeypatch, options):
    constructor = MagicMock()
    monkeypatch.setattr(command, "create_backend", constructor)
    with pytest.raises(SystemExit) as caught:
        command.main(["--model", "nemotron", *options])
    assert caught.value.code == 2
    constructor.assert_not_called()


def test_existing_result_rejected_before_model_creation(monkeypatch, tmp_path):
    constructor = MagicMock()
    monkeypatch.setattr(command, "create_backend", constructor)
    output = tmp_path / "existing.jsonl"
    output.write_text("old run\n")
    assert command.main(["--model", "qwen", "--root", str(ROOT), "--output", str(output)]) == 1
    assert output.read_text() == "old run\n"
    constructor.assert_not_called()


def test_backend_failure_is_persisted_with_nonzero_exit(monkeypatch, tmp_path, corpus):
    backend = MagicMock()
    backend.generate.side_effect = BackendError("Timed out.")
    monkeypatch.setattr(command, "create_backend", MagicMock(return_value=backend))
    output = tmp_path / "failed.jsonl"
    assert command.main(["--model", "nemotron", "--root", str(ROOT), "--task", corpus[0].id,
                         "--output", str(output)]) == 1
    assert json.loads(output.read_text())["status"] == "backend_error"
