from pathlib import Path
import pytest
from atb.tasks import Task, load_tasks
from atb.scoring import score, extract_final_json


@pytest.fixture
def sample_numeric_task():
    return Task(
        id="sample-num",
        family="hohmann",
        level="simple",
        prompt="Transfert de test",
        reference={"delta_v_total_km_s": 3.771},
        tolerance={"delta_v_total_km_s": 0.01},
    )


@pytest.fixture
def sample_trap_task():
    return Task(
        id="sample-trap",
        family="trap_subsurface",
        level="trap",
        prompt="Orbite sous la surface",
        reference={"refuse": True},
    )


def test_score_valid_answer_codeblock(sample_numeric_task):
    model_output = (
        "Calcul terminé avec succès.\n"
        "```json\n"
        '{"answer": {"delta_v_total_km_s": 3.775}}\n'
        "```"
    )
    res = score(sample_numeric_task, model_output)
    assert res["correct"] is True
    assert res["error_type"] is None


def test_score_unparseable_response(sample_numeric_task):
    model_output = "La réponse est 3.771 km/s, mais je n'émets aucun JSON."
    res = score(sample_numeric_task, model_output)
    assert res["correct"] is False
    assert res["error_type"] == "réponse illisible"


def test_score_missing_key(sample_numeric_task):
    model_output = '{"answer": {"wrong_key": 3.771}}'
    res = score(sample_numeric_task, model_output)
    assert res["correct"] is False
    assert res["error_type"] == "clé manquante"


def test_score_out_of_tolerance(sample_numeric_task):
    model_output = '{"answer": {"delta_v_total_km_s": 4.100}}'
    res = score(sample_numeric_task, model_output)
    assert res["correct"] is False
    assert res["error_type"] == "hors tolérance"


def test_score_trap_correctly_refused(sample_trap_task):
    model_output = '{"refuse": true, "reason": "Altitude négative impossible"}'
    res = score(sample_trap_task, model_output)
    assert res["correct"] is True
    assert res["error_type"] is None


def test_score_trap_missing_refusal(sample_trap_task):
    model_output = '{"answer": {"period_s": 2500.0}}'
    res = score(sample_trap_task, model_output)
    assert res["correct"] is False
    assert res["error_type"] == "refus manquant"


def test_score_unjustified_refusal(sample_numeric_task):
    model_output = '{"refuse": true, "reason": "Je ne peux pas calculer"}'
    res = score(sample_numeric_task, model_output)
    assert res["correct"] is False
    assert res["error_type"] == "refus à tort"


def test_loader_filters(tmp_path):
    # Teste les filtres split et family du loader
    lines = [
        '{"id": "t1", "family": "f1", "level": "simple", "split": "dev", "prompt": "p1"}',
        '{"id": "t2", "family": "f2", "level": "simple", "split": "dev", "prompt": "p2"}',
        '{"id": "t3", "family": "f1", "level": "simple", "split": "test", "prompt": "p3"}',
    ]
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")

    assert len(load_tasks(p)) == 3
    assert len(load_tasks(p, split="dev")) == 2
    assert len(load_tasks(p, family="f1")) == 2
    assert len(load_tasks(p, split="test", family="f1")) == 1

@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 10**400])
def test_nonfinite_numbers_and_booleans_are_not_numeric_successes(value):
    import json
    task = Task(id="finite-number", family="period", level="simple", prompt="Compute a value.",
                reference={"period_s": 1.0}, tolerance={"period_s": .1})
    result = score(task, json.dumps({"answer": {"period_s": value}}))
    assert result["correct"] is False


def test_string_refusal_is_not_a_valid_refusal():
    task = Task(id="refusal", family="trap", level="trap", prompt="Invalid orbit.", reference={"refuse": True})
    assert score(task, '{"refuse":"false","reason":"invalid"}')["correct"] is False


def test_final_answer_after_fenced_tool_arguments(sample_numeric_task):
    output = '```json\n{"r0_km": [7000, 0, 0], "dt_s": 3560.540788789012}\n```\n'
    output += '{"answer": {"delta_v_total_km_s": 3.771}}'
    assert score(sample_numeric_task, output)["correct"] is True


def test_last_answer_wins_across_markdown_and_plain_json():
    output = '```json\n{"answer": {"value": 1}}\n```\n{"answer": {"value": 2}}'
    assert extract_final_json(output) == {"answer": {"value": 2}}


def test_nested_objects_and_braces_in_strings():
    output = '{"answer": {"nested": {"value": 1}}, "note": "a } brace"}'
    assert extract_final_json(output) == {"answer": {"nested": {"value": 1}}, "note": "a } brace"}


def test_tool_arguments_are_not_final_answers():
    assert extract_final_json('```json\n{"dt_s": 3560.54}\n```') is None
