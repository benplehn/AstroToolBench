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