"""Trace adaptation and tokenizer-backed checks of scientific tool serialization."""

from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from atb import template_inspection as inspection
from atb.tools import TOOLS_B, TOOLS_C


ROOT = Path(__file__).resolve().parents[1]
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


@pytest.fixture
def trace():
    return json.loads((ROOT / "docs/examples/nvidia-nemotron-agent.json").read_text())


@pytest.fixture
def tokenizer():
    pytest.importorskip("transformers", reason="Install the tokenizer extra for rendering checks")
    pytest.importorskip("jinja2", reason="Install the tokenizer extra for rendering checks")
    try:
        return inspection.load_tokenizer(revision=REVISION, local_files_only=True)
    except OSError:
        pytest.skip("Pinned Qwen tokenizer is not cached; tests never download assets")


def test_api_arguments_become_objects_without_mutating_trace(trace):
    original = deepcopy(trace)
    messages = inspection.prepare_messages(trace["messages"])
    assert trace == original
    assert messages[2]["content"] == ""
    first = messages[2]["tool_calls"][0]
    second = messages[4]["tool_calls"][0]
    assert first["function"]["arguments"] == {"initial_altitude_km": 621.863, "final_altitude_km": 2621.863}
    assert isinstance(second["function"]["arguments"], dict)
    first_result = json.loads(messages[3]["content"])
    assert second["function"]["arguments"]["dt_s"] == first_result["transfer_time_s"]
    assert first["id"] == messages[3]["tool_call_id"]
    # Calling the adapter on an already prepared conversation is also safe.
    assert inspection.prepare_messages(messages) == messages


@pytest.mark.parametrize("arguments", ['{broken', '[]', 'null', '{"dt_s":NaN}', '{"dt_s":1e999}'])
def test_unrenderable_arguments_raise_clear_errors(trace, arguments):
    trace["messages"][2]["tool_calls"][0]["function"]["arguments"] = arguments
    with pytest.raises(ValueError):
        inspection.prepare_messages(trace["messages"])


@pytest.mark.parametrize("messages", [None, [], [{"role": "other", "content": "text"}], [{"role": "user", "content": None}]])
def test_invalid_message_structure_is_rejected(messages):
    with pytest.raises(ValueError):
        inspection.prepare_messages(messages)


def test_description_counterfactual_keeps_interface_and_constraints():
    original = deepcopy(TOOLS_C)
    stripped = inspection._without_descriptions(TOOLS_C)
    assert TOOLS_C == original
    assert '"description"' not in json.dumps(stripped)
    assert stripped[0]["function"]["name"] == "orbital_period"
    assert stripped[0]["function"]["parameters"]["required"] == ["altitude_km"]
    assert stripped[0]["function"]["parameters"]["properties"]["altitude_km"]["minimum"] == 0.0
    assert stripped[2]["function"]["parameters"]["properties"]["v0_km_s"]["minItems"] == 3


def test_real_template_embeds_objects_results_and_completed_answer(tokenizer, trace):
    rendered = inspection.render_chat(tokenizer, trace["messages"], tools=trace["tools"], add_generation_prompt=False)
    blocks = re.findall(r"<tool_call>\n(.*?)\n</tool_call>", rendered, re.DOTALL)
    actual_calls = []
    for block in blocks:
        try:
            actual_calls.append(json.loads(block))
        except ValueError:
            pass  # The system message also contains a non-JSON format example.
    assert [call["name"] for call in actual_calls] == ["hohmann_transfer", "propagate_orbit"]
    assert all(isinstance(call["arguments"], dict) for call in actual_calls)
    assert actual_calls[1]["arguments"]["dt_s"] == 3560.540788789012
    for message in trace["messages"]:
        if message["role"] == "tool":
            assert f"<tool_response>\n{message['content']}\n</tool_response>" in rendered
    assert rendered.endswith(trace["final_answer"] + "<|im_end|>\n")
    assert not rendered.endswith("<|im_start|>assistant\n")


def test_token_ids_match_direct_template_tokenization(tokenizer, trace):
    prepared = inspection.prepare_messages(trace["messages"])
    rendered = inspection.render_chat(tokenizer, prepared, tools=trace["tools"], add_generation_prompt=False)
    direct = tokenizer.apply_chat_template(prepared, tools=trace["tools"], tokenize=True, add_generation_prompt=False)
    assert tokenizer.encode(rendered, add_special_tokens=False) == direct
    assert inspection.count_tokens(tokenizer, rendered) == len(direct)


def test_comparison_uses_initial_prompt_and_isolates_descriptions(tokenizer, trace):
    comparison = inspection.compare_tools(tokenizer, trace["messages"])
    assert comparison["messages"] == trace["messages"][:2]
    for condition, tools in (("B", TOOLS_B), ("C", TOOLS_C)):
        counts = comparison["conditions"][condition]
        assert counts["prompt_tokens"] > comparison["baseline_prompt_tokens"]
        assert counts["tool_overhead_tokens"] == counts["prompt_tokens"] - comparison["baseline_prompt_tokens"]
        assert counts["description_fields_cost_tokens"] > 0
        assert counts["prompt_tokens_without_descriptions"] < counts["prompt_tokens"]
        # Definitions present in the template must retain the full interface.
        text = inspection.render_chat(tokenizer, comparison["messages"], tools=tools)
        block = text.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
        assert [json.loads(line) for line in block.splitlines()] == tools
    assert comparison["C_minus_B_prompt_tokens"] > 0


def test_command_renders_trace_and_persists_reproducible_report(tokenizer, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(inspection, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    output, report = tmp_path / "chat.txt", tmp_path / "report.json"
    arguments = ["--trace", str(ROOT / "docs/examples/nvidia-nemotron-agent.json"), "--compare-tools",
                 "--revision", REVISION, "--output", str(output), "--report", str(report)]
    assert inspection.main(arguments, project_root=tmp_path) == 0
    measurements = json.loads(report.read_text())
    assert measurements["tool_call_count"] == 2
    assert measurements["message_count"] == 7
    assert measurements["add_generation_prompt"] is False
    assert measurements["revision"] == REVISION
    assert measurements["rendered_tokens"] == inspection.count_tokens(tokenizer, output.read_text())
    assert "C minus B prompt tokens:" in capsys.readouterr().out
    before = output.read_bytes(), report.read_bytes()
    assert inspection.main(arguments) == 1
    assert (output.read_bytes(), report.read_bytes()) == before
    assert "Output already exists" in capsys.readouterr().err


def test_command_rejects_non_agent_trace_before_loading_tokenizer(tmp_path, monkeypatch, capsys):
    path = tmp_path / "trace.json"
    path.write_text('{"trace_type":"completion"}')
    monkeypatch.setattr(inspection, "load_tokenizer", lambda *args, **kwargs: pytest.fail("Must validate the source first"))
    assert inspection.main(["--trace", str(path)]) == 1
    assert "agent-run trace" in capsys.readouterr().err
