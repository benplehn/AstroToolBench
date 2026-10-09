"""The local adapter follows the checkpoint's tool-call envelope."""

import json

import pytest

from astrotoolbench.models import BackendProtocolError
from astrotoolbench.models.qwen import parse_tool_calls


def test_parallel_envelopes_and_text_preserve_arguments_and_assign_unique_ids():
    content, calls = parse_tool_calls('I will calculate.\n<tool_call>{"name":"delta_v","arguments":{"v_initial":[0,0,0],'
                                    '"v_final":[1,0,0]}}</tool_call>\n<tool_call>{"name":"orbital_period",'
                                    '"arguments":"{bad JSON"}</tool_call>')
    assert content == "I will calculate."
    assert len(calls) == len({call.id for call in calls}) == 2
    assert json.loads(calls[0].arguments)["v_final"] == [1, 0, 0]
    assert calls[1].arguments == "{bad JSON"


def test_closing_tag_inside_argument_string_is_not_a_delimiter():
    _, calls = parse_tool_calls('<tool_call>{"name":"example","arguments":{"text":"</tool_call>"}}</tool_call>')
    assert json.loads(calls[0].arguments)["text"] == "</tool_call>"


@pytest.mark.parametrize("text", [
    '<tool_call>bad JSON</tool_call>', '<tool_call>{"name":"x","arguments":{}}', '</tool_call>',
    '<tool_call>{"name":"","arguments":{}}</tool_call>', '<tool_call>{"name":"x","arguments":[]}</tool_call>',
    '<tool_call>{"name":"x","name":"y","arguments":{}}</tool_call>',
])
def test_malformed_envelopes_raise_a_protocol_error_with_original_text(text):
    with pytest.raises(BackendProtocolError) as caught:
        parse_tool_calls(text)
    assert caught.value.raw_response["content"] == text


def test_plain_final_answer_is_not_converted_to_a_function_call():
    assert parse_tool_calls('{"outputs":{"delta_v":1.0}}') == ('{"outputs":{"delta_v":1.0}}', ())
