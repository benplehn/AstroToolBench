"""Exercise local generation without downloading weights or importing Torch."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock
import sys

import pytest

from astrotoolbench.models import BackendError, HuggingFaceBackend, Message, ToolDefinition

REVISION = "a" * 40


class Tokens(list):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Tokens(value) if isinstance(key, slice) else SimpleNamespace(item=lambda: value)


@pytest.fixture
def local_runtime(monkeypatch):
    model = MagicMock()
    model.to.return_value = model
    model.device = "cpu"
    model.generation_config.eos_token_id = 99
    model.generate.return_value = [Tokens([1, 2, 3, 4, 99])]
    inputs = MagicMock()
    inputs.to.return_value = {"input_ids": SimpleNamespace(shape=(1, 3))}
    tokenizer = MagicMock(return_value=inputs)
    tokenizer.apply_chat_template.return_value = "rendered conversation"
    tokenizer.decode.return_value = '{"outputs":{"delta_v":1.0}}'
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model
    tokenizer_class = MagicMock()
    tokenizer_class.from_pretrained.return_value = tokenizer
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        inference_mode=nullcontext, float32="float32", float16="float16",
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=tokenizer_class, AutoModelForCausalLM=model_class,
    ))
    return SimpleNamespace(model=model, tokenizer=tokenizer, model_class=model_class, tokenizer_class=tokenizer_class)


def test_pinned_load_greedy_generation_and_usage(local_runtime):
    backend = HuggingFaceBackend("test/model", revision=REVISION, max_tokens=2, local_files_only=True)
    response = backend.generate([Message(role="user", content="Calculate.")])
    local_runtime.tokenizer_class.from_pretrained.assert_called_once_with(
        "test/model", revision=REVISION, local_files_only=True, trust_remote_code=False,
    )
    local_runtime.model_class.from_pretrained.assert_called_once_with(
        "test/model", revision=REVISION, local_files_only=True, trust_remote_code=False, torch_dtype="float32",
    )
    local_runtime.model.eval.assert_called_once()
    request = local_runtime.model.generate.call_args.kwargs
    assert request["do_sample"] is False
    assert request["max_new_tokens"] == 2
    assert request["num_beams"] == request["num_return_sequences"] == 1
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 2
    assert response.finish_reason == "stop"
    assert response.content == '{"outputs":{"delta_v":1.0}}'


def test_token_limit_without_eos_is_truncated(local_runtime):
    local_runtime.model.generate.return_value = [Tokens([1, 2, 3, 4, 5])]
    response = HuggingFaceBackend("test/model", revision=REVISION, max_tokens=2).generate(
        [Message(role="user", content="Calculate.")],
    )
    assert response.finish_reason == "length"


def test_multiple_eos_ids_and_device_dtype(local_runtime):
    local_runtime.model.generation_config.eos_token_id = [98, 99]
    backend = HuggingFaceBackend("test/model", revision=REVISION, device="mps", max_tokens=2)
    assert backend.generate([Message(role="user", content="hello")]).finish_reason == "stop"
    assert local_runtime.model_class.from_pretrained.call_args.kwargs["torch_dtype"] == "float16"
    local_runtime.model.to.assert_called_once_with("mps")


def test_tools_are_rendered_and_normalized_using_the_native_template(local_runtime):
    backend = HuggingFaceBackend("test/model", revision=REVISION)
    local_runtime.tokenizer.decode.return_value = '<tool_call>{"name":"period","arguments":{"a":7000}}</tool_call>'
    tools = [ToolDefinition(name="period", description="Period.", parameters={"type": "object"})]
    response = backend.generate([Message(role="user", content="hello")], tools)
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "period"
    assert response.raw_content == local_runtime.tokenizer.decode.return_value
    assert local_runtime.tokenizer.apply_chat_template.call_args.kwargs["tools"] == [
        {"type": "function", "function": tools[0].model_dump()}
    ]
    backend.generate([response.as_message(), Message(role="tool", tool_call_id=response.tool_calls[0].id,
                                                    content='{"success":true,"result":5828.5}')], tools)
    replay = local_runtime.tokenizer.apply_chat_template.call_args.args[0]
    assert replay[0]["tool_calls"][0]["function"]["arguments"] == response.tool_calls[0].arguments
    assert replay[1]["tool_call_id"] == response.tool_calls[0].id


def test_truncated_local_call_is_kept_as_text_and_never_decoded(local_runtime):
    local_runtime.model.generate.return_value = [Tokens([1, 2, 3, 4, 5])]
    local_runtime.tokenizer.decode.return_value = '<tool_call>{"name":"period"'
    response = HuggingFaceBackend("test/model", revision=REVISION, max_tokens=2).generate(
        [Message(role="user", content="hello")], [ToolDefinition(name="period", description="", parameters={})],
    )
    assert response.finish_reason == "length"
    assert not response.tool_calls
    assert response.raw_content == local_runtime.tokenizer.decode.return_value


def test_malformed_local_tool_envelope_is_returned_as_a_model_mistake(local_runtime):
    local_runtime.tokenizer.decode.return_value = '<tool_call>bad JSON</tool_call>'
    response = HuggingFaceBackend("test/model", revision=REVISION).generate(
        [Message(role="user", content="hello")], [ToolDefinition(name="period", description="", parameters={})],
    )
    assert response.finish_reason == "invalid_tool_call"
    assert not response.tool_calls
    assert response.content == response.raw_content == local_runtime.tokenizer.decode.return_value


def test_local_runtime_errors_use_the_common_error(local_runtime):
    local_runtime.model.generate.side_effect = RuntimeError("out of memory")
    with pytest.raises(BackendError, match="Local model generation failed"):
        HuggingFaceBackend("test/model", revision=REVISION).generate([Message(role="user", content="hello")])


@pytest.mark.parametrize("options", [
    {"revision": "main"}, {"revision": ""}, {"revision": "a" * 39},
    {"revision": REVISION, "max_tokens": True}, {"revision": REVISION, "max_tokens": 0},
    {"revision": REVISION, "device": "unknown"},
])
def test_invalid_local_configuration_rejected_before_loading(local_runtime, options):
    with pytest.raises(ValueError):
        HuggingFaceBackend("test/model", **options)
    local_runtime.model_class.from_pretrained.assert_not_called()


def test_empty_history_rejected(local_runtime):
    with pytest.raises(ValueError, match="At least one message"):
        HuggingFaceBackend("test/model", revision=REVISION).generate([])
