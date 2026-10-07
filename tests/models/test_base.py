"""The harness contract is usable without a provider or SDK."""

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from astrotoolbench.models import Message, ModelBackend, ModelResponse, TokenUsage, ToolCall, ToolDefinition


def test_other_backend_uses_the_same_conversation_contract():
    class LocalBackend(ModelBackend):
        def generate(self, messages, tools=None):
            assert messages[-1].content == "Compute a period."
            assert tools[0].name == "period"
            return ModelResponse(model="local-test", tool_calls=(
                ToolCall(id="call-1", name="period", arguments='{"radius": 7000}'),
            ), finish_reason="tool_calls")

    response = LocalBackend().generate(
        [Message(role="user", content="Compute a period.")],
        [ToolDefinition(name="period", description="Orbital period.", parameters={"type": "object"})],
    )
    history = [response.as_message(), Message(role="tool", content="5828.5", tool_call_id="call-1")]
    assert history[0].tool_calls[0].id == history[1].tool_call_id
    assert ModelResponse.model_validate_json(response.model_dump_json()) == response


def test_backend_requires_an_implementation():
    with pytest.raises(TypeError):
        ModelBackend()


def test_malformed_arguments_are_preserved_for_execution_layer():
    call = ToolCall(id="call-1", name="period", arguments="{bad JSON")
    assert call.arguments == "{bad JSON"


@pytest.mark.parametrize("kwargs", [
    {"role": "tool", "content": "1"},
    {"role": "user", "content": "hello", "tool_call_id": "call-1"},
    {"role": "system"},
    {"role": "user", "content": 123},
    {"role": "tool", "content": "1", "tool_call_id": " "},
    {"role": "user", "content": "hi", "tool_calls": (ToolCall(id="1", name="period", arguments="{}"),)},
])
def test_invalid_message_roles_and_payloads_rejected(kwargs):
    with pytest.raises(ValidationError):
        Message(**kwargs)


def test_duplicate_call_ids_rejected():
    calls = (ToolCall(id="1", name="period", arguments="{}"),) * 2
    with pytest.raises(ValidationError, match="unique"):
        ModelResponse(model="test", tool_calls=calls)
    with pytest.raises(ValidationError, match="unique"):
        Message(role="assistant", tool_calls=calls)


@pytest.mark.parametrize("count", [-1, True, 1.5, "123"])
def test_invalid_usage_rejected(count):
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=count)


def test_absent_usage_is_not_zero():
    assert ModelResponse(model="test").usage.input_tokens is None
    assert TokenUsage(input_tokens=0).input_tokens == 0


def test_models_import_with_api_dependencies_blocked():
    code = """
import sys
from importlib.abc import MetaPathFinder
sys.path.insert(0, sys.argv[1])
class NoSDK(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'openai', 'dotenv', 'transformers', 'atb'}:
            raise ImportError('API dependency blocked: ' + fullname)
sys.meta_path.insert(0, NoSDK())
from astrotoolbench.models import Message, ModelBackend, ModelResponse, OpenAIBackend
from astrotoolbench.benchmark import load_frozen_benchmark
assert ModelResponse(model='offline', content='ok').content == 'ok'
assert len(load_frozen_benchmark()) == 50
"""
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-c", code, str(root / "src")], cwd=root, check=True)
