"""Raw signatures, native result shapes and strict execution boundaries."""

import inspect
import json

import numpy as np
import pytest

from astrotoolbench.eval.raw_tools import (
    FUNCTIONS, MAX_ARGUMENT_BYTES, MAX_SEARCH_SAMPLES, execute_call, result_message, tool_definitions,
)
from astrotoolbench.models import ToolCall

STATE = {"r0": [7000, 0, 0], "v0": [0, 7.5, 0]}
CASES = {
    "orbital_period": {"a": 7000},
    "solve_kepler": {"M": 0.4, "e": 0.2},
    "coe_to_rv": {"a": 7000, "e": 0.1, "i": 0, "raan": 0, "argp": 0, "nu": 0},
    "rv_to_coe": {"r": STATE["r0"], "v": STATE["v0"]},
    "propagate_kepler": STATE | {"dt": 100},
    "delta_v": {"v_initial": [0, 7.5, 0], "v_final": [0, 8, 0]},
    "hohmann_transfer": {"r1": 7000, "r2": 9000},
    "in_cylindrical_shadow": {"r_sat": [-7000, 0, 0], "sun_dir": [1, 0, 0]},
    "eclipse_windows": STATE | {"sun_dir": [1, 0, 0], "duration": 0},
    "closest_approach": {"r1": STATE["r0"], "v1": STATE["v0"], "r2": [7001, 0, 0],
                         "v2": STATE["v0"], "window": 0},
}


def call(name, arguments, id="call-1"):
    return ToolCall(id=id, name=name, arguments=json.dumps(arguments) if not isinstance(arguments, str) else arguments)


def test_all_ten_original_signatures_are_exposed_once():
    definitions = tool_definitions()
    assert {tool.name for tool in definitions} == set(CASES) == set(FUNCTIONS)
    for tool in definitions:
        signature = inspect.signature(FUNCTIONS[tool.name])
        assert set(tool.parameters["properties"]) == set(signature.parameters)
        assert set(tool.parameters.get("required", [])) == {
            name for name, parameter in signature.parameters.items() if parameter.default is inspect.Parameter.empty
        }
        assert "\n" not in tool.description
        assert tool.parameters["additionalProperties"] is False


@pytest.mark.parametrize("name", CASES)
def test_execution_retains_native_json_results(name):
    record = execute_call(call(name, CASES[name]))
    assert record["success"] is True
    assert record["executed"] is True
    assert record["arguments"] == CASES[name]
    assert record["execution_time_ms"] >= 0
    assert record["error"] is None
    expected = FUNCTIONS[name](**CASES[name])
    expected = json.loads(json.dumps(expected, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value))
    assert record["result"] == expected
    message = result_message(record)
    assert message.tool_call_id == record["id"]
    assert json.loads(message.content) == {"success": True, "result": record["result"]}


@pytest.mark.parametrize("arguments", [
    "{bad JSON", "[]", "null", '{"a":7000,"a":8000}', '{"a":NaN}', '{"a":1e309}',
    {"a": True}, {"a": "7000"}, {"a": 7000, "invented_parameter": 1}, {},
    {"a": [7000]}, " " * (MAX_ARGUMENT_BYTES + 1),
])
def test_invalid_arguments_remain_visible_and_are_not_executed(arguments):
    request = call("orbital_period", arguments)
    record = execute_call(request)
    assert record["success"] is False
    assert record["executed"] is False
    assert record["arguments_json"] == request.arguments
    assert record["error"]["type"] == "invalid_arguments"
    json.dumps(record, allow_nan=False)
    assert json.loads(result_message(record).content)["success"] is False


def test_hallucinated_function_never_dispatches_to_python():
    record = execute_call(call("__import__", {"name": "os"}))
    assert record["error"]["type"] == "unknown_tool"
    assert not record["executed"]


@pytest.mark.parametrize("arguments", ["{bad JSON", " " * (MAX_ARGUMENT_BYTES + 1)])
def test_hallucinated_function_counts_even_with_broken_arguments(arguments):
    record = execute_call(call("propagate", arguments))
    assert record["error"]["type"] == "unknown_tool"
    assert not record["executed"]


@pytest.mark.parametrize("arguments", [
    {"v_initial": [True, 0, 0], "v_final": [0, 0, 0]},
    {"v_initial": [0, 0], "v_final": [0, 0, 0]},
    {"v_initial": [0, "0", 0], "v_final": [0, 0, 0]},
])
def test_vector_type_and_shape_checked_before_science(arguments):
    record = execute_call(call("delta_v", arguments))
    assert not record["executed"] and not record["success"]


def test_defaults_are_recorded_and_physical_input_errors_are_returned():
    record = execute_call(call("hohmann_transfer", {"r1": -1, "r2": 9000}))
    assert record["executed"] is True
    assert record["success"] is False
    assert record["bound_arguments"]["mu"] == 398600.4418
    assert record["error"]["type"] == "execution_error"


def test_work_limits_reject_huge_sample_allocations_and_iteration_budgets():
    record = execute_call(call("eclipse_windows", STATE | {
        "sun_dir": [1, 0, 0], "duration": MAX_SEARCH_SAMPLES, "step": 1,
    }))
    assert record["error"]["type"] == "resource_limit"
    assert record["executed"] is False
    record = execute_call(call("solve_kepler", {"M": 1, "e": 0.1, "max_iter": 10001}))
    assert record["error"]["type"] == "resource_limit"


def test_nonfinite_tool_output_is_not_sent_to_the_model(monkeypatch):
    monkeypatch.setitem(FUNCTIONS, "orbital_period", lambda **kwargs: float("nan"))
    record = execute_call(call("orbital_period", {"a": 7000}))
    assert record["error"]["type"] == "invalid_result"
    assert record["result"] is None
    json.dumps(record, allow_nan=False)
