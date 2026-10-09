"""Original scientific functions exposed without task-specific recipes."""

import inspect
import json
import math
from time import perf_counter_ns
from typing import Annotated

import numpy as np
from numpy.typing import ArrayLike
from pydantic import ConfigDict, Field, create_model

from .. import tools
from ..benchmark.loader import _unique_object
from ..models import Message, ToolCall, ToolDefinition

MAX_SEARCH_SAMPLES = 10_000
MAX_SOLVER_ITERATIONS = 10_000
MAX_ARGUMENT_BYTES = 64_000
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Vector = Annotated[list[Number], Field(min_length=3, max_length=3)]

# Constants and private helpers are not callable model tools.
FUNCTIONS = {name: getattr(tools, name) for name in tools.__all__
             if callable(getattr(tools, name))}


def _parameter_model(name, function):
    fields = {}
    for parameter in inspect.signature(function).parameters.values():
        annotation = Vector if parameter.annotation == ArrayLike else (
            Annotated[int, Field(strict=True)] if parameter.annotation is int else Number
        )
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (annotation, default)
    return create_model(name + "Arguments", __config__=ConfigDict(extra="forbid"), **fields)


PARAMETERS = {name: _parameter_model(name, function) for name, function in FUNCTIONS.items()}


def tool_definitions() -> list[ToolDefinition]:
    return [ToolDefinition(
        name=name, description=inspect.getdoc(function).splitlines()[0],
        parameters=PARAMETERS[name].model_json_schema(),
    ) for name, function in FUNCTIONS.items()]


def _finite_float(text):
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("Arguments must contain finite numbers.")
    return number


def _nonfinite(text):
    raise ValueError("Arguments must contain finite numbers.")


def _check_work(name, parameters):
    if name in {"eclipse_windows", "closest_approach"}:
        duration = parameters["duration" if name == "eclipse_windows" else "window"]
        step = parameters["step"]
        if duration >= 0 and step > 0 and duration / step > MAX_SEARCH_SAMPLES - 1:
            raise ValueError(f"Search exceeds {MAX_SEARCH_SAMPLES} samples.")
    if name == "solve_kepler" and parameters["max_iter"] > MAX_SOLVER_ITERATIONS:
        raise ValueError(f"Solver exceeds {MAX_SOLVER_ITERATIONS} iterations.")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {name: _json_value(item) for name, item in value.items()}
    return value


def requested_call(call: ToolCall, step: int) -> dict:
    return {
        "step": step, "id": call.id, "tool": call.name,
        "arguments_json": call.arguments, "arguments": None, "bound_arguments": None,
        "status": "pending", "executed": False, "success": None,
        "result": None, "error": None, "execution_time_ms": None,
    }


def execute_call(call: ToolCall, *, step: int = 1) -> dict:
    record = requested_call(call, step)
    started = perf_counter_ns()
    error_type = "unknown_tool"
    try:
        # Check the name first: an invented tool is a hallucination even if its arguments are broken.
        if call.name not in FUNCTIONS:
            raise ValueError(f"Unknown scientific tool: {call.name}")
        error_type = "invalid_arguments"
        if len(call.arguments.encode("utf-8")) > MAX_ARGUMENT_BYTES:
            raise ValueError(f"Arguments exceed {MAX_ARGUMENT_BYTES} bytes.")
        arguments = json.loads(call.arguments, object_pairs_hook=_unique_object,
                               parse_constant=_nonfinite, parse_float=_finite_float)
        record["arguments"] = arguments
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        parameters = PARAMETERS[call.name].model_validate(arguments).model_dump()
        record["bound_arguments"] = parameters
        error_type = "resource_limit"
        _check_work(call.name, parameters)
        record["executed"] = True
        error_type = "execution_error"
        result = _json_value(FUNCTIONS[call.name](**parameters))
        error_type = "invalid_result"
        # Reject nonfinite solver output before it enters a trace or conversation.
        json.dumps(result, allow_nan=False)
        record.update(result=result, status="completed", success=True)
    except (ValueError, RuntimeError, ArithmeticError) as error:
        record.update(status="error", success=False, error={"type": error_type, "message": str(error)})
    record["execution_time_ms"] = (perf_counter_ns() - started) / 1_000_000
    return record


def result_message(record: dict) -> Message:
    payload = {"success": record["success"]}
    payload["result" if record["success"] else "error"] = record["result"] if record["success"] else record["error"]
    return Message(role="tool", tool_call_id=record["id"],
                   content=json.dumps(payload, ensure_ascii=False, allow_nan=False))
