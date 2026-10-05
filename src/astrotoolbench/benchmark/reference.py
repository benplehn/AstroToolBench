"""Execute fixed scientific recipes without reading their expected values."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from astrotoolbench.tools import (
    closest_approach, delta_v, eclipse_windows, hohmann_transfer,
    in_cylindrical_shadow, propagate_kepler, rv_to_coe,
)

from .catalog import RECIPES
from .schema import TaskMetadata


@dataclass(frozen=True)
class Outcome:
    values: dict[str, Any]
    error: str | None = None


class TaskInputError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _parameters(task: TaskMetadata) -> dict[str, Any]:
    for name, unit in RECIPES[task.recipe].inputs.items():
        quantity = task.inputs[name]
        if quantity.value is None:
            raise TaskInputError("missing_input")
        if quantity.unit != unit:
            raise TaskInputError("inconsistent_units")
    if task.context.dynamics != "two_body":
        raise TaskInputError("unsupported_model")
    values = {name: quantity.value for name, quantity in task.inputs.items()}
    requested_accuracy = task.context.requested_time_accuracy_s
    if requested_accuracy is not None and "tol" in values and requested_accuracy < values["tol"]:
        raise TaskInputError("unachievable_precision")
    _surface_check(task, values)
    return values


def _surface_check(task, values):
    """The benchmark describes exterior satellites, unlike a bare point mass."""
    body_radius = task.context.body_radius_km
    for name in ("r0", "r1", "r2", "r_start", "r_target_a", "r_target_b", "r_sat"):
        if name not in values:
            continue
        value = np.asarray(values[name])
        radius = float(value) if value.ndim == 0 else np.linalg.norm(value)
        # Zero/negative scalars are malformed tool inputs, not underground orbits.
        if 0 < radius < body_radius:
            raise TaskInputError("physical_impossibility")
    for position_name, velocity_name in (("r0", "v0"), ("r1", "v1"), ("r2", "v2")):
        if velocity_name in values:
            a, e, *_ = rv_to_coe(values[position_name], values[velocity_name], values["mu"])
            if a * (1 - e) < body_radius:
                raise TaskInputError("physical_impossibility")


def _propagation(p):
    position, velocity = propagate_kepler(p["r0"], p["v0"], p["dt"], p["mu"])
    return {"position": position.tolist(), "velocity": velocity.tolist()}


def _circularization(p):
    position, velocity = np.asarray(p["r0"]), np.asarray(p["v0"])
    tangent = velocity - np.dot(position, velocity) / np.dot(position, position) * position
    speed = np.sqrt(p["mu"] / np.linalg.norm(position))
    target_velocity = tangent / np.linalg.norm(tangent) * speed
    return {"delta_v": delta_v(velocity, target_velocity)}


def _compare_transfers(p):
    first = hohmann_transfer(p["r_start"], p["r_target_a"], p["mu"])["dv_total"]
    second = hohmann_transfer(p["r_start"], p["r_target_b"], p["mu"])["dv_total"]
    return {"dv_a": first, "dv_b": second, "difference": second - first}


def _eclipse(p):
    windows = eclipse_windows(p["r0"], p["v0"], p["sun_dir"], p["duration"], p["step"], p["tol"],
                              mu=p["mu"], r_body=p["r_body"])
    return {"total_duration": sum(end - start for start, end in windows),
            "boundaries": [boundary for window in windows for boundary in window]}


def _separation(p):
    first, _ = propagate_kepler(p["r1"], p["v1"], p["dt"], p["mu"])
    second, _ = propagate_kepler(p["r2"], p["v2"], p["dt"], p["mu"])
    return {"distance": float(np.linalg.norm(first - second))}


def _transfer_propagation(p):
    transfer = hohmann_transfer(p["r1"], p["r2"], p["mu"])
    return transfer | {"position": _propagation(p | {"dt": transfer["tof"]})["position"]}


def _transfer_eclipse(p):
    duration = hohmann_transfer(p["r1"], p["r2"], p["mu"])["tof"]
    return {"tof": duration, "total_duration": _eclipse(p | {"duration": duration})["total_duration"]}


def _transfer_arrival_shadow(p):
    transfer = hohmann_transfer(p["r1"], p["r2"], p["mu"])
    initial_speed = np.sqrt(p["mu"] / p["r1"])
    sign = np.sign(p["r2"] - p["r1"])
    # Start on +x; apply the departure impulse tangentially before propagation.
    position, _ = propagate_kepler([p["r1"], 0, 0], [0, initial_speed + sign * transfer["dv1"], 0],
                                  transfer["tof"], p["mu"])
    return {"tof": transfer["tof"], "arrival_position": position.tolist(),
            "in_shadow": in_cylindrical_shadow(position, p["sun_dir"], p["r_body"])}


_CALCULATORS = {
    "propagation": _propagation,
    "delta_v": lambda p: {"delta_v": delta_v(p["v_initial"], p["v_final"])},
    "circularization": _circularization,
    "hohmann": lambda p: hohmann_transfer(p["r1"], p["r2"], p["mu"]),
    "compare_transfers": _compare_transfers,
    "shadow": lambda p: {"in_shadow": in_cylindrical_shadow(p["r_sat"], p["sun_dir"], p["r_body"])},
    "eclipse": _eclipse,
    "proximity": lambda p: closest_approach(p["r1"], p["v1"], p["r2"], p["v2"], p["window"],
                                          p["step"], p["tol"], mu=p["mu"]),
    "separation": _separation,
    "transfer_propagation": _transfer_propagation,
    "transfer_eclipse": _transfer_eclipse,
    "transfer_arrival_shadow": _transfer_arrival_shadow,
}


def calculate(task: TaskMetadata) -> Outcome:
    """Compute an outcome; unexpected solver failures propagate instead of becoming references."""
    try:
        parameters = _parameters(task)
        computed = _CALCULATORS[task.recipe](parameters)
        return Outcome({name: computed[name] for name in task.outputs})
    except TaskInputError as error:
        return Outcome({}, error.code)
    except ValueError:
        return Outcome({}, "invalid_input")
