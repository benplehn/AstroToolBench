"""Finite catalog of scientific recipes, with explicit input and output units.

Recipes combine the existing numerical tools; they are not model-facing APIs.
Keep family labels separate: several problem families may use the same recipe.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    inputs: dict[str, str]
    outputs: dict[str, str]
    tools: tuple[str, ...]


_STATE = {"r0": "km", "v0": "km/s", "mu": "km^3/s^2"}
_TRANSFER = {"r1": "km", "r2": "km", "mu": "km^3/s^2"}
_TRANSFER_OUTPUTS = {"dv1": "km/s", "dv2": "km/s", "dv_total": "km/s", "tof": "s"}
_SHADOW = {"sun_dir": "1", "r_body": "km"}
_SEARCH = {"step": "s", "tol": "s"}
_TWO_STATES = {"r1": "km", "v1": "km/s", "r2": "km", "v2": "km/s", "mu": "km^3/s^2"}

RECIPES = {
    "propagation": Recipe(_STATE | {"dt": "s"}, {"position": "km", "velocity": "km/s"}, ("propagate_kepler",)),
    "delta_v": Recipe({"v_initial": "km/s", "v_final": "km/s"}, {"delta_v": "km/s"}, ("delta_v",)),
    "circularization": Recipe(_STATE, {"delta_v": "km/s"}, ("delta_v",)),
    "hohmann": Recipe(_TRANSFER, _TRANSFER_OUTPUTS, ("hohmann_transfer",)),
    "compare_transfers": Recipe(
        {"r_start": "km", "r_target_a": "km", "r_target_b": "km", "mu": "km^3/s^2"},
        {"dv_a": "km/s", "dv_b": "km/s", "difference": "km/s"}, ("hohmann_transfer",),
    ),
    "shadow": Recipe({"r_sat": "km"} | _SHADOW, {"in_shadow": "1"}, ("in_cylindrical_shadow",)),
    "eclipse": Recipe(_STATE | _SHADOW | _SEARCH | {"duration": "s"},
                      {"total_duration": "s", "boundaries": "s"}, ("eclipse_windows",)),
    "proximity": Recipe(_TWO_STATES | _SEARCH | {"window": "s"},
                        {"tca": "s", "miss_distance": "km"}, ("closest_approach",)),
    "separation": Recipe(_TWO_STATES | {"dt": "s"}, {"distance": "km"}, ("propagate_kepler",)),
    "transfer_propagation": Recipe(_TRANSFER | _STATE, _TRANSFER_OUTPUTS | {"position": "km"},
                                   ("hohmann_transfer", "propagate_kepler")),
    "transfer_eclipse": Recipe(_TRANSFER | _STATE | _SHADOW | _SEARCH,
                               {"tof": "s", "total_duration": "s"}, ("hohmann_transfer", "eclipse_windows")),
    "transfer_arrival_shadow": Recipe(_TRANSFER | _SHADOW,
                                      {"tof": "s", "arrival_position": "km", "in_shadow": "1"},
                                      ("hohmann_transfer", "propagate_kepler", "in_cylindrical_shadow")),
}
