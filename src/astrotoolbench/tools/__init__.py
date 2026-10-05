"""Numerical flight dynamics for AstroToolBench.

Units and coordinate frame:
    distance  -> km
    velocity  -> km/s
    time      -> s
    angles    -> rad
    frame     -> body-centered inertial (ECI with the Earth defaults)

The scientific contract is documented in docs/scientific-tools.md. No model
provider, network access or optional LLM dependency is needed for these tools.
"""
from .constants import MU_EARTH, R_EARTH
from .propagation import coe_to_rv, rv_to_coe, solve_kepler, propagate_kepler, orbital_period
from .maneuvers import delta_v, hohmann_transfer
from .eclipse import in_cylindrical_shadow, eclipse_windows
from .proximity import closest_approach

__all__ = [
    "MU_EARTH", "R_EARTH", "coe_to_rv", "rv_to_coe", "solve_kepler",
    "propagate_kepler", "orbital_period", "delta_v", "hohmann_transfer",
    "in_cylindrical_shadow", "eclipse_windows", "closest_approach",
]
