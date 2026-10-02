"""Core flight-dynamics functions for AstroToolBench (working name).

Conventions (used everywhere, no exceptions):
    distance  -> km
    velocity  -> km/s
    time      -> s
    angles    -> rad
    frame     -> Earth-centered inertial (ECI)
"""
from .constants import MU_EARTH, R_EARTH
from .kepler import coe_to_rv, rv_to_coe, solve_kepler, propagate_kepler, orbital_period
from .maneuvers import hohmann_transfer
from .eclipse import in_cylindrical_shadow, eclipse_windows
from .conjunction import closest_approach
