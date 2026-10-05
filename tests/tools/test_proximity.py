"""Closest approach with analytically constructed simultaneous encounters."""

import numpy as np
import pytest

from astrodyn_tools import MU_EARTH, closest_approach, coe_to_rv, orbital_period


def crossing_states(meeting_time, mu=MU_EARTH):
    radius = 7000.0
    phase = -2 * np.pi * meeting_time / orbital_period(radius, mu)
    first = coe_to_rv(radius, 0, 0, 0, 0, phase, mu)
    second = coe_to_rv(radius, 0, np.pi / 2, 0, 0, phase, mu)
    return (*first, *second)


@pytest.mark.parametrize("meeting_time", [0.0, 300.0])
def test_minimum_at_window_endpoint_is_preserved(meeting_time):
    result = closest_approach(*crossing_states(meeting_time), 300, step=17)
    assert result["tca"] == meeting_time
    assert result["miss_distance"] < 1e-8


def test_zero_window_returns_exact_initial_distance():
    states = crossing_states(100)
    result = closest_approach(*states, 0)
    assert result["tca"] == 0
    assert result["miss_distance"] == np.linalg.norm(states[0] - states[2])


def test_identical_trajectories_choose_initial_instant():
    position, velocity = coe_to_rv(8000, 0.1, 0.2, 0.3, 0.4, 0.5)
    result = closest_approach(position, velocity, position, velocity, 50)
    assert result == {"tca": 0.0, "miss_distance": 0.0}


def test_custom_mu_controls_both_trajectories():
    mu = 4 * MU_EARTH
    result = closest_approach(*crossing_states(101.25, mu), 250, step=13, tol=1e-5, mu=mu)
    assert result["tca"] == pytest.approx(101.25, abs=1e-5)
    assert result["miss_distance"] < 0.0002


def test_short_window_still_refines_an_interior_encounter():
    result = closest_approach(*crossing_states(2.3), 5, step=10, tol=1e-6)
    assert result["tca"] == pytest.approx(2.3, abs=1e-6)
    assert result["miss_distance"] < 0.00002


@pytest.mark.parametrize("kwargs", [
    {"window": -1}, {"window": np.nan}, {"window": np.inf},
    {"step": 0}, {"step": np.nan}, {"tol": -1}, {"tol": np.inf},
    {"mu": 0}, {"r1": [0, 0, 0]}, {"v2": [0, np.nan, 0]},
])
def test_invalid_closest_approach_inputs(kwargs):
    r1, v1, r2, v2 = crossing_states(100)
    arguments = dict(r1=r1, v1=v1, r2=r2, v2=v2, window=0)
    with pytest.raises(ValueError):
        closest_approach(**(arguments | kwargs))
