"""Analytic cylindrical geometry and search-boundary checks."""

import numpy as np
import pytest

from astrotoolbench.tools import (
    MU_EARTH, R_EARTH, coe_to_rv, eclipse_windows, in_cylindrical_shadow, orbital_period,
)


@pytest.mark.parametrize("position, expected", [
    ([7000, 0, 0], False),   # Sunward side.
    ([-7000, 0, 0], True),
    ([-7000, R_EARTH, 0], False),  # Cylinder boundary is illuminated.
    ([0, 7000, 0], False),  # Day/night plane is illuminated.
])
def test_shadow_geometry_and_direction_normalization(position, expected):
    assert in_cylindrical_shadow(position, [1, 0, 0]) is expected
    assert in_cylindrical_shadow(position, [12, 0, 0]) is expected
    assert in_cylindrical_shadow(position, [1e-300, 0, 0]) is expected


@pytest.mark.parametrize("kwargs", [
    {"sun_dir": [0, 0, 0]}, {"sun_dir": [np.nan, 0, 0]},
    {"sun_dir": [1, 0]}, {"r_sat": [0, 0, 0]}, {"r_body": 0},
    {"r_body": np.inf}, {"r_sat": [-7000, np.inf, 0]},
])
def test_invalid_shadow_geometry(kwargs):
    with pytest.raises(ValueError):
        in_cylindrical_shadow(**({"r_sat": [-7000, 0, 0], "sun_dir": [1, 0, 0]} | kwargs))


def test_custom_body_and_mu_against_analytic_eclipse_times():
    radius, body_radius, mu = 7000.0, 3500.0, 4 * MU_EARTH
    state = coe_to_rv(radius, 0, 0, 0, 0, 0, mu)
    period = orbital_period(radius, mu)
    half_angle = np.arcsin(body_radius / radius)
    expected = np.array([np.pi - half_angle, np.pi + half_angle]) * period / (2 * np.pi)
    windows = eclipse_windows(*state, [1, 0, 0], period, step=7, tol=1e-5, mu=mu, r_body=body_radius)
    assert len(windows) == 1
    np.testing.assert_allclose(windows[0], expected, rtol=0, atol=1e-5)


def test_eclipse_clipped_at_both_window_boundaries():
    state = coe_to_rv(7000, 0, 0, 0, 0, np.pi)
    assert eclipse_windows(*state, [1, 0, 0], 1, step=10) == [(0.0, 1.0)]
    assert eclipse_windows(*state, [1, 0, 0], 0) == []


def test_multiple_periods_include_every_detectable_eclipse():
    radius = 7000.0
    state = coe_to_rv(radius, 0, 0, 0, 0, 0)
    period = orbital_period(radius)
    windows = eclipse_windows(*state, [1, 0, 0], 2 * period, step=30)
    assert len(windows) == 2
    expected_duration = period * np.arcsin(R_EARTH / radius) / np.pi
    for start, end in windows:
        assert end - start == pytest.approx(expected_duration, abs=0.002)


@pytest.mark.parametrize("kwargs", [
    {"duration": -1}, {"duration": np.nan}, {"step": 0}, {"step": -1},
    {"step": np.inf}, {"tol": 0}, {"tol": np.nan}, {"mu": -1},
    {"r_body": 0}, {"sun_dir": [0, 0, 0]},
])
def test_invalid_eclipse_search(kwargs):
    position, velocity = coe_to_rv(7000, 0, 0, 0, 0, 0)
    arguments = dict(r0=position, v0=velocity, sun_dir=[1, 0, 0], duration=100)
    with pytest.raises(ValueError):
        eclipse_windows(**(arguments | kwargs))


def test_body_intersection_rejected_even_when_initial_position_is_outside():
    state = coe_to_rv(9000, 0.4, 0, 0, 0, np.pi)
    with pytest.raises(ValueError, match="periapsis"):
        eclipse_windows(*state, [1, 0, 0], 0)
