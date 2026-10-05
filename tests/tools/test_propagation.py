"""Analytic checks and edge cases for the elliptic two-body contract."""

import numpy as np
import pytest

from astrotoolbench.tools import (
    MU_EARTH, coe_to_rv, orbital_period, propagate_kepler, rv_to_coe, solve_kepler,
)


@pytest.mark.parametrize("inclination", [0.0, 0.7, np.pi])
@pytest.mark.parametrize("eccentricity", [0.0, 0.2])
def test_singular_element_conventions_preserve_state(inclination, eccentricity):
    position, velocity = coe_to_rv(9000, eccentricity, inclination, 1.1, 0.9, 2.0)
    rebuilt = coe_to_rv(*rv_to_coe(position, velocity))
    np.testing.assert_allclose(rebuilt[0], position, rtol=0, atol=1e-8)
    np.testing.assert_allclose(rebuilt[1], velocity, rtol=0, atol=1e-11)


@pytest.mark.parametrize("mu", [MU_EARTH, 42828.375214])
@pytest.mark.parametrize("direction", [-1, 1])
def test_circular_quarter_period_matches_analytic_rotation(mu, direction):
    radius = 7000.0
    speed = np.sqrt(mu / radius)
    position, velocity = propagate_kepler(
        [radius, 0, 0], [0, speed, 0], direction * orbital_period(radius, mu) / 4, mu,
    )
    np.testing.assert_allclose(position, [0, direction * radius, 0], rtol=0, atol=1e-8)
    np.testing.assert_allclose(velocity, [-direction * speed, 0, 0], rtol=0, atol=1e-11)


def test_zero_duration_returns_independent_exact_copies():
    position, velocity = coe_to_rv(9000, 0.2, 0.3, 0.4, 0.5, 0.6)
    final_position, final_velocity = propagate_kepler(position, velocity, 0)
    np.testing.assert_array_equal(final_position, position)
    np.testing.assert_array_equal(final_velocity, velocity)
    assert not np.shares_memory(position, final_position)
    assert not np.shares_memory(velocity, final_velocity)


def test_many_periods_preserve_phase():
    radius = 7000.0
    period = orbital_period(radius)
    initial = ([radius, 0, 0], [0, np.sqrt(MU_EARTH / radius), 0])
    short = propagate_kepler(*initial, period / 4)
    long = propagate_kepler(*initial, 100000 * period + period / 4)
    np.testing.assert_allclose(long[0], short[0], rtol=0, atol=1e-5)
    np.testing.assert_allclose(long[1], short[1], rtol=0, atol=1e-8)


@pytest.mark.parametrize("mean_anomaly", [1e-10, 1e-6, 0.1, np.pi, 2*np.pi - 1e-6])
def test_kepler_high_eccentricity_against_independent_bisection(mean_anomaly):
    eccentricity = 0.9999
    left, right = 0.0, 2 * np.pi
    for _ in range(100):
        middle = (left + right) / 2
        if middle - eccentricity * np.sin(middle) < mean_anomaly:
            left = middle
        else:
            right = middle
    expected = (left + right) / 2
    assert solve_kepler(mean_anomaly, eccentricity) == pytest.approx(expected, abs=2e-11)


def test_kepler_nonconvergence_is_explicit():
    with pytest.raises(RuntimeError, match="converge"):
        solve_kepler(0.1, 0.99, max_iter=1)


@pytest.mark.parametrize("kwargs", [
    {"M": np.nan}, {"e": 1.0}, {"e": -0.1}, {"tol": 0},
    {"max_iter": 0}, {"max_iter": 1.5}, {"max_iter": True},
])
def test_invalid_kepler_inputs(kwargs):
    arguments = {"M": 0.1, "e": 0.2, **kwargs}
    with pytest.raises(ValueError):
        solve_kepler(**arguments)


@pytest.mark.parametrize("position, velocity", [
    ([0, 0, 0], [0, 7, 0]),
    ([7000, 0, 0], [0, 0, 0]),
    ([7000, 0, 0], [7, 0, 0]),
    ([7000, 0, 0], [0, 12, 0]),  # Unbound Earth orbit.
    ([7000, 0], [0, 7, 0]),
    ([[7000, 0, 0]], [0, 7, 0]),
    ([7000, np.nan, 0], [0, 7, 0]),
    ([7000, 0, 0], [0, np.inf, 0]),
])
def test_invalid_state_is_rejected_even_at_zero_time(position, velocity):
    with pytest.raises(ValueError):
        propagate_kepler(position, velocity, 0)


@pytest.mark.parametrize("mu", [0, -1, np.nan, np.inf, True, "398600", [398600]])
def test_invalid_gravitational_parameter(mu):
    with pytest.raises(ValueError):
        orbital_period(7000, mu)
    with pytest.raises(ValueError):
        propagate_kepler([7000, 0, 0], [0, 7, 0], 1, mu)


@pytest.mark.parametrize("kwargs", [
    {"a": 0}, {"a": np.inf}, {"e": 1}, {"e": -1},
    {"i": -0.1}, {"i": np.pi + 0.1}, {"raan": np.nan},
    {"argp": np.inf}, {"nu": np.nan}, {"mu": -1},
])
def test_invalid_classical_elements(kwargs):
    arguments = dict(a=7000, e=0.1, i=0.2, raan=0.3, argp=0.4, nu=0.5)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        coe_to_rv(**arguments)


@pytest.mark.parametrize("duration", [np.nan, np.inf, [1.0], True])
def test_invalid_propagation_duration(duration):
    with pytest.raises(ValueError):
        propagate_kepler([7000, 0, 0], [0, 7, 0], duration)
