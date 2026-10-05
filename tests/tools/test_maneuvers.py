"""Independent impulse geometry and Hohmann symmetry checks."""

import numpy as np
import pytest

from astrotoolbench.tools import MU_EARTH, delta_v, hohmann_transfer, orbital_period


def test_delta_v_is_vector_difference_not_speed_difference():
    assert delta_v([3, 0, 0], [0, 4, 0]) == 5.0
    assert delta_v([0, 3, 4], [0, 3, 4]) == 0.0


def test_delta_v_velocity_reversal():
    assert delta_v([0, 7.5, 0], [0, -7.5, 0]) == 15.0


@pytest.mark.parametrize("bad", [
    [1, 2], [[1, 2, 3]], [0, np.inf, 0], [True]*3, [True, 0, 0], ["0"]*3,
])
def test_delta_v_invalid_vectors(bad):
    with pytest.raises(ValueError):
        delta_v(bad, [0, 0, 0])
    with pytest.raises(ValueError):
        delta_v([0, 0, 0], bad)


def test_hohmann_equal_radii_half_period_convention():
    result = hohmann_transfer(7000, 7000)
    assert result["dv1"] == result["dv2"] == result["dv_total"] == 0
    assert result["tof"] == pytest.approx(orbital_period(7000) / 2, abs=1e-10)


def test_hohmann_lowering_reverses_impulses():
    up = hohmann_transfer(7000, 42164)
    down = hohmann_transfer(42164, 7000)
    assert up["dv1"] == down["dv2"]
    assert up["dv2"] == down["dv1"]
    assert up["dv_total"] == down["dv_total"]
    assert up["tof"] == down["tof"]


def test_hohmann_custom_mu_scaling():
    original = hohmann_transfer(7000, 9000)
    scaled = hohmann_transfer(7000, 9000, mu=4 * MU_EARTH)
    for name in ("dv1", "dv2", "dv_total"):
        assert scaled[name] == pytest.approx(2 * original[name], abs=1e-12)
    assert scaled["tof"] == pytest.approx(original["tof"] / 2, abs=1e-10)


@pytest.mark.parametrize("kwargs", [
    {"r1": 0}, {"r2": -1}, {"r1": np.nan}, {"r2": np.inf},
    {"mu": 0}, {"mu": -1}, {"mu": np.nan}, {"mu": np.inf},
])
def test_invalid_transfer_inputs(kwargs):
    with pytest.raises(ValueError):
        hohmann_transfer(**({"r1": 7000, "r2": 9000} | kwargs))
