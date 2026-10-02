"""Unit tests against analytic results and textbook values."""
import numpy as np
import pytest

from astrodyn_tools import (MU_EARTH, R_EARTH, coe_to_rv, rv_to_coe, propagate_kepler,
                            orbital_period, solve_kepler, hohmann_transfer,
                            eclipse_windows, closest_approach)


def test_coe_rv_roundtrip():
    coe = (7500.0, 0.1, np.radians(51.6), np.radians(40.0), np.radians(30.0), np.radians(120.0))
    r, v = coe_to_rv(*coe)
    back = rv_to_coe(r, v)
    assert np.allclose(back, coe, atol=1e-9)


def test_kepler_equation():
    for e in [0.0, 0.3, 0.9, 0.99]:
        for M in np.linspace(0.1, 6.2, 7):
            E = solve_kepler(M, e)
            assert abs(E - e * np.sin(E) - M) < 1e-10


def test_one_period_returns_to_start():
    r0, v0 = coe_to_rv(8000.0, 0.2, 0.5, 1.0, 2.0, 0.3)
    r1, v1 = propagate_kepler(r0, v0, orbital_period(8000.0))
    assert np.allclose(r1, r0, atol=1e-6) and np.allclose(v1, v0, atol=1e-9)


def test_energy_and_momentum_conserved():
    r0, v0 = coe_to_rv(9000.0, 0.3, 1.2, 0.4, 0.8, 0.0)
    r1, v1 = propagate_kepler(r0, v0, 3456.7)
    energy = lambda r, v: np.dot(v, v) / 2 - MU_EARTH / np.linalg.norm(r)
    assert abs(energy(r1, v1) - energy(r0, v0)) < 1e-9
    assert np.allclose(np.cross(r1, v1), np.cross(r0, v0), atol=1e-6)


def test_hohmann_leo_to_geo_vallado_example_6_1():
    # Vallado, Fundamentals of Astrodynamics, Example 6-1:
    # 191.34411 km altitude -> 35,781.34857 km altitude
    out = hohmann_transfer(R_EARTH + 191.34411, R_EARTH + 35781.34857)
    assert out["dv1"] == pytest.approx(2.457, abs=2e-3)
    assert out["dv2"] == pytest.approx(1.478, abs=2e-3)
    assert out["dv_total"] == pytest.approx(3.935, abs=2e-3)
    assert out["tof"] / 3600 == pytest.approx(5.256, abs=2e-3)


def test_eclipse_duration_circular_sun_in_plane():
    # Circular orbit, Sun in the orbital plane: eclipse fraction = asin(R/r) / pi
    r = R_EARTH + 500.0
    r0, v0 = coe_to_rv(r, 0.0, 0.0, 0.0, 0.0, 0.0)
    T = orbital_period(r)
    w = eclipse_windows(r0, v0, sun_dir=[1.0, 0.0, 0.0], duration=T, step=5.0)
    expected = np.arcsin(R_EARTH / r) / np.pi * T
    assert len(w) == 1
    assert (w[0][1] - w[0][0]) == pytest.approx(expected, abs=0.05)


def test_no_eclipse_when_orbit_faces_sun():
    # Circular polar orbit whose plane is perpendicular to the Sun direction (dawn-dusk)
    r = R_EARTH + 8000.0
    r0, v0 = coe_to_rv(r, 0.0, np.pi / 2, np.pi / 2, 0.0, 0.0)   # raan=90 deg -> orbit in the y-z plane
    w = eclipse_windows(r0, v0, sun_dir=[1.0, 0.0, 0.0], duration=orbital_period(r))
    assert w == []


def test_closest_approach_designed_crossing():
    # Two circular orbits (equatorial and polar) of the same radius, both reaching
    # the node on the +x axis at t = t_meet.
    r = R_EARTH + 700.0
    T = orbital_period(r)
    t_meet = 1000.0
    nu0 = -2 * np.pi * t_meet / T
    r1, v1 = coe_to_rv(r, 0.0, 0.0, 0.0, 0.0, nu0)
    r2, v2 = coe_to_rv(r, 0.0, np.pi / 2, 0.0, 0.0, nu0)
    out = closest_approach(r1, v1, r2, v2, window=2000.0, step=5.0)
    assert out["tca"] == pytest.approx(t_meet, abs=0.01)
    assert out["miss_distance"] < 0.05
