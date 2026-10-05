"""Instantaneous velocity changes and coplanar circular Hohmann transfers."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import positive_scalar, vector3
from .constants import MU_EARTH


def delta_v(v_initial: ArrayLike, v_final: ArrayLike) -> float:
    """Return ||v_final - v_initial|| [km/s] for an impulsive maneuver.

    Inputs are finite shape-(3,) velocities [km/s] at the same position and
    instant, in the same inertial frame. The result is a nonnegative magnitude;
    the signed impulse vector is v_final - v_initial. No gravity or spacecraft
    mass is needed. Invalid vectors raise ValueError. Frame/unit consistency is
    the caller's responsibility; numerical values cannot reveal mixed units.
    """
    initial = vector3(v_initial, "v_initial [km/s]")
    final = vector3(v_final, "v_final [km/s]")
    return float(np.linalg.norm(final - initial))


def hohmann_transfer(r1: float, r2: float, mu: float = MU_EARTH) -> dict[str, float]:
    """Transfer between coplanar circular orbits around the same point mass.

    r1 and r2 are positive finite center radii [km], not altitudes; mu is
    positive finite [km^3/s^2]. Return dv1, dv2, dv_total as nonnegative impulse
    magnitudes [km/s], and tof [s] for half the transfer ellipse. Raising and
    lowering are supported. Equal radii give zero impulses and half a circular
    period (the chosen half-ellipse convention, not a zero-duration maneuver).
    No inclination change, phasing or surface clearance is modeled. Invalid
    scalars raise ValueError.
    """
    r1 = positive_scalar(r1, "r1 [km]")
    r2 = positive_scalar(r2, "r2 [km]")
    mu = positive_scalar(mu, "mu [km^3/s^2]")
    a_t = 0.5 * r1 + 0.5 * r2
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    # The speed ratios avoid subtracting nearly equal vis-viva terms.
    v_departure = v1 * np.sqrt(r2 / a_t)
    v_arrival = v2 * np.sqrt(r1 / a_t)
    dv1 = float(abs(v_departure - v1))
    dv2 = float(abs(v2 - v_arrival))
    tof = float(np.pi * a_t * np.sqrt(a_t / mu))
    return {"dv1": dv1, "dv2": dv2, "dv_total": dv1 + dv2, "tof": tof}
