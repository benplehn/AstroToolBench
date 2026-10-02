"""Impulsive maneuvers."""
import numpy as np
from .constants import MU_EARTH


def hohmann_transfer(r1, r2, mu=MU_EARTH):
    """Hohmann transfer between coplanar circular orbits of radii r1, r2 [km].

    Returns a dict with dv1, dv2, dv_total [km/s] and time of flight tof [s].
    Delta-v values are magnitudes (always positive).
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError("Orbit radii must be positive (km, from Earth's center).")
    a_t = 0.5 * (r1 + r2)
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    v_tp = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))   # transfer orbit speed at r1
    v_ta = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))   # transfer orbit speed at r2
    dv1 = abs(v_tp - v1)
    dv2 = abs(v2 - v_ta)
    tof = np.pi * np.sqrt(a_t**3 / mu)
    return {"dv1": dv1, "dv2": dv2, "dv_total": dv1 + dv2, "tof": tof}
