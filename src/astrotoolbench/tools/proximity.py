"""Simultaneous closest approach of two elliptic two-body trajectories."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import time_grid
from .constants import MU_EARTH
from .propagation import propagate_kepler, rv_to_coe


def _refine_minimum(distance, left, right, tol):
    """Golden-section search on one sampled, locally unimodal basin."""
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = distance(c), distance(d)
    while right - left > tol:
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = distance(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = distance(d)
        if not left < c < d < right:
            if right - left > tol:
                raise RuntimeError("Closest-approach tolerance is below floating-point resolution.")
            break
    time = 0.5 * left + 0.5 * right
    return float(time), distance(time)


def closest_approach(
    r1: ArrayLike, v1: ArrayLike, r2: ArrayLike, v2: ArrayLike, window: float,
    step: float = 10.0, tol: float = 1e-3, *, mu: float = MU_EARTH,
) -> dict[str, float]:
    """Return tca [s] and miss_distance [km] over the closed [0, window].

    Both initial states share the same epoch, body-centered inertial frame and
    positive mu [km^3/s^2]. r1/r2 are km; v1/v2 are km/s. Both orbits must be
    nondegenerate ellipses. window [s] is nonnegative; step and tol [s] positive.
    For window=0, return the initial separation. Endpoints and all sampled local
    minima are candidates; exact distance ties select the earliest candidate.

    Each sampled basin is refined to time-bracket width <= tol. Detection is
    limited by step: each bracket must contain one minimum and narrow encounters
    between samples may be missed. This is a sampled search, not an unconditional
    global-minimum guarantee; tol is not a distance-error bound. This point-mass
    model does not check surface clearance or collision probability. Invalid
    inputs raise ValueError; propagation/refinement failures raise RuntimeError.
    """
    times, tol = time_grid(window, step, tol)
    # Validate both states even for a zero-duration search.
    rv_to_coe(r1, v1, mu)
    rv_to_coe(r2, v2, mu)

    def distance(time):
        first, _ = propagate_kepler(r1, v1, time, mu=mu)
        second, _ = propagate_kepler(r2, v2, time, mu=mu)
        return float(np.linalg.norm(first - second))

    distances = [distance(time) for time in times]
    candidates = [(float(time), value) for time, value in zip(times, distances)]
    # Refining only the smallest sample can select the wrong encounter when
    # another, deeper minimum falls farther from a grid point.
    for index in range(len(times)):
        no_larger_than_left = index == 0 or distances[index] <= distances[index - 1]
        no_larger_than_right = index == len(times) - 1 or distances[index] <= distances[index + 1]
        if no_larger_than_left and no_larger_than_right and len(times) > 1:
            left = times[max(index - 1, 0)]
            right = times[min(index + 1, len(times) - 1)]
            candidates.append(_refine_minimum(distance, left, right, tol))
    time, separation = min(candidates, key=lambda candidate: (candidate[1], candidate[0]))
    return {"tca": time, "miss_distance": separation}
