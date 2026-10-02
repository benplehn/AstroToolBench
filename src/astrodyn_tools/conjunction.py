"""Closest approach between two objects (two-body propagation)."""
import numpy as np
from .kepler import propagate_kepler


def closest_approach(r1, v1, r2, v2, window, step=10.0, tol=1e-3):
    """Time of closest approach [s] and miss distance [km] within [0, window].

    Coarse scan with `step` [s], then golden-section refinement around the best sample.
    Note: with a too large step, a fast fly-by between two samples can be missed.
    """
    def dist(t):
        ra, _ = propagate_kepler(r1, v1, t)
        rb, _ = propagate_kepler(r2, v2, t)
        return np.linalg.norm(ra - rb)

    times = np.arange(0.0, window + step, step)
    times[-1] = min(times[-1], window)
    d = np.array([dist(t) for t in times])
    k = int(np.argmin(d))
    lo, hi = times[max(k - 1, 0)], times[min(k + 1, len(times) - 1)]

    g = (np.sqrt(5) - 1) / 2
    c, e_ = hi - g * (hi - lo), lo + g * (hi - lo)
    while hi - lo > tol:
        if dist(c) < dist(e_):
            hi = e_
        else:
            lo = c
        c, e_ = hi - g * (hi - lo), lo + g * (hi - lo)
    tca = 0.5 * (lo + hi)
    return {"tca": tca, "miss_distance": dist(tca)}
