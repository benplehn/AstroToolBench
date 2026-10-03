"""Eclipse prediction with a cylindrical Earth-shadow model.

Model assumptions:
    - the Sun direction is fixed during the analysis window,
    - the shadow is a cylinder of radius R_EARTH (no penumbra),
    - two-body propagation.
"""
import numpy as np
from .constants import R_EARTH
from .kepler import propagate_kepler


def in_cylindrical_shadow(r_sat, sun_dir, r_body=R_EARTH):
    """True if the satellite at ECI position r_sat [km] is in the cylindrical shadow."""
    s = np.asarray(sun_dir, float)
    s = s / np.linalg.norm(s)
    r = np.asarray(r_sat, float)
    proj = np.dot(r, s)
    if proj >= 0:          # day side
        return False
    perp = np.linalg.norm(r - proj * s)
    return perp < r_body


def eclipse_windows(r0, v0, sun_dir, duration, step=10.0, tol=1e-3):
    """List of (t_start, t_end) eclipse intervals [s] within [0, duration].

    Coarse scan with `step` [s], then each entry/exit refined by bisection to `tol` [s].
    Intervals cut by the window boundaries start at 0 or end at `duration`.
    """
    def shadow(t):
        r, _ = propagate_kepler(r0, v0, t)
        return in_cylindrical_shadow(r, sun_dir)

    def refine(t_a, t_b, state_a):
        while t_b - t_a > tol:
            t_m = 0.5 * (t_a + t_b)
            if shadow(t_m) == state_a:
                t_a = t_m
            else:
                t_b = t_m
        return 0.5 * (t_a + t_b)

    times = np.arange(0.0, duration + step, step)
    times[-1] = min(times[-1], duration)
    windows, start = [], (0.0 if shadow(0.0) else None)
    prev_t, prev_s = 0.0, shadow(0.0)
    for t in times[1:]:
        s = shadow(t)
        if s != prev_s:
            t_cross = refine(prev_t, t, prev_s)
            if s:
                start = t_cross
            else:
                windows.append((start, t_cross))
                start = None
        prev_t, prev_s = t, s
    if start is not None:
        windows.append((start, duration))
    return windows
