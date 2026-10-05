"""Eclipse detection in a fixed cylindrical shadow, without penumbra.

The Sun direction is fixed in the body's inertial frame throughout the window.
The body is spherical, rays parallel, and motion elliptic and Keplerian.
"""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import positive_scalar, time_grid, unit_direction, vector3
from .constants import MU_EARTH, R_EARTH
from .propagation import propagate_kepler, rv_to_coe


def _in_shadow(position, sun_unit, body_radius):
    projection = float(np.dot(position, sun_unit))
    if projection >= 0.0:
        return False
    perpendicular = np.linalg.norm(position - projection * sun_unit)
    return bool(perpendicular < body_radius)


def in_cylindrical_shadow(
    r_sat: ArrayLike, sun_dir: ArrayLike, r_body: float = R_EARTH,
) -> bool:
    """Return whether an exterior position r_sat [km] is inside the shadow.

    Vectors have shape (3,) in the same body-centered inertial frame. sun_dir
    points FROM body TO Sun, is dimensionless, nonzero and normalized internally.
    r_body [km] is a finite positive spherical radius. The day plane and cylinder
    boundary are illuminated (strict inequalities). Positions inside the body,
    invalid vectors and invalid radii raise ValueError.
    """
    position = vector3(r_sat, "r_sat [km]")
    sun_unit = unit_direction(sun_dir, "sun_dir")
    body_radius = positive_scalar(r_body, "r_body [km]")
    if np.linalg.norm(position) < body_radius:
        raise ValueError("r_sat must be on or outside the body surface.")
    return _in_shadow(position, sun_unit, body_radius)


def eclipse_windows(
    r0: ArrayLike, v0: ArrayLike, sun_dir: ArrayLike, duration: float,
    step: float = 10.0, tol: float = 1e-3, *,
    mu: float = MU_EARTH, r_body: float = R_EARTH,
) -> list[tuple[float, float]]:
    """Return eclipse intervals (start, end) [s] within [0, duration].

    r0 [km], v0 [km/s] define an elliptic inertial state at t=0. sun_dir is a
    nonzero body-to-Sun direction in that frame. mu [km^3/s^2] and r_body [km]
    are positive, defaulting to Earth. The full orbit must clear the body.
    duration [s] is nonnegative; step and tol [s] are strictly positive. A zero
    window returns []. Boundary-clipped intervals start at 0 or end at duration;
    total eclipse duration is sum(end - start for start, end in intervals).

    Sampling detects changes of shadow state; bisection brackets each detected
    transition to width <= tol. step is the detection resolution: an eclipse or
    illuminated gap entirely between samples can be missed, especially near a
    grazing geometry. Use a step smaller than every relevant event/gap; tol alone
    does not guarantee detection. Invalid inputs, a body-intersecting orbit or
    unsupported orbit raise ValueError; propagation may raise RuntimeError.
    """
    times, tol = time_grid(duration, step, tol)
    sun_unit = unit_direction(sun_dir, "sun_dir")
    body_radius = positive_scalar(r_body, "r_body [km]")
    a, e, *_ = rv_to_coe(r0, v0, mu)
    if a * (1.0 - e) < body_radius:
        raise ValueError("The orbit periapsis must be on or outside the body surface.")
    if duration == 0.0:
        return []

    def shadow(time):
        position, _ = propagate_kepler(r0, v0, time, mu=mu)
        return _in_shadow(position, sun_unit, body_radius)

    def refine(left, right, left_state):
        while right - left > tol:
            middle = 0.5 * left + 0.5 * right
            if middle == left or middle == right:
                raise RuntimeError("Eclipse time tolerance is below floating-point resolution.")
            if shadow(middle) == left_state:
                left = middle
            else:
                right = middle
        return 0.5 * left + 0.5 * right

    previous_time = 0.0
    previous_state = shadow(0.0)
    start = 0.0 if previous_state else None
    windows = []
    for time in times[1:]:
        state = shadow(time)
        if state != previous_state:
            crossing = refine(previous_time, time, previous_state)
            if state:
                start = crossing
            else:
                windows.append((float(start), float(crossing)))
                start = None
        previous_time, previous_state = time, state
    if start is not None:
        windows.append((float(start), float(duration)))
    return windows
