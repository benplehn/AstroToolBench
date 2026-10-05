"""Shared input checks for the numerical backend (no unit conversion)."""

import numpy as np


def finite_scalar(value, name):
    """Return a finite real scalar, rejecting arrays, strings and booleans."""
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar.")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar.")
    return result


def positive_scalar(value, name):
    result = finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def nonnegative_scalar(value, name):
    result = finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def vector3(value, name):
    """Return a copy of a finite three-component real vector."""
    array = np.asarray(value)
    if array.shape != (3,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain exactly three finite real components.")
    # NumPy would otherwise silently turn a mixed [True, 0, 0] into integers.
    if any(isinstance(component, (bool, np.bool_)) for component in np.asarray(value, dtype=object)):
        raise ValueError(f"{name} must contain exactly three finite real components.")
    result = array.astype(float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain exactly three finite real components.")
    return result


def unit_direction(value, name):
    vector = vector3(value, name)
    # Scale first so normalizing a large or tiny nonzero direction is safe.
    scale = np.max(np.abs(vector))
    if scale == 0.0:
        raise ValueError(f"{name} must be a nonzero direction vector.")
    vector /= scale
    return vector / np.linalg.norm(vector)


def time_grid(duration, step, tol):
    """Validate a forward search and sample both endpoints without overshoot."""
    duration = nonnegative_scalar(duration, "duration/window [s]")
    step = positive_scalar(step, "step [s]")
    tol = positive_scalar(tol, "tol [s]")
    # Use integer multiples instead of arange(duration + step): the latter can
    # overflow or introduce a repeated endpoint for exact multiples of step.
    times = np.arange(np.ceil(duration / step), dtype=float) * step
    return np.append(times, duration), tol
