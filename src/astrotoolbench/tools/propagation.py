"""Deterministic two-body mechanics for nondegenerate elliptic orbits.

Distances are km, velocities km/s, times s, angles rad and mu km^3/s^2.
States use one right-handed body-centered inertial frame (ECI for Earth).
The central body is a point mass: surface clearance is not checked here.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import finite_scalar, positive_scalar, vector3
from .constants import MU_EARTH

_TWO_PI = 2.0 * np.pi
_SINGULARITY_TOL = 1e-12


def _elliptic_elements(a, e, mu):
    a = positive_scalar(a, "a [km]")
    e = finite_scalar(e, "e")
    mu = positive_scalar(mu, "mu [km^3/s^2]")
    if not 0.0 <= e < 1.0:
        raise ValueError("Only elliptic orbits are supported (0 <= e < 1).")
    return a, e, mu


def orbital_period(a: float, mu: float = MU_EARTH) -> float:
    """Return the elliptic orbital period [s] for semi-major axis a [km].

    a is a center distance, not an altitude. Both a and mu must be finite and
    strictly positive; invalid inputs raise ValueError. No body radius is assumed.
    """
    a = positive_scalar(a, "a [km]")
    mu = positive_scalar(mu, "mu [km^3/s^2]")
    return float(_TWO_PI * a * np.sqrt(a / mu))


def solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    """Solve M = E - e sin(E), returning E in [0, 2*pi) [rad].

    Finite M is reduced modulo 2*pi; eccentricity must satisfy 0 <= e < 1.
    tol is the eccentric-anomaly accuracy [rad], not a time or residual tolerance.
    max_iter is a positive integer. Invalid inputs raise ValueError; exhaustion
    raises RuntimeError. A bracket safeguards Newton steps near e = 1.
    """
    M = finite_scalar(M, "M [rad]") % _TWO_PI
    e = finite_scalar(e, "e")
    tol = positive_scalar(tol, "tol [rad]")
    if not 0.0 <= e < 1.0:
        raise ValueError("solve_kepler only handles elliptic orbits (0 <= e < 1).")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if M == 0.0 or e == 0.0:
        return float(M)

    lo, hi = 0.0, _TWO_PI
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        residual = E - e * np.sin(E) - M
        derivative = 1.0 - e * np.cos(E)
        correction = residual / derivative
        if abs(correction) <= tol:
            return float(E)
        if residual > 0.0:
            hi = E
        else:
            lo = E
        candidate = E - correction
        E = candidate if lo < candidate < hi else 0.5 * (lo + hi)
        if hi - lo <= tol:
            return float(0.5 * (lo + hi))
    raise RuntimeError("Kepler's equation did not converge within max_iter.")


def coe_to_rv(
    a: float, e: float, i: float, raan: float, argp: float, nu: float,
    mu: float = MU_EARTH,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert (a, e, i, RAAN, argument of periapsis, true anomaly) to (r, v).

    a [km] > 0, 0 <= e < 1, 0 <= i <= pi; all angles are finite [rad].
    RAAN, argp and nu are periodic and may be outside [0, 2*pi). Rotation is
    Rz(RAAN) Rx(i) Rz(argp) from perifocal to inertial coordinates. Returned
    arrays have shape (3,), in km and km/s. Invalid inputs raise ValueError.
    """
    a, e, mu = _elliptic_elements(a, e, mu)
    i = finite_scalar(i, "i [rad]")
    if not 0.0 <= i <= np.pi:
        raise ValueError("Inclination i must be in [0, pi] rad.")
    raan = finite_scalar(raan, "raan [rad]") % _TWO_PI
    argp = finite_scalar(argp, "argp [rad]") % _TWO_PI
    nu = finite_scalar(nu, "nu [rad]") % _TWO_PI
    p = a * (1.0 - e) * (1.0 + e)
    r_pf = p / (1.0 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)
    rotation = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return rotation @ r_pf, rotation @ v_pf


def rv_to_coe(r: ArrayLike, v: ArrayLike, mu: float = MU_EARTH) -> tuple[float, ...]:
    """Convert an inertial state to (a, e, i, raan, argp, nu), km and rad.

    r and v are finite vectors of shape (3,); mu is finite and positive.
    Zero position, radial/near-radial states and unbound states raise ValueError.
    Circular (e <= 1e-12): argp = 0, nu is argument of latitude. Equatorial
    (sin(i) <= 1e-12): raan = 0, angles use the +x axis and direction of motion,
    including retrograde orbits. Other output angles are in [0, 2*pi).
    """
    r, v = vector3(r, "r [km]"), vector3(v, "v [km/s]")
    mu = positive_scalar(mu, "mu [km^3/s^2]")
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    if rn == 0.0 or vn == 0.0:
        raise ValueError("An elliptic state requires nonzero position and velocity.")
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    if hn <= _SINGULARITY_TOL * rn * vn:
        raise ValueError("Radial or near-radial states do not define an orbital plane.")
    n = np.cross([0.0, 0.0, 1.0], h)
    nn = np.linalg.norm(n)
    e_vec = ((vn**2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = float(np.linalg.norm(e_vec))
    energy = vn**2 / 2.0 - mu / rn
    if not np.isfinite(energy) or energy >= 0.0 or not e < 1.0:
        raise ValueError("State is not on a supported elliptic orbit (energy must be < 0, e < 1).")
    a = -mu / (2.0 * energy)
    i = np.arctan2(np.linalg.norm(h[:2]), h[2])

    equatorial = nn <= _SINGULARITY_TOL * hn
    raan = 0.0 if equatorial else np.arctan2(n[1], n[0]) % _TWO_PI
    ref = np.array([1.0, 0.0, 0.0]) if equatorial else n / nn
    if e <= _SINGULARITY_TOL:
        argp = 0.0
        nu = np.arctan2(np.dot(np.cross(ref, r), h) / hn, np.dot(ref, r)) % _TWO_PI
    else:
        argp = np.arctan2(np.dot(np.cross(ref, e_vec), h) / hn, np.dot(ref, e_vec)) % _TWO_PI
        nu = np.arctan2(np.dot(np.cross(e_vec, r), h) / hn, np.dot(e_vec, r)) % _TWO_PI
    return tuple(float(value) for value in (a, e, i, raan, argp, nu))


def propagate_kepler(
    r0: ArrayLike, v0: ArrayLike, dt: float, mu: float = MU_EARTH,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagate an elliptic inertial state by a finite signed duration dt [s].

    Return (position [km], velocity [km/s]) as new shape-(3,) arrays. dt = 0
    returns exact copies after validating the state; negative dt propagates
    backward. mu [km^3/s^2] defaults explicitly to MU_EARTH. This point-mass
    model omits perturbations and surface collisions. Input errors raise
    ValueError; failure of the Kepler solver raises RuntimeError.
    """
    dt = finite_scalar(dt, "dt [s]")
    a, e, i, raan, argp, nu0 = rv_to_coe(r0, v0, mu)
    if dt == 0.0:
        return vector3(r0, "r0 [km]"), vector3(v0, "v0 [km/s]")
    E0 = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu0 / 2), np.sqrt(1 + e) * np.cos(nu0 / 2))
    M0 = E0 - e * np.sin(E0)
    # Reducing time before computing phase limits precision loss over many laps.
    period = orbital_period(a, mu)
    M = M0 + _TWO_PI * (np.fmod(dt, period) / period)
    E = solve_kepler(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return coe_to_rv(a, e, i, raan, argp, nu, mu)
