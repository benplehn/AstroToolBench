"""Two-body (Keplerian) orbital mechanics, elliptic orbits only."""
import numpy as np
from .constants import MU_EARTH


def orbital_period(a, mu=MU_EARTH):
    """Orbital period [s] of an elliptic orbit with semi-major axis a [km]."""
    if a <= 0:
        raise ValueError("Semi-major axis must be positive for an elliptic orbit.")
    return 2.0 * np.pi * np.sqrt(a**3 / mu)


def solve_kepler(M, e, tol=1e-12, max_iter=50):
    """Solve Kepler's equation M = E - e sin E for the eccentric anomaly E [rad] (Newton)."""
    if not 0.0 <= e < 1.0:
        raise ValueError("solve_kepler only handles elliptic orbits (0 <= e < 1).")
    M = np.mod(M, 2.0 * np.pi)
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        dE = -f / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            return E
    raise RuntimeError("Kepler's equation did not converge.")


def coe_to_rv(a, e, i, raan, argp, nu, mu=MU_EARTH):
    """Classical orbital elements -> ECI position [km] and velocity [km/s]."""
    p = a * (1.0 - e**2)
    r_pf = p / (1.0 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)
    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                  ci],
    ])
    return R @ r_pf, R @ v_pf


def rv_to_coe(r, v, mu=MU_EARTH):
    """ECI state -> (a, e, i, raan, argp, nu). Angles in rad.

    Edge cases: for circular orbits argp is set to 0 and nu is measured from the
    ascending node; for equatorial orbits raan is set to 0.
    """
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    n = np.cross([0.0, 0.0, 1.0], h)
    nn = np.linalg.norm(n)
    e_vec = ((vn**2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)
    energy = vn**2 / 2.0 - mu / rn
    if energy >= 0:
        raise ValueError("State is not on an elliptic orbit (energy >= 0).")
    a = -mu / (2.0 * energy)
    i = np.arccos(np.clip(h[2] / hn, -1.0, 1.0))

    eps = 1e-10
    raan = 0.0 if nn < eps else np.arctan2(n[1], n[0]) % (2 * np.pi)
    if e < eps:  # circular
        argp = 0.0
        ref = n / nn if nn >= eps else np.array([1.0, 0.0, 0.0])
        nu = np.arctan2(np.dot(np.cross(ref, r), h) / hn, np.dot(ref, r)) % (2 * np.pi)
    else:
        ref = n / nn if nn >= eps else np.array([1.0, 0.0, 0.0])
        argp = np.arctan2(np.dot(np.cross(ref, e_vec), h) / hn, np.dot(ref, e_vec)) % (2 * np.pi)
        nu = np.arctan2(np.dot(np.cross(e_vec, r), h) / hn, np.dot(e_vec, r)) % (2 * np.pi)
    return a, e, i, raan, argp, nu


def propagate_kepler(r0, v0, dt, mu=MU_EARTH):
    """Propagate an ECI state by dt [s] on a Keplerian (two-body) elliptic orbit."""
    a, e, i, raan, argp, nu0 = rv_to_coe(r0, v0, mu)
    E0 = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu0 / 2), np.sqrt(1 + e) * np.cos(nu0 / 2))
    M0 = E0 - e * np.sin(E0)
    M = M0 + np.sqrt(mu / a**3) * dt
    E = solve_kepler(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return coe_to_rv(a, e, i, raan, argp, nu, mu)
