# Scientific tools — milestone 1

The deterministic scientific backend lives in `src/astrotoolbench/tools`. Importing and
using it requires NumPy only: no credentials, model provider or network access.
This milestone stabilizes the five core operations and their contracts. Dataset
expansion, task-schema changes, reference generation and release work belong to
later milestones. Existing package names, positional parameters and return keys
are preserved for the prototype's callers.

## Shared conventions

- Distances and orbit radii: **km**, measured from the central body's center.
  Altitudes must be explicitly converted by the caller.
- Velocities and delta-v: **km/s**; time: **s**; angles: **rad**.
- Gravitational parameter `mu`: **km³/s²**, finite and strictly positive.
- States: three-component Cartesian vectors in one **right-handed,
  body-centered inertial frame**, with a common epoch. The Earth default is ECI.
- `MU_EARTH = 398600.4418 km³/s²`, `R_EARTH = 6378.137 km`.
  These constants are explicit defaults; another body can be supplied through
  `mu` and, for shadow calculations, `r_body`.
- Numerical scalars and vectors must be real and finite. Boolean values, numeric
  strings, NaN, infinity and wrong-shaped vectors are rejected with `ValueError`.
  Inputs are not mutated. Outputs use float64 computation with no rounding.
- Units, coordinate frames and epochs are caller contracts. Bare numerical arrays
  cannot reveal meters passed as kilometers or velocities in different frames.
  There is no automatic conversion or inference of units.
- Motion is an unperturbed **elliptic two-body orbit**. Atmospheric drag, J2,
  third-body gravity, radiation pressure and finite-duration burns are omitted.
  Parabolic, hyperbolic and radial states are unsupported. Near-radial states
  with `|r × v| / (|r| |v|) <= 1e-12` are also rejected.

The point-mass orbital tools do not assume a body radius and therefore do not
reject mathematical orbits below Earth's surface. Shadow calculations require
exterior positions and a periapsis at least equal to the supplied body radius.
Use surface checks separately when constructing physically valid tasks.

## Core operations

| Operation | Inputs | Outputs | Domain and conventions |
| --- | --- | --- | --- |
| `propagate_kepler(r0, v0, dt, mu=MU_EARTH)` | Position `[x,y,z]` km; velocity `[vx,vy,vz]` km/s; signed elapsed time s; mu km³/s² | Tuple of new position and velocity arrays, each shape `(3,)` | Nondegenerate elliptic state; negative time propagates backward; zero time returns exact copies after validation. |
| `delta_v(v_initial, v_final)` | Two velocities km/s in the same frame at the same position and instant | Nonnegative scalar km/s | Instantaneous impulse; Euclidean norm of the vector difference, not the difference of speed magnitudes. Signed impulse is `v_final - v_initial`. |
| `hohmann_transfer(r1, r2, mu=MU_EARTH)` | Positive departure/arrival **center radii** km; mu km³/s² | Dict: `dv1`, `dv2`, `dv_total` km/s; `tof` s | Coplanar circular orbits, two tangential impulses, half an ellipse. Raising and lowering supported; delta-v values are magnitudes. Equal radii give zero impulses and **half a circular period**. No phasing or plane change. |
| `in_cylindrical_shadow(r_sat, sun_dir, r_body=R_EARTH)` | Exterior position km; dimensionless body-to-Sun direction; positive body radius km | Python `bool` | Direction normalized internally. Shadow requires a negative Sun-axis projection and perpendicular distance strictly less than body radius. Cylinder boundary and day/night plane are illuminated. |
| `eclipse_windows(r0, v0, sun_dir, duration, step=10, tol=1e-3, *, mu=MU_EARTH, r_body=R_EARTH)` | Elliptic state; fixed Sun direction; nonnegative duration s; positive sampling step and transition tolerance s | List of `(start_s, end_s)` tuples within `[0, duration]` | Cylindrical shadow with parallel rays and no penumbra. Body-intersecting orbits rejected. Intervals are clipped to observation boundaries. Zero duration gives `[]`. Total duration is the sum of interval lengths. |
| `closest_approach(r1, v1, r2, v2, window, step=10, tol=1e-3, *, mu=MU_EARTH)` | Two elliptic states at the same epoch; shared mu; nonnegative window s; positive step and tolerance s | Dict: `tca` s after the initial epoch; `miss_distance` km | Minimum of **simultaneous** separation in the closed time interval, not geometric distance between arbitrary points on the two orbital paths. Zero window gives initial separation; endpoints are retained. Exact distance ties choose the earliest candidate. |

All core operations reject invalid inputs with `ValueError`. Propagation raises
`RuntimeError` if the Kepler solver fails to converge; the two time searches can
also raise `RuntimeError` when the requested tolerance is below floating-point
time resolution. Like other float64 scientific routines, these tools are intended
for representable orbital scales; values near floating-point overflow and
ill-conditioned limiting orbits are outside the validated numerical regime.

## Sampling accuracy is part of the contract

The eclipse search samples both observation boundaries, detects changes in the
shadow predicate, then bisects each detected transition. Its final transition
bracket has width at most `tol` seconds. An entire eclipse or illuminated gap
between samples can be missed, particularly near a grazing geometry. Choose
`step` smaller than every eclipse and illuminated gap relevant to the task.

The proximity search retains all sampled positions and both endpoints. It
refines **every sampled local minimum**, including boundary basins, with a
golden-section search. Each refinement assumes its bracket contains a single
minimum and stops at time-bracket width at most `tol` seconds. A narrow encounter
can be missed, and multiple minima within one bracket violate that assumption.

Neither sampled search has an unconditional global guarantee for arbitrary
steps. `tol` bounds a time bracket, not detection completeness or miss-distance
error. Benchmark references should use analytically checkable configurations and
steps resolving the events; reduce the step and check stability when using new
configurations. A smaller tolerance cannot recover an undetected event.

## Orbital helpers

`orbital_period(a, mu=MU_EARTH)` returns `2*pi*sqrt(a³/mu)` in seconds for a
positive semi-major axis in km. A semi-major axis is not an altitude.

`solve_kepler(M, e, tol=1e-12, max_iter=100)` solves `M = E - e*sin(E)` in radians
for `0 <= e < 1`, reducing finite mean anomaly modulo `2*pi`. It returns eccentric
anomaly in `[0, 2*pi)`. The tolerance is in eccentric anomaly radians. Newton steps
are safeguarded by a bracket; exhaustion raises `RuntimeError`. `max_iter` must
be a positive integer.

`coe_to_rv(a, e, i, raan, argp, nu, mu=MU_EARTH)` converts classical elements to
inertial position and velocity. `a > 0`, `0 <= e < 1`, and `0 <= i <= pi`.
RAAN, argument of periapsis and true anomaly are finite periodic angles. The
perifocal-to-inertial rotation is `Rz(raan) Rx(i) Rz(argp)`.

`rv_to_coe(r, v, mu=MU_EARTH)` returns `(a, e, i, raan, argp, nu)`. Inclination is
in `[0, pi]`; other output angles are in `[0, 2*pi)`. For circular states
(`e <= 1e-12`), `argp=0` and `nu` is argument of latitude. For equatorial states
(`sin(i) <= 1e-12`), `raan=0` and angular coordinates use the inertial +x axis in
the direction of motion. This convention also covers retrograde equatorial
orbits. These limiting conventions are numerical approximations at that threshold.

## Offline example

```python
import numpy as np
from astrotoolbench.tools import (
    MU_EARTH, closest_approach, delta_v, eclipse_windows,
    hohmann_transfer, propagate_kepler,
)

position_km = [7000.0, 0.0, 0.0]
velocity_km_s = [0.0, np.sqrt(MU_EARTH / 7000.0), 0.0]

position, velocity = propagate_kepler(position_km, velocity_km_s, 600.0)
impulse_km_s = delta_v(velocity_km_s, [0.0, 8.0, 0.0])
transfer = hohmann_transfer(7000.0, 9000.0)
windows = eclipse_windows(position_km, velocity_km_s, [1.0, 0.0, 0.0], 6000.0)
total_eclipse_s = sum(end - start for start, end in windows)
encounter = closest_approach(
    position_km, velocity_km_s, position, velocity, 600.0,
)
```

## Verification

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/tools tests/test_core.py -q
python -m pytest -q
```

The scientific checks use analytic circular motion, independent bisection of
Kepler's equation at high eccentricity, conserved energy and angular momentum,
orbital-element round trips including retrograde/equatorial states, textbook
Hohmann values, vector impulse geometry, analytic eclipse entry/exit times and
constructed simultaneous encounters. Boundary and invalid-input checks cover
zero durations, exact interval endpoints, custom mu/body radius and malformed
inputs. No LLM supplies these expected results.

Lambert is deferred; it is optional in milestone 1 and needs a separate solver
contract and validation effort.
