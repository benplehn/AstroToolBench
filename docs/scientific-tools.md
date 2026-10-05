# Scientific tools

The tools live in `src/astrotoolbench/tools` and only need NumPy. No network,
no API key. Function names, argument order and return keys haven't changed
since the first prototype.

## Conventions

- Distances and radii in **km**, from the center of the body. Altitudes must be
  converted by the caller.
- Velocities and ΔV in **km/s**, time in **s**, angles in **rad**.
- `mu` in **km³/s²**, finite and > 0.
- State vectors are 3D Cartesian, in a right-handed inertial frame centered on
  the body, at the same epoch (ECI for Earth).
- `MU_EARTH = 398600.4418 km³/s²`, `R_EARTH = 6378.137 km`. For another body,
  pass `mu` (and `r_body` for shadows).
- Inputs must be real finite numbers. Booleans, numeric strings, NaN, inf and
  wrong-shaped vectors raise `ValueError`. Inputs aren't modified and outputs
  aren't rounded.
- There's no unit detection: if you pass meters as kilometers, the tools can't
  tell.
- Unperturbed **elliptic two-body** motion only: no drag, J2, third body, solar
  pressure or finite burns. Parabolic, hyperbolic and (near-)radial states are
  rejected (`|r × v| / (|r| |v|) <= 1e-12`).

The orbit tools don't know the Earth's radius, so they won't complain about an
orbit below the surface. The shadow tools do require the satellite (and the
periapsis) to be outside the body.

## Main functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `propagate_kepler(r0, v0, dt, mu=MU_EARTH)` | position km, velocity km/s, time s | new `(r, v)`, shape `(3,)` each | negative `dt` goes backward; `dt=0` returns copies |
| `delta_v(v_initial, v_final)` | two velocities, same frame/point/time | scalar km/s ≥ 0 | norm of the vector difference, not difference of speeds |
| `hohmann_transfer(r1, r2, mu=MU_EARTH)` | **center radii** km | dict `dv1`, `dv2`, `dv_total` (km/s), `tof` (s) | coplanar circular orbits, up or down; equal radii → 0 ΔV and half a period |
| `in_cylindrical_shadow(r_sat, sun_dir, r_body=R_EARTH)` | position km, Sun direction | `bool` | Sun direction is normalized; the cylinder edge counts as lit |
| `eclipse_windows(r0, v0, sun_dir, duration, step=10, tol=1e-3, *, mu, r_body)` | state, Sun direction, duration ≥ 0, step and tolerance s | list of `(start_s, end_s)` | cylindrical shadow, no penumbra; clipped to `[0, duration]` |
| `closest_approach(r1, v1, r2, v2, window, step=10, tol=1e-3, *, mu)` | two states at the same epoch, window ≥ 0 | dict `tca` (s), `miss_distance` (km) | minimum of the distance **at the same time**, endpoints included; ties → earliest |

Bad inputs raise `ValueError`. Propagation raises `RuntimeError` if Kepler's
equation doesn't converge, and the two searches raise it if `tol` is below what
float64 time can resolve. Extreme values (close to overflow, nearly degenerate
orbits) aren't validated.

## Sampling

`eclipse_windows` samples the trajectory, finds where the shadow test flips, and
bisects each change down to `tol` seconds. An eclipse (or a lit gap) shorter than
`step` can be missed entirely, especially in grazing geometries, so pick `step`
smaller than the shortest event you care about.

`closest_approach` samples too, then refines every local minimum (including at
the boundaries) with a golden-section search down to `tol`. Very short
encounters can be missed, and two minima in the same bracket break the
assumption.

So neither search is guaranteed to find everything. `tol` only controls the
precision of a time once found. For benchmark tasks I use geometries that can be
checked analytically; for new cases, reduce the step and check the result doesn't
move.

## Helpers

- `orbital_period(a, mu=MU_EARTH)`: `2π√(a³/μ)` in s. `a` is the semi-major
  axis, not an altitude.
- `solve_kepler(M, e, tol=1e-12, max_iter=100)`: solves `M = E − e·sin(E)`,
  `0 ≤ e < 1`, returns `E` in `[0, 2π)`. Newton with a bracket fallback for
  high eccentricity.
- `coe_to_rv(a, e, i, raan, argp, nu, mu=MU_EARTH)`: orbital elements to
  `(r, v)`, rotation `Rz(raan) Rx(i) Rz(argp)`.
- `rv_to_coe(r, v, mu=MU_EARTH)`: the reverse. Circular orbits (`e <= 1e-12`)
  get `argp=0` and `nu` = argument of latitude; equatorial orbits
  (`sin(i) <= 1e-12`) get `raan=0` and angles measured from +x.

## Example

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

## Tests

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/tools tests/test_core.py -q
```

The tests compare against things I can compute independently: circular motion
in closed form, a bisection Kepler solver at high eccentricity, energy and
angular momentum conservation, element round trips (including retrograde and
equatorial), textbook Hohmann values, analytic eclipse entry/exit times and
constructed encounters. Plus edge cases (zero durations, interval boundaries,
custom μ/radius) and bad inputs.

Lambert isn't implemented yet; it needs its own solver and tests.
