# Ground truth

Tasks start as specifications without answers (`tasks/astrodynamics/specifications.jsonl`).
The tools compute the expected values at full float64 precision. But running the
same code twice proves nothing, so `astrotoolbench.benchmark.verification`
computes a **second answer independently** (closed-form equations or Cartesian
integration), without looking at stored values. `verify_task` then checks that
the tool result, the independent result and the committed reference all agree
within a tighter error budget than the scoring tolerance.

The [verification report](reference-verification.json) lists, for each task, the
method, the observed errors, the budgets and the dataset SHA-256.

38 tasks have a numerical/boolean answer and 12 have an expected error. Methods:
33 analytic, 5 RK4, 12 input checks. No API, no randomness, and no reference
number comes from an LLM.

## Evidence per tool

| Tool | Normal case | Edge cases | Bad input | Independent check |
| --- | --- | --- | --- | --- |
| Propagation | quarter circular period, periapsis → apoapsis, arbitrary elliptic time | zero/negative time, full period, inclined, retrograde | wrong shape, radial, unbound, bad μ | closed-form circle, RK4, energy/momentum conservation |
| ΔV | 3 and 4 km/s orthogonal → 5 km/s | same velocity, full reversal | non-finite/malformed vectors | component-wise norm, radial/transverse circularization formula |
| Hohmann | up/down, textbook LEO–GEO | equal radii: 0 ΔV, half period | radius or μ ≤ 0 | transfer energy + vis-viva (not the same formula as the tool) |
| Shadow | points on the Sun axis | cylinder edge and terminator plane are lit | null Sun direction, point inside Earth | sign of Sun projection + perpendicular distance |
| Eclipse windows | one and two circular periods | clipped window, zero window, Sun normal to orbit | negative window, null direction, orbit hits Earth | closed-form entry/exit angles, no sampling |
| Closest approach | inclined circular encounters | min at start/end, zero window, identical orbits, constant distance | negative window, bad state | analytic distance² sinusoid, all minima + endpoints |

Period, orbital elements and Kepler's equation also have their own tests
(Kepler at high eccentricity is compared with a plain bisection solver).

Tests: `tests/tools`, `tests/test_core.py`, `tests/benchmark`. The analytic checks
only cover the geometries below; a new family needs its own check.

## Equations

Same point-mass model as the tools:

- **Circular motion.** `n=sqrt(mu/r³)`, `r0 ⟂ v0`, `|v0|²=mu/r`:
  `r(t)=r0*cos(n*t)+(v0/n)*sin(n*t)`, `v(t)=v0*cos(n*t)-n*r0*sin(n*t)`.
  Computed directly in Cartesian, no Kepler solver.
- **RK4.** Integrate `r'=v`, `v'=-mu*r/|r|³` with max step 2 s. I tested step
  halving to 1 s on circular and elliptic orbits (e up to 0.65, ±durations up to
  6000 s); the RK4 tasks in the dataset have `e <= 0.25` and shorter durations.
  2 s vs 1 s and RK4 vs tool must both agree within 5e-5 km / 5e-8 km/s. That's
  validated for these cases only.
- **Hohmann.** Transfer energy `epsilon=-mu/(r1+r2)`, speed
  `v_transfer(r)=sqrt(2*(mu/r+epsilon))`, circular speed `sqrt(mu/r)`, time
  `pi/sqrt(mu/((r1+r2)/2)³)`. ΔV = differences of speeds.
- **Circularization.** Split velocity into radial `v_r=(r·v)/|r|` and transverse
  `v_t=|r×v|/|r|`: `ΔV=sqrt(v_r²+(v_t-sqrt(mu/|r|))²)`. Doesn't go through
  `delta_v`.
- **Shadow.** With unit Sun direction `s`: shadow iff `r·s < 0` and
  `r·r-(r·s)² < R²`. Edge counts as lit.
- **Circular eclipses.** With `u=r0/|r0|`, `w=v0/|v0|`, `A=u·s`, `B=w·s`, the
  normalized Sun projection is `C*cos(n*t-phi)` with `C=hypot(A,B)`,
  `phi=atan2(B,A)`. Let `q=sqrt(1-(R/r)²)`. No eclipse if `C <= q`, otherwise
  entry/exit at `phi+pi ± acos(q/C)`, every `2*pi/n`, clipped to the window.
  This catches eclipses a coarse sampling step would miss.
- **Equal-radius proximity.** The dot product of the two unit positions is
  `(a+b)/2 + (a-b)/2*cos(2*n*t) + c/2*sin(2*n*t)` with `a=u1·u2`, `b=w1·w2`,
  `c=u1·w2+w1·u2`. Max dot product = min distance. Candidates:
  `2*n*t=atan2(c,a-b)+2*k*pi`, plus both endpoints. If the distance is constant,
  the task only asks for the distance; identical orbits ask for t=0.
- **Departure → arrival.** A Hohmann departure from +x moving toward +y arrives at
  `[-r2,0,0]` after half the ellipse. The first burn is applied before
  propagating; position and shadow are checked analytically (circularizing at
  arrival doesn't change them).

## Tolerances and errors

Each numerical output has a unit and a scoring tolerance; the verification
budget is tighter, and references aren't rounded. Some examples (scoring /
verification): position 1e-3 / 5e-5 km, ΔV 1e-6 / 1e-10 km/s, eclipse duration
5e-3 / 5e-6 s, closest-approach time 1e-3 / 2e-6 s. Booleans and error codes are
exact.

The tasks are about **satellites outside the Earth**, so on top of the
point-mass tools there's a surface rule: a radius below the surface or an orbit
crossing it → `physical_impossibility`. Zero/negative scalars, malformed states,
unbound states, negative windows → `invalid_input`. Negative propagation time is
fine.

Also: missing value → `missing_input`, mixed units → `inconsistent_units`,
asking for J2 → `unsupported_model`, asking for a time precision finer than the
search step allows → `unachievable_precision`. The error is decided from the
inputs, not from the expected answer, and the independent checker uses its own
checks (energy/momentum, shapes, units), not the tools' validation code.

Eclipse and proximity searches are still sampled. Dataset tasks use an 8 s step
and 1e-6 s bracket, on geometries solved analytically. Tests halve both and
re-verify. One test deliberately uses a full-period step to show that a missed
eclipse can't sneak into the references just by regenerating. Grazing eclipses
or unequal-radius encounters would need new checks first.
