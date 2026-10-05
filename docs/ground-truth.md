# Ground truth — milestones 2–7

Answer-free specifications in `tasks/astrodynamics/specifications.jsonl` supply inputs and contracts.
Expected numerical values are computed by the deterministic backend, at full
floating-point precision. **Re-running the same backend is not the verification.**
`astrotoolbench.benchmark.verification` computes a second answer from Cartesian integration
or independent analytical equations, without consulting stored expected values.
`verify_task` compares both answers and the committed reference against a stricter
per-output scientific error budget.

The generated [accuracy report](reference-verification.json) records every task's
method, observed absolute errors, verification budgets and the dataset SHA-256.
The corpus has 38 successful outcomes and 12 independently reproduced error
outcomes: 33 analytic certificates, 5 Cartesian integration certificates and 12
independent input checks. No model API, credentials, random draws or LLM-generated
numerical reference values enter this process.

## Scientific evidence

| Tool | Nominal / known solution | Limit case | Invalid input | Independent evidence |
| --- | --- | --- | --- | --- |
| Propagation | Circular quarter period; elliptic periapsis to apogee; arbitrary elliptic epoch | Zero/signed duration, full period, inclined and retrograde states | Wrong shape, radial state, unbound state, invalid mu | Closed-form Cartesian circle; fixed-step Cartesian RK4; energy/momentum conservation. |
| ΔV | Orthogonal 3 and 4 km/s velocities require 5 km/s | Identical velocities; full reversal | Nonfinite or malformed vectors | Componentwise Euclidean norm; radial/transverse circularization formula. |
| Hohmann | Raising/lowering; existing textbook LEO–GEO check | Equal radii: zero impulses, half a circular period | Nonpositive radius or mu | Transfer energy and vis-viva, independent of the backend's speed-ratio implementation. |
| Shadow detection | Day/night axis positions | Cylinder tangent and day/night plane are illuminated | Null Sun direction; position inside body | Negative Sun projection and squared perpendicular-distance inequality. |
| Eclipse windows | One/two complete circular periods | Clipped window, zero window, Sun normal to orbit | Negative window, null direction, body-intersecting orbit | Closed-form boundary angles and periodic interval clipping, without sampling. |
| Closest approach | Constructed inclined circular encounters | Initial/final minima, zero window, identical trajectories, constant separation | Negative window or malformed state | Analytic separation-squared sinusoid; all stationary minima and both endpoints. |

The orbital helpers also retain nominal, limiting and invalid-input tests for
period, orbital-element conversions and Kepler's equation. Kepler's equation at
high eccentricity is compared with an independent bracketed bisection.

Tests are in `tests/tools`, `tests/test_core.py` and `tests/benchmark`. The analytic
certificates are deliberately limited to the geometries documented below; a new
family must supply suitable evidence instead of reusing an inapplicable method.

## Equations and derivations

For the same point-mass model as the tools:

- **Circular motion.** With `n=sqrt(mu/r³)`, initial `r0 ⟂ v0` and
  `|v0|²=mu/r`, `r(t)=r0*cos(n*t)+(v0/n)*sin(n*t)` and
  `v(t)=v0*cos(n*t)-n*r0*sin(n*t)`. These are evaluated directly in Cartesian
  coordinates; no Kepler solver or orbital-element conversion is reused.
- **Cartesian integration.** Integrate `r'=v`, `v'=-mu*r/|r|³` with classical
  fourth-order Runge–Kutta. The reference checker uses a maximum step of 2 s;
  step-halving tests at 1 s cover circular and elliptic motion, eccentricities
  through 0.65, positive/negative durations and horizons through 6000 s. The
  corpus's integration cases have `e <= 0.25` and shorter horizons. Both the
  2-versus-1-second difference and the independent-versus-backend difference must
  be within 5e-5 km in position and 5e-8 km/s in velocity. This is validation for
  these regimes, not an unrestricted accuracy guarantee for RK4.
- **Hohmann.** The transfer ellipse's specific energy is
  `epsilon=-mu/(r1+r2)`. Endpoint speeds follow from
  `v_transfer(r)=sqrt(2*(mu/r+epsilon))`, circular speed is `sqrt(mu/r)`,
  and time is `pi/sqrt(mu/((r1+r2)/2)³)`. Differences of the endpoint speeds
  give the impulse magnitudes. Equal radii preserve the half-period convention.
- **Circularization.** At a fixed position, decompose the velocity into radial
  speed `v_r=(r·v)/|r|` and transverse speed `v_t=|r×v|/|r|`. Maintaining the
  plane and direction of motion, `ΔV=sqrt(v_r²+(v_t-sqrt(mu/|r|))²)`. This is
  independent of constructing a target velocity vector and calling `delta_v`.
- **Shadow.** For normalized body-to-Sun direction `s`, a position is shadowed
  iff `r·s < 0` and `r·r-(r·s)² < R²`. The cylinder boundary is illuminated.
- **Circular eclipse intervals.** Let `u=r0/|r0|`, `w=v0/|v0|`. Sun-axis
  projection divided by orbital radius is `A*cos(n*t)+B*sin(n*t)`, where
  `A=u·s`, `B=w·s`. Write it as `C*cos(n*t-phi)`, with `C=hypot(A,B)` and
  `phi=atan2(B,A)`. Define `q=sqrt(1-(R/r)²)`. There is no finite-duration eclipse
  when `C <= q`. Otherwise entry/exit phases are
  `phi+pi ± acos(q/C)`, repeated every `2*pi/n` and clipped to the observation
  window. This certificate can detect eclipses missed by a coarse sampling step.
- **Equal-radius circular proximity.** Using each orbit's basis `(u,w)`, their
  normalized dot product is
  `(a+b)/2 + (a-b)/2*cos(2*n*t) + c/2*sin(2*n*t)`, with
  `a=u1·u2`, `b=w1·w2`, `c=u1·w2+w1·u2`.
  Maximizing that dot product minimizes simultaneous squared separation.
  Candidate times satisfy `2*n*t=atan2(c,a-b)+2*k*pi`; include both observation
  boundaries. A constant separation has no unique minimizing time; that problem
  requests only the distance. Identical trajectories explicitly request t=0.
- **Multi-step arrival.** A tangential Hohmann departure from +x with velocity
  toward +y reaches `[-r2,0,0]` after half the transfer ellipse. The first impulse
  is applied before production propagation. Position and eclipse are checked
  analytically; circularization at arrival does not change either quantity.

## Tolerances and failure policy

Every numeric output stores its unit and scoring tolerance. Independently checked
references use tighter absolute budgets, with no reference rounding. Examples:
position 1e-3 km scoring / 5e-5 km verification; impulse 1e-6 km/s / 1e-10 km/s;
eclipse durations 5e-3 s / 5e-6 s; closest-approach time 1e-3 s / 2e-6 s.
See each record for the full contract. Booleans and error codes are exact,
non-numerical outcomes and therefore have no invented numeric tolerance.

The benchmark context describes **exterior Earth satellites**, so it adds an
explicit surface/periapsis policy on top of the mathematical point-mass tools.
A positive radius below the body surface, or an ellipse that intersects it, gives
`physical_impossibility`. Zero/negative tool scalars, malformed states, unbound
states for the elliptic solver and negative observation windows give
`invalid_input`. Signed negative propagation time remains valid.

A null required quantity gives `missing_input`; incompatible units give
`inconsistent_units`; requesting J2 from the two-body implementation gives
`unsupported_model`. Asking for a guaranteed event-time accuracy tighter than
the configured bracket width gives `unachievable_precision`. Classification does
not depend on the task's expected error field. The independent checker uses
energy/angular-momentum inequalities and direct shape/unit/context checks, not
the backend's validation helpers.

The production eclipse/proximity tools remain sampled searches. Corpus searches
use an 8 s step and 1e-6 s time bracket, with analytically resolved event geometry.
Tests halve both settings and reverify the references. An intentional whole-period
sampling test proves that a missed eclipse cannot become accepted ground truth
merely by regenerating a backend result. New grazing or unequal-radius proximity
families need new certificates and resolution studies before they are admitted.
