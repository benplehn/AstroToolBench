# Problem families

50 tasks in 6 categories. I tried to make each one a different physical or
diagnostic situation, not the same prompt with another radius. The 22 `family`
labels describe the problem structure; `category` is just the broad topic.

| Category | Count | Situations |
| --- | ---: | --- |
| A / propagation | 8 | quarter and full period, backward and zero time, periapsis → apoapsis, arbitrary elliptic time, comparing two propagated states, unbound state |
| B / maneuvers | 9 | vector turn, reversal, Hohmann up and down, comparing transfers, circularization at periapsis/apoapsis, mixed velocity units, equal-radius transfer |
| C / eclipse | 8 | lit/shadowed points, one and two periods, clipped and zero window, Sun normal to the orbit, null Sun direction |
| D / proximity | 8 | distance at a given time, min inside/at the edge of the window, constant distance, zero window, identical orbits, invalid window |
| E / multistep | 8 | transfer time → circular/elliptic propagation, transfer time → eclipse, real departure burn → arrival position and shadow |
| F / diagnostic | 9 | missing velocity, J2 requested, meters given as km, ellipse hitting the Earth, negative duration, impossible time precision, transfer below the surface, μ = 0, valid zero ΔV |

Difficulty: **20 simple, 12 multistep, 8 diagnostic, 10 trap** (40/24/16/20 %).
Outcomes: **38 answers, 12 expected errors**. A diagnostic task can still have a
numerical answer (e.g. a zero-length observation), and backward propagation is
included as a valid case: negative time isn't automatically wrong.

The definitions are in `astrotoolbench.benchmark.families.problem_definitions`.
Inputs are built from simple Cartesian geometry and orbital relations.
`build_tasks` computes the references with the tools and refuses to write a task
that has no independent check. Some tasks also check intermediate results, each
with its own unit and tolerance.

In category E, the first six tasks use the transfer only to get a duration: the
initial orbit is propagated **without** the Hohmann burns. The last two actually
apply the departure burn and follow the transfer ellipse to check the arrival
shadow. The prompts say which one it is.

Split: **24 train / 6 validation / 20 test**, 11 family groups, no overlap. Tools
are shared between splits on purpose. See [data splits](data-splits.md).

To rebuild and check:

```bash
python -m astrotoolbench.benchmark.build
python -m astrotoolbench.generate_references --check
python -m pytest tests/tools tests/test_core.py tests/benchmark -q
```

This regenerates task sources, schemas, split manifest and the verification
report. A test pins a digest of the reference values, so any change to an
answer shows up and has to be updated on purpose.
