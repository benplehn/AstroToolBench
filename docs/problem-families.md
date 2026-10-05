# Problem families — milestones 4–6

The official corpus contains **50 tasks**, covering all six requested authoring
groups. Milestone 5 adds equal-radius Hohmann and zero-impulse limits. These are
authored physical/diagnostic situations rather than radius-only variations of one
prompt. The 22 `family` labels describe problem structure and
are distinct from the six `category` groups.

| Group / category | Count | Situations |
| --- | ---: | --- |
| A / propagation | 8 | Quarter/full period, backward/zero propagation, periapsis-to-apogee, arbitrary elliptic epoch, comparison of propagated states, unbound state. |
| B / maneuvers | 9 | Vector turn, reversal, raising/lowering Hohmann, comparison of transfers, periapsis/apogee circularization, incompatible velocity units, equal-radius transfer. |
| C / eclipse | 8 | Illuminated/shadowed positions, one/two periods, clipped/zero window, Sun normal to orbital plane, null solar direction. |
| D / proximity | 8 | Separation at a specified epoch, interior/boundary minimum, constant separation, zero window, identical trajectories, invalid interval. |
| E / multistep | 8 | Transfer time then circular/elliptic propagation; rising/lowering transfer time then eclipse window; actual transfer departure impulse, arrival position and shadow. |
| F / diagnostic | 9 | Missing velocity, unsupported J2, meters passed as kilometers, body-intersecting ellipse, negative observation duration, incompatible time accuracy, underground transfer, zero mu, valid zero impulse. |

Difficulty distribution: **20 simple, 12 multistep, 8 diagnostic, 10 trap** (40%, 24%, 16%, 20%).
Outcome distribution: **38 successes and 12 justified errors**. A diagnostic can
have a numerical result (for example a zero-duration observation), and valid
backward propagation is included rather than incorrectly treating every negative
time as physically invalid.

Families are defined in `astrotoolbench.benchmark.families.problem_definitions`. Inputs are
constructed by direct Cartesian geometry and elementary orbital relations.
`build_tasks` computes numerical references from the scientific tools and refuses
to emit a record without an independent certificate. Selected intermediate and
final results have their own units and tolerances; all requested results must pass.

The eight category E problems contain dependent calculations. For the first six,
the stated initial orbit is propagated/observed **without applying the Hohmann
impulses**: the transfer supplies the analysis duration. The last two apply the
departure impulse and propagate the transfer ellipse before checking arrival
shadow. Their prompts make that physical distinction explicit.

The corpus now has **24 train, 6 validation and 20 test** tasks. The 22 labels
are grouped into 11 reviewed equivalence/composition groups, with zero local
cross-partition family, successful-recipe or duplicate-problem overlaps. Primitive
tool reuse in held-out compositions is intentional. See
[family split policy](data-splits.md) for the registry, enforced checks and scope.

To regenerate and inspect the corpus from the repository root:

```bash
python -m astrotoolbench.benchmark.build
python -m astrotoolbench.generate_references --check
python -m pytest tests/tools tests/test_core.py tests/benchmark -q
```

The output includes answer-free task sources, machine-readable schemas, split
manifest and the per-task independent accuracy report. The committed reference
digest protects against silent result
changes; updating it is an explicit review step after re-verification.
