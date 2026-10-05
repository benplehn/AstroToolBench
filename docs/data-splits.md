# Family-separated data — milestone 6

The 50-task corpus has **24 train, 6 validation and 20 test** records. Splits are
assigned deterministically by a reviewed registry, not by random task IDs,
parameter values or requested split-size percentages. Whole families stay together.
The smaller validation set is a consequence of the available families; it is not
intended to support precise statistical estimates on its own.

The 22 descriptive `family` labels are collected into **11 conservative split
groups**. The grouping also keeps differently named versions of the same full
computation in one partition. The machine-readable manifest is
[`tasks/astrodynamics/splits.json`](../tasks/astrodynamics/splits.json); `astrotoolbench.benchmark.splits` is its authoring
policy and executable audit.

| Group | Split | Families / intended scope |
| --- | --- | --- |
| `kepler_propagation` | train | Circular and elliptic propagation, including signed/zero/full-period variants. |
| `impulsive_maneuvers` | train | Vector ΔV and circularization; zero impulses stay with other velocity-change problems. |
| `circular_transfers` | train | Raising/lowering/equal-radius Hohmann and comparisons of the same transfer solution. |
| `cylindrical_shadow` | train | Instantaneous shadow geometry and interval/duration questions. |
| `unit_contract` | train | Mixed velocity and position units; consistent dimensional-check failure. |
| `state_separation` | validation | Both propagated-state comparison and simultaneous separation at a specified time. |
| `input_contract` | validation | Missing data, invalid observation windows, invalid gravity parameter. |
| `closest_approach` | test | A minimum over time, including endpoints, constant separation and coincident trajectories. |
| `transfer_composition` | test | Transfer time then propagation/eclipse, or actual transfer departure then arrival shadow. |
| `orbital_geometry` | test | Undefined solar direction, body-intersecting states/transfers and unsupported unbound states. |
| `model_limits` | test | Requested J2 and time guarantees tighter than the configured bracket width. |

This is a **family/composition generalization split**. A held-out workflow can
reuse tools learned from the training partition; for example, Hohmann and Kepler
propagation occur as prerequisites in held-out transfer compositions. The full
successful computation recipe is not shared across partitions. This policy does
not claim that every primitive formula is unseen at test time.

In particular, changing a Hohmann radius does not produce a test family. A new
Hohmann record still belongs to `circular_transfers/train`. `kepler_comparison`
and `separation_at_epoch` are also grouped together because both calculate two
propagated positions followed by a norm, despite their different descriptive
family names.

## Enforced checks

Both source and reference loaders audit the complete file before filtering:

- A family must have exactly one reviewed group and split.
- Unknown family labels require an explicit policy addition, not an automatic
  hash/random split.
- A family must use its reviewed recipes and outcome classes; renaming a Hohmann
  calculation as `closest_approach` cannot bypass the policy.
- A complete successful recipe cannot appear in multiple partitions.
- Cross-partition duplicate problems are rejected using a fingerprint that ignores
  ID, wording, output subset and numerical search step/tolerance. Physical inputs,
  units and context remain part of the fingerprint; harmless float serialization
  differences and signed zeros are normalized.

The fingerprint is an additional duplicate check. The conservative family/group
registry is what prevents numerical variants from leaking between partitions.
Expected failures can use a familiar primitive tool while testing a different
failure mechanism: a null solar direction and a negative observation window are
not the same diagnostic problem merely because both eventually concern eclipses.

The manifest and verification report record zero local family overlaps, successful
recipe overlaps and duplicate problems across splits. Tests deliberately attempt
family moves, fake labels, a radius-only split and duplicate questions, and require
these attempts to fail.

## Scope and use

`benchmark/tasks.jsonl` and recorded model traces are legacy development material.
They are excluded from the official training data; the strict official loader
rejects that format. Do not mix them into future training exports. Use
`load_benchmark(..., split="train")` and keep the official test partition out of
training or tuning. Tune against `validation`; evaluate the frozen test partition
only when the protocol is ready.

These checks prepare a local split policy. They cannot prove that a public task,
physical formula or earlier development example is absent from an external model's
pretraining corpus, and do not undo prior public exposure. Record that provenance
when interpreting future evaluation results. No model training or evaluation is
performed by these milestones.
