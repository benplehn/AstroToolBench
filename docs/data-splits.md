# Data splits

The 50 tasks are split **24 train / 6 validation / 20 test**. The split isn't
random: it comes from a hand-written registry so that whole families stay
together. Validation is small because there aren't many families to give it;
don't read too much into validation scores alone.

The 22 `family` labels are grouped into **11 split groups**. Some families with
different names are really the same computation, so they're grouped too. The
manifest is [`tasks/astrodynamics/splits.json`](../tasks/astrodynamics/splits.json),
and the rules + checks are in `astrotoolbench.benchmark.splits`.

| Group | Split | What's in it |
| --- | --- | --- |
| `kepler_propagation` | train | circular/elliptic propagation, negative/zero/full-period durations |
| `impulsive_maneuvers` | train | vector ΔV and circularization, including zero ΔV |
| `circular_transfers` | train | Hohmann up/down/equal radii, comparing transfers |
| `cylindrical_shadow` | train | shadow at a point, eclipse intervals and durations |
| `unit_contract` | train | mixed units in velocities and positions |
| `state_separation` | validation | comparing two propagated states, distance at a given time |
| `input_contract` | validation | missing data, bad observation window, bad μ |
| `closest_approach` | test | minimum distance over time, endpoints, constant distance, identical orbits |
| `transfer_composition` | test | transfer time → propagation/eclipse, or real departure → arrival shadow |
| `orbital_geometry` | test | undefined Sun direction, orbits/transfers hitting the Earth, unbound states |
| `model_limits` | test | asking for J2, or for more time precision than the search allows |

So it's a split by **family / composition**. Test tasks can still use tools seen
in train (e.g. Hohmann and propagation appear inside the held-out transfer
compositions), but the same full computation never appears in two splits.

Changing a Hohmann radius doesn't make a new test task: it stays in
`circular_transfers/train`. `kepler_comparison` and `separation_at_epoch` are
grouped because both are "propagate two states, take the distance".

## What's checked

The loaders check the whole file before filtering:

- each family has exactly one group and one split;
- an unknown family is an error (no automatic hash/random split);
- a family can only use its own recipes and outcome types, so you can't rename
  a Hohmann task to `closest_approach` to move it;
- a successful recipe can't appear in two splits;
- no duplicate problem across splits. The fingerprint ignores ID, wording,
  selected outputs and search step/tolerance, but keeps physical inputs, units
  and context (with float formatting and `-0.0` normalized).

The fingerprint is a backup; the family registry is what actually prevents
variants from leaking. Two error tasks can use the same tool and still test
different things (a null Sun direction vs. a negative window).

The manifest and verification report both show zero family, recipe or duplicate
overlap. The tests try to break this on purpose (moving a family, fake labels,
radius-only variants, duplicate questions) and check that it fails.

## Using it

`benchmark/tasks.jsonl` and the recorded traces are dev material from the first
prototype; they're not training data, and the official loader refuses that
format. Use `load_benchmark(..., split="train")` for training, tune on
`validation`, and only touch `test` for the final evaluation.

None of this tells you whether a model saw these formulas or tasks during
pretraining (the repo is public). That has to be kept in mind when reading
future results. No training or evaluation has been done yet.
