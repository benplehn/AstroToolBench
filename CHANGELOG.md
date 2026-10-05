# Changelog

## 0.1.0 — 2026-10-05

First reproducible scientific benchmark release.

- Stabilized propagation, ΔV, Hohmann transfer, cylindrical shadow/eclipse and
  closest-approach tools with explicit scientific contracts and input validation.
- Added 50 versioned tasks across six categories and 22 descriptive families,
  with a 20/12/8/10 difficulty mix and 24/6/20 family-based partitions.
- Added answer-free source records, strict JSON Schemas and a reviewed split
  registry that rejects cross-partition numerical variants and duplicates.
- Added independent analytic, Cartesian RK4 and input-contract verification for
  all 38 successful computations and 12 expected error outcomes.
- Added `python -m astrotoolbench.generate_references` and
  `python -m astrotoolbench.validate`, deterministic artifacts and diagnostics.
- Added scientific, dataset and validator regression tests plus scientific CI
  on Python 3.10 and 3.12 without optional LLM dependencies.
- Restricted default pytest discovery to `tests/`, so installing only `.[dev]`
  does not collect optional live-model experiment scripts.
- Organized scientific code under `astrotoolbench`, retained `astrodyn_tools`
  compatibility imports, and included data, schemas, examples and MIT licensing
  in source distributions.
- Documented reproducibility, assumptions, tolerances and contamination scope.

The existing legacy model integration and development traces remain available.
This release makes no comparative LLM-quality, fine-tuning or GPU-performance claim.
