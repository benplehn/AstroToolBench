# AstroToolBench v0.1.0 — release checklist

The week-end 2 deliverable is a reproducible scientific benchmark. It establishes
tools, task contracts, independently verified references and validation before
comparative model evaluation. The optional legacy integrations retain their own
development data and results.

## Definition of Done

| Requirement | Evidence for v0.1.0 |
| --- | --- |
| At least 40 tasks | 50 strict official records in `tasks/astrodynamics/tasks.jsonl`. |
| At least five problem families | 22 descriptive families, conservatively grouped into 11 split groups. |
| At least four task types | 20 simple, 12 multistep, 8 diagnostic and 10 trap. |
| Every reference verified | 33 analytic, 5 Cartesian RK4 and 12 independent input certificates; 50 verified outcomes. |
| Every numerical result has a tolerance | Per-output scoring tolerances plus independent verification budgets. Boolean results and error codes use exact comparison. |
| Every task has a family | Strict schema and reviewed family registry; missing or unknown labels fail validation. |
| `pytest` passes | Full local suite: 422 passed on both Python 3.10.22 and 3.12.13. Clean editable installation with only `.[dev]`: 352 passed, 8 optional-integration skips. GitHub CI state is recorded separately on the release page. |
| `validate` reports no error | 50 valid, 0 invalid, 0 issues; source, manifest and accuracy report checked. |
| README understandable | What, why, why astrodynamics, roadmap and a reproducible quick start appear before optional integrations. |
| Public repository | `benplehn/AstroToolBench` visibility verified as `PUBLIC` through GitHub. |
| Release `v0.1.0` | Publication gate: matching local/remote tag, verified source/tests and a published GitHub release. Final publication evidence and any pending external CI checks are recorded on the release page. |

Expected errors are valid benchmark tasks only when scientific execution and the
independent checker reproduce their declared code. They do not require fabricated
numeric answers or tolerances. The complete dataset has 38 successful computations
and 12 expected error outcomes.

## Reproduction

From a clean checkout with Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m astrotoolbench.generate_references
python -m astrotoolbench.validate
python examples/scientific_workflow.py
```

Pytest is a development dependency, so a fresh environment installs `.[dev]`
when running tests. Base installation with `pip install -e .` provides NumPy and
Pydantic for the scientific commands. Optional model-client tests are skipped
without `llm`; rendering tests need `tokenizer` and cached pinned assets. Test
skips for these optional integrations do not skip scientific validation.

`generate_references --check` and `benchmark.build --check` compare exact artifacts
within the same numerical environment. Cross-platform scientific checks use
explicit error budgets; the validator does not assume bit-identical floating
point calculations across platforms.

## Publication gates

Before creating the release tag:

- Run the full suite and complete official validation.
- Verify both generation and authoring check modes without changing artifacts.
- Test editable installation and public commands from a clean clone in a fresh
  environment with no optional LLM packages.
- Build source and wheel distributions and verify reviewed data, schemas,
  examples, documentation and licensing in the source archive.
- Verify all local documentation links and exclude runtime secrets/generated
  files from the release commit.
- Push the reviewed commit and check the Python 3.10/3.12 scientific and
  integration CI jobs at that commit. An external runner incident is disclosed
  as pending CI, with independent local verification retained; a queued job is
  never recorded as a successful job.

Publish an annotated `v0.1.0` tag and a GitHub release at the tested commit.
After publication, verify repository visibility, tag-to-commit agreement, release
draft/prerelease state and downloadable assets. The release page records this
final publication evidence, rather than treating a proposed publication as done.

During preparation on 5 October 2026, GitHub reported an
[Actions runner-assignment incident](https://www.githubstatus.com/incidents/3q1yb5m7ltvb).
The Linux scientific Python 3.12 job passed at the verified scientific source
commit; the remaining jobs were queued. Full local suites passed on Python 3.10
and 3.12, and a fresh minimal installation passed its tests, generation and
validation. Release evidence distinguishes these successful checks from external
CI still waiting for runners. Any later documentation-only commit is checked for
unchanged scientific source, tests, data, schemas and workflow configuration.

[Repository](https://github.com/benplehn/AstroToolBench) ·
[Release](https://github.com/benplehn/AstroToolBench/releases/tag/v0.1.0) ·
[Changelog](../CHANGELOG.md) · [Validation](validation.md) ·
[Scientific evidence](ground-truth.md) · [Split policy](data-splits.md)

## Scientific scope

The production assumptions are two-body dynamics and a fixed-Sun cylindrical
shadow; production eclipse/proximity searches are sampled. Independent certificates
cover the committed geometries and regimes. This release does not claim J2,
Lambert, arbitrary grazing-event guarantees or unrestricted RK4 accuracy.

Family/composition separation rejects local leakage and radius-only variants.
It permits primitive tool reuse inside held-out compositions and cannot prove
absence from external pretraining data. No new live LLM run, fine-tuning or GPU
experiment is needed to validate this scientific release.
