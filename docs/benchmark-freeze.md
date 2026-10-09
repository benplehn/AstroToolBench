# Frozen benchmark

All model runs use `benchmark-v0.1`, which points to commit
`222125788897206010e9f4ba72ddf3945ce18efa` (same as `v0.1.0`). 50 tasks, 22
families, 24 train / 6 validation / 20 test. The validator passed with no
errors before freezing.

[benchmark-v0.1.json](../tasks/astrodynamics/benchmark-v0.1.json) stores the
SHA-256 of the corpus and its companion files. The corpus hash covers whole
records (IDs, prompts, families, references, tolerances), not just the answers.

For experiments, use the frozen loader:

```python
from astrotoolbench.benchmark import BENCHMARK_TAG, load_frozen_benchmark

tasks = load_frozen_benchmark(".", split="test")
```

`root` is the folder that contains `tasks/`, `schemas/` and `docs/` (doesn't
need to be a git checkout). All six files are checked before filtering, so a
changed or missing file stops everything, even if the task you want isn't
affected. The manifest's own hash is pinned in the code, so editing it doesn't
sneak in a new baseline.

Don't regenerate the references for model comparisons: on another machine the
last bits can change. Use temporary output paths if you want to test the
generator elsewhere. Changing the benchmark on purpose means a new version and
new results.

`load_benchmark` is still there for working on other datasets; it checks the
format and splits but not this snapshot.
