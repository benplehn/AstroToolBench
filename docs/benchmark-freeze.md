# Benchmark used for model runs

`benchmark-v0.1` points to commit
`222125788897206010e9f4ba72ddf3945ce18efa`, also tagged `v0.1.0` when the
benchmark was frozen. The corpus has 50 tasks in 22 families, with the same
24 train / 6 validation / 20 test split as the scientific release.

Before freezing it, the offline validator checked all 50 references and their
source specifications, split manifest and verification certificates: no errors.

The file [benchmark-v0.1.json](../tasks/astrodynamics/benchmark-v0.1.json)
records SHA-256 hashes for the corpus and its companion files. The corpus hash
covers the entire record, including IDs, prompts, families, references and
tolerances. It is not just a hash of the answers.

Use the frozen loader for experiments:

```python
from astrotoolbench.benchmark import BENCHMARK_TAG, load_frozen_benchmark

tasks = load_frozen_benchmark(".", split="test")
```

The root argument is the directory containing `tasks/`, `schemas/` and `docs/`;
it does not need to be a Git checkout. All six files are checked before filtering
by split or family. A changed or missing file stops loading, even if the affected
task would have been excluded from the run. The manifest itself has a pinned
hash, so editing its recorded hashes does not approve a new baseline.

Keep these committed references for model comparisons. Regenerating references
on another numerical platform may change their last bits; use temporary output
paths when checking the generator there. A deliberate benchmark change needs a
new snapshot/version and separately identified results.

`load_benchmark` remains available for developing and validating other datasets.
It checks the task format and split policy, without requiring this snapshot.
