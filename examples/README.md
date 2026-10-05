# Examples

From the repo root:

```bash
python -m pip install ".[dev]"
python examples/scientific_workflow.py
python -m astrotoolbench.validate
```

`scientific_workflow.py` raises a circular orbit from 7000 km to 14000 km: it
applies the departure burn, propagates half the transfer ellipse and checks
whether the arrival point is in Earth's shadow. With the Sun along +x, arrival
is near −x, so it is. Output is JSON with units. Everything runs locally.

Loading one split of the dataset:

```python
from astrotoolbench.benchmark import load_benchmark

tasks = load_benchmark("tasks/astrodynamics/tasks.jsonl", split="validation")
for task in tasks:
    print(task.id, task.family, task.expected.kind)
```

More in the [validation guide](../docs/validation.md).
