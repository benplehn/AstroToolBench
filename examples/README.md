# Offline scientific examples

Install the package from the repository root and run:

```bash
python -m pip install ".[dev]"
python examples/scientific_workflow.py
python -m astrotoolbench.validate
```

`scientific_workflow.py` raises a circular orbit from 7000 km to 14000 km. It
applies the departure impulse, propagates the transfer ellipse for half its
period, then evaluates the arrival shadow. Its JSON output names units explicitly.
The Sun points along +x; arrival is near −x and inside the cylindrical shadow.
All calculations are local and require no API key.

Load only the intended official partition when inspecting the corpus:

```python
from astrotoolbench.benchmark import load_benchmark

tasks = load_benchmark("tasks/astrodynamics/tasks.jsonl", split="validation")
for task in tasks:
    print(task.id, task.family, task.expected.kind)
```

The [validator guide](../docs/validation.md) covers machine-readable reports,
custom datasets, independently checked errors and publication coverage.
