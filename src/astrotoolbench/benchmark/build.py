"""Author the v0.1 specifications, schemas, partitions and verified references."""

import argparse
from pathlib import Path
import sys

from .artifacts import pretty_json, serialize_records, write_artifacts
from .families import build_specifications
from .coverage import audit_coverage
from .generation import accuracy_report, generate_records
from .paths import DEFAULT_REPORT, DEFAULT_SPECIFICATIONS, DEFAULT_SPLITS, DEFAULT_TASKS
from .schema import BenchmarkTask, TaskSpecification
from .splits import audit_splits

# Preserve the original public helper used for official record serialization.
serialize_tasks = serialize_records


def _schema(model, identifier):
    schema = model.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = identifier
    return pretty_json(schema)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specifications", type=Path, default=DEFAULT_SPECIFICATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--schema", type=Path, default=Path("schemas/task-v0.1.schema.json"))
    parser.add_argument("--specification-schema", type=Path, default=Path("schemas/task-spec-v0.1.schema.json"))
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check", action="store_true", help="Verify every authored artifact without changing files.")
    arguments = parser.parse_args(argv)
    try:
        paths = (arguments.specifications, arguments.output, arguments.schema,
                 arguments.specification_schema, arguments.splits, arguments.report)
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError("All artifact paths must be distinct.")
        specifications = build_specifications()
        generated = generate_records(specifications)
        audit_coverage(generated.tasks)
        source = serialize_records(specifications)
        manifest = audit_splits(generated.tasks)
        artifacts = {
            arguments.specifications: source,
            arguments.output: serialize_records(generated.tasks),
            arguments.schema: _schema(BenchmarkTask, "urn:astrotoolbench:task:0.1"),
            arguments.specification_schema: _schema(TaskSpecification, "urn:astrotoolbench:task-spec:0.1"),
            arguments.splits: pretty_json(manifest),
            arguments.report: pretty_json(accuracy_report(generated, source)),
        }
        write_artifacts(artifacts, check=arguments.check)
    except (ValueError, OSError) as error:
        print(f"Benchmark authoring failed:\n{error}", file=sys.stderr)
        return 1
    print(f"{len(generated.tasks)} tasks independently verified; " +
          ("artifacts unchanged." if arguments.check else "specifications, splits, schemas and references written."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
