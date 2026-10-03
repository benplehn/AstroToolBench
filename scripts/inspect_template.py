"""Repository entry point for rendering a benchmark chat template."""

from pathlib import Path

from atb.template_inspection import main


if __name__ == "__main__":
    raise SystemExit(main(project_root=Path(__file__).resolve().parents[1]))
