"""Repository entry point for inspecting a raw model completion."""

from pathlib import Path

from atb.inspection import main


if __name__ == "__main__":
    raise SystemExit(main(project_root=Path(__file__).resolve().parents[1]))
