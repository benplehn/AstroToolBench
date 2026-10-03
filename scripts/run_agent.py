"""Repository entry point for a single-task agent run."""

from pathlib import Path

from atb.agent_cli import main


if __name__ == "__main__":
    raise SystemExit(main(project_root=Path(__file__).resolve().parents[1]))
