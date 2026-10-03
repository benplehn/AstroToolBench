"""List model IDs advertised by the configured OpenAI-compatible provider."""

import argparse
from pathlib import Path
import sys

from openai import APIError

from atb.client import ClientSettings, get_client


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    parser.add_argument("--filter", default="", help="Case-insensitive substring in model IDs")
    args = parser.parse_args(argv)
    try:
        settings = ClientSettings.from_env(args.env_file)
        client, _ = get_client(settings)
        with client:
            models = sorted(model.id for model in client.models.list())
        for model in models:
            if args.filter.lower() in model.lower():
                print(model)
        return 0
    except APIError:
        print("Could not list models. Check the endpoint, credential and provider support for /models.", file=sys.stderr)
        return 1
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
