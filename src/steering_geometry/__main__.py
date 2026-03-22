"""CLI entry point for steering_geometry package."""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for main CLI."""
    parser = argparse.ArgumentParser(
        prog="steering_geometry",
        description="Steering vector extraction for LLM representation engineering",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Output bash variable exports for ALL_MODELS, ALL_CONCEPTS, DEFAULT_MODEL",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()

    if args.shell:
        try:
            from steering_geometry.config import (
                DEFAULT_MODEL,
                SUPPORTED_CONCEPTS,
                SUPPORTED_MODELS,
            )
        except ImportError as e:
            print(f"Error: Failed to import config: {e}", file=sys.stderr)
            sys.exit(1)

        # Format bash array with properly quoted strings
        models_quoted = " ".join(f'"{m}"' for m in SUPPORTED_MODELS)
        concepts_quoted = " ".join(f'"{c}"' for c in SUPPORTED_CONCEPTS)

        print(f"ALL_MODELS=({models_quoted})")
        print(f"ALL_CONCEPTS=({concepts_quoted})")
        print(f'DEFAULT_MODEL="{DEFAULT_MODEL}"')
        sys.exit(0)

    # No action specified - show help
    _build_parser().print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
