"""Command-line entry point for the project."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer project CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run neural style transfer once the core pipeline is implemented.",
    )
    run_parser.add_argument("--content", help="Path to the content image.")
    run_parser.add_argument("--style", help="Path to the style image.")
    run_parser.add_argument("--mask", help="Optional path to a mask image.")
    run_parser.add_argument(
        "--output",
        help="Optional output image path. Defaults will be added later.",
    )

    subparsers.add_parser(
        "warmup",
        help="Warm up model weights once the offline-demo flow is implemented.",
    )
    return parser


def main() -> int:
    """Run the CLI command dispatcher."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        parser.error("The run command is not implemented yet.")

    if args.command == "warmup":
        parser.error("The warmup command is not implemented yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
