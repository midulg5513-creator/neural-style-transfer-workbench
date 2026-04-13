"""Command-line entry point for the project."""

from __future__ import annotations

import argparse
import sys

from neural_style.validation import (
    ValidationError,
    normalize_output_path,
    require_cuda,
    validate_image_path,
    validate_image_size,
    validate_num_steps,
    validate_optional_image_path,
    validate_style_strength,
)
from neural_style.config import DEFAULT_IMAGE_SIZE, DEFAULT_NUM_STEPS, DEFAULT_STYLE_STRENGTH


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer project CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one style-transfer pass and save the image plus metadata.",
    )
    run_parser.add_argument(
        "--content",
        required=True,
        help="Path to the content image.",
    )
    run_parser.add_argument(
        "--style",
        required=True,
        help="Path to the style image.",
    )
    run_parser.add_argument("--mask", help="Optional path to a mask image.")
    run_parser.add_argument(
        "--output",
        help="Optional output image path. Defaults to outputs/result.png.",
    )
    run_parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Optimization steps to run.",
    )
    run_parser.add_argument(
        "--style-strength",
        type=float,
        default=DEFAULT_STYLE_STRENGTH,
        help="Multiplier applied to the base style-loss weight.",
    )
    run_parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Resize target for the longest image edge.",
    )
    run_parser.add_argument(
        "--keep-color",
        action="store_true",
        help="Preserve the content image chroma after stylization.",
    )

    subparsers.add_parser(
        "warmup",
        help="Pre-cache the pretrained VGG19 weights for offline demo use.",
    )
    return parser


def handle_run(args: argparse.Namespace) -> int:
    """Validate CLI inputs, run style transfer, and save output artifacts."""
    from neural_style.engine import EngineError, run_style_transfer
    from neural_style.mask import load_mask_tensor
    from neural_style.metadata import build_run_metadata, save_run_metadata
    from neural_style.utils import build_output_paths, load_image_tensor, save_tensor_image

    device = require_cuda()
    content_path = validate_image_path(args.content, "content image")
    style_path = validate_image_path(args.style, "style image")
    mask_path = validate_optional_image_path(args.mask, "mask image")
    num_steps = validate_num_steps(args.steps)
    style_strength = validate_style_strength(args.style_strength)
    image_size = validate_image_size(args.image_size)
    output_path = normalize_output_path(args.output)
    image_output_path, metadata_output_path = build_output_paths(output_path)

    content_tensor = load_image_tensor(content_path, target_size=image_size, device=device)
    style_tensor = load_image_tensor(style_path, target_size=image_size, device=device)
    mask_tensor = (
        load_mask_tensor(mask_path, target_size=image_size, device=device)
        if mask_path is not None
        else None
    )

    result = run_style_transfer(
        content_tensor,
        style_tensor,
        num_steps=num_steps,
        style_strength=style_strength,
        keep_color=args.keep_color,
        mask=mask_tensor,
        device=device,
    )
    saved_image_path = save_tensor_image(result.output_tensor, image_output_path)
    metadata = build_run_metadata(
        content_path=content_path,
        style_path=style_path,
        mask_path=mask_path,
        output_image_path=saved_image_path,
        metadata_path=metadata_output_path,
        parameters={
            "num_steps": num_steps,
            "style_strength": style_strength,
            "image_size": image_size,
            "keep_color": args.keep_color,
        },
        device=device,
    )
    saved_metadata_path = save_run_metadata(metadata)

    print(f"Output image: {saved_image_path}")
    print(f"Metadata JSON: {saved_metadata_path}")
    print(f"Device: {result.device}")
    print(f"Content loss: {result.content_loss:.6f}")
    print(f"Style loss: {result.style_loss:.6f}")
    return 0


def handle_warmup() -> int:
    """Pre-cache VGG19 weights for the offline demo path."""
    from neural_style.model import load_vgg19_features

    device = require_cuda()
    load_vgg19_features(progress=True)
    print("VGG19 weights are cached and ready for offline demo use.")
    print(f"Validated device: {device}")
    return 0


def main() -> int:
    """Run the CLI command dispatcher."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run":
            return handle_run(args)

        if args.command == "warmup":
            return handle_warmup()
    except ValidationError as exc:
        print(f"Environment error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Engine error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
