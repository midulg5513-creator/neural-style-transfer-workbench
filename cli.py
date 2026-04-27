"""Command-line entry point for the project."""

from __future__ import annotations

import argparse
import sys

from neural_style.validation import (
    ValidationError,
    validate_histogram_weight,
    validate_init_mode,
    normalize_output_path,
    require_cuda,
    validate_content_blend,
    validate_image_path,
    validate_image_size,
    validate_num_steps,
    validate_optional_image_path,
    validate_style_strength,
    validate_tv_weight,
)
from neural_style.config import (
    DEFAULT_CONTENT_BLEND,
    DEFAULT_HISTOGRAM_WEIGHT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INIT_MODE,
    DEFAULT_NUM_STEPS,
    DEFAULT_STYLE_STRENGTH,
    DEFAULT_TV_WEIGHT,
    ENHANCED_MODE_CONTENT_BLEND,
    ENHANCED_MODE_INIT_MODE,
    ENHANCED_MODE_NUM_STEPS,
    ENHANCED_MODE_STYLE_STRENGTH,
    ENHANCED_MODE_TV_WEIGHT,
    INIT_MODE_CHOICES,
    PAPER_MODE_CONTENT_BLEND,
    PAPER_MODE_HISTOGRAM_WEIGHT,
    PAPER_MODE_INIT_MODE,
    PAPER_MODE_NUM_STEPS,
    PAPER_MODE_STYLE_STRENGTH,
    PAPER_MODE_TV_WEIGHT,
)
from neural_style.model import BACKBONE_CHOICES, DEFAULT_BACKBONE


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
        "--backbone",
        choices=BACKBONE_CHOICES,
        default=DEFAULT_BACKBONE,
        help="Feature backbone used for style transfer: VGG19 baseline or ResNet50 extension.",
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
        "--content-blend",
        type=float,
        default=DEFAULT_CONTENT_BLEND,
        help="How much of the original content image to mix back into the final result (0-1).",
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
    run_parser.add_argument(
        "--tv-weight",
        type=float,
        default=DEFAULT_TV_WEIGHT,
        help="Total variation regularization strength used to smooth brush strokes.",
    )
    run_parser.add_argument(
        "--histogram-weight",
        type=float,
        default=DEFAULT_HISTOGRAM_WEIGHT,
        help="Histogram-matching loss weight used to stabilize activation statistics.",
    )
    run_parser.add_argument(
        "--init-mode",
        choices=INIT_MODE_CHOICES,
        default=DEFAULT_INIT_MODE,
        help="Optimization seed image: content, content_noise, or noise.",
    )
    run_parser.add_argument(
        "--avg-pool",
        action="store_true",
        help="Replace VGG max pooling layers with average pooling for a softer paper-like result.",
    )
    run_parser.add_argument(
        "--enhanced-mode",
        action="store_true",
        help="Apply a stronger preset closer to classic Gatys-style results.",
    )
    run_parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Use the paper-oriented layer preset plus coarse-to-fine optimization.",
    )

    subparsers.add_parser(
        "warmup",
        help="Pre-cache the supported backbone weights for offline demo use.",
    )
    return parser


def handle_run(args: argparse.Namespace) -> int:
    """Validate CLI inputs, run style transfer, and save output artifacts."""
    from neural_style.engine import (
        EngineError,
        build_progressive_scale_schedule,
        run_style_transfer,
    )
    from neural_style.mask import load_mask_tensor
    from neural_style.metadata import build_run_metadata, save_run_metadata
    from neural_style.model import DEFAULT_LAYER_PRESET, PAPER_LAYER_PRESET
    from neural_style.utils import build_output_paths, load_image_tensor, save_tensor_image

    device = require_cuda()
    backbone = args.backbone
    content_path = validate_image_path(args.content, "content image")
    style_path = validate_image_path(args.style, "style image")
    mask_path = validate_optional_image_path(args.mask, "mask image")
    num_steps = validate_num_steps(args.steps)
    style_strength = validate_style_strength(args.style_strength)
    content_blend = validate_content_blend(args.content_blend)
    tv_weight = validate_tv_weight(args.tv_weight)
    histogram_weight = validate_histogram_weight(args.histogram_weight)
    init_mode = validate_init_mode(args.init_mode)
    image_size = validate_image_size(args.image_size)
    use_avg_pool = args.avg_pool
    keep_color = args.keep_color
    layer_preset = DEFAULT_LAYER_PRESET
    scale_schedule = None

    if args.enhanced_mode and args.paper_mode:
        raise ValidationError("enhanced mode and paper mode cannot be enabled together.")

    if args.enhanced_mode:
        if num_steps == DEFAULT_NUM_STEPS:
            num_steps = ENHANCED_MODE_NUM_STEPS
        if style_strength == DEFAULT_STYLE_STRENGTH:
            style_strength = ENHANCED_MODE_STYLE_STRENGTH
        if content_blend == DEFAULT_CONTENT_BLEND:
            content_blend = ENHANCED_MODE_CONTENT_BLEND
        if tv_weight == DEFAULT_TV_WEIGHT:
            tv_weight = ENHANCED_MODE_TV_WEIGHT
        if init_mode == DEFAULT_INIT_MODE:
            init_mode = ENHANCED_MODE_INIT_MODE
        use_avg_pool = True
    elif args.paper_mode:
        if num_steps == DEFAULT_NUM_STEPS:
            num_steps = PAPER_MODE_NUM_STEPS
        if style_strength == DEFAULT_STYLE_STRENGTH:
            style_strength = PAPER_MODE_STYLE_STRENGTH
        if content_blend == DEFAULT_CONTENT_BLEND:
            content_blend = PAPER_MODE_CONTENT_BLEND
        if tv_weight == DEFAULT_TV_WEIGHT:
            tv_weight = PAPER_MODE_TV_WEIGHT
        if histogram_weight == DEFAULT_HISTOGRAM_WEIGHT:
            histogram_weight = PAPER_MODE_HISTOGRAM_WEIGHT
        if init_mode == DEFAULT_INIT_MODE:
            init_mode = PAPER_MODE_INIT_MODE
        use_avg_pool = True
        layer_preset = PAPER_LAYER_PRESET
        scale_schedule = build_progressive_scale_schedule(
            image_size,
            enabled=True,
        )

    output_path = normalize_output_path(args.output)
    image_output_path, metadata_output_path = build_output_paths(output_path)

    content_tensor = load_image_tensor(content_path, target_size=image_size, device=device)
    style_tensor = load_image_tensor(style_path, target_size=image_size, device=device)
    mask_tensor = (
        load_mask_tensor(
            mask_path,
            target_size=image_size,
            target_shape=content_tensor.shape[-2:],
            device=device,
        )
        if mask_path is not None
        else None
    )

    result = run_style_transfer(
        content_tensor,
        style_tensor,
        num_steps=num_steps,
        style_strength=style_strength,
        content_blend=content_blend,
        tv_weight=tv_weight,
        histogram_weight=histogram_weight,
        init_mode=init_mode,
        use_avg_pool=use_avg_pool,
        keep_color=keep_color,
        mask=mask_tensor,
        device=device,
        backbone=backbone,
        layer_preset=layer_preset,
        scale_schedule=scale_schedule,
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
            "content_blend": content_blend,
            "tv_weight": tv_weight,
            "histogram_weight": histogram_weight,
            "init_mode": init_mode,
            "use_avg_pool": use_avg_pool,
            "enhanced_mode": args.enhanced_mode,
            "paper_mode": args.paper_mode,
            "image_size": image_size,
            "keep_color": keep_color,
            "backbone": result.backbone,
            "layer_preset": result.layer_preset,
            "scale_schedule": list(result.scale_schedule),
        },
        device=device,
    )
    saved_metadata_path = save_run_metadata(metadata)

    print(f"Output image: {saved_image_path}")
    print(f"Metadata JSON: {saved_metadata_path}")
    print(f"Device: {result.device}")
    print(f"Backbone: {result.backbone}")
    print(f"Content loss: {result.content_loss:.6f}")
    print(f"Style loss: {result.style_loss:.6f}")
    print(f"Histogram loss: {result.histogram_loss:.6f}")
    print(f"Layer preset: {result.layer_preset}")
    print(f"Scale schedule: {list(result.scale_schedule)}")
    return 0


def handle_warmup() -> int:
    """Pre-cache supported backbone weights for the offline demo path."""
    from neural_style.model import load_backbone_features

    device = require_cuda()
    for backbone in BACKBONE_CHOICES:
        load_backbone_features(backbone=backbone, progress=True)
        print(f"{backbone} weights are cached and ready for offline demo use.")
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
