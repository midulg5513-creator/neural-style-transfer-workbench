"""Core Gatys-style optimization engine for neural style transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .color import apply_color_preservation
from .config import (
    DEFAULT_HISTOGRAM_WEIGHT,
    DEFAULT_INIT_MODE,
    DEFAULT_NUM_STEPS,
    DEFAULT_TV_WEIGHT,
    INIT_MODE_CHOICES,
)
from .mask import blend_with_mask, normalize_mask_tensor
from .model import (
    DEFAULT_BACKBONE,
    DEFAULT_LAYER_PRESET,
    FeatureBackbone,
    build_style_transfer_model,
    load_backbone_features,
    normalize_backbone_name,
    resolve_histogram_layers,
    resolve_layer_preset,
)

DEFAULT_CONTENT_WEIGHT = 1.0
DEFAULT_STYLE_WEIGHT = 1_000_000.0


class EngineError(RuntimeError):
    """Raised when the NST engine cannot execute safely."""


class StyleTransferCancelled(EngineError):
    """Raised when a running NST job is cancelled cooperatively."""


@dataclass(frozen=True)
class StyleTransferResult:
    """Structured result returned by the NST engine."""

    output_tensor: torch.Tensor
    content_loss: float
    style_loss: float
    num_steps: int
    device: str
    backbone: str
    content_blend: float
    applied_keep_color: bool
    applied_mask: bool
    histogram_loss: float
    histogram_weight: float
    layer_preset: str
    scale_schedule: tuple[int, ...]


ProgressCallback = Callable[[int, int, float, float], None]
CancelCallback = Callable[[], bool]


def build_initial_image(
    content_image: torch.Tensor,
    *,
    init_mode: str = DEFAULT_INIT_MODE,
    noise_ratio: float = 0.15,
) -> torch.Tensor:
    """Create the optimization seed image for style transfer."""
    if init_mode not in INIT_MODE_CHOICES:
        supported = ", ".join(INIT_MODE_CHOICES)
        raise EngineError(f"init_mode must be one of: {supported}.")
    if not 0.0 <= noise_ratio <= 1.0:
        raise EngineError("noise_ratio must be between 0.0 and 1.0.")

    if init_mode == "content":
        return content_image.clone()

    random_image = torch.rand_like(content_image)
    if init_mode == "noise":
        return random_image

    return torch.lerp(content_image, random_image, noise_ratio)


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    """Compute isotropic total variation to encourage coherent brush strokes."""
    if image.shape[-2] < 2 or image.shape[-1] < 2:
        return image.new_tensor(0.0)

    horizontal = torch.mean(torch.abs(image[..., :, 1:] - image[..., :, :-1]))
    vertical = torch.mean(torch.abs(image[..., 1:, :] - image[..., :-1, :]))
    return horizontal + vertical


def resize_tensor_to_longest_edge(
    image: torch.Tensor,
    target_longest_edge: int,
) -> torch.Tensor:
    """Resize a tensor so its longest spatial edge matches the requested size."""
    if target_longest_edge <= 0:
        raise EngineError("target_longest_edge must be a positive integer.")

    current_height, current_width = image.shape[-2:]
    current_longest_edge = max(current_height, current_width)
    if current_longest_edge == target_longest_edge:
        return image.clone()

    scale = target_longest_edge / current_longest_edge
    target_height = max(1, round(current_height * scale))
    target_width = max(1, round(current_width * scale))
    return F.interpolate(
        image,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def build_progressive_scale_schedule(
    target_longest_edge: int,
    *,
    enabled: bool = False,
) -> tuple[int, ...]:
    """Build a small coarse-to-fine schedule ending at the requested output size."""
    if target_longest_edge <= 0:
        raise EngineError("target_longest_edge must be a positive integer.")
    if not enabled or target_longest_edge < 384:
        return (target_longest_edge,)

    schedule: list[int] = []
    if target_longest_edge >= 1024:
        schedule.append(max(256, target_longest_edge // 4))
    if target_longest_edge >= 512:
        schedule.append(max(256, target_longest_edge // 2))
    schedule.append(target_longest_edge)
    return tuple(sorted(set(schedule)))


def allocate_progressive_steps(
    total_steps: int,
    num_scales: int,
) -> tuple[int, ...]:
    """Distribute the total optimization budget across progressive scales."""
    if total_steps <= 0:
        raise EngineError("total_steps must be a positive integer.")
    if num_scales <= 0:
        raise EngineError("num_scales must be a positive integer.")
    if total_steps < num_scales:
        raise EngineError("total_steps must be at least as large as num_scales.")

    weights = list(range(1, num_scales + 1))
    total_weight = sum(weights)
    steps = [max(1, (total_steps * weight) // total_weight) for weight in weights]
    remainder = total_steps - sum(steps)
    index = num_scales - 1

    while remainder > 0:
        steps[index] += 1
        remainder -= 1
        index = (index - 1) % num_scales

    index = num_scales - 1
    while remainder < 0:
        if steps[index] > 1:
            steps[index] -= 1
            remainder += 1
        index = (index - 1) % num_scales

    return tuple(steps)


def _normalize_scale_schedule(
    scale_schedule: Sequence[int] | None,
    *,
    target_longest_edge: int,
) -> tuple[int, ...]:
    """Normalize a user-provided scale schedule and ensure it ends at target size."""
    if scale_schedule is None:
        return (target_longest_edge,)

    normalized = sorted(
        {
            int(scale)
            for scale in scale_schedule
            if int(scale) > 0 and int(scale) <= target_longest_edge
        }
    )
    if not normalized or normalized[-1] != target_longest_edge:
        normalized.append(target_longest_edge)
    return tuple(normalized)


def ensure_cuda_device(device: torch.device | str | None = None) -> torch.device:
    """Return a CUDA device or fail fast with a clear engine-level error."""
    if not torch.cuda.is_available():
        raise EngineError("CUDA is required for style transfer, but no CUDA device is available.")

    resolved_device = torch.device("cuda" if device is None else device)
    if resolved_device.type != "cuda":
        raise EngineError("The style transfer engine only supports CUDA devices.")
    return resolved_device


def validate_image_tensor(name: str, tensor: torch.Tensor) -> None:
    """Validate an image tensor for engine execution."""
    if tensor.dim() != 4:
        raise EngineError(f"{name} must be a 4D tensor shaped [N, C, H, W].")
    if tensor.shape[0] != 1:
        raise EngineError(f"{name} must contain exactly one image batch.")
    if tensor.shape[1] != 3:
        raise EngineError(f"{name} must contain exactly 3 RGB channels.")
    if tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
        raise EngineError(f"{name} must have positive spatial dimensions.")


def _run_style_transfer_single_scale(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    *,
    backbone: str,
    mask: torch.Tensor | None,
    input_image: torch.Tensor | None,
    num_steps: int,
    style_strength: float,
    tv_weight: float,
    histogram_weight: float,
    init_mode: str,
    use_avg_pool: bool,
    device: torch.device,
    cnn: torch.nn.Sequential | FeatureBackbone | None,
    content_weight: float,
    style_weight: float,
    content_layers: Sequence[str],
    style_layers: Sequence[str],
    histogram_layers: Sequence[str],
    progress_callback: ProgressCallback | None,
    progress_offset: int,
    progress_total: int,
    cancel_callback: CancelCallback | None,
) -> tuple[torch.Tensor, float, float, float]:
    """Run one optimization stage for a single resolution."""
    feature_stack = (
        cnn
        if cnn is not None
        else load_backbone_features(backbone=backbone, device=device)
    )
    model, style_losses, histogram_losses, content_losses = build_style_transfer_model(
        feature_stack,
        content_image,
        style_image,
        content_layers=content_layers,
        style_layers=style_layers,
        histogram_layers=histogram_layers,
        style_mask=mask,
        use_avg_pool=use_avg_pool,
    )
    model.eval()

    if input_image is None:
        input_image = build_initial_image(content_image, init_mode=init_mode)
    elif input_image.shape[-2:] != content_image.shape[-2:]:
        raise EngineError("input_image must match the current scale dimensions.")

    input_image = (
        input_image.detach()
        .to(device=device, dtype=torch.float32)
        .requires_grad_(True)
    )
    optimizer = torch.optim.LBFGS([input_image], max_iter=1)

    last_content_loss = 0.0
    last_style_loss = 0.0
    last_histogram_loss = 0.0
    scaled_style_weight = style_weight * style_strength

    def raise_if_cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise StyleTransferCancelled("Style transfer was cancelled.")

    for step in range(1, num_steps + 1):
        raise_if_cancelled()

        def closure() -> torch.Tensor:
            nonlocal last_content_loss, last_style_loss, last_histogram_loss
            raise_if_cancelled()
            with torch.no_grad():
                input_image.clamp_(0.0, 1.0)

            optimizer.zero_grad()
            model(input_image)

            zero = input_image.new_tensor(0.0)
            style_score = sum(
                (style_loss.loss for style_loss in style_losses),
                start=zero,
            )
            histogram_score = sum(
                (histogram_loss.loss for histogram_loss in histogram_losses),
                start=zero,
            )
            content_score = sum(
                (content_loss.loss for content_loss in content_losses),
                start=zero,
            )
            total_loss = (
                (content_weight * content_score)
                + (scaled_style_weight * style_score)
            )
            if histogram_weight > 0.0:
                total_loss = total_loss + (histogram_weight * histogram_score)
            if tv_weight > 0.0:
                total_loss = total_loss + (tv_weight * total_variation_loss(input_image))
            last_content_loss = float(content_score.detach().item())
            last_style_loss = float(style_score.detach().item())
            last_histogram_loss = float(histogram_score.detach().item())
            total_loss.backward()
            return total_loss

        optimizer.step(closure)
        raise_if_cancelled()
        if progress_callback is not None:
            progress_callback(
                progress_offset + step,
                progress_total,
                last_content_loss,
                last_style_loss,
            )

    with torch.no_grad():
        output_image = input_image.detach().clamp_(0.0, 1.0)
    return output_image, last_content_loss, last_style_loss, last_histogram_loss


def run_style_transfer(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    *,
    num_steps: int = DEFAULT_NUM_STEPS,
    style_strength: float = 1.0,
    content_blend: float = 0.0,
    tv_weight: float = DEFAULT_TV_WEIGHT,
    histogram_weight: float = DEFAULT_HISTOGRAM_WEIGHT,
    init_mode: str = DEFAULT_INIT_MODE,
    use_avg_pool: bool = False,
    keep_color: bool = False,
    mask: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    cnn: torch.nn.Sequential | FeatureBackbone | None = None,
    backbone: str = DEFAULT_BACKBONE,
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    style_weight: float = DEFAULT_STYLE_WEIGHT,
    layer_preset: str = DEFAULT_LAYER_PRESET,
    content_layers: Sequence[str] | None = None,
    style_layers: Sequence[str] | None = None,
    histogram_layers: Sequence[str] | None = None,
    scale_schedule: Sequence[int] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> StyleTransferResult:
    """Run neural style transfer and return the stylized tensor plus summary."""
    if num_steps <= 0:
        raise EngineError("num_steps must be a positive integer.")
    if style_strength <= 0:
        raise EngineError("style_strength must be greater than 0.")
    if not 0.0 <= content_blend <= 1.0:
        raise EngineError("content_blend must be between 0.0 and 1.0.")
    if tv_weight < 0.0:
        raise EngineError("tv_weight must be greater than or equal to 0.0.")
    if histogram_weight < 0.0:
        raise EngineError("histogram_weight must be greater than or equal to 0.0.")
    if init_mode not in INIT_MODE_CHOICES:
        supported = ", ".join(INIT_MODE_CHOICES)
        raise EngineError(f"init_mode must be one of: {supported}.")
    if content_weight <= 0 or style_weight <= 0:
        raise EngineError("Loss weights must be greater than 0.")

    resolved_device = ensure_cuda_device(device)
    resolved_backbone = normalize_backbone_name(backbone)
    if isinstance(cnn, FeatureBackbone):
        resolved_backbone = cnn.backbone_name

    validate_image_tensor("content_image", content_image)
    validate_image_tensor("style_image", style_image)
    if mask is not None and mask.shape[-2:] != content_image.shape[-2:]:
        raise EngineError("mask must match the content image spatial dimensions.")

    if content_layers is None or style_layers is None:
        preset_content_layers, preset_style_layers = resolve_layer_preset(
            layer_preset,
            backbone=resolved_backbone,
        )
        resolved_content_layers = (
            tuple(content_layers)
            if content_layers is not None
            else preset_content_layers
        )
        resolved_style_layers = (
            tuple(style_layers)
            if style_layers is not None
            else preset_style_layers
        )
        resolved_layer_preset = layer_preset
    else:
        resolved_content_layers = tuple(content_layers)
        resolved_style_layers = tuple(style_layers)
        resolved_layer_preset = "custom"

    if histogram_weight <= 0.0:
        resolved_histogram_layers = ()
    elif histogram_layers is None:
        if resolved_layer_preset in (DEFAULT_LAYER_PRESET, "paper"):
            resolved_histogram_layers = resolve_histogram_layers(
                resolved_layer_preset,
                backbone=resolved_backbone,
            )
        else:
            resolved_histogram_layers = ()
    else:
        resolved_histogram_layers = tuple(histogram_layers)

    content_image = content_image.to(device=resolved_device, dtype=torch.float32)
    style_image = style_image.to(device=resolved_device, dtype=torch.float32)
    if mask is not None:
        normalized_mask = normalize_mask_tensor(mask).to(
            device=resolved_device,
            dtype=content_image.dtype,
        )
    else:
        normalized_mask = None

    resolved_scale_schedule = _normalize_scale_schedule(
        scale_schedule,
        target_longest_edge=max(content_image.shape[-2:]),
    )
    if len(resolved_scale_schedule) > num_steps:
        resolved_scale_schedule = resolved_scale_schedule[-num_steps:]
    scale_steps = allocate_progressive_steps(num_steps, len(resolved_scale_schedule))
    feature_stack = (
        cnn
        if cnn is not None
        else load_backbone_features(backbone=resolved_backbone, device=resolved_device)
    )
    input_image: torch.Tensor | None = None
    last_content_loss = 0.0
    last_style_loss = 0.0
    last_histogram_loss = 0.0
    completed_steps = 0

    for current_longest_edge, current_steps in zip(
        resolved_scale_schedule,
        scale_steps,
    ):
        current_content = resize_tensor_to_longest_edge(
            content_image,
            current_longest_edge,
        )
        current_style = resize_tensor_to_longest_edge(
            style_image,
            current_longest_edge,
        )
        if normalized_mask is not None:
            current_mask = resize_tensor_to_longest_edge(
                normalized_mask,
                current_longest_edge,
            )
        else:
            current_mask = None

        if input_image is not None:
            input_image = resize_tensor_to_longest_edge(
                input_image,
                current_longest_edge,
            )

        input_image, last_content_loss, last_style_loss, last_histogram_loss = _run_style_transfer_single_scale(
            current_content,
            current_style,
            backbone=resolved_backbone,
            mask=current_mask,
            input_image=input_image,
            num_steps=current_steps,
            style_strength=style_strength,
            tv_weight=tv_weight,
            histogram_weight=histogram_weight,
            init_mode=init_mode,
            use_avg_pool=use_avg_pool,
            device=resolved_device,
            cnn=feature_stack,
            content_weight=content_weight,
            style_weight=style_weight,
            content_layers=resolved_content_layers,
            style_layers=resolved_style_layers,
            histogram_layers=resolved_histogram_layers,
            progress_callback=progress_callback,
            progress_offset=completed_steps,
            progress_total=num_steps,
            cancel_callback=cancel_callback,
        )
        completed_steps += current_steps

    output_image = input_image
    if output_image is None:
        raise EngineError("Style transfer did not produce an output image.")

    if keep_color:
        output_image = apply_color_preservation(output_image, content_image)
    if normalized_mask is not None:
        output_image = blend_with_mask(output_image, content_image, normalized_mask)
    if content_blend > 0.0:
        output_image = torch.lerp(output_image, content_image, content_blend)

    return StyleTransferResult(
        output_tensor=output_image.detach().clamp_(0.0, 1.0),
        content_loss=last_content_loss,
        style_loss=last_style_loss,
        num_steps=num_steps,
        device=str(resolved_device),
        backbone=resolved_backbone,
        content_blend=content_blend,
        applied_keep_color=keep_color,
        applied_mask=normalized_mask is not None,
        histogram_loss=last_histogram_loss,
        histogram_weight=histogram_weight,
        layer_preset=resolved_layer_preset,
        scale_schedule=resolved_scale_schedule,
    )
