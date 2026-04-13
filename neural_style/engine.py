"""Core Gatys-style optimization engine for neural style transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .color import apply_color_preservation
from .config import DEFAULT_NUM_STEPS
from .mask import blend_with_mask, normalize_mask_tensor
from .model import build_style_transfer_model, load_vgg19_features

DEFAULT_CONTENT_WEIGHT = 1.0
DEFAULT_STYLE_WEIGHT = 1_000_000.0


class EngineError(RuntimeError):
    """Raised when the NST engine cannot execute safely."""


@dataclass(frozen=True)
class StyleTransferResult:
    """Structured result returned by the NST engine."""

    output_tensor: torch.Tensor
    content_loss: float
    style_loss: float
    num_steps: int
    device: str
    applied_keep_color: bool
    applied_mask: bool


ProgressCallback = Callable[[int, int, float, float], None]


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


def run_style_transfer(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    *,
    num_steps: int = DEFAULT_NUM_STEPS,
    style_strength: float = 1.0,
    keep_color: bool = False,
    mask: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    cnn: torch.nn.Sequential | None = None,
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    style_weight: float = DEFAULT_STYLE_WEIGHT,
    progress_callback: ProgressCallback | None = None,
) -> StyleTransferResult:
    """Run neural style transfer and return the stylized tensor plus summary."""
    if num_steps <= 0:
        raise EngineError("num_steps must be a positive integer.")
    if style_strength <= 0:
        raise EngineError("style_strength must be greater than 0.")
    if content_weight <= 0 or style_weight <= 0:
        raise EngineError("Loss weights must be greater than 0.")

    resolved_device = ensure_cuda_device(device)

    validate_image_tensor("content_image", content_image)
    validate_image_tensor("style_image", style_image)
    if mask is not None and mask.shape[-2:] != content_image.shape[-2:]:
        raise EngineError("mask must match the content image spatial dimensions.")

    content_image = content_image.to(device=resolved_device, dtype=torch.float32)
    style_image = style_image.to(device=resolved_device, dtype=torch.float32)
    if mask is not None:
        normalized_mask = normalize_mask_tensor(mask).to(
            device=resolved_device,
            dtype=content_image.dtype,
        )
    else:
        normalized_mask = None

    feature_stack = cnn if cnn is not None else load_vgg19_features(device=resolved_device)
    model, style_losses, content_losses = build_style_transfer_model(
        feature_stack,
        content_image,
        style_image,
    )
    model.eval()

    input_image = content_image.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([input_image], max_iter=1)

    last_content_loss = 0.0
    last_style_loss = 0.0
    scaled_style_weight = style_weight * style_strength

    for step in range(1, num_steps + 1):
        def closure() -> torch.Tensor:
            nonlocal last_content_loss, last_style_loss
            with torch.no_grad():
                input_image.clamp_(0.0, 1.0)

            optimizer.zero_grad()
            model(input_image)

            style_score = sum(style_loss.loss for style_loss in style_losses)
            content_score = sum(content_loss.loss for content_loss in content_losses)
            total_loss = (
                (content_weight * content_score)
                + (scaled_style_weight * style_score)
            )
            last_content_loss = float(content_score.detach().item())
            last_style_loss = float(style_score.detach().item())
            total_loss.backward()
            return total_loss

        optimizer.step(closure)
        if progress_callback is not None:
            progress_callback(step, num_steps, last_content_loss, last_style_loss)

    with torch.no_grad():
        output_image = input_image.detach().clamp_(0.0, 1.0)

    if keep_color:
        output_image = apply_color_preservation(output_image, content_image)
    if normalized_mask is not None:
        output_image = blend_with_mask(output_image, content_image, normalized_mask)

    return StyleTransferResult(
        output_tensor=output_image.detach().clamp_(0.0, 1.0),
        content_loss=last_content_loss,
        style_loss=last_style_loss,
        num_steps=num_steps,
        device=str(resolved_device),
        applied_keep_color=keep_color,
        applied_mask=normalized_mask is not None,
    )
