"""Mask helpers for local style-transfer control and blending."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import DEFAULT_IMAGE_SIZE
from .utils import load_image_tensor, pil_image_to_tensor


def normalize_mask_tensor(mask_tensor: torch.Tensor) -> torch.Tensor:
    """Convert an arbitrary mask tensor into a clamped [1, 1, H, W] mask."""
    normalized = mask_tensor
    if normalized.dim() == 2:
        normalized = normalized.unsqueeze(0).unsqueeze(0)
    elif normalized.dim() == 3:
        if normalized.shape[0] in (1, 3):
            normalized = normalized.unsqueeze(0)
        else:
            raise ValueError("3D mask tensors must be shaped [C, H, W].")
    elif normalized.dim() != 4:
        raise ValueError("Mask tensors must be 2D, 3D, or 4D.")

    if normalized.shape[0] != 1:
        raise ValueError("Only single-image masks are supported.")
    if normalized.shape[1] not in (1, 3):
        raise ValueError("Mask tensors must contain 1 or 3 channels.")

    normalized = normalized.float()
    if normalized.shape[1] == 3:
        normalized = normalized.mean(dim=1, keepdim=True)
    return normalized.clamp(0.0, 1.0)


def load_mask_tensor(
    path: str | Path,
    target_size: int = DEFAULT_IMAGE_SIZE,
    target_shape: tuple[int, int] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Load a mask image from disk and normalize it for blending."""
    if target_shape is not None:
        target_height, target_width = target_shape
        if target_height <= 0 or target_width <= 0:
            raise ValueError("target_shape must contain positive height and width.")

        from PIL import Image

        source_path = Path(path).expanduser()
        with Image.open(source_path) as image:
            resized = image.convert("RGB").resize(
                (target_width, target_height),
                resample=Image.Resampling.LANCZOS,
            )
            return normalize_mask_tensor(pil_image_to_tensor(resized, device=device))

    return normalize_mask_tensor(
        load_image_tensor(path, target_size=target_size, device=device)
    )


def blend_with_mask(
    stylized_tensor: torch.Tensor,
    content_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
) -> torch.Tensor:
    """Blend stylized and content images according to a normalized mask."""
    if stylized_tensor.shape != content_tensor.shape:
        raise ValueError("stylized_tensor and content_tensor must share the same shape.")
    normalized_mask = normalize_mask_tensor(mask_tensor).to(
        device=stylized_tensor.device,
        dtype=stylized_tensor.dtype,
    )
    if normalized_mask.shape[-2:] != stylized_tensor.shape[-2:]:
        raise ValueError("mask_tensor must share the same HxW shape as the images.")
    return normalized_mask * stylized_tensor + (1.0 - normalized_mask) * content_tensor
