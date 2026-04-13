"""Color-preservation helpers for stylized outputs."""

from __future__ import annotations

import numpy as np
import torch


def _validate_color_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape [1, 3, H, W].")


def apply_color_preservation(
    stylized_tensor: torch.Tensor,
    content_tensor: torch.Tensor,
) -> torch.Tensor:
    """Preserve the content image chroma while keeping stylized lightness."""
    from skimage.color import lab2rgb, rgb2lab

    _validate_color_tensor("stylized_tensor", stylized_tensor)
    _validate_color_tensor("content_tensor", content_tensor)
    if stylized_tensor.shape[-2:] != content_tensor.shape[-2:]:
        raise ValueError("stylized_tensor and content_tensor must share HxW shape.")

    stylized_rgb = (
        stylized_tensor.detach()
        .cpu()
        .clamp(0.0, 1.0)[0]
        .permute(1, 2, 0)
        .numpy()
    )
    content_rgb = (
        content_tensor.detach()
        .cpu()
        .clamp(0.0, 1.0)[0]
        .permute(1, 2, 0)
        .numpy()
    )

    stylized_lab = rgb2lab(stylized_rgb)
    content_lab = rgb2lab(content_rgb)
    merged_lab = stylized_lab.copy()
    merged_lab[..., 1:] = content_lab[..., 1:]

    merged_rgb = np.clip(lab2rgb(merged_lab), 0.0, 1.0).astype(np.float32)
    merged_tensor = torch.from_numpy(merged_rgb).permute(2, 0, 1).unsqueeze(0)
    return merged_tensor.to(device=stylized_tensor.device, dtype=stylized_tensor.dtype)
