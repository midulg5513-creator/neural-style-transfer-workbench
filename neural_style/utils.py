"""Shared utility helpers used across CLI and GUI entry points."""

from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_STEM, SUPPORTED_IMAGE_SUFFIXES


def get_cuda_device() -> "torch.device":
    """Return the required CUDA device object."""
    import torch

    return torch.device("cuda")


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_path(suffix: str = ".png") -> Path:
    """Build the default output path for generated images."""
    return ensure_directory(DEFAULT_OUTPUT_DIR) / f"{DEFAULT_OUTPUT_STEM}{suffix}"


def is_supported_image_path(path: Path) -> bool:
    """Return whether the path points to a supported image suffix."""
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
