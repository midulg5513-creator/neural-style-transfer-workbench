"""Validation helpers shared by CLI and GUI flows."""

from __future__ import annotations

from pathlib import Path

from .config import (
    DEFAULT_OUTPUT_DIR,
    MAX_NUM_STEPS,
    MAX_STYLE_STRENGTH,
    MIN_NUM_STEPS,
    MIN_STYLE_STRENGTH,
)
from .utils import build_output_paths, ensure_directory, is_supported_image_path


class ValidationError(RuntimeError):
    """Raised when project input or environment validation fails."""


CUDA_REQUIRED_MESSAGE = (
    "A CUDA-capable NVIDIA GPU is required for this project. "
    "PyTorch did not report CUDA as available on this machine."
)
TORCH_MISSING_MESSAGE = (
    "PyTorch is not installed in the current environment. Install the project "
    "dependencies on the target CUDA-capable machine before running the app."
)


def is_cuda_ready() -> bool:
    """Return whether CUDA is available for runtime use."""
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.cuda.is_available()


def require_cuda() -> "torch.device":
    """Return the CUDA device or raise a validation error."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ValidationError(TORCH_MISSING_MESSAGE) from exc
    if not is_cuda_ready():
        raise ValidationError(CUDA_REQUIRED_MESSAGE)
    return torch.device("cuda")


def validate_image_path(value: str | Path, field_name: str) -> Path:
    """Validate an image-like filesystem path."""
    path = Path(value).expanduser()
    if not path.exists():
        raise ValidationError(f"{field_name} does not exist: {path}")
    if not path.is_file():
        raise ValidationError(f"{field_name} is not a file: {path}")
    if not is_supported_image_path(path):
        raise ValidationError(
            f"{field_name} must be one of the supported image formats: {path}"
        )
    return path


def validate_optional_image_path(
    value: str | Path | None, field_name: str
) -> Path | None:
    """Validate an optional image path."""
    if value in (None, ""):
        return None
    return validate_image_path(value, field_name)


def validate_num_steps(value: int) -> int:
    """Validate the optimization-step count."""
    if not MIN_NUM_STEPS <= value <= MAX_NUM_STEPS:
        raise ValidationError(
            f"num_steps must be between {MIN_NUM_STEPS} and {MAX_NUM_STEPS}."
        )
    return value


def validate_style_strength(value: float) -> float:
    """Validate the user-selected style strength."""
    if not MIN_STYLE_STRENGTH <= value <= MAX_STYLE_STRENGTH:
        raise ValidationError(
            "style_strength must be between "
            f"{MIN_STYLE_STRENGTH} and {MAX_STYLE_STRENGTH}."
        )
    return value


def validate_image_size(value: int) -> int:
    """Validate the requested image resize target."""
    if value <= 0:
        raise ValidationError("image_size must be greater than 0.")
    return value


def normalize_output_path(value: str | Path | None) -> Path:
    """Return a safe output path, defaulting into the output directory."""
    if value in (None, ""):
        ensure_directory(DEFAULT_OUTPUT_DIR)
        return DEFAULT_OUTPUT_DIR / "result.png"

    image_path, _ = build_output_paths(value)
    return image_path


def build_startup_status_message() -> str:
    """Summarize the startup environment state for the UI."""
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return TORCH_MISSING_MESSAGE
    if is_cuda_ready():
        return "CUDA detected. The project can continue with GPU-only execution."
    return CUDA_REQUIRED_MESSAGE
