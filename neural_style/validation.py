"""Validation helpers shared by CLI and GUI flows."""

from __future__ import annotations

from pathlib import Path

from .config import (
    DEFAULT_OUTPUT_DIR,
    INIT_MODE_CHOICES,
    MAX_CONTENT_BLEND,
    MAX_NUM_STEPS,
    MAX_STYLE_STRENGTH,
    MIN_CONTENT_BLEND,
    MIN_NUM_STEPS,
    MIN_STYLE_STRENGTH,
    DEFAULT_HISTOGRAM_WEIGHT,
    MIN_TV_WEIGHT,
    SUPPORTED_IMAGE_SUFFIXES,
)
from .utils import build_output_paths, ensure_directory, is_supported_image_path


class ValidationError(RuntimeError):
    """Raised when project input or environment validation fails."""


CUDA_REQUIRED_MESSAGE = (
    "当前项目必须使用支持 CUDA 的 NVIDIA 显卡。"
    "PyTorch 在本机环境中未检测到可用的 CUDA。"
)
TORCH_MISSING_MESSAGE = (
    "当前环境未安装 PyTorch。请先在目标 CUDA 机器上安装项目依赖，"
    "再启动本程序。"
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
    if value in (None, ""):
        raise ValidationError(f"请选择{field_name}。")

    path = Path(value).expanduser()
    if not path.exists():
        raise ValidationError(f"{field_name}不存在：{path}")
    if not path.is_file():
        raise ValidationError(f"{field_name}不是有效文件：{path}")
    if not is_supported_image_path(path):
        raise ValidationError(
            f"{field_name}格式不受支持：{path}"
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
            f"迭代步数必须在 {MIN_NUM_STEPS} 到 {MAX_NUM_STEPS} 之间。"
        )
    return value


def validate_style_strength(value: float) -> float:
    """Validate the user-selected style strength."""
    if not MIN_STYLE_STRENGTH <= value <= MAX_STYLE_STRENGTH:
        raise ValidationError(
            "风格强度必须在 "
            f"{MIN_STYLE_STRENGTH} 到 {MAX_STYLE_STRENGTH} 之间。"
        )
    return value


def validate_content_blend(value: float) -> float:
    """Validate the requested content-preservation blend ratio."""
    if not MIN_CONTENT_BLEND <= value <= MAX_CONTENT_BLEND:
        raise ValidationError(
            "原图保留度必须在 "
            f"{MIN_CONTENT_BLEND} 到 {MAX_CONTENT_BLEND} 之间。"
        )
    return value


def validate_tv_weight(value: float) -> float:
    """Validate the total-variation regularization weight."""
    if value < MIN_TV_WEIGHT:
        raise ValidationError(
            f"TV 正则权重必须大于或等于 {MIN_TV_WEIGHT}。"
        )
    return value


def validate_histogram_weight(value: float) -> float:
    """Validate the histogram-loss weight."""
    if value < DEFAULT_HISTOGRAM_WEIGHT:
        raise ValidationError("直方图损失权重必须大于或等于 0。")
    return value


def validate_init_mode(value: str) -> str:
    """Validate the image initialization mode."""
    if value not in INIT_MODE_CHOICES:
        supported = ", ".join(INIT_MODE_CHOICES)
        raise ValidationError(f"初始化模式必须是以下之一：{supported}")
    return value


def validate_image_size(value: int) -> int:
    """Validate the requested image resize target."""
    if value <= 0:
        raise ValidationError("输出尺寸必须大于 0。")
    return value


def normalize_output_path(value: str | Path | None) -> Path:
    """Return a safe output path, defaulting into the output directory."""
    if value in (None, ""):
        ensure_directory(DEFAULT_OUTPUT_DIR)
        return DEFAULT_OUTPUT_DIR / "result.png"

    image_path, _ = build_output_paths(value)
    return image_path


def validate_output_image_path(value: str | Path | None) -> Path:
    """Validate the requested output image path and enforce a supported suffix."""
    image_path = normalize_output_path(value)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValidationError(
            "输出图像必须使用以下受支持的后缀之一："
            f"{supported}"
        )
    return image_path


def build_startup_status_message() -> str:
    """Summarize the startup environment state for the UI."""
    try:
        import torch
    except ModuleNotFoundError:
        return TORCH_MISSING_MESSAGE
    if is_cuda_ready():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return (
            "已检测到 CUDA 环境，可以按仅 GPU 模式运行。\n"
            f"当前显卡：{device_name}"
        )
    return CUDA_REQUIRED_MESSAGE
