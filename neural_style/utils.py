"""Shared utility helpers used across CLI and GUI entry points."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from .config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_METADATA_SUFFIX,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_STEM,
    GUI_PREVIEW_HEIGHT,
    GUI_PREVIEW_WIDTH,
    SUPPORTED_IMAGE_SUFFIXES,
)


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


def metadata_output_path(image_path: str | Path) -> Path:
    """Return the JSON sidecar path for a generated image."""
    return Path(image_path).expanduser().with_suffix(DEFAULT_METADATA_SUFFIX)


def build_output_paths(
    output_path: str | Path | None = None,
    suffix: str = ".png",
) -> tuple[Path, Path]:
    """Resolve the generated image path and matching metadata sidecar path."""
    if output_path in (None, ""):
        image_path = default_output_path(suffix)
    else:
        image_path = Path(output_path).expanduser()
        if not image_path.suffix:
            image_path = image_path.with_suffix(suffix)
        ensure_directory(image_path.parent)
    return image_path, metadata_output_path(image_path)


def is_supported_image_path(path: Path) -> bool:
    """Return whether the path points to a supported image suffix."""
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def calculate_resize_shape(
    width: int,
    height: int,
    target_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[int, int]:
    """Scale an image so its longest edge matches the configured target size."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive integers.")
    if target_size <= 0:
        raise ValueError("target_size must be a positive integer.")

    scale = target_size / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_image(
    image: "Image.Image",
    target_size: int = DEFAULT_IMAGE_SIZE,
) -> "Image.Image":
    """Resize an image with preserved aspect ratio."""
    from PIL import Image

    resized_shape = calculate_resize_shape(*image.size, target_size=target_size)
    if image.size == resized_shape:
        return image.copy()
    return image.resize(resized_shape, resample=Image.Resampling.LANCZOS)


def load_rgb_image(
    path: str | Path,
    target_size: int = DEFAULT_IMAGE_SIZE,
) -> "Image.Image":
    """Load an image, convert it to RGB, and resize it for model input."""
    from PIL import Image

    source_path = Path(path).expanduser()
    with Image.open(source_path) as image:
        rgb_image = image.convert("RGB")
        resized = resize_image(rgb_image, target_size=target_size)
        return resized.copy()


def move_tensor_to_device(
    tensor: "torch.Tensor",
    device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    """Move a tensor to the requested device, defaulting to CUDA."""
    resolved_device = get_cuda_device() if device is None else device
    return tensor.to(resolved_device)


def pil_image_to_tensor(
    image: "Image.Image",
    device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    """Convert a PIL RGB image to a float tensor in NCHW layout."""
    import numpy as np
    import torch

    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    image_array = np.array(rgb_image, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
    return move_tensor_to_device(tensor.unsqueeze(0), device=device)


def load_image_tensor(
    path: str | Path,
    target_size: int = DEFAULT_IMAGE_SIZE,
    device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    """Load an image from disk and convert it to a model-ready tensor."""
    return pil_image_to_tensor(
        load_rgb_image(path, target_size=target_size),
        device=device,
    )


def tensor_to_pil_image(tensor: "torch.Tensor") -> "Image.Image":
    """Convert a tensor in CHW or NCHW format back to a clamped RGB PIL image."""
    import numpy as np
    from PIL import Image

    image_tensor = tensor
    if image_tensor.dim() == 4:
        if image_tensor.shape[0] != 1:
            raise ValueError("Only single-image batches can be converted to PIL.")
        image_tensor = image_tensor[0]
    if image_tensor.dim() != 3:
        raise ValueError("Expected a tensor with shape [C, H, W] or [1, C, H, W].")
    if image_tensor.shape[0] != 3:
        raise ValueError("Expected exactly 3 channels for RGB image conversion.")

    image_array = (
        image_tensor.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    uint8_array = (np.rint(image_array * 255.0)).astype("uint8")
    return Image.fromarray(uint8_array, mode="RGB")


def build_preview_image(
    image: "Image.Image",
    max_size: tuple[int, int] = (GUI_PREVIEW_WIDTH, GUI_PREVIEW_HEIGHT),
) -> "Image.Image":
    """Resize an image copy so it fits within the preview viewport."""
    from PIL import Image

    preview = image.copy()
    preview.thumbnail(max_size, resample=Image.Resampling.LANCZOS)
    return preview


def pil_image_to_png_bytes(image: "Image.Image") -> bytes:
    """Encode a PIL image as PNG bytes for Qt preview loading."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_preview_png_bytes(
    path: str | Path,
    max_size: tuple[int, int] = (GUI_PREVIEW_WIDTH, GUI_PREVIEW_HEIGHT),
) -> bytes:
    """Load a disk image and return preview-sized PNG bytes."""
    return pil_image_to_png_bytes(
        build_preview_image(load_rgb_image(path, target_size=max(max_size)), max_size=max_size)
    )


def tensor_to_preview_png_bytes(
    tensor: "torch.Tensor",
    max_size: tuple[int, int] = (GUI_PREVIEW_WIDTH, GUI_PREVIEW_HEIGHT),
) -> bytes:
    """Convert a generated tensor into preview-sized PNG bytes."""
    return pil_image_to_png_bytes(
        build_preview_image(tensor_to_pil_image(tensor), max_size=max_size)
    )


def save_tensor_image(
    tensor: "torch.Tensor",
    output_path: str | Path | None = None,
) -> Path:
    """Save a tensor image to disk and return the resolved path."""
    image_path, _ = build_output_paths(output_path)
    tensor_to_pil_image(tensor).save(image_path)
    return image_path
