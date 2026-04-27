"""JSON metadata helpers for generated style-transfer runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import platform
from typing import Any, Mapping

from .utils import ensure_directory

SCHEMA_VERSION = "1.0"


def _now_isoformat() -> str:
    """Return a timezone-aware timestamp for metadata output."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_path(value: str | Path | None) -> str | None:
    """Convert filesystem-like values to absolute string paths."""
    if value in (None, ""):
        return None
    return str(Path(value).expanduser().resolve())


def _normalize_json_value(value: Any) -> Any:
    """Recursively coerce metadata values into JSON-safe primitives."""
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return value


def collect_device_metadata(
    device: "torch.device | str | None" = None,
) -> dict[str, Any]:
    """Collect runtime device information for a generated run."""
    requested_device = "cuda" if device is None else str(device)
    metadata: dict[str, Any] = {
        "device": requested_device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch
    except ModuleNotFoundError:
        metadata.update(
            {
                "torch_available": False,
                "torch_version": None,
                "cuda_available": False,
            }
        )
        return metadata

    resolved_device = torch.device(requested_device)
    metadata.update(
        {
            "device": str(resolved_device),
            "torch_available": True,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        device_index = (
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        )
        metadata["cuda_device_index"] = device_index
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device_index)
    return metadata


@dataclass(frozen=True)
class RunMetadata:
    """Structured metadata for one generated result."""

    content_image: Path
    style_image: Path
    output_image: Path
    metadata_file: Path
    parameters: Mapping[str, Any]
    device: Mapping[str, Any]
    mask_image: Path | None = None
    created_at: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the metadata into a stable JSON-friendly dictionary."""
        created_at = self.created_at or _now_isoformat()
        return {
            "schema_version": self.schema_version,
            "created_at": created_at,
            "inputs": {
                "content_image": _normalize_path(self.content_image),
                "style_image": _normalize_path(self.style_image),
                "mask_image": _normalize_path(self.mask_image),
            },
            "parameters": _normalize_json_value(dict(self.parameters)),
            "runtime": _normalize_json_value(dict(self.device)),
            "artifacts": {
                "output_image": _normalize_path(self.output_image),
                "metadata_file": _normalize_path(self.metadata_file),
            },
        }


def build_run_metadata(
    *,
    content_path: str | Path,
    style_path: str | Path,
    output_image_path: str | Path,
    metadata_path: str | Path,
    parameters: Mapping[str, Any],
    mask_path: str | Path | None = None,
    device: "torch.device | str | None" = None,
) -> RunMetadata:
    """Build a structured metadata object for a generated NST run."""
    return RunMetadata(
        content_image=Path(content_path),
        style_image=Path(style_path),
        mask_image=None if mask_path in (None, "") else Path(mask_path),
        output_image=Path(output_image_path),
        metadata_file=Path(metadata_path),
        parameters=dict(parameters),
        device=collect_device_metadata(device),
        created_at=_now_isoformat(),
    )


def save_run_metadata(
    metadata: RunMetadata | Mapping[str, Any],
    metadata_path: str | Path | None = None,
) -> Path:
    """Write a metadata sidecar JSON file and return its resolved path."""
    payload = (
        metadata.to_dict()
        if isinstance(metadata, RunMetadata)
        else _normalize_json_value(dict(metadata))
    )
    target_path = (
        Path(metadata_path).expanduser()
        if metadata_path not in (None, "")
        else Path(payload["artifacts"]["metadata_file"]).expanduser()
    )
    ensure_directory(target_path.parent)
    target_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target_path


def format_run_summary(metadata: RunMetadata | Mapping[str, Any]) -> str:
    """Build a short text summary suitable for the desktop UI."""
    payload = (
        metadata.to_dict()
        if isinstance(metadata, RunMetadata)
        else _normalize_json_value(dict(metadata))
    )
    inputs = payload.get("inputs", {})
    parameters = payload.get("parameters", {})
    runtime = payload.get("runtime", {})
    artifacts = payload.get("artifacts", {})
    keep_color = parameters.get("keep_color", None)
    content_blend = parameters.get("content_blend", None)
    tv_weight = parameters.get("tv_weight", None)
    histogram_weight = parameters.get("histogram_weight", None)
    histogram_enabled = parameters.get("histogram_loss", None)
    init_mode = parameters.get("init_mode", None)
    use_avg_pool = parameters.get("use_avg_pool", None)
    enhanced_mode = parameters.get("enhanced_mode", None)
    paper_mode = parameters.get("paper_mode", None)
    backbone = parameters.get("backbone", None)
    layer_preset = parameters.get("layer_preset", None)
    scale_schedule = parameters.get("scale_schedule", None)
    mask_image = inputs.get("mask_image", None)

    lines = [
        f"生成时间：{payload.get('created_at', '未知')}",
        f"内容图像：{inputs.get('content_image', '未知')}",
        f"风格图像：{inputs.get('style_image', '未知')}",
        f"输出图像：{artifacts.get('output_image', '未知')}",
        f"参数记录：{artifacts.get('metadata_file', '未知')}",
        f"运行设备：{runtime.get('cuda_device_name') or runtime.get('device', '未知')}",
        f"优化步数：{parameters.get('num_steps', '未知')}",
        f"风格强度：{parameters.get('style_strength', '未知')}",
        f"原图保留度：{content_blend if content_blend is not None else '未知'}",
        f"TV 正则：{tv_weight if tv_weight is not None else '未知'}",
        f"直方图损失：{'是' if histogram_enabled else '否' if histogram_enabled is not None else '未知'}",
        f"直方图权重：{histogram_weight if histogram_weight is not None else '未知'}",
        f"初始化模式：{init_mode if init_mode is not None else '未知'}",
        f"平均池化：{'是' if use_avg_pool else '否' if use_avg_pool is not None else '未知'}",
        f"强化模式：{'是' if enhanced_mode else '否' if enhanced_mode is not None else '未知'}",
        f"论文模式：{'是' if paper_mode else '否' if paper_mode is not None else '未知'}",
        f"特征骨干：{backbone if backbone is not None else '未知'}",
        f"层配置：{layer_preset if layer_preset is not None else '未知'}",
        f"多尺度：{scale_schedule if scale_schedule is not None else '未知'}",
        f"图像尺寸：{parameters.get('image_size', '未知')}",
        f"保留原色：{'是' if keep_color is True else '否' if keep_color is False else '未知'}",
        f"使用遮罩：{'是' if mask_image else '否'}",
    ]

    return "\n".join(lines)
