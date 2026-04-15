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
    payload = metadata.to_dict() if isinstance(metadata, RunMetadata) else _normalize_json_value(dict(metadata))
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
    payload = metadata.to_dict() if isinstance(metadata, RunMetadata) else _normalize_json_value(dict(metadata))
    parameters = payload.get("parameters", {})
    runtime = payload.get("runtime", {})
    artifacts = payload.get("artifacts", {})

    lines = [
        f"Created at: {payload.get('created_at', 'unknown')}",
        f"Output image: {artifacts.get('output_image', 'unknown')}",
        f"Metadata JSON: {artifacts.get('metadata_file', 'unknown')}",
        f"Device: {runtime.get('cuda_device_name') or runtime.get('device', 'unknown')}",
        f"Steps: {parameters.get('num_steps', 'unknown')}",
        f"Style strength: {parameters.get('style_strength', 'unknown')}",
        f"Image size: {parameters.get('image_size', 'unknown')}",
        f"Keep color: {parameters.get('keep_color', 'unknown')}",
    ]

    return "\n".join(lines)
