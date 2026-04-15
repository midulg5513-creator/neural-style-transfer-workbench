"""Qt worker helpers for running NST jobs off the GUI thread."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Signal, Slot

from .engine import EngineError, StyleTransferCancelled, run_style_transfer
from .mask import load_mask_tensor
from .metadata import build_run_metadata, format_run_summary, save_run_metadata
from .utils import (
    build_output_paths,
    load_image_tensor,
    save_tensor_image,
    tensor_to_preview_png_bytes,
)
from .validation import (
    ValidationError,
    require_cuda,
    validate_image_path,
    validate_image_size,
    validate_num_steps,
    validate_optional_image_path,
    validate_output_image_path,
    validate_style_strength,
)


@dataclass(frozen=True)
class StyleTransferRunRequest:
    """Validated parameters required for one GUI-triggered NST run."""

    content_path: Path
    style_path: Path
    output_path: Path
    num_steps: int
    style_strength: float
    image_size: int
    keep_color: bool
    mask_path: Path | None = None


@dataclass(frozen=True)
class StyleTransferRunProgress:
    """Progress payload emitted back to the GUI thread."""

    stage: str
    message: str
    percent: int
    current_step: int | None = None
    total_steps: int | None = None
    content_loss: float | None = None
    style_loss: float | None = None


@dataclass(frozen=True)
class StyleTransferRunResult:
    """Structured data returned after a successful GUI-triggered run."""

    output_image_path: Path
    metadata_path: Path
    preview_png_bytes: bytes
    metadata_summary: str
    device: str
    content_loss: float
    style_loss: float
    applied_keep_color: bool
    applied_mask: bool


ProgressHandler = Callable[[StyleTransferRunProgress], None]
CancelHandler = Callable[[], bool]


def execute_style_transfer_request(
    request: StyleTransferRunRequest,
    *,
    progress_handler: ProgressHandler | None = None,
    cancel_handler: CancelHandler | None = None,
) -> StyleTransferRunResult:
    """Run the full NST pipeline for one request and return saved artifacts."""
    def emit(stage: str, message: str, percent: int, **kwargs: object) -> None:
        if progress_handler is None:
            return
        progress_handler(
            StyleTransferRunProgress(
                stage=stage,
                message=message,
                percent=max(0, min(100, percent)),
                current_step=kwargs.get("current_step"),
                total_steps=kwargs.get("total_steps"),
                content_loss=kwargs.get("content_loss"),
                style_loss=kwargs.get("style_loss"),
            )
        )

    emit("setup", "Validating CUDA environment...", 2)

    def raise_if_cancelled() -> None:
        if cancel_handler is not None and cancel_handler():
            raise StyleTransferCancelled("Style transfer was cancelled.")

    device = require_cuda()
    raise_if_cancelled()

    content_path = validate_image_path(request.content_path, "content image")
    style_path = validate_image_path(request.style_path, "style image")
    mask_path = validate_optional_image_path(request.mask_path, "mask image")
    num_steps = validate_num_steps(request.num_steps)
    style_strength = validate_style_strength(request.style_strength)
    image_size = validate_image_size(request.image_size)
    output_path = validate_output_image_path(request.output_path)
    image_output_path, metadata_output_path = build_output_paths(output_path)

    emit("setup", "Loading content image...", 8)
    content_tensor = load_image_tensor(content_path, target_size=image_size, device=device)
    raise_if_cancelled()

    emit("setup", "Loading style image...", 12)
    style_tensor = load_image_tensor(style_path, target_size=image_size, device=device)
    raise_if_cancelled()

    if mask_path is not None:
        emit("setup", "Loading mask image...", 16)
        mask_tensor = load_mask_tensor(mask_path, target_size=image_size, device=device)
        raise_if_cancelled()
    else:
        mask_tensor = None

    def on_engine_progress(step: int, total: int, content_loss: float, style_loss: float) -> None:
        percent = 20 + round((step / total) * 70)
        emit(
            "optimizing",
            f"Optimizing on GPU: step {step}/{total}",
            percent,
            current_step=step,
            total_steps=total,
            content_loss=content_loss,
            style_loss=style_loss,
        )

    emit("setup", "Starting style transfer...", 20)
    result = run_style_transfer(
        content_tensor,
        style_tensor,
        num_steps=num_steps,
        style_strength=style_strength,
        keep_color=request.keep_color,
        mask=mask_tensor,
        device=device,
        progress_callback=on_engine_progress,
        cancel_callback=cancel_handler,
    )

    emit("saving", "Saving output image...", 94)
    raise_if_cancelled()
    saved_image_path = save_tensor_image(result.output_tensor, image_output_path)
    preview_png_bytes = tensor_to_preview_png_bytes(result.output_tensor)

    metadata = build_run_metadata(
        content_path=content_path,
        style_path=style_path,
        mask_path=mask_path,
        output_image_path=saved_image_path,
        metadata_path=metadata_output_path,
        parameters={
            "num_steps": num_steps,
            "style_strength": style_strength,
            "image_size": image_size,
            "keep_color": request.keep_color,
        },
        device=device,
    )

    emit("saving", "Writing metadata sidecar...", 98)
    raise_if_cancelled()
    saved_metadata_path = save_run_metadata(metadata)

    emit("complete", "Run completed successfully.", 100)
    return StyleTransferRunResult(
        output_image_path=saved_image_path,
        metadata_path=saved_metadata_path,
        preview_png_bytes=preview_png_bytes,
        metadata_summary=format_run_summary(metadata),
        device=result.device,
        content_loss=result.content_loss,
        style_loss=result.style_loss,
        applied_keep_color=result.applied_keep_color,
        applied_mask=result.applied_mask,
    )


class StyleTransferWorker(QObject):
    """Worker object moved into a QThread for background NST execution."""

    progress = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    finished = Signal()

    def __init__(self, request: StyleTransferRunRequest) -> None:
        super().__init__()
        self._request = request
        self._cancel_requested = False

    @Slot()
    def run(self) -> None:
        """Execute the NST request and emit worker lifecycle signals."""
        try:
            result = execute_style_transfer_request(
                self._request,
                progress_handler=self.progress.emit,
                cancel_handler=self.is_cancel_requested,
            )
        except StyleTransferCancelled as exc:
            self.cancelled.emit(str(exc))
        except (ValidationError, EngineError, OSError, RuntimeError, ValueError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive boundary for GUI use
            self.failed.emit(f"Unexpected error: {exc}")
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()

    @Slot()
    def cancel(self) -> None:
        """Request cooperative cancellation for the running NST job."""
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        """Return whether the user has requested cancellation."""
        return self._cancel_requested
