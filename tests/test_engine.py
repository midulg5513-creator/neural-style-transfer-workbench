from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import nn

from neural_style.engine import (
    EngineError,
    StyleTransferCancelled,
    ensure_cuda_device,
    run_style_transfer,
)


def test_ensure_cuda_device_rejects_cpu_request(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.engine.torch.cuda.is_available", lambda: True)

    with pytest.raises(EngineError, match="only supports CUDA"):
        ensure_cuda_device("cpu")


def test_run_style_transfer_rejects_nonpositive_style_strength(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.engine.ensure_cuda_device", lambda device=None: torch.device("cpu"))

    content = torch.rand(1, 3, 8, 8)
    style = torch.rand(1, 3, 8, 8)

    with pytest.raises(EngineError, match="style_strength"):
        run_style_transfer(content, style, style_strength=0.0)


def test_run_style_transfer_rejects_mask_shape_mismatch(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.engine.ensure_cuda_device", lambda device=None: torch.device("cpu"))

    content = torch.rand(1, 3, 8, 8)
    style = torch.rand(1, 3, 8, 8)
    mask = torch.rand(1, 1, 4, 4)

    with pytest.raises(EngineError, match="mask must match"):
        run_style_transfer(content, style, mask=mask)


def test_run_style_transfer_executes_with_injected_backbone(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.engine.ensure_cuda_device", lambda device=None: torch.device("cpu"))

    toy_backbone = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
    )
    content = torch.rand(1, 3, 8, 8)
    style = torch.rand(1, 3, 8, 8)
    progress_events: list[tuple[int, int, float, float]] = []

    result = run_style_transfer(
        content,
        style,
        num_steps=2,
        device="cuda",
        cnn=toy_backbone,
        progress_callback=lambda step, total, content_loss, style_loss: progress_events.append(
            (step, total, content_loss, style_loss)
        ),
    )

    assert result.output_tensor.shape == content.shape
    assert result.output_tensor.device.type == "cpu"
    assert result.num_steps == 2
    assert result.applied_keep_color is False
    assert result.applied_mask is False
    assert len(progress_events) == 2


def test_run_style_transfer_can_be_cancelled(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.engine.ensure_cuda_device", lambda device=None: torch.device("cpu"))

    toy_backbone = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
    )
    content = torch.rand(1, 3, 8, 8)
    style = torch.rand(1, 3, 8, 8)

    with pytest.raises(StyleTransferCancelled, match="cancelled"):
        run_style_transfer(
            content,
            style,
            num_steps=2,
            device="cuda",
            cnn=toy_backbone,
            cancel_callback=lambda: True,
        )
