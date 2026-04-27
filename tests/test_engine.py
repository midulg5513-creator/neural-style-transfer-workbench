from __future__ import annotations

from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import nn

from neural_style.engine import (
    allocate_progressive_steps,
    build_progressive_scale_schedule,
    build_initial_image,
    EngineError,
    StyleTransferCancelled,
    ensure_cuda_device,
    run_style_transfer,
    total_variation_loss,
)
from neural_style.model import FeatureBackbone


class ToyStage(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(image))


class ToyResidualBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = ToyStage(3, 4)
        self.layer1 = ToyStage(4, 4)
        self.layer2 = ToyStage(4, 4)
        self.layer3 = ToyStage(4, 4)
        self.layer4 = ToyStage(4, 4)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stem_output = self.stem(image)
        layer1_output = self.layer1(stem_output) + stem_output
        layer2_output = self.layer2(layer1_output) + layer1_output
        layer3_output = self.layer3(layer2_output)
        return self.layer4(layer3_output)


def build_toy_feature_backbone() -> FeatureBackbone:
    return FeatureBackbone(
        "resnet50",
        ToyResidualBackbone(),
        OrderedDict(
            [
                ("stem", "stem.relu"),
                ("layer1", "layer1.relu"),
                ("layer2", "layer2.relu"),
                ("layer3", "layer3.relu"),
                ("layer4", "layer4.relu"),
            ]
        ),
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


def test_run_style_transfer_rejects_out_of_range_content_blend(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.engine.ensure_cuda_device", lambda device=None: torch.device("cpu"))

    content = torch.rand(1, 3, 8, 8)
    style = torch.rand(1, 3, 8, 8)

    with pytest.raises(EngineError, match="content_blend"):
        run_style_transfer(content, style, content_blend=1.5)


def test_run_style_transfer_rejects_negative_tv_weight(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.engine.ensure_cuda_device", lambda device=None: torch.device("cpu"))

    content = torch.rand(1, 3, 8, 8)
    style = torch.rand(1, 3, 8, 8)

    with pytest.raises(EngineError, match="tv_weight"):
        run_style_transfer(content, style, tv_weight=-0.1)


def test_run_style_transfer_rejects_negative_histogram_weight(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.engine.ensure_cuda_device", lambda device=None: torch.device("cpu"))

    content = torch.rand(1, 3, 8, 8)
    style = torch.rand(1, 3, 8, 8)

    with pytest.raises(EngineError, match="histogram_weight"):
        run_style_transfer(content, style, histogram_weight=-0.1)


def test_run_style_transfer_rejects_unknown_init_mode(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.engine.ensure_cuda_device", lambda device=None: torch.device("cpu"))

    content = torch.rand(1, 3, 8, 8)
    style = torch.rand(1, 3, 8, 8)

    with pytest.raises(EngineError, match="init_mode"):
        run_style_transfer(content, style, init_mode="bad-mode")


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
    assert result.backbone == "vgg19"
    assert result.content_blend == 0.0
    assert len(progress_events) == 2


def test_build_initial_image_supports_content_noise_mode() -> None:
    content = torch.full((1, 3, 4, 4), 0.5)

    initialized = build_initial_image(content, init_mode="content_noise", noise_ratio=0.5)

    assert initialized.shape == content.shape
    assert not torch.allclose(initialized, content)
    assert torch.all(initialized >= 0.0)
    assert torch.all(initialized <= 1.0)


def test_total_variation_loss_is_nonnegative() -> None:
    image = torch.rand(1, 3, 8, 8)

    loss = total_variation_loss(image)

    assert loss.item() >= 0.0


def test_build_progressive_scale_schedule_returns_expected_sizes() -> None:
    assert build_progressive_scale_schedule(256, enabled=True) == (256,)
    assert build_progressive_scale_schedule(768, enabled=True) == (384, 768)
    assert build_progressive_scale_schedule(1024, enabled=True) == (256, 512, 1024)


def test_allocate_progressive_steps_preserves_total_budget() -> None:
    steps = allocate_progressive_steps(10, 3)

    assert sum(steps) == 10
    assert steps[-1] >= steps[0]


def test_run_style_transfer_can_mix_back_original_content(monkeypatch) -> None:
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

    result = run_style_transfer(
        content,
        style,
        num_steps=1,
        device="cuda",
        cnn=toy_backbone,
        content_blend=1.0,
    )

    assert torch.allclose(result.output_tensor, content, atol=1e-6)
    assert result.content_blend == 1.0


def test_run_style_transfer_supports_paper_like_options(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.engine.ensure_cuda_device", lambda device=None: torch.device("cpu"))

    toy_backbone = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2),
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

    result = run_style_transfer(
        content,
        style,
        num_steps=2,
        device="cuda",
        cnn=toy_backbone,
        tv_weight=1e-4,
        histogram_weight=0.5,
        init_mode="content_noise",
        use_avg_pool=True,
        content_layers=("conv_2",),
        style_layers=("conv_1", "conv_2"),
        histogram_layers=("conv_1",),
        scale_schedule=(4, 8),
    )

    assert result.output_tensor.shape == content.shape
    assert result.scale_schedule == (4, 8)
    assert result.backbone == "vgg19"
    assert result.layer_preset == "custom"
    assert result.histogram_weight == 0.5
    assert result.histogram_loss >= 0.0


def test_run_style_transfer_supports_feature_backbone_objects(monkeypatch) -> None:
    monkeypatch.setattr(
        "neural_style.engine.ensure_cuda_device",
        lambda device=None: torch.device("cpu"),
    )

    content = torch.rand(1, 3, 8, 8)
    style = torch.rand(1, 3, 8, 8)

    result = run_style_transfer(
        content,
        style,
        num_steps=1,
        device="cuda",
        cnn=build_toy_feature_backbone(),
        backbone="resnet50",
        content_layers=("layer3",),
        style_layers=("stem", "layer1", "layer2"),
        histogram_layers=("stem",),
        histogram_weight=0.5,
    )

    assert result.output_tensor.shape == content.shape
    assert result.backbone == "resnet50"
    assert result.layer_preset == "custom"


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
