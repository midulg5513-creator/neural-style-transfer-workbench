from __future__ import annotations

from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import nn

from neural_style.model import (
    BACKBONE_LABELS,
    ContentLoss,
    DEFAULT_LAYER_PRESET,
    FeatureBackbone,
    masked_gram_matrix,
    match_activation_histograms,
    Normalization,
    PAPER_LAYER_PRESET,
    resize_spatial_mask,
    StyleLoss,
    build_style_transfer_model,
    gram_matrix,
    load_backbone_features,
    load_resnet50_features,
    load_vgg19_features,
    resolve_layer_preset,
)


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


def test_gram_matrix_returns_batched_channel_square() -> None:
    features = torch.arange(1, 1 + (2 * 3 * 4 * 5), dtype=torch.float32).reshape(
        2, 3, 4, 5
    )

    gram = gram_matrix(features)

    assert gram.shape == (2, 3, 3)
    assert torch.allclose(gram, gram.transpose(1, 2))


def test_masked_gram_matrix_respects_spatial_mask() -> None:
    features = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)

    gram = masked_gram_matrix(features, mask)

    expected = torch.tensor([[[0.5, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    assert torch.allclose(gram, expected)


def test_resize_spatial_mask_matches_requested_shape() -> None:
    mask = torch.ones(1, 1, 4, 4)

    resized = resize_spatial_mask(mask, (2, 3))

    assert resized.shape == (1, 1, 2, 3)
    assert torch.all(resized >= 0.0)
    assert torch.all(resized <= 1.0)


def test_match_activation_histograms_reorders_values_to_reference_distribution() -> None:
    source = torch.tensor([[[[0.1, 0.4], [0.2, 0.3]]]], dtype=torch.float32)
    reference = torch.tensor([[[[1.0, 4.0], [2.0, 3.0]]]], dtype=torch.float32)

    matched = match_activation_histograms(source, reference)

    assert torch.allclose(
        torch.sort(matched.reshape(-1)).values,
        torch.sort(reference.reshape(-1)).values,
    )


def test_content_and_style_loss_targets_are_detached() -> None:
    target = torch.randn(1, 3, 4, 4, requires_grad=True)

    content_loss = ContentLoss(target)
    style_loss = StyleLoss(target)

    assert content_loss.target.requires_grad is False
    assert style_loss.target.requires_grad is False


def test_normalization_uses_registered_buffers() -> None:
    module = Normalization()

    assert "mean" in dict(module.named_buffers())
    assert "std" in dict(module.named_buffers())
    assert len(list(module.parameters())) == 0


def test_build_style_transfer_model_inserts_losses_and_trims_tail() -> None:
    cnn = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)

    model, style_losses, histogram_losses, content_losses = build_style_transfer_model(
        cnn,
        content,
        style,
        content_layers=("conv_2",),
        style_layers=("conv_1", "conv_2"),
        histogram_layers=("conv_1",),
    )

    module_names = list(model._modules.keys())
    assert module_names[0] == "normalization"
    assert "style_loss_1" in module_names
    assert "content_loss_2" in module_names
    assert isinstance(model._modules[module_names[-1]], (ContentLoss, StyleLoss))
    assert len(style_losses) == 2
    assert len(histogram_losses) == 1
    assert len(content_losses) == 1


def test_build_style_transfer_model_accepts_different_image_shapes() -> None:
    cnn = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    content = torch.randn(1, 3, 16, 12)
    style = torch.randn(1, 3, 16, 16)

    model, style_losses, histogram_losses, content_losses = build_style_transfer_model(
        cnn,
        content,
        style,
        content_layers=("conv_1",),
        style_layers=("conv_1", "conv_2"),
    )

    assert len(style_losses) == 2
    assert len(histogram_losses) == 0
    assert len(content_losses) == 1
    assert model(content).shape == (1, 4, 8, 6)


def test_build_style_transfer_model_can_replace_max_pool_with_avg_pool() -> None:
    cnn = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)

    model, _, _, _ = build_style_transfer_model(
        cnn,
        content,
        style,
        content_layers=("conv_2",),
        style_layers=("conv_1",),
        use_avg_pool=True,
    )

    assert any(isinstance(module, nn.AvgPool2d) for module in model.modules())
    assert not any(isinstance(module, nn.MaxPool2d) for module in model.modules())


def test_build_style_transfer_model_accepts_style_mask() -> None:
    cnn = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    style_mask = torch.ones(1, 1, 16, 16)

    model, style_losses, histogram_losses, _ = build_style_transfer_model(
        cnn,
        content,
        style,
        content_layers=("conv_2",),
        style_layers=("conv_1", "conv_2"),
        style_mask=style_mask,
        histogram_layers=("conv_2",),
    )

    assert len(style_losses) == 2
    assert len(histogram_losses) == 1
    assert all(style_loss.mask is not None for style_loss in style_losses)
    assert model(content).shape == (1, 4, 16, 16)


def test_build_style_transfer_model_supports_feature_backbones() -> None:
    backbone = build_toy_feature_backbone()
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)

    model, style_losses, histogram_losses, content_losses = build_style_transfer_model(
        backbone,
        content,
        style,
        content_layers=("layer3",),
        style_layers=("stem", "layer1", "layer2"),
        histogram_layers=("stem",),
    )

    assert len(style_losses) == 3
    assert len(histogram_losses) == 1
    assert len(content_losses) == 1
    assert model(content).shape == (1, 4, 16, 16)


def test_resolve_layer_preset_returns_expected_paper_mapping() -> None:
    default_content_layers, default_style_layers = resolve_layer_preset(
        DEFAULT_LAYER_PRESET
    )
    paper_content_layers, paper_style_layers = resolve_layer_preset(
        PAPER_LAYER_PRESET
    )
    resnet_content_layers, resnet_style_layers = resolve_layer_preset(
        DEFAULT_LAYER_PRESET,
        backbone="resnet50",
    )

    assert default_content_layers == ("conv_4",)
    assert default_style_layers == ("conv_1", "conv_2", "conv_3", "conv_4", "conv_5")
    assert paper_content_layers == ("conv_10",)
    assert paper_style_layers == ("conv_1", "conv_3", "conv_5", "conv_9", "conv_13")
    assert resnet_content_layers == ("layer3",)
    assert resnet_style_layers == ("stem", "layer1", "layer2", "layer3")


def test_load_vgg19_features_freezes_parameters(monkeypatch) -> None:
    called: dict[str, object] = {}

    class DummyVGG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1))

    def fake_vgg19(*, weights, progress):
        called["weights"] = weights
        called["progress"] = progress
        return DummyVGG()

    monkeypatch.setattr("neural_style.model.vgg19", fake_vgg19)

    features = load_vgg19_features(progress=True)

    assert called["progress"] is True
    assert features.training is False
    assert all(parameter.requires_grad is False for parameter in features.parameters())


def test_load_resnet50_features_freezes_parameters(monkeypatch) -> None:
    called: dict[str, object] = {}

    class DummyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=1)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.conv1(image)

    def fake_resnet50(*, weights, progress):
        called["weights"] = weights
        called["progress"] = progress
        return DummyResNet()

    monkeypatch.setattr("neural_style.model.resnet50", fake_resnet50)

    features = load_resnet50_features(progress=True)

    assert called["progress"] is True
    assert isinstance(features, FeatureBackbone)
    assert features.training is False
    assert all(
        parameter.requires_grad is False
        for parameter in features.model.parameters()
    )


def test_load_backbone_features_dispatches_to_expected_loader(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "neural_style.model.load_resnet50_features",
        lambda **_kwargs: sentinel,
    )

    loaded = load_backbone_features("resnet50")

    assert loaded is sentinel
    assert BACKBONE_LABELS["resnet50"] == "ResNet50（扩展实验）"
