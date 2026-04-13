from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import nn

from neural_style.model import (
    ContentLoss,
    Normalization,
    StyleLoss,
    build_style_transfer_model,
    gram_matrix,
    load_vgg19_features,
)


def test_gram_matrix_returns_batched_channel_square() -> None:
    features = torch.arange(1, 1 + (2 * 3 * 4 * 5), dtype=torch.float32).reshape(
        2, 3, 4, 5
    )

    gram = gram_matrix(features)

    assert gram.shape == (2, 3, 3)
    assert torch.allclose(gram, gram.transpose(1, 2))


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

    model, style_losses, content_losses = build_style_transfer_model(
        cnn,
        content,
        style,
        content_layers=("conv_2",),
        style_layers=("conv_1", "conv_2"),
    )

    module_names = list(model._modules.keys())
    assert module_names[0] == "normalization"
    assert "style_loss_1" in module_names
    assert "content_loss_2" in module_names
    assert isinstance(model._modules[module_names[-1]], (ContentLoss, StyleLoss))
    assert len(style_losses) == 2
    assert len(content_losses) == 1


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
