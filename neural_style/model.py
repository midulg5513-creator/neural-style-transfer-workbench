"""VGG-backed model assembly helpers for neural style transfer."""

from __future__ import annotations

from collections import OrderedDict
import copy
from typing import Sequence

import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19

DEFAULT_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALIZATION_STD = (0.229, 0.224, 0.225)
DEFAULT_CONTENT_LAYERS = ("conv_4",)
DEFAULT_STYLE_LAYERS = ("conv_1", "conv_2", "conv_3", "conv_4", "conv_5")


class Normalization(nn.Module):
    """Normalize image tensors to the pretrained VGG19 input distribution."""

    def __init__(
        self,
        mean: Sequence[float] = DEFAULT_NORMALIZATION_MEAN,
        std: Sequence[float] = DEFAULT_NORMALIZATION_STD,
    ) -> None:
        super().__init__()
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        std_tensor = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute a normalized Gram matrix for a batch of feature maps."""
    if features.dim() != 4:
        raise ValueError("Expected a 4D tensor shaped [N, C, H, W].")

    batch_size, channels, height, width = features.shape
    flattened = features.reshape(batch_size, channels, height * width)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    return gram / (channels * height * width)


class ContentLoss(nn.Module):
    """Content-loss layer that preserves the forward data path."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target", target.detach())
        self.criterion = nn.MSELoss()
        self.loss = torch.tensor(0.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.loss = self.criterion(input_tensor, self.target)
        return input_tensor


class StyleLoss(nn.Module):
    """Style-loss layer that compares Gram matrices without mutating inputs."""

    def __init__(self, target_feature: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target", gram_matrix(target_feature).detach())
        self.criterion = nn.MSELoss()
        self.loss = torch.tensor(0.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.loss = self.criterion(gram_matrix(input_tensor), self.target)
        return input_tensor


def load_vgg19_features(
    device: torch.device | str | None = None,
    weights: VGG19_Weights | None = None,
    progress: bool = False,
) -> nn.Sequential:
    """Load pretrained VGG19 feature layers with frozen parameters."""
    resolved_weights = weights or VGG19_Weights.DEFAULT
    feature_stack = vgg19(weights=resolved_weights, progress=progress).features.eval()
    for parameter in feature_stack.parameters():
        parameter.requires_grad_(False)
    if device is not None:
        feature_stack = feature_stack.to(device)
    return feature_stack


def build_style_transfer_model(
    cnn: nn.Sequential,
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    normalization_mean: Sequence[float] = DEFAULT_NORMALIZATION_MEAN,
    normalization_std: Sequence[float] = DEFAULT_NORMALIZATION_STD,
    content_layers: Sequence[str] = DEFAULT_CONTENT_LAYERS,
    style_layers: Sequence[str] = DEFAULT_STYLE_LAYERS,
) -> tuple[nn.Sequential, list[StyleLoss], list[ContentLoss]]:
    """Insert normalization and loss layers into a pretrained feature stack."""
    content_image = content_image.detach()
    style_image = style_image.detach()

    cnn = copy.deepcopy(cnn).eval().to(content_image.device)
    for parameter in cnn.parameters():
        parameter.requires_grad_(False)

    normalization = Normalization(normalization_mean, normalization_std).to(
        content_image.device
    )

    content_losses: list[ContentLoss] = []
    style_losses: list[StyleLoss] = []
    model = nn.Sequential(OrderedDict([("normalization", normalization)]))

    conv_index = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv_index += 1
            name = f"conv_{conv_index}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{conv_index}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{conv_index}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_index}"
        else:
            raise RuntimeError(f"Unrecognized layer type: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{conv_index}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{conv_index}", style_loss)
            style_losses.append(style_loss)

    if not content_losses:
        raise ValueError("No content-loss layers were inserted into the model.")
    if not style_losses:
        raise ValueError("No style-loss layers were inserted into the model.")

    trimmed_children = list(model.named_children())
    last_loss_index = max(
        index
        for index, (_, module) in enumerate(trimmed_children)
        if isinstance(module, (ContentLoss, StyleLoss))
    )
    trimmed_model = nn.Sequential(
        OrderedDict(trimmed_children[: last_loss_index + 1])
    )
    return trimmed_model, style_losses, content_losses
