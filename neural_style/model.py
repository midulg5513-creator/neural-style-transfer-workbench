"""Backbone-backed model assembly helpers for neural style transfer."""

from __future__ import annotations

from collections import OrderedDict
import copy
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    ResNet50_Weights,
    VGG19_Weights,
    resnet50,
    vgg19,
)
from torchvision.models.feature_extraction import create_feature_extractor

DEFAULT_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALIZATION_STD = (0.229, 0.224, 0.225)
DEFAULT_BACKBONE = "vgg19"
DEFAULT_LAYER_PRESET = "legacy"
PAPER_LAYER_PRESET = "paper"
BACKBONE_CHOICES = (DEFAULT_BACKBONE, "resnet50")
BACKBONE_LABELS: OrderedDict[str, str] = OrderedDict(
    [
        (DEFAULT_BACKBONE, "VGG19（论文基线）"),
        ("resnet50", "ResNet50（扩展实验）"),
    ]
)

_VGG19_CONV_FEATURE_INDICES = (
    0,
    2,
    5,
    7,
    10,
    12,
    14,
    16,
    19,
    21,
    23,
    25,
    28,
    30,
    32,
    34,
)

BACKBONE_FEATURE_NODES: dict[str, OrderedDict[str, str]] = {
    DEFAULT_BACKBONE: OrderedDict(
        (
            f"conv_{index}",
            f"features.{feature_index}",
        )
        for index, feature_index in enumerate(_VGG19_CONV_FEATURE_INDICES, start=1)
    ),
    "resnet50": OrderedDict(
        [
            ("stem", "relu"),
            ("layer1", "layer1.2.relu_2"),
            ("layer2", "layer2.3.relu_2"),
            ("layer3", "layer3.5.relu_2"),
            ("layer4", "layer4.2.relu_2"),
        ]
    ),
}

BACKBONE_LAYER_PRESETS: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    DEFAULT_BACKBONE: {
        DEFAULT_LAYER_PRESET: (
            ("conv_4",),
            ("conv_1", "conv_2", "conv_3", "conv_4", "conv_5"),
        ),
        PAPER_LAYER_PRESET: (
            ("conv_10",),
            ("conv_1", "conv_3", "conv_5", "conv_9", "conv_13"),
        ),
    },
    "resnet50": {
        DEFAULT_LAYER_PRESET: (
            ("layer3",),
            ("stem", "layer1", "layer2", "layer3"),
        ),
        PAPER_LAYER_PRESET: (
            ("layer4",),
            ("stem", "layer1", "layer2", "layer3", "layer4"),
        ),
    },
}

LAYER_PRESETS = BACKBONE_LAYER_PRESETS[DEFAULT_BACKBONE]
DEFAULT_CONTENT_LAYERS, DEFAULT_STYLE_LAYERS = BACKBONE_LAYER_PRESETS[
    DEFAULT_BACKBONE
][DEFAULT_LAYER_PRESET]

BACKBONE_HISTOGRAM_LAYERS: dict[str, dict[str, tuple[str, ...]]] = {
    DEFAULT_BACKBONE: {
        DEFAULT_LAYER_PRESET: ("conv_1", "conv_5"),
        PAPER_LAYER_PRESET: ("conv_1", "conv_9"),
    },
    "resnet50": {
        DEFAULT_LAYER_PRESET: ("stem", "layer2"),
        PAPER_LAYER_PRESET: ("stem", "layer3"),
    },
}
DEFAULT_HISTOGRAM_LAYERS = BACKBONE_HISTOGRAM_LAYERS[DEFAULT_BACKBONE][
    DEFAULT_LAYER_PRESET
]
PAPER_HISTOGRAM_LAYERS = BACKBONE_HISTOGRAM_LAYERS[DEFAULT_BACKBONE][
    PAPER_LAYER_PRESET
]


def normalize_backbone_name(backbone: str | None = DEFAULT_BACKBONE) -> str:
    """Normalize and validate a backbone identifier."""
    resolved_backbone = (
        DEFAULT_BACKBONE if backbone in (None, "") else str(backbone).strip().lower()
    )
    if resolved_backbone not in BACKBONE_CHOICES:
        available = ", ".join(BACKBONE_CHOICES)
        raise ValueError(
            f"Unknown backbone: {backbone}. Available backbones: {available}."
        )
    return resolved_backbone


def get_backbone_label(backbone: str | None = DEFAULT_BACKBONE) -> str:
    """Return the UI-friendly label for a backbone."""
    return BACKBONE_LABELS[normalize_backbone_name(backbone)]


def resolve_layer_preset(
    preset_name: str = DEFAULT_LAYER_PRESET,
    *,
    backbone: str = DEFAULT_BACKBONE,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the content/style layer tuples for a named preset."""
    resolved_backbone = normalize_backbone_name(backbone)
    presets = BACKBONE_LAYER_PRESETS[resolved_backbone]
    try:
        return presets[preset_name]
    except KeyError as exc:
        available = ", ".join(sorted(presets))
        raise ValueError(
            "Unknown layer preset: "
            f"{preset_name} for backbone {resolved_backbone}. "
            f"Available presets: {available}."
        ) from exc


def resolve_histogram_layers(
    preset_name: str = DEFAULT_LAYER_PRESET,
    *,
    backbone: str = DEFAULT_BACKBONE,
) -> tuple[str, ...]:
    """Return the histogram-loss layers for a named preset."""
    resolved_backbone = normalize_backbone_name(backbone)
    presets = BACKBONE_HISTOGRAM_LAYERS[resolved_backbone]
    try:
        return presets[preset_name]
    except KeyError as exc:
        available = ", ".join(sorted(presets))
        raise ValueError(
            "Unknown histogram preset: "
            f"{preset_name} for backbone {resolved_backbone}. "
            f"Available presets: {available}."
        ) from exc


class Normalization(nn.Module):
    """Normalize image tensors to the pretrained model input distribution."""

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


def resize_spatial_mask(
    mask: torch.Tensor,
    spatial_size: tuple[int, int],
) -> torch.Tensor:
    """Resize a single-channel spatial mask to match a feature map size."""
    if mask.dim() != 4 or mask.shape[0] != 1 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [1, 1, H, W].")
    return F.interpolate(
        mask.float(),
        size=spatial_size,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)


def masked_gram_matrix(
    features: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a Gram matrix, optionally restricted to a spatial mask."""
    if mask is None:
        return gram_matrix(features)
    if mask.dim() != 4 or mask.shape[0] != features.shape[0] or mask.shape[1] != 1:
        raise ValueError("mask must have shape [N, 1, H, W].")
    if mask.shape[-2:] != features.shape[-2:]:
        raise ValueError("mask must match the feature map spatial dimensions.")

    batch_size, channels, _, _ = features.shape
    flattened = features.reshape(batch_size, channels, -1)
    flattened_mask = mask.reshape(batch_size, 1, -1)
    weighted = flattened * flattened_mask
    gram = torch.bmm(weighted, weighted.transpose(1, 2))
    normalization = (
        channels * flattened_mask.sum(dim=2).clamp_min(1.0)
    ).view(batch_size, 1, 1)
    return gram / normalization


def _resample_sorted_values(values: torch.Tensor, target_count: int) -> torch.Tensor:
    """Linearly resample sorted 1D values to a requested sample count."""
    if values.dim() != 1:
        raise ValueError("values must be a 1D tensor.")
    if target_count <= 0:
        raise ValueError("target_count must be a positive integer.")
    if values.numel() == target_count:
        return values
    if values.numel() == 1:
        return values.expand(target_count)

    positions = torch.linspace(
        0,
        values.numel() - 1,
        steps=target_count,
        device=values.device,
        dtype=values.dtype,
    )
    lower = positions.floor().long()
    upper = positions.ceil().long()
    upper = torch.clamp(upper, max=values.numel() - 1)
    weights = positions - lower.to(dtype=values.dtype)
    return values[lower] * (1.0 - weights) + values[upper] * weights


def match_activation_histograms(
    source: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Histogram-match source activations to the reference activations channel-wise."""
    if source.dim() != 4 or reference.dim() != 4:
        raise ValueError("source and reference must be shaped [N, C, H, W].")
    if source.shape[:2] != reference.shape[:2]:
        raise ValueError("source and reference must share batch and channel dimensions.")

    matched = torch.empty_like(source)
    batch_size, channels, _, _ = source.shape
    for batch_index in range(batch_size):
        for channel_index in range(channels):
            source_flat = source[batch_index, channel_index].reshape(-1)
            reference_flat = reference[batch_index, channel_index].reshape(-1)
            _, source_order = torch.sort(source_flat)
            sorted_reference, _ = torch.sort(reference_flat)
            resampled_reference = _resample_sorted_values(
                sorted_reference,
                source_flat.numel(),
            )
            matched_flat = torch.empty_like(source_flat)
            matched_flat[source_order] = resampled_reference
            matched[batch_index, channel_index] = matched_flat.view_as(
                source[batch_index, channel_index]
            )
    return matched


class ContentLoss(nn.Module):
    """Content-loss layer that preserves the forward data path."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target", target.detach())
        self.criterion = nn.MSELoss()
        self.loss = torch.tensor(0.0)
        self.enabled = True

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            self.loss = self.criterion(input_tensor, self.target)
        return input_tensor


class StyleLoss(nn.Module):
    """Style-loss layer that compares Gram matrices without mutating inputs."""

    def __init__(
        self,
        target_feature: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if mask is not None:
            self.register_buffer("mask", mask.detach())
        else:
            self.mask = None
        self.register_buffer("target", masked_gram_matrix(target_feature, mask).detach())
        self.criterion = nn.MSELoss()
        self.loss = torch.tensor(0.0)
        self.enabled = True

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            self.loss = self.criterion(
                masked_gram_matrix(input_tensor, self.mask),
                self.target,
            )
        return input_tensor


class HistogramLoss(nn.Module):
    """Histogram-matching loss layer for stabilizing activation statistics."""

    def __init__(self, target_feature: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target", target_feature.detach())
        self.criterion = nn.MSELoss()
        self.loss = torch.tensor(0.0)
        self.enabled = True

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            with torch.no_grad():
                matched_target = match_activation_histograms(
                    input_tensor.detach(),
                    self.target,
                )
            self.loss = self.criterion(input_tensor, matched_target)
        return input_tensor


class FeatureBackbone(nn.Module):
    """Frozen backbone wrapper with stable layer aliases for feature extraction."""

    def __init__(
        self,
        backbone_name: str,
        model: nn.Module,
        feature_nodes: Mapping[str, str],
    ) -> None:
        super().__init__()
        self.backbone_name = normalize_backbone_name(backbone_name)
        self.model = model.eval()
        self.feature_nodes = OrderedDict(feature_nodes)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def resolve_feature_order(self, requested_layers: Sequence[str]) -> tuple[str, ...]:
        """Return requested layers ordered by natural backbone depth."""
        missing_layers = [
            layer_name
            for layer_name in requested_layers
            if layer_name not in self.feature_nodes
        ]
        if missing_layers:
            available = ", ".join(self.feature_nodes)
            missing = ", ".join(sorted(set(missing_layers)))
            raise ValueError(
                f"Unknown feature layers for {self.backbone_name}: {missing}. "
                f"Available layers: {available}."
            )

        requested = set(requested_layers)
        return tuple(
            layer_name
            for layer_name in self.feature_nodes
            if layer_name in requested
        )


class FeatureExtractionStyleTransferModel(nn.Module):
    """Loss-network wrapper backed by a graph feature extractor."""

    def __init__(
        self,
        normalization: Normalization,
        extractor: nn.Module,
        feature_order: Sequence[str],
        content_loss_modules: Mapping[str, ContentLoss],
        style_loss_modules: Mapping[str, StyleLoss],
        histogram_loss_modules: Mapping[str, HistogramLoss],
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.extractor = extractor
        self.feature_order = tuple(feature_order)
        self.content_loss_modules = nn.ModuleDict(content_loss_modules)
        self.style_loss_modules = nn.ModuleDict(style_loss_modules)
        self.histogram_loss_modules = nn.ModuleDict(histogram_loss_modules)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        normalized = self.normalization(input_tensor)
        features = self.extractor(normalized)
        last_output = normalized
        for layer_name in self.feature_order:
            last_output = features[layer_name]
            if layer_name in self.content_loss_modules:
                last_output = self.content_loss_modules[layer_name](last_output)
            if layer_name in self.style_loss_modules:
                last_output = self.style_loss_modules[layer_name](last_output)
            if layer_name in self.histogram_loss_modules:
                last_output = self.histogram_loss_modules[layer_name](last_output)
        return last_output


def set_loss_modules_enabled(model: nn.Module, enabled: bool) -> None:
    """Temporarily enable or disable loss computation inside a model stack."""
    for module in model.modules():
        if isinstance(module, (ContentLoss, StyleLoss, HistogramLoss)):
            module.enabled = enabled


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


def load_resnet50_features(
    device: torch.device | str | None = None,
    weights: ResNet50_Weights | None = None,
    progress: bool = False,
) -> FeatureBackbone:
    """Load a frozen ResNet50 backbone with stable stage aliases."""
    resolved_weights = weights or ResNet50_Weights.DEFAULT
    base_model = resnet50(weights=resolved_weights, progress=progress).eval()
    feature_backbone = FeatureBackbone(
        "resnet50",
        base_model,
        BACKBONE_FEATURE_NODES["resnet50"],
    )
    if device is not None:
        feature_backbone = feature_backbone.to(device)
    return feature_backbone


def load_backbone_features(
    backbone: str = DEFAULT_BACKBONE,
    *,
    device: torch.device | str | None = None,
    progress: bool = False,
) -> nn.Sequential | FeatureBackbone:
    """Load one of the supported pretrained backbones."""
    resolved_backbone = normalize_backbone_name(backbone)
    if resolved_backbone == DEFAULT_BACKBONE:
        return load_vgg19_features(device=device, progress=progress)
    if resolved_backbone == "resnet50":
        return load_resnet50_features(device=device, progress=progress)
    raise AssertionError(f"Unhandled backbone: {resolved_backbone}")


def _ordered_unique_layers(*layer_groups: Sequence[str]) -> tuple[str, ...]:
    """Combine layer sequences while preserving first-seen order."""
    return tuple(
        OrderedDict.fromkeys(
            layer_name
            for layer_group in layer_groups
            for layer_name in layer_group
        )
    )


def _replace_model_max_pool_with_avg_pool(model: nn.Module) -> nn.Module:
    """Swap a top-level max-pool layer for average pooling when available."""
    if hasattr(model, "maxpool") and isinstance(model.maxpool, nn.MaxPool2d):
        max_pool = model.maxpool
        model.maxpool = nn.AvgPool2d(
            kernel_size=max_pool.kernel_size,
            stride=max_pool.stride,
            padding=max_pool.padding,
            ceil_mode=max_pool.ceil_mode,
        )
    return model


def _build_feature_extractor_style_transfer_model(
    backbone: FeatureBackbone,
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
    content_layers: Sequence[str],
    style_layers: Sequence[str],
    histogram_layers: Sequence[str],
    style_mask: torch.Tensor | None,
    use_avg_pool: bool,
) -> tuple[nn.Module, list[StyleLoss], list[HistogramLoss], list[ContentLoss]]:
    """Build a style-transfer model around a graph feature extractor."""
    content_image = content_image.detach()
    style_image = style_image.detach()

    requested_layers = _ordered_unique_layers(
        content_layers,
        style_layers,
        histogram_layers,
    )
    feature_order = backbone.resolve_feature_order(requested_layers)

    base_model = backbone.model
    if use_avg_pool:
        base_model = _replace_model_max_pool_with_avg_pool(copy.deepcopy(base_model))
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)

    return_nodes = {
        backbone.feature_nodes[layer_name]: layer_name
        for layer_name in feature_order
    }
    extractor = create_feature_extractor(base_model, return_nodes=return_nodes)
    extractor = extractor.eval().to(content_image.device)
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    normalization = Normalization(normalization_mean, normalization_std).to(
        content_image.device
    )
    content_layer_set = set(content_layers)
    style_layer_set = set(style_layers)
    histogram_layer_set = set(histogram_layers)

    with torch.no_grad():
        content_features = extractor(normalization(content_image))
        style_features = extractor(normalization(style_image))

    content_loss_modules: dict[str, ContentLoss] = {}
    style_loss_modules: dict[str, StyleLoss] = {}
    histogram_loss_modules: dict[str, HistogramLoss] = {}
    content_losses: list[ContentLoss] = []
    style_losses: list[StyleLoss] = []
    histogram_losses: list[HistogramLoss] = []

    for layer_name in feature_order:
        if layer_name in content_layer_set:
            content_loss = ContentLoss(content_features[layer_name])
            content_loss_modules[layer_name] = content_loss
            content_losses.append(content_loss)

        if layer_name in style_layer_set:
            target_feature = style_features[layer_name]
            if style_mask is not None:
                layer_style_mask = resize_spatial_mask(
                    style_mask.to(
                        device=target_feature.device,
                        dtype=target_feature.dtype,
                    ),
                    target_feature.shape[-2:],
                )
            else:
                layer_style_mask = None
            style_loss = StyleLoss(target_feature, mask=layer_style_mask)
            style_loss_modules[layer_name] = style_loss
            style_losses.append(style_loss)

        if layer_name in histogram_layer_set:
            histogram_loss = HistogramLoss(style_features[layer_name])
            histogram_loss_modules[layer_name] = histogram_loss
            histogram_losses.append(histogram_loss)

    if not content_losses:
        raise ValueError("No content-loss layers were inserted into the model.")
    if not style_losses:
        raise ValueError("No style-loss layers were inserted into the model.")

    model = FeatureExtractionStyleTransferModel(
        normalization=normalization,
        extractor=extractor,
        feature_order=feature_order,
        content_loss_modules=content_loss_modules,
        style_loss_modules=style_loss_modules,
        histogram_loss_modules=histogram_loss_modules,
    )
    return model, style_losses, histogram_losses, content_losses


def _build_sequential_style_transfer_model(
    cnn: nn.Sequential,
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
    content_layers: Sequence[str],
    style_layers: Sequence[str],
    histogram_layers: Sequence[str],
    style_mask: torch.Tensor | None,
    use_avg_pool: bool,
) -> tuple[nn.Sequential, list[StyleLoss], list[HistogramLoss], list[ContentLoss]]:
    """Insert normalization and loss layers into a sequential feature stack."""
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
    histogram_losses: list[HistogramLoss] = []
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
            if use_avg_pool:
                layer = nn.AvgPool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    ceil_mode=layer.ceil_mode,
                )
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_index}"
        else:
            raise RuntimeError(f"Unrecognized layer type: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            set_loss_modules_enabled(model, False)
            try:
                target = model(content_image).detach()
            finally:
                set_loss_modules_enabled(model, True)
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{conv_index}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers or name in histogram_layers:
            set_loss_modules_enabled(model, False)
            try:
                target_feature = model(style_image).detach()
            finally:
                set_loss_modules_enabled(model, True)

        if name in style_layers:
            if style_mask is not None:
                layer_style_mask = resize_spatial_mask(
                    style_mask.to(
                        device=target_feature.device,
                        dtype=target_feature.dtype,
                    ),
                    target_feature.shape[-2:],
                )
            else:
                layer_style_mask = None
            style_loss = StyleLoss(target_feature, mask=layer_style_mask)
            model.add_module(f"style_loss_{conv_index}", style_loss)
            style_losses.append(style_loss)

        if name in histogram_layers:
            histogram_loss = HistogramLoss(target_feature)
            model.add_module(f"histogram_loss_{conv_index}", histogram_loss)
            histogram_losses.append(histogram_loss)

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
    return trimmed_model, style_losses, histogram_losses, content_losses


def build_style_transfer_model(
    cnn: nn.Sequential | FeatureBackbone,
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    normalization_mean: Sequence[float] = DEFAULT_NORMALIZATION_MEAN,
    normalization_std: Sequence[float] = DEFAULT_NORMALIZATION_STD,
    content_layers: Sequence[str] = DEFAULT_CONTENT_LAYERS,
    style_layers: Sequence[str] = DEFAULT_STYLE_LAYERS,
    histogram_layers: Sequence[str] = (),
    style_mask: torch.Tensor | None = None,
    use_avg_pool: bool = False,
) -> tuple[nn.Module, list[StyleLoss], list[HistogramLoss], list[ContentLoss]]:
    """Insert normalization and loss layers into a pretrained feature stack."""
    if isinstance(cnn, FeatureBackbone):
        return _build_feature_extractor_style_transfer_model(
            cnn,
            content_image,
            style_image,
            normalization_mean,
            normalization_std,
            content_layers,
            style_layers,
            histogram_layers,
            style_mask,
            use_avg_pool,
        )
    if isinstance(cnn, nn.Sequential):
        return _build_sequential_style_transfer_model(
            cnn,
            content_image,
            style_image,
            normalization_mean,
            normalization_std,
            content_layers,
            style_layers,
            histogram_layers,
            style_mask,
            use_avg_pool,
        )
    raise TypeError(
        "cnn must be an nn.Sequential or FeatureBackbone instance."
    )
