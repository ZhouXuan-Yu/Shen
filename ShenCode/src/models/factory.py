"""Model factory for creating pretrained models with custom classification heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
    ShuffleNet_V2_X1_0_Weights,
    EfficientNet_B0_Weights,
)

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


AVAILABLE_MODELS = {
    # P0 模型（必须）
    "resnet18": "torchvision",
    "resnet34": "torchvision",
    "mobilenetv2": "torchvision",
    # P1 模型（可选）
    "shufflenetv2": "torchvision",
    "efficientnet_b0": "torchvision",
    # 扩展模型
    "resnet50": "torchvision",
    "mobilenetv3_small": "torchvision",
    "mobilenetv3_large": "torchvision",
    # timm 模型
    "efficientnet_b1": "timm",
    "convnext_tiny": "timm",
    "swin_tiny": "timm",
}


def get_available_models() -> list[str]:
    """Get list of available model names."""
    available = []
    for name, backend in AVAILABLE_MODELS.items():
        if backend == "torchvision":
            available.append(name)
        elif backend == "timm" and TIMM_AVAILABLE:
            available.append(name)
    return available


class FrameAggregator(nn.Module):
    """Aggregates frame-level features into video-level features."""

    def __init__(self, method: str = "mean"):
        super().__init__()
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate frame features.

        Args:
            x: Tensor of shape (batch, num_frames, num_features).

        Returns:
            Aggregated tensor of shape (batch, num_features).
        """
        if self.method == "mean":
            return x.mean(dim=1)
        elif self.method == "max":
            return x.max(dim=1).values
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")


class VideoClassifier(nn.Module):
    """Video classifier that processes frames and aggregates features."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = FrameAggregator(method=aggregation)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Video tensor of shape (batch, num_frames, C, H, W).

        Returns:
            Logits of shape (batch, num_classes).
        """
        batch_size, num_frames, C, H, W = x.shape

        x = x.view(batch_size * num_frames, C, H, W)

        features = self.backbone(x)

        features = features.view(batch_size, num_frames, -1)

        pooled = self.aggregator(features)

        return pooled


def _create_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create ResNet-18 model."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def _create_resnet34(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create ResNet-34 model."""
    weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet34(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def _create_mobilenetv2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create MobileNetV2 model."""
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def _create_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create ResNet-50 model."""
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def _create_shufflenetv2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create ShuffleNetV2 1.0x model."""
    weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.shufflenet_v2_x1_0(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def _create_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create EfficientNet-B0 model."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def _create_mobilenetv3_small(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create MobileNetV3-Small model."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def _create_mobilenetv3_large(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create MobileNetV3-Large model."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def _create_timm_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Create model using timm library."""
    if not TIMM_AVAILABLE:
        raise RuntimeError(f"timm is required for {model_name} but not installed.")

    timm_names = {
        "efficientnet_b1": "efficientnet_b1",
        "convnext_tiny": "convnext_tiny",
        "swin_tiny": "swin_tiny_patch4_window7_224",
    }

    timm_name = timm_names.get(model_name, model_name)
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

    return model


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    aggregation: str = "mean",
    for_video: bool = True,
) -> nn.Module:
    """Create a model for sign language recognition.

    Args:
        model_name: Name of the model ('resnet18', 'resnet34', 'mobilenetv2', etc.).
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        aggregation: Frame aggregation method ('mean', 'max').
        for_video: If True, wrap in VideoClassifier for multi-frame input.

    Returns:
        Model instance.
    """
    model_name = model_name.lower()

    if model_name not in AVAILABLE_MODELS:
        available = get_available_models()
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    backend = AVAILABLE_MODELS[model_name]

    if backend == "timm" and not TIMM_AVAILABLE:
        raise RuntimeError(f"timm is required for {model_name} but not installed.")

    if model_name == "resnet18":
        backbone = _create_resnet18(num_classes, pretrained)
    elif model_name == "resnet34":
        backbone = _create_resnet34(num_classes, pretrained)
    elif model_name == "resnet50":
        backbone = _create_resnet50(num_classes, pretrained)
    elif model_name == "mobilenetv2":
        backbone = _create_mobilenetv2(num_classes, pretrained)
    elif model_name == "mobilenetv3_small":
        backbone = _create_mobilenetv3_small(num_classes, pretrained)
    elif model_name == "mobilenetv3_large":
        backbone = _create_mobilenetv3_large(num_classes, pretrained)
    elif model_name == "shufflenetv2":
        backbone = _create_shufflenetv2(num_classes, pretrained)
    elif model_name == "efficientnet_b0":
        backbone = _create_efficientnet_b0(num_classes, pretrained)
    else:
        backbone = _create_timm_model(model_name, num_classes, pretrained)

    if for_video:
        return VideoClassifier(backbone, num_classes, aggregation)
    else:
        return backbone
