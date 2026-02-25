"""Download all pretrained models for offline use."""

from __future__ import annotations

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


def download_all_models():
    """Download all pretrained models to local cache."""

    print("=" * 60)
    print("下载预训练模型到本地缓存")
    print("=" * 60)

    # ==================== P0 模型（必须）====================
    print("\n--- P0 模型（必须）---")

    print("\n[P0] ResNet-18 (ImageNet)...")
    models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print("  ✓ 已下载")

    print("\n[P0] ResNet-34 (ImageNet)...")
    models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    print("  ✓ 已下载")

    print("\n[P0] MobileNetV2 (ImageNet)...")
    models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    print("  ✓ 已下载")

    # ==================== P1 模型（可选）====================
    print("\n--- P1 模型（可选）---")

    print("\n[P1] ShuffleNetV2 1.0x (ImageNet) via torchvision...")
    models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    print("  ✓ 已下载")

    print("\n[P1] EfficientNet-B0 (ImageNet) via torchvision...")
    models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    print("  ✓ 已下载")

    # ==================== 扩展模型（备选）====================
    print("\n--- 扩展模型（备选）---")

    print("\n[扩展] ResNet-50 (ImageNet)...")
    models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    print("  ✓ 已下载")

    print("\n[扩展] MobileNetV3-Small (ImageNet)...")
    models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    print("  ✓ 已下载")

    print("\n[扩展] MobileNetV3-Large (ImageNet)...")
    models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    print("  ✓ 已下载")

    # timm 扩展模型
    if TIMM_AVAILABLE:
        print("\n--- timm 扩展模型 ---")

        timm_models = [
            ("efficientnet_b1", "EfficientNet-B1"),
            ("convnext_tiny", "ConvNeXt-Tiny"),
            ("swin_tiny_patch4_window7_224", "Swin-Tiny"),
        ]

        for model_name, display_name in timm_models:
            try:
                print(f"\n[timm] {display_name}...")
                timm.create_model(model_name, pretrained=True)
                print("  ✓ 已下载")
            except Exception as e:
                print(f"  ✗ 下载失败: {e}")

    print("\n" + "=" * 60)
    print("所有模型下载完成！")
    print("缓存位置: ~/.cache/torch/hub/checkpoints/")
    print("=" * 60)

    # 打印模型汇总表
    print("\n模型汇总：")
    print("-" * 40)
    print("P0（必须）: ResNet-18, ResNet-34, MobileNetV2")
    print("P1（可选）: ShuffleNetV2, EfficientNet-B0")
    print("扩展：ResNet-50, MobileNetV3-S/L, EfficientNet-B1, ConvNeXt-Tiny, Swin-Tiny")


if __name__ == "__main__":
    download_all_models()
