"""Data transforms for sign language recognition."""

from __future__ import annotations

from torchvision import transforms


def get_train_transforms(
    input_size: int = 224,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Get training data transforms with augmentation.

    Args:
        input_size: Target image size (square).
        mean: Normalization mean (ImageNet default).
        std: Normalization std (ImageNet default).

    Returns:
        Composed transforms for training.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_eval_transforms(
    input_size: int = 224,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Get evaluation data transforms (no augmentation).

    Args:
        input_size: Target image size (square).
        mean: Normalization mean (ImageNet default).
        std: Normalization std (ImageNet default).

    Returns:
        Composed transforms for evaluation.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
