from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.transforms import functional as transform_functional


class ColourRetention:
    """Retain a deterministic fraction of image chroma."""

    def __init__(self, retention: float = 1.0):
        retention = float(retention)
        if not 0.0 <= retention <= 1.0:
            raise ValueError(
                f"colour_retention must be between 0 and 1, got {retention}."
            )
        self.retention = retention

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise TypeError(
                "ColourRetention must be applied after transforms.ToTensor()."
            )
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected an RGB tensor with shape [3, H, W], got {tuple(image.shape)}."
            )
        if self.retention == 1.0:
            return image
        greyscale = transform_functional.rgb_to_grayscale(
            image, num_output_channels=3
        )
        return torch.lerp(greyscale, image, self.retention).clamp_(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(retention={self.retention:.3f})"


def build_transforms(
    image_size: int,
    train: bool,
    colour_retention: float = 1.0,
) -> transforms.Compose:
    """Build the legacy ordinary multi-task transform sequence."""
    operations = [transforms.Resize((image_size, image_size))]
    if train:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=270),
        ])
    operations.extend([
        transforms.ToTensor(),
        ColourRetention(colour_retention),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transforms.Compose(operations)
