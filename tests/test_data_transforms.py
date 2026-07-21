from __future__ import annotations

import unittest

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.worm_species.data.conditions import ChannelShuffle
from src.worm_species.data.conditions import (
    ColourRetention as ConditionColourRetention,
)
from src.worm_species.data.conditions import PatchShuffle
from src.worm_species.data.conditions import build_condition_transform
from src.worm_species.data.conditions import build_test_condition_transform
from src.worm_species.data.transforms import ColourRetention
from src.worm_species.data.transforms import build_split_transform
from src.worm_species.data.transforms import build_transforms


def operation_names(transform: transforms.Compose) -> list[str]:
    return [type(operation).__name__ for operation in transform.transforms]


class CanonicalTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        pixels = np.arange(19 * 23 * 3, dtype=np.uint8).reshape(19, 23, 3)
        self.image = Image.fromarray(pixels, mode="RGB")

    def test_colour_retention_legacy_export_is_the_canonical_class(self) -> None:
        self.assertIs(ColourRetention, ConditionColourRetention)

    def test_legacy_wrappers_preserve_default_order_and_parameters(self) -> None:
        training = build_transforms(16, train=True)
        evaluation = build_transforms(16, train=False)

        self.assertEqual(
            operation_names(training),
            [
                "Resize",
                "RandomHorizontalFlip",
                "RandomVerticalFlip",
                "RandomRotation",
                "ToTensor",
                "ColourRetention",
                "Normalize",
            ],
        )
        self.assertEqual(
            operation_names(evaluation),
            ["Resize", "ToTensor", "ColourRetention", "Normalize"],
        )
        self.assertEqual(training.transforms[0].size, (16, 16))
        self.assertEqual(training.transforms[1].p, 0.5)
        self.assertEqual(training.transforms[2].p, 0.5)
        self.assertEqual(training.transforms[3].degrees, [-270.0, 270.0])
        self.assertEqual(training.transforms[5].retention, 1.0)
        self.assertIsInstance(training.transforms[-1].mean, list)
        self.assertIsInstance(training.transforms[-1].std, list)
        self.assertEqual(
            list(training.transforms[-1].mean),
            [0.485, 0.456, 0.406],
        )
        self.assertEqual(
            list(training.transforms[-1].std),
            [0.229, 0.224, 0.225],
        )

    def test_legacy_training_pixels_match_the_historical_sequence(self) -> None:
        expected = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=270),
            transforms.ToTensor(),
            ColourRetention(0.4),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        actual = build_transforms(16, train=True, colour_retention=0.4)

        torch.manual_seed(2026)
        expected_output = expected(self.image)
        torch.manual_seed(2026)
        actual_output = actual(self.image)
        self.assertTrue(torch.equal(actual_output, expected_output))

    def test_split_composer_resizes_and_can_disable_normalisation(self) -> None:
        transform = build_split_transform(
            split="validation",
            preprocessing={
                "image_size": 12,
                "normalisation": {"enabled": False},
            },
        )
        self.assertEqual(
            operation_names(transform),
            ["Resize", "ToTensor", "ColourRetention"],
        )
        output = transform(self.image)
        self.assertEqual(tuple(output.shape), (3, 12, 12))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_augmentation_is_train_only_and_disabled_mode_is_deterministic(self) -> None:
        training = build_split_transform(
            split="train",
            preprocessing={"image_size": 16},
        )
        validation = build_split_transform(
            split="validation",
            preprocessing={"image_size": 16},
        )
        test = build_split_transform(
            split="test",
            preprocessing={"image_size": 16},
        )
        disabled = build_split_transform(
            split="train",
            preprocessing={"image_size": 16},
            augmentation={"enabled": False},
        )

        self.assertIn("RandomRotation", operation_names(training))
        for deterministic in (validation, test, disabled):
            self.assertNotIn("RandomRotation", operation_names(deterministic))
            self.assertTrue(torch.equal(
                deterministic(self.image),
                deterministic(self.image),
            ))

    def test_diagnostic_override_is_explicit_and_rotation_zero_is_retained(self) -> None:
        transform = build_split_transform(
            split="test",
            preprocessing={"image_size": 16},
            augmentation={
                "horizontal_flip": {"enabled": False},
                "vertical_flip": {"enabled": False},
                "rotation": {"enabled": True, "degrees": 0},
            },
            apply_augmentation=True,
        )
        self.assertEqual(
            operation_names(transform),
            [
                "Resize",
                "RandomRotation",
                "ToTensor",
                "ColourRetention",
                "Normalize",
            ],
        )
        self.assertEqual(transform.transforms[1].degrees, [-0.0, 0.0])

    def test_gaussian_blur_augmentation_is_random_and_train_only(self) -> None:
        augmentation = {
            "horizontal_flip": {"enabled": False},
            "vertical_flip": {"enabled": False},
            "rotation": {"enabled": False},
            "gaussian_blur": {
                "enabled": True,
                "probability": 0.5,
                "kernel_size": 5,
                "sigma": [0.1, 2.0],
            },
        }
        training = build_split_transform(
            split="train",
            preprocessing={"image_size": 16},
            augmentation=augmentation,
        )
        validation = build_split_transform(
            split="validation",
            preprocessing={"image_size": 16},
            augmentation=augmentation,
        )

        self.assertEqual(
            operation_names(training),
            ["Resize", "RandomApply", "ToTensor", "ColourRetention", "Normalize"],
        )
        random_apply = training.transforms[1]
        self.assertEqual(random_apply.p, 0.5)
        self.assertIsInstance(random_apply.transforms[0], transforms.GaussianBlur)
        self.assertEqual(random_apply.transforms[0].kernel_size, (5, 5))
        self.assertEqual(random_apply.transforms[0].sigma, (0.1, 2.0))
        self.assertNotIn("RandomApply", operation_names(validation))

    def test_condition_is_after_tensor_conversion_and_before_normalisation(self) -> None:
        transform = build_split_transform(
            split="validation",
            preprocessing={"image_size": 16},
            condition={
                "transform": "channel_shuffle",
                "parameters": {"order": [1, 2, 0]},
            },
        )
        self.assertEqual(
            operation_names(transform),
            [
                "Resize",
                "ToTensor",
                "ColourRetention",
                "ChannelShuffle",
                "Normalize",
            ],
        )
        self.assertIsInstance(transform.transforms[3], ChannelShuffle)
        self.assertEqual(transform.transforms[3].order, (1, 2, 0))

    def test_matched_condition_is_on_all_splits_but_only_train_is_random(self) -> None:
        condition = {
            "transform": "patch_shuffle",
            "parameters": {"grid_size": 4, "seed": 2026},
        }
        transforms_by_split = {
            split: build_split_transform(
                split=split,
                preprocessing={"image_size": 16},
                condition=condition,
            )
            for split in ("train", "validation", "test")
        }
        expected_permutation = next(
            operation.permutation
            for operation in transforms_by_split["test"].transforms
            if isinstance(operation, PatchShuffle)
        )

        for transform in transforms_by_split.values():
            shuffles = [
                operation
                for operation in transform.transforms
                if isinstance(operation, PatchShuffle)
            ]
            self.assertEqual(len(shuffles), 1)
            self.assertEqual(shuffles[0].seed, 2026)
            self.assertEqual(shuffles[0].permutation, expected_permutation)
        self.assertIn(
            "RandomRotation",
            operation_names(transforms_by_split["train"]),
        )
        self.assertNotIn(
            "RandomRotation",
            operation_names(transforms_by_split["validation"]),
        )
        self.assertNotIn(
            "RandomRotation",
            operation_names(transforms_by_split["test"]),
        )

    def test_fixed_rgb_stress_wrapper_is_deterministic_and_test_only(self) -> None:
        condition = {"transform": "channel_shuffle", "order": [2, 0, 1]}
        training = build_transforms(16, train=True)
        stress = build_test_condition_transform(16, condition)
        matched_validation = build_condition_transform(16, False, condition)

        self.assertNotIn("ChannelShuffle", operation_names(training))
        self.assertIn("ChannelShuffle", operation_names(stress))
        self.assertTrue(torch.equal(stress(self.image), stress(self.image)))
        self.assertTrue(torch.equal(
            stress(self.image),
            matched_validation(self.image),
        ))

    def test_invalid_split_and_image_size_fail_at_the_composer_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "split must be"):
            build_split_transform(split="val")  # type: ignore[arg-type]
        for image_size in (0, -1, 1.5, True):
            with self.subTest(image_size=image_size):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    build_split_transform(
                        split="test",
                        preprocessing={"image_size": image_size},
                    )


if __name__ == "__main__":
    unittest.main()
