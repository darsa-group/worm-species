from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
import yaml
from torchvision.transforms import functional as tv_functional

from src.worm_species.cache.condition_variants import (
    MANIFEST_FILE,
    READY_MARKER,
    TENSOR_COLUMN,
    attach_condition_cache,
    build_condition_cache,
    cacheable_conditions,
    condition_cache_directory,
    verify_condition_cache,
)
from src.worm_species.cache.maintenance import build_persistent_cache
from src.worm_species.config.loading import load_config
from src.worm_species.data.conditions import ResolutionLoss


ROOT = Path(__file__).resolve().parents[1]


class ConditionVariantCacheTests(unittest.TestCase):
    def test_visual_config_selects_every_deterministic_expensive_condition(self) -> None:
        config = load_config(ROOT / "dev" / "genome_visual_ablation.yaml")
        selected = cacheable_conditions(config)
        transforms = [condition["transform"] for condition in selected]
        self.assertEqual(len(selected), 21)
        self.assertEqual(transforms.count("gaussian_blur_percent"), 10)
        self.assertEqual(transforms.count("patch_shuffle"), 4)
        self.assertEqual(transforms.count("resolution_loss"), 7)

    def test_cache_identity_is_condition_based_and_attaches_complete_tensors(self) -> None:
        condition = {
            "condition": "resolution_loss_087.5pct",
            "feature": "spatial_detail",
            "transform": "resolution_loss",
            "percent": 87.5,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = condition_cache_directory(root, condition)
            base_paths = [
                root / "base" / "aa" / "aa001.png",
                root / "base" / "bb" / "bb002.png",
            ]
            for base in base_paths:
                base.parent.mkdir(parents=True, exist_ok=True)
                base.write_bytes(b"base")
                tensor_path = (
                    cache / "tensors" / base.stem[:2] / f"{base.stem}.pt"
                )
                tensor_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.zeros(3, 224, 224), tensor_path)
            manifest = {
                "schema_version": 1,
                "protocol_version": 1,
                "status": "complete",
                "condition": {
                    "name": "resolution_loss_087.5pct",
                    "feature": "spatial_detail",
                    "transform": "resolution_loss",
                    "strength": 0.0,
                    "parameters": {"percent": 87.5},
                },
                "rows": 2,
                "cached_rows": 2,
            }
            (cache / MANIFEST_FILE).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (cache / READY_MARKER).write_text("", encoding="utf-8")
            frame = pd.DataFrame(
                {"_cached_image_path": [str(path) for path in base_paths]}
            )
            attached = attach_condition_cache(
                frame,
                cache_root=root,
                condition=condition,
                protocol_version=1,
            )
            self.assertTrue(
                all(Path(path).is_file() for path in attached[TENSOR_COLUMN])
            )

    def test_builder_materialises_one_complete_float32_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            image = data / "segmented.png"
            pixels = torch.arange(8 * 8 * 3, dtype=torch.uint8).reshape(
                8, 8, 3
            ).numpy()
            Image.fromarray(pixels).save(image)
            metadata = data / "metadata.csv"
            pd.DataFrame(
                [
                    {
                        "barcode": "Aporrectodea_longa_Adult_1",
                        "rel_path_seg": image.name,
                    }
                ]
            ).to_csv(metadata, index=False)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "root_dir": str(data),
                            "metadata_csv": str(metadata),
                            "image_col": "rel_path_seg",
                            "mask_col": None,
                            "target_col": "genus",
                            "group_col": "barcode",
                            "min_individuals_per_class": 1,
                            "crop_to_foreground": False,
                            "target_cols": {
                                "genus": "genus",
                                "species": "species_label",
                                "age": "life_stage",
                            },
                        },
                        "preprocessing": {"image_size": 4},
                        "cache": {
                            "enabled": True,
                            "format": "png",
                            "num_workers": 1,
                            "condition_variants": {
                                "enabled": True,
                                "protocol_version": 1,
                                "storage": "torch_float32",
                            },
                        },
                        "sweep": {
                            "enabled": True,
                            "parameters": {},
                            "conditions": [
                                {
                                    "name": "resolution_loss_050pct",
                                    "feature": "spatial_detail",
                                    "transform": "resolution_loss",
                                    "strength": 50.0,
                                    "parameters": {"percent": 50},
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            base = root / "base_cache"
            variants = root / "condition_cache"
            base_result = build_persistent_cache(
                config_path,
                data_root=data,
                metadata_csv=metadata,
                cache_dir=base,
            )
            self.assertEqual(base_result.cached_rows, 1)
            result = build_condition_cache(
                config_path,
                data_root=data,
                metadata_csv=metadata,
                base_cache_dir=base,
                condition_cache_root=variants,
                condition_index=0,
                num_workers=1,
            )
            self.assertEqual(result.status, "built")
            verified = verify_condition_cache(result.cache_dir)
            self.assertEqual(verified.cached_rows, 1)
            tensor_path = next(Path(result.cache_dir).rglob("*.pt"))
            tensor = torch.load(
                tensor_path, map_location="cpu", weights_only=True
            )
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tuple(tensor.shape), (3, 4, 4))
            with Image.open(next(base.rglob("*.png"))) as cached_image:
                expected = ResolutionLoss(50)(
                    tv_functional.to_tensor(cached_image.convert("RGB"))
                )
            torch.testing.assert_close(
                tensor, expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
