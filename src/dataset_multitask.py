"""Compatibility exports for the historical multi-task data module.

The implementation is split by responsibility under ``src.worm_species.data``.
All previously importable public names and the detected private helper aliases
remain available here.
"""

from src.worm_species.data.conditions import (
    ChannelShuffle,
    PatchShuffle,
    TensorBilateralFilter,
    TensorGaussianBlur,
    build_condition_transform,
    build_test_condition_transform,
)
from src.worm_species.data.cropping import (
    foreground_bbox_from_image,
    foreground_bbox_from_mask,
    pad_square_bbox,
)
from src.worm_species.data.datasets import MultiTaskWormImageDataset
from src.worm_species.data.image_validation import resolve_path
from src.worm_species.data.labels import (
    DEFAULT_MISSING_VALUES,
    MISSING_LABEL,
    MISSING_LABEL_INDEX,
    clean_label_value,
    get_target_cols,
    is_missing_label,
    missing_values_from_config,
    normalise_missing_series,
)
from src.worm_species.data.metadata import (
    mask_rare_classes_per_task,
    prepare_metadata,
)
from src.worm_species.data.taxonomy import (
    apply_taxonomic_uncertainty_rules,
    derive_taxonomy_and_stage,
    parse_taxonomy_from_barcode,
    strip_final_number,
)
from src.worm_species.data.transforms import ColourRetention, build_transforms

_clean_label_value = clean_label_value
_derive_taxonomy_and_stage = derive_taxonomy_and_stage
_missing_values_from_cfg = missing_values_from_config
_normalise_missing_series = normalise_missing_series

__all__ = [
    "ChannelShuffle",
    "ColourRetention",
    "DEFAULT_MISSING_VALUES",
    "MISSING_LABEL",
    "MISSING_LABEL_INDEX",
    "MultiTaskWormImageDataset",
    "PatchShuffle",
    "TensorBilateralFilter",
    "TensorGaussianBlur",
    "apply_taxonomic_uncertainty_rules",
    "build_condition_transform",
    "build_test_condition_transform",
    "build_transforms",
    "foreground_bbox_from_image",
    "foreground_bbox_from_mask",
    "get_target_cols",
    "is_missing_label",
    "mask_rare_classes_per_task",
    "pad_square_bbox",
    "parse_taxonomy_from_barcode",
    "prepare_metadata",
    "resolve_path",
    "strip_final_number",
]
