# Inputs and generated artifacts

The public dataset has one global `metadata.csv` alongside extracted `2025/` and `2026/` photo directories. Its 9,356 rows link 9,295 available segmented photographs. It contains no calibration/reference photographs, masks or per-photo sidecars. `configs/release_inputs.json` fixes the metadata and checkpoint hashes for this release.

`make prepare-data` creates a runtime view under `work/inputs/`. It writes the analysis manifest at `dataset/metadata/images.csv`, restores `splits/split_csv/{train,val,test}_split.csv` in the frozen order and writes reference metadata. The year directories link to the public dataset. No image bytes are copied. These are generated runtime tables, not additional downloaded annotation files. A missing mask field is added as empty for compatibility. Classification uses the supplied segmented images without masks or another foreground crop.

`configs/reference_split_order.json` stores the original ordered image IDs. The 6,191 reference images belong to 718/166/221 disjoint worms. `configs/reference_control_selections.json` stores the original 25 condition definitions and removed-worm sets. New control plans must reproduce those sets. Unresolved species targets, source taxon and string location codes remain explicit. Species complexes remain operational labels.

The model directory contains the selected ConvNeXt checkpoint, its class maps/settings, and the segmentation checkpoint/settings. The single-model CLI checks the relevant checkpoint hash, uses saved preprocessing and strictly loads the classifier state dictionary. It does not require all 90 baseline checkpoints.

Full reproduction first trains the omitted grid. Each fit produces a checkpoint, config, class map, run summary, original-test predictions, completion marker and receipt. Inference planning requires all 90 baseline fits. The 30 ConvNeXt fits supply native features and worm-disjoint calibration plans. Visual rescoring requires every condition/seed pair. Control compilation requires all 750 validated prediction artifacts. Incomplete grids fail and are not filled with the supplied single checkpoint.

Generated analysis paths keep inputs, predictions, features, fixed-rank analysis, adaptive-rank analysis and controls separate. Figures read those outputs. No saved paper predictions, features or result tables are bundled in this source release. Existing results may be configured as read-only inputs when their plans, maps, hashes and split files are also available.

The acquisition-layout dataset builder is retained as historical source. It requires original capture archives outside this reduced release. Training, supplied-model inference and mask generation use the public photographs directly. Segmentation retraining needs independent human annotations and their original partition; generated masks are not a replacement.
