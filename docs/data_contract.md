# Required artifacts

Files are external inputs, not committed research data. Paths are configured in `configs/paper.yaml` and resolve from the repository root. Absolute paths work as overrides. Separate output folders preserve reference results.

## Dataset and original partitions

`paths.dataset` must contain:

```
metadata/images.csv
<raw, segmented and mask paths referenced by the manifest>
```

Manifest fields used by this pipeline include `image_id`, `individual_id`, `barcode`, `filename`, `year`, `camera`, `kind`, `taxon`, `life_stage`, `location_code`, `rel_path_raw`, `rel_path_seg`, `rel_path_segmask`, `segmentation_status`, `original_paper_split`. The camera values are `original_camera` (2025 Canon), `gphoto2` (2026 Canon), and `webcam` (2026 USB). `image_id` must be globally unique and a safe filename component. Biological identity, taxon and stage must agree within each individual. Failed segmentation rows must remain in the metadata.

`paths.splits` must be a directory named **`split_csv`** containing frozen `train_split.csv`, `val_split.csv`, `test_split.csv`. These contain the original training labels, image paths, `barcode`, `filename`, `genus`, `species_label`, `life_stage`. Keep all photographs of a worm in one split. Missing species labels must remain missing, not be reassigned from a genus label.

`paths.original_data` contains the original segmented archive and `01_Segmented/global_metadata.csv`, with paths consistent with those split files. The dataset preparation command combines that archive with `acquisition.new_captures` without altering either source.

## Baseline and visual training artifacts

`paths.trained_results` follows the existing training layout:

```
runs/<stage>/run_<...>/
  run_status.txt                    # Exactly 0 for successful historical fits
  <run-name>/
    config.json
    label_to_index_by_task.json
    run_summary.json
    test_predictions_best.csv
    best_model.pt
```

Baseline discovery requires 3 architectures × 30 seeds, contiguous fixed class maps, consistent taxonomy/preprocessing, original RGB inputs, no biological holdout and no hierarchy loss. Visual rescoring uses configs, class maps, completion markers and hard prediction CSVs; it does not load checkpoints. Baseline saved summaries supply **original validation loss** for architecture/seed selection. Do not select using external-camera test scores.

Supply completed baseline/visual training outputs as external inputs, or create a separate new grid with the [training commands](training.md). The generated analysis config uses the same file layout. For CSV-only analysis of an existing run, use a preserved inference plan plus all sidecar JSON/CSV files it references. Neither `make analysis` nor `make all` discovers/checks neural checkpoints; they read the frozen plan, saved predictions and features. Fresh `make plan` and `make inference features` do need weights and image files. For relocated CSV-only archives, validation summaries are resolved from `trained_results/runs/baseline` when a frozen absolute checkpoint-parent path is unavailable. Keep the original run-directory names. Legacy plans may list additional inputs, but these commands read only RGB/native files and never analyse the excluded external-transfer conditions.

## Inference

`paths.predictions` contains a frozen `plan.json` and:

```
predictions/<model>/seed_<N>/<original|gphoto2|webcam>/rgb/
  predictions.csv
  complete.json
```

Prediction columns include task, worm/image IDs, true/predicted indices, `probabilities_json` and `logits_json`. Unknown labels are recorded as `-1` and are not silently turned into species truths. All camera domains share the trained output vocabulary.

## Features and calibration

`paths.features` contains `plan.json`, `individual_splits.csv`, `experiment_matrix.csv` and:

```
rgb/seed_<N>/features/<domain>/native/
  features.npz
  metadata.csv
  complete.json
```

NPZ arrays include `image_id`, `early`, `intermediate`, `final`, `probabilities_age`, `probabilities_genus`, `probabilities_species`, and each task's `<task>_weight` and `<task>_bias`. CSV and NPZ image order must agree. Features are extracted from the saved classifier; heads must be identical across cameras. Receipts record file SHA-256 values and checkpoint/runtime provenance.

Calibration pairs averages of **all usable photographs within a worm and camera**. It does not pair capture indices or assume identical poses. Calibration uses all views from calibration worms; prediction uses all usable views from different test worms. The five-fold split and nested calibration subsets are frozen before correction fitting.

## Matched-removal controls and CSV-only return

`paths.controls/training` contains the frozen `plan.json`, `conditions.csv`, `full_train.csv`, per-condition splits and per-job config JSONs. Preserve these JSON bytes when moving results: receipts compare configuration hashes. Put returned output directories under `paths.received_controls/<condition>/seed_<N>`.

Each returned fit needs its `audit_complete.json` plus one run directory containing `test_predictions_best.csv`, `label_to_index_by_task.json` and `run_summary.json`. Include referenced logs if the completion receipt names them. Checkpoints are unnecessary for compiling saved decisions.

The compiler can resolve frozen remote config paths to the local `controls/training/configs/<condition>/seed_<N>.json`; it never rewrites the frozen JSON. The supplied local test partition must have the original SHA-256. After compilation, `control_summary` points to `received_csv_summary/latest.json`, which references the timestamped, validated CSV snapshot.

## Failures and not-estimable cases

Incomplete fits, duplicate condition/seed results, different class maps, checksum failures and overlapping calibration/test worms cause explicit errors. A fixed-rank mapping with too few calibration worms is recorded as not estimable. In historical biological exclusions, a species removed from the output vocabulary is not scored as a zero-performing available output.

A manual mask review is separate from a measured segmentation-overlap test. This package can report recorded failure coverage and create quantitative pipeline abstentions; it cannot reconstruct missing human ground-truth masks or acquisition settings.
