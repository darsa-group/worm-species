# Worm Species paper-ablation pipeline

## Combine the 2025 and 2026 images and segment both cameras

```bash
make dataset-setup
make dataset-plan
make dataset-gpu-check
make dataset DEVICE=cuda:0
```

`dataset-setup` creates `.venv-segmentation` using the existing wormspecies
environment's PyTorch. It installs the segmentation package; NVIDIA drivers
must already work on the host. Fix the driver before the GPU commands, or use
`make dataset DEVICE=cpu`. Override `PYTHON=/path/to/python` if needed.

The inputs default to `/mnt/extssd/Earthworms/petridish-worm-images` and
`/mnt/extssd/images`. The output is a **new, separate folder** at
`/mnt/extssd/Earthworms/publication_dataset`:

```text
publication_dataset/
├── README.md
├── metadata/
│   ├── images.csv
│   ├── segmentation_status.csv
│   ├── summary.json
│   ├── sources/
│   └── original_paper_splits/
├── 2025/original_camera/
│   ├── 00_RawData/metadata.csv
│   ├── 01_Segmented/metadata.csv
│   └── masks/
└── 2026/
    ├── gphoto2/
    │   ├── 00_RawData/metadata.csv
    │   ├── 01_Segmented/metadata.csv
    │   ├── masks/
    │   └── calibration/metadata.csv
    └── webcam/
        ├── 00_RawData/metadata.csv
        ├── 01_Segmented/metadata.csv
        ├── masks/
        └── calibration/metadata.csv
```

Image files sit under `specimen barcode/capture session/filename` within each
folder (`legacy` is the session for old images). Every raw and segmented folder
has its **own CSV**: `image_path` is relative to that folder, and `rel_path_*`
columns are relative to the dataset root. Location codes come from `captures.csv`
and remain text, including leading zeros. Both cameras share the same specimen
identity. Source metadata and original paper splits are preserved.

The script copies images without changing the source folders, reuses existing
historical segmentations, and runs the two-stage rotated-square method from
`segment/inference_coco.py` on both new camera datasets and missing historical
segmentations. It saves the segmented RGB, a mask in original coordinates, and
a mask aligned with the segmented RGB. Historical masks retain their original
coordinates and are not presented as masks aligned with segmented RGB.
Calibration photos are retained separately and excluded from segmentation.
Only successful segmentations enter each `01_Segmented/metadata.csv`; failures
remain visible in the global status CSV. CSVs refresh every 25 attempts and at
exit; completed per-image records allow interruption and resume. A rerun retries
failed images. Exit code 2 means some images still have failed segmentation.

Useful commands:

```bash
make dataset-prepare                  # Copy and organize only; no model required
make dataset-segment DEVICE=cuda:0    # Resume segmentation without copying again
make dataset-segment DEVICE=cpu SEGMENT_LIMIT=10  # Small check of pending images
make dataset DATASET_ROOT=/path/to/new/folder DEVICE=cuda:0
```

The full dataset needs approximately 60 GB for copied inputs and historical
outputs, plus space for the new segmentations. The script reserves new gphoto2
worms as external test and webcam worms for later evaluation in metadata; these
commands perform dataset preparation only. They do not train, evaluate classifiers,
delete experiments, or modify the saved 30-seed results.

### Explore the dataset while segmentation runs

Open [`notebooks/publication_dataset_overview.ipynb`](notebooks/publication_dataset_overview.ipynb)
and select **Run All**. Its settings cell defaults to the combined dataset above;
change `DATASET_ROOT` for a different location. Use the wormspecies Python kernel
(pandas, numpy, matplotlib, Pillow and IPython are required).

The notebook reads a single metadata snapshot and covers image versus individual
counts, recorded taxa and life stages, locations, camera overlap, segmentation
progress and failures, capture dates, specimen weights, original split checks,
sampled image properties, mask overlays, and paired camera examples. It preserves
leading zeros in location codes, excludes calibration photos from worm counts,
and distinguishes pending images from failed segmentations. Every plotting cell
is editable. Image checks use small samples so they can run during segmentation.
It never writes into the dataset or changes the running job.

Rerun all cells to refresh progress; the metadata is updated every 25 segmentation
attempts. `make dataset-notebook` regenerates the notebook from its builder and
replaces notebook edits and saved outputs, so use that command only when you want
to reset it to the generated version.

## Test the new dataset with all 30 saved baseline seeds on Genome

This evaluates the saved **ConvNeXt Base, ViT-B/16 and ResNet-50** checkpoints:
30 seeds per model (`40, 140, ..., 2940`), separately on **gphoto2 and webcam**.
There are 90 GPU array tasks, each evaluating both cameras, for 180 evaluations.
Each task requests one GPU; at most 12 tasks run concurrently. This performs
inference only and writes into a separate `publication_external_2026` directory.

First finish segmentation and transfer the combined dataset to Genome, preserving
its directory structure. The required inputs are `metadata/images.csv` and all
segmented images it references. The default Genome dataset location is
`/faststorage/project/worm-species/publication_dataset`; override it below if needed.
The original `publication_30seed_result/runs/baseline` tree must contain all 90
completed runs, their `best_model.pt`, configs, class maps, and original-test
prediction CSVs. Symlinked checkpoints must resolve on Genome.

Run these commands **in the Genome terminal**, from the updated source checkout:

```bash
conda activate wormspecies
make publication-test-plan \
  EXTERNAL_DATASET=/faststorage/project/worm-species/publication_dataset \
  PUBLICATION_RESULT=/faststorage/project/worm-species/source/publication_30seed_result
make publication-test-submit \
  EXTERNAL_DATASET=/faststorage/project/worm-species/publication_dataset \
  PUBLICATION_RESULT=/faststorage/project/worm-species/source/publication_30seed_result
```

Adjust the two input paths to their actual locations on Genome. Planning validates
the inputs, freezes the cohort and renders jobs without submitting. Submission
queues the GPU array and a CPU report dependent on successful array completion.
Jobs use the Python executable from the activated environment; set
`EXTERNAL_PYTHON=/path/to/python` to choose it explicitly. The script preserves
checkpoint preprocessing, task vocabularies and taxonomy uncertainty rules.
It does not apply the training rare-class filter to new images.

Pending segmentation blocks planning. Failed segmentations are listed explicitly
in `cohorts/excluded.csv`; successful images from each camera form its cohort.
Unknown or uncertain labels remain in predictions, with an exclusion reason for
each unscored task. They can still be scored on other known tasks. Specimens
overlapping the original paper splits block planning.

Resource overrides include `EXTERNAL_CPUS=8`, `EXTERNAL_MEMORY=32G`,
`EXTERNAL_TIME=02:00:00`, `EXTERNAL_BATCH_SIZE=64`, `EXTERNAL_WORKERS=6`, and
`EXTERNAL_MAX_ACTIVE=12`. Account and GPU partition are set with
`EXTERNAL_ACCOUNT` and `EXTERNAL_PARTITION`. The CPU report has independent
`EXTERNAL_REPORT_CPUS`, `EXTERNAL_REPORT_MEMORY`, `EXTERNAL_REPORT_TIME` and
optional `EXTERNAL_REPORT_PARTITION` settings.

```bash
make publication-test-status       # File completion, not live scheduler status
make publication-test-report       # Regenerate report after all evaluations finish
```

To resume after jobs stop, rerun the same `publication-test-submit` command.
It refuses to resubmit while recorded jobs remain active, submits only checkpoint
indices with incomplete outputs, and skips completed cameras before loading a
model or building its node cache. If all evaluations completed but reporting failed,
submission queues only the CPU report. Checksummed per-camera completion records
detect missing or changed outputs. A changed input snapshot requires a new
`EXTERNAL_RESULT`; keep code and inputs fixed while jobs run.

Each camera/model/seed directory contains predictions, metrics and confusion
matrices for age, genus and species. The `summary/` directory contains per-seed
metrics, means and 95% intervals across seeds, paired differences from each
checkpoint's original test results, and confusion-matrix CSV/PNG/SVG/PDF files.
Balanced accuracy and the checkpoint's `1/K` uniform-chance reference are included.
Metrics are image-level; intervals describe variation across training seeds,
not specimen sampling uncertainty. Camera cohorts include all their successful
images and can differ in composition. The report never counts repeated seed
evaluations as additional specimens.

## Existing paper experiment commands

This branch contains only the code, configuration, documentation, and focused
tests reachable from the Genome paper pipeline:

1. build one deterministic segmented-image base cache;
2. precompute deterministic Gaussian blur, patch shuffle, resolution-loss, and
   pairwise compound variants on the persistent shared filesystem;
3. train 90 original-image baselines;
4. train 660 matched standalone visual-ablation models;
5. train 600 Gaussian pairwise-interaction models;
6. train 120 biological holdout models;
7. collect completed results and rebuild the paper tables and figures.

The pipeline contains 1,470 model fits. Every phase uses seeds 40, 41, and 42,
and every fit is repeated with hierarchy loss disabled (`h=0`) and enabled at
weight `h=0.2`. Baselines use five backbones and three complete task-loss
recipes; downstream phases use the fixed genus-1/species-0.5/age-2 recipe.
Baseline controls are matched by backbone, seed, task-loss recipe, and
hierarchy-loss weight. The report retains the original `h=0` figures and emits
an `_hloss_comparison` counterpart for each scientific performance figure.

## Run it

Render and inspect the complete dependency chain without contacting SLURM:

```bash
make ablation-pipeline
```

Submit only after inspecting the generated artifacts:

```bash
make ablation-pipeline PIPELINE_MODE=submit
```

The entrypoint is
[`dev/genome_ablation_pipeline.yaml`](dev/genome_ablation_pipeline.yaml).
It selects the Genome cluster profile, all four experiment stages, both cache
jobs, and the final report job. `afterok` dependencies prevent downstream
stages from running after a failed prerequisite.

## Shared cache design

The persistent base cache stores resized, segmented RGB inputs with foreground
cropping disabled. Its identity includes the resolved preprocessing settings,
source metadata, image source stamp, image size, and crop settings, so a cache
created under an older crop policy cannot be silently reused.

The condition cache stores exact float32 tensors for deterministic expensive
conditions only:

- Gaussian blur;
- seeded patch shuffle;
- resolution loss;
- composed Gaussian-blur × colour or patch conditions.

Random train augmentation remains live. Each condition has a versioned,
content-addressed directory, manifest, ready marker, file lock, and atomic
publication. Multiple SLURM jobs and nodes can safely read the same persistent
cache. A training task copies only its required condition directory to
node-local scratch and validates the shared ready marker before use.
Original-image baseline tasks keep the persistent cache as their source and
stage only the test-split tensors needed by each post-training condition. Those
subsets are locked and shared per node, so concurrent baseline tasks reuse them
without copying every train and validation tensor or recomputing the transform.

Saturation remains on-the-fly because it is inexpensive.

## Gaussian and interaction schedules

Standalone Gaussian severity uses percentages
`2, 5, 10, 25, 40, 50, 60, 75, 90, 100` with `max_sigma=64`. The four
interaction levels are 25, 50, 75, and 100 percent (sigma 16, 32, 48, and 64).
Each is crossed separately with colour removal and four patch grids, creating
20 interpretable pairwise conditions. Resolution is deliberately excluded from
this interaction matrix and evaluated in its standalone three-way control.

## Resolution-loss schedule

The configured 30-seed publication loss percentages are:

| Lost linear resolution | Retained linear dimension | 224 px intermediate |
|---:|---:|---:|
| 0% | 100% | 224 × 224 |
| 25% | 75% | 168 × 168 |
| 50% | 50% | 112 × 112 |
| 75% | 25% | 56 × 56 |
| 87.5% | 12.5% | 28 × 28 |
| 90% | 10% | 22 × 22 |
| 93.75% | 6.25% | 14 × 14 |
| 95% | 5% | 11 × 11 |
| 97% | 3% | 7 × 7 |
| 98% | 2% | 4 × 4 |
| 99% | 1% | 2 × 2 |
| 100% | 0% | 1 × 1 |

The existing `ResolutionLoss` implementation is unchanged: it uses bilinear
resizing with anti-aliasing and `max(1, ...)`. The 100% setting is an extreme
spatial-information control that retains mean colour but no spatial structure.

## Paper outputs

Regenerate figures, tables, and readiness manifests from completed runs:

```bash
make paper-report
```

Styling is editable in
[`dev/paper_report_style.yaml`](dev/paper_report_style.yaml). The report also
writes `resolution_loss_schedule.csv`, and resolution plots label both retained
linear dimension and the corresponding 224-pixel intermediate size.

Every metric graph is aggregated across seeds with 95% t-confidence intervals
and a class-count-derived chance reference. PNG, PDF, and SVG versions are
written. Exact plotted rows, seed summaries, style settings, representative
source images, transformed level images, hashes, and manifests are saved below
`paper_result/figure_sources/`. The reproducible notebook is
[`notebooks/worm_species_figures_tables_confusion_matrices.ipynb`](notebooks/worm_species_figures_tables_confusion_matrices.ipynb).

For the editable main and supplementary model/ablation figures, use
[`notebooks/holdouts_and_visual_combinations.ipynb`](notebooks/holdouts_and_visual_combinations.ipynb)
or run:

```bash
make holdout-visual-report
```

The notebook is standalone: its result readers, metric calculations, exact
image transformations, confidence intervals, and plotting functions are
embedded in the notebook. It imports no local project module or reporting
script. Set the paths in its editable settings cell, run the preparation cell,
then run any figure cell independently. If the reporting implementation
changes, regenerate the embedded copy with:

```bash
/home/devd/miniconda3/envs/wormspecies/bin/python \
  scripts/build_standalone_holdout_visual_notebook.py
```

The output contains: baseline task scores and seed-mean confusion matrices with
cellwise confidence intervals; a dedicated Adult/Juvenile diagnostic; linear
and log2 pixel-resolution visual-ablation variants; a paired-seed mixed-cue
comparison; configurable single-species recall and precision/recall/F1 views;
four cross-cohort biological-transfer questions; and complete all-species
supplementary figures. Each notebook section contains an editable plotting
cell immediately above its graph. The standardized data-ablation figures use
the normal model's sample seed SD as one shared unit. Chance is plotted at zero,
the ablated point is `d_retained`, the normal point is `d_total`, and the gap is
`d_ablation`, with `d_total = d_ablation + d_retained`. Figures 5–6 provide the
matching raw-margin plots: `M_total = normal - chance`,
`M_lost = normal - ablated`, and `M_retained = ablated - chance`, with
`M_total = M_lost + M_retained`. Chance is derived as `1/K` from each task's
saved class map under uniform random prediction, not hard-coded. Visual
macro-F1 panels instead use the expected macro-F1 under uniform prediction on
the fixed test-label distribution. The data-ablation figures
use pointwise 95% paired-seed percentile-bootstrap intervals (10,000
resamples): the baseline and ablated runs are resampled together by training
seed. The visible whiskers cover the retained and total positions, while the
lost-gap interval is preserved in the figure-source CSV. These confidence
intervals come from variation across seeds; they are separate from and do not
use the class-count `1/K` chance reference. They are polished plots, not metric
tables; the underlying seed recalls and exact effects are saved as figure-source
CSVs for reproducibility. The confirmed publication design uses
30 seeds spaced from 40 through 2940. All metric figures are test-only and use
only hierarchy loss `h=0` and loss weights
`genus=1.0/species=0.5/age=2.0`. Figure 1 adds mean row-normalized
ConvNeXt-Base genus/species/age confusion matrices. Representative panels show
the same ten transformations for five reproducibly sampled test worms.
PNG/PDF/SVG outputs and exact
plotted CSV inputs are written under
`paper_result/notebook_holdout_visual_figures/`. Override the defaults with
`HOLDOUT_VISUAL_MODEL=...` or `SPECIES_ABLATION=...` on the `make` command.

## Confirmed 30-seed publication pipeline

[`dev/genome_publication_30seed_pipeline.yaml`](dev/genome_publication_30seed_pipeline.yaml)
is the single orchestration entry point for the confirmed design. It plans
1,890 fits: 90 three-model baselines, 840 ConvNeXt-Base visual ablations,
600 visual interactions, 30 full-data controls, and 330 taxon-stage holdouts.
The five new resolution levels account for 150 of the visual fits. Every run uses
validation total weighted loss for early stopping and best-checkpoint
selection, retains only `best_model.pt`, and reports test results only.

```bash
# Safe default: render the full dependency chain without submitting.
make publication-pipeline

# Read filesystem completion state.
make publication-status

# Explicit initial submission.
make publication-pipeline PUBLICATION_PIPELINE_MODE=submit

# Explicit recovery submission; completed run IDs skip before cache staging.
make publication-resume

# Dry-run only the five new pixel-resolution levels (150 fits).
make publication-resolution-gapfill

# Explicitly submit only those 150 fits.
make publication-resolution-gapfill-submit

# Recover target precision/recall/F1 from retained test predictions; no training.
make publication-data-metrics

# Build the main/supplementary figures and auditable publication bundle.
make publication-report
```

The bundle under `publication_30seed_result/publication_bundle/` contains
PNG/PDF/SVG figures, figure-source CSVs, exact test predictions, and checksum
inventories for best checkpoints, resolved configs, split files, label maps,
metrics, and training histories. The implementation checklist is
[`PUBLICATION_PIPELINE_TASKS.md`](PUBLICATION_PIPELINE_TASKS.md).

Holdout runs report both the cohort removed from train/validation and the
independent matching test cohort. Corresponding baseline checkpoints are
evaluated on the exact same cohorts. Resolution plots likewise compare matched
resolution training/testing, resolution-trained models on original images, and
original-trained baselines on the same transformed test images.

## Exhaustive Adult and Juvenile taxon-stage ablations

[`dev/genome_adult_taxon_ablation_pipeline.yaml`](dev/genome_adult_taxon_ablation_pipeline.yaml)
tests every observed `genus × species × developmental stage` combination for
Adults and Juveniles. It contains 30 full-data controls and 330 combination
holdouts: five backbones, seeds 40/41/42, eleven combinations (eight Adult and
three Juvenile), and hierarchy loss `h=0` and `h=0.2`.

```bash
# Render only; does not contact SLURM.
make adult-taxon-ablation-pipeline

# Explicit submission after inspecting the render.
make adult-taxon-ablation-pipeline ADULT_TAXON_PIPELINE_MODE=submit

# Rebuild the report from completed runs.
make adult-taxon-report
```

The report evaluates each removed Adult or Juvenile development cohort and the
matching fixed test cohort. It exports raw target recall and paired
withheld-minus-full-data effects, both as retained `h=0` figures and
hierarchy-loss comparisons. Every
figure is saved as PNG, PDF, and SVG with seed observations, seed-level 95%
confidence intervals, chance or zero references, and its exact source CSVs.
Enhanced runs also retain target precision, specificity, F1, aggregate
macro/micro/weighted metrics, confusion counts, per-class probabilities,
AUROC, average precision, Brier score, and a ten-bin calibration error where
the target and non-target classes are both defined. Historical hard predictions
can be upgraded without training via `make publication-data-metrics`;
probability metrics require checkpoint re-evaluation if probabilities were not
originally saved.

## Verification

```bash
make test
```

This runs the retained configuration, cache, transform, model, evaluation,
logging, loss, and end-to-end dry-run/report tests. A dry-run proves planning
and rendering, not real training or live cluster execution.
