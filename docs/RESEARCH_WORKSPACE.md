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

## Local individual-level and paired-camera domain analysis

Edit [`dev/publication_domain_analysis.yaml`](dev/publication_domain_analysis.yaml).
The local defaults use `/mnt/extssd/Earthworms/publication_dataset`, saved models
under `/mnt/extssd/Earthworms/worm-species/publication_30seed_result`, CUDA device
0, batch size 32, two data workers and six CPU threads. One checkpoint is loaded
at a time. `output` defaults to this checkout's `publication_domain_analysis_2026`.
Paths are absolute or relative to the YAML file's directory.

```bash
make domain-analysis-plan
make domain-analysis
make domain-analysis-status
```

Run these from this updated checkout. These commands use the local wormspecies
Python environment; override `DOMAIN_PYTHON=/path/to/python` if necessary.
Use `device: cpu` in the YAML or `DOMAIN_ARGS='--device cpu'` for CPU execution.
Reduce `batch_size` if GPU memory is tight. Device, batch size, thread count and
worker count can change on resume; scientific settings require a new output root.
No Slurm or training jobs are used. The runner builds a locked, resumable cache
of lossless resized RGB images under the output folder, shared by all checkpoints.
Greyscale is applied after this common resize, before channel normalization.

This implements the analysis hierarchy using the **existing 90 checkpoints**:

| Analysis | Outputs |
| --- | --- |
| Image versus individual | Mean probabilities, mean logits, majority vote, median probabilities; macro-F1 for age, genus and species |
| Colour ablation | RGB and three-channel greyscale on identical original-test, gphoto2 and webcam images |
| Factorial | RGB/greyscale × image/individual mean, with all 30 seeds per model |
| Paired cameras | Separate full-cohort and shared-individual scores, camera gaps and greyscale gap changes |
| Representation | Early/intermediate/final features and each task's probabilities for one best original-validation seed |
| Paired geometry | Cosine and normalized Euclidean distances; within-individual and different-individual controls; connected PCA plots |

The architecture for representation analysis is explicitly configured (default
ConvNeXt Base). Its seed is selected solely using `best_val_score` from original
validation, with matching selection metric/direction and a deterministic seed
tie break. The ranking and selected seed are saved before inference. That seed
runs first; once it finishes, use `make domain-analysis-representations` to
generate detailed figures while the remaining checkpoints run.

The complete run has **540 inference units**: 90 checkpoints × three domains ×
RGB/greyscale. All views of an individual are aggregated **within camera only**.
No camera pairs enter model selection. Probabilities and logits are saved for
every image, including unscored uncertain or out-of-vocabulary labels. Metadata
conflicts within an individual fail before inference. Original scoring preserves
the exact scorable image population of the saved original-test predictions.

The original reference experiments remain in place. The new report compares RGB
re-inference with saved original predictions. If the completed external gphoto2
and webcam reference results are available locally, set `external_reference` to
their folder to validate and compare them as well; `null` records no external
reference comparison. Existing ablations are not retrained or overwritten.

```bash
# Optional bounded run: one checkpoint index from plan.json.
make domain-analysis DOMAIN_ARGS='--index 0'
# After all checkpoints finish, rebuild tables/plots without inference.
make domain-analysis-report
```

Rerun `make domain-analysis` after interruption: completed checkpoint/domain/input
units skip before loading model weights. Partial units are recomputed. Results
and code identities are frozen; changing analysis code or input metadata requires
a new output directory. `plan` writes metadata and validates inputs but performs
no model inference. A full report requires every planned checkpoint.

Results include:

- `summary/per_seed_metrics.csv`, `seed_summary.csv` and per-individual predictions.
- `summary/rgb_greyscale_by_observation_unit.csv` and factorial figures.
- `summary/paired_camera_gaps.csv`, `camera_gap_summary.csv`, colour and aggregation contrasts.
- `summary/reference_comparison.csv`, explicitly exposing RGB re-inference differences.
- `representations/per_individual_distances.csv`, distance intervals and colour effects.
- Paired PCA PNG/SVG/PDF figures coloured by recorded taxon or genus, with camera
  marker shapes and lines connecting the same worm. PCA coordinates are also saved.

Performance intervals describe variation across training seeds. Representation
intervals resample paired individuals, and distances are computed in normalized
feature space before PCA. Same-taxon/stage different-individual controls help
separate camera differences from taxonomic differences. The report documents
metric definitions, missing labels and ties. Smaller camera gaps in greyscale
must be considered alongside absolute performance: equal degradation can also
shrink a gap. PCA is descriptive and cannot establish the cause of camera shifts.

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
## Local RGB/greyscale camera-transfer study

Run the complete frozen-checkpoint study on the local GPU, including W&B tracking,
all correction experiments, tables, captioned PNG/SVG/PDF figures, and UMAPs:

```bash
make domain-transfer
```

This uses the existing `wormspecies` Python environment and
[`dev/publication_domain_transfer.yaml`](dev/publication_domain_transfer.yaml).
W&B defaults to online logging in project `worm-species-domain-transfer`; if this
machine is not already authenticated, run `wandb login` from the activated
`wormspecies` environment first. Set `wandb.entity` in the config to use a team.
No credentials belong in this repository. GPU inference and correction fitting
run locally; W&B receives progress, summary metrics, tables, and report figures.
Image files, model weights, and cached embeddings stay on the local disk.

The command validates and freezes a plan, processes both validation-selected
checkpoints first, runs all **30 matched RGB-trained and 30 greyscale-trained
ConvNeXt checkpoints**, then builds the complete report. This does not retrain
networks. Greyscale uses genuine greyscale-trained weights and greyscale inputs.
The earlier RGB-checkpoint/greyscale-input experiment remains a separate reference.
Both camera directions use the same split definitions for both model conditions.
Repeat the command after interruption: checksummed completed units are reused.

```bash
make domain-transfer-plan       # Read-only input validation; design table and diagram
make domain-transfer-status     # Local completion, without contacting W&B
make domain-transfer-report     # Regenerate all tables, plots and W&B report artifact
make domain-transfer-figures    # Selected-seed UMAPs once those checkpoints have finished
make test-domain-transfer
```

Outputs are in `publication_domain_transfer_2026/`:

| Output | Meaning |
|---|---|
| `experiment_overview.md` / `.csv` | Calibration data, evaluation data and scientific question for each experiment |
| `figures/experiment_overview.png` / `.svg` / `.pdf` | Captioned diagram explaining the full study |
| `report.html` | Local report linking figures, explanations and tables |
| `tables/main_summary.csv` | Before/after macro-F1, accuracy, balanced accuracy, paired distances and original-test reference |
| `tables/baseline_summary.csv` | Original, gphoto2 and webcam scores for image and all individual aggregation methods |
| `tables/original_saved_prediction_comparison.csv` | Saved original-test scores versus reconstructed frozen-head scores, making inference-precision differences visible |
| `tables/correction_per_seed.csv` | Per-seed, protocol and biological-group results |
| `tables/individual_transfer_*.csv` | Donor-specific results; repeated evaluation worms are not independent replicates |
| `tables/paired_representation_distances.csv` | Early, intermediate, final and probability-space camera/individual distances |
| `umap_coordinates/` | Reproducible projected point coordinates and projection settings |
| `captions.json`, `figures/*.caption.md` | Plain-language explanations of every figure and its terms |
| `training_recipe_differences.json` | Existing RGB/greyscale training differences that limit colour-only causal claims |

The methods are no correction, per-image foreground colour balancing (gray-world),
paired mean displacement, regularised CORAL covariance alignment, and paired ridge
regression. Learned corrections fit **only calibration worm means**, in a bounded
calibration-only feature subspace, and act on final classifier-input embeddings.
The existing classifier then scores each corrected image. CORAL and ridge need at
least three paired calibration worms; unsupported trials appear in each chunk's
`not_estimable.csv`, rather than masquerading as successful or zero-effect fits.
Gray-world is an input control; it does not learn a population transformation.

Experiments test held-out views of the same worm, other worms of the same or other
species, species-to-species transfer, leave-one-species-out correction, global
held-out correction, and calibration-size curves. No pose correspondence between
camera photographs is assumed. Whole worms remain disjoint between calibration
and evaluation, except the explicit same-worm experiment, which holds out images
independently within each camera. Species held out from correction may have been
seen in original classifier training. Unresolved taxa remain explicitly labelled.

Disjoint folds are pooled by confusion counts before calculating macro-F1.
Overlapping donor experiments retain donor-specific results and report means;
repeat fits are averaged within a seed before 95% intervals across training seeds.
Single-species macro-F1 is accompanied by recall/balanced accuracy and accuracy.
These intervals describe training-seed variation on a fixed cohort, not uncertainty
from sampling new earthworms. Every correction reports both classification and
representation outcomes: a smaller camera distance alone is not evidence that
classification improved.

UMAPs use the best **original-validation** seed separately for each training
condition, never a new-camera test ranking. Native original/DSLR/webcam worm means
define a descriptive projection; correction panels use its transform with common
axes before/after. Final features and all three task-probability spaces accompany
each generalisation protocol, with early/intermediate native diagnostics. Distances
and metrics are computed in the original feature spaces, never on UMAP coordinates.
Separate networks/layers have separate projections. Each detailed trial is chosen
by frozen order, not by its measured improvement.

Defaults are batch size 32, 2 loader workers and 6 CPU threads for the local 16 GB
GPU. Resource settings and W&B mode can change on resume; scientific settings and
source hashes are frozen. To change methods/splits after planning, choose a new
output directory. Examples:

```bash
make domain-transfer TRANSFER_WANDB_MODE=offline    # Save W&B events locally for later sync
make domain-transfer TRANSFER_WANDB_MODE=disabled   # No W&B logging
make domain-transfer TRANSFER_ARGS='--index 0'       # One checkpoint for a bounded run
make domain-transfer TRANSFER_ARGS='--device cpu'    # Functional but slow fallback
```

The all-checkpoint workflow is sequential and never submits Slurm/Genome jobs.

### MEE manuscript figures and notebook

With the publication source images mounted, run `make mee-review` to regenerate the main and supplementary PDFs, the figures/tables-only proof PDFs and the executed [figure notebook](notebooks/mee_manuscript_figures.ipynb). The notebook includes the complete plotting code immediately before each figure. The new view-count sensitivity uses 1–11 supplied photographs; when fewer distinct views exist, available photographs are cycled, with reuse counts recorded in the supporting CSVs. No model training is required for this rebuild.

### Separate deployment audit (v2)

The new common-species comparisons, specimen bootstraps, task-specific visual
results, segmentation/abstention analysis and fixed-rank calibration baselines
are saved separately in `publication_deployment_audit_v2/README.md`. Reference
experiments and the existing manuscript are preserved.

Activate `wormspecies`, then use `make publication-deployment-local` for local
evaluation/calibration only. Neural-network training runs on GenomeDK: use
`make publication-genome-launch` to upload and submit, then
`make publication-genome-download` to import and analyse completed outputs. Use
`make publication-deployment-status` to check verified completions and
`make publication-deployment-controls-stop` to stop the detached queue and worker.
The new checkpoints live in a separate directory on the external SSD; W&B logs
are offline by default. See the result folder for the figure PDF and executed
notebook. No new biological-coverage conclusion is reported before its matched
controls have genuinely completed.
