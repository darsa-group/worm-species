# Worm Species paper-ablation pipeline

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
