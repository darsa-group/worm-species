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

Saturation remains on-the-fly because it is inexpensive.

## Gaussian and interaction schedules

Standalone Gaussian severity uses percentages
`2, 5, 10, 25, 40, 50, 60, 75, 90, 100` with `max_sigma=64`. The four
interaction levels are 25, 50, 75, and 100 percent (sigma 16, 32, 48, and 64).
Each is crossed separately with colour removal and four patch grids, creating
20 interpretable pairwise conditions. Resolution is deliberately excluded from
this interaction matrix and evaluated in its standalone three-way control.

## Resolution-loss schedule

The configured loss percentages are:

| Lost linear resolution | Retained linear dimension | 224 px intermediate |
|---:|---:|---:|
| 0% | 100% | 224 × 224 |
| 25% | 75% | 168 × 168 |
| 50% | 50% | 112 × 112 |
| 75% | 25% | 56 × 56 |
| 87.5% | 12.5% | 28 × 28 |
| 93.75% | 6.25% | 14 × 14 |
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

Holdout runs report both the cohort removed from train/validation and the
independent matching test cohort. Corresponding baseline checkpoints are
evaluated on the exact same cohorts. Resolution plots likewise compare matched
resolution training/testing, resolution-trained models on original images, and
original-trained baselines on the same transformed test images.

## Exhaustive Adult taxon data ablations

The standalone Adult taxon pipeline removes every fully labelled
`genus × species × Adult` combination observed in the predefined train,
validation, and test splits. It currently covers eight combinations. Each is
removed from train and validation while the predefined test split remains
unchanged. A matching full-data model evaluates the same development and test
cohorts, so the report can show both raw recall and the paired effect of
withholding the combination.

Render the 270-fit plan (30 full-data controls plus 240 holdouts):

```bash
make adult-taxon-ablation-pipeline
```

Submit explicitly after inspecting the generated plan:

```bash
make adult-taxon-ablation-pipeline ADULT_TAXON_PIPELINE_MODE=submit
```

Rebuild its report from completed runs:

```bash
make adult-taxon-report
```

The report writes separate `h=0` and `h=0` versus `h=0.2` figures, seed-level
points, 95% seed-level t-confidence intervals, chance or zero references, and
PNG/PDF/SVG exports. Plot rows and seed summaries are retained under
`adult_taxon_ablation_result/figure_sources/`. Sparse combinations can leave a
species absent from the training head; these unsupported values remain missing
and are not converted into zero recall.

## Verification

```bash
make test
```

This runs the retained configuration, cache, transform, model, evaluation,
logging, loss, and end-to-end dry-run/report tests. A dry-run proves planning
and rendering, not real training or live cluster execution.

## Task-specific multitask diagnostics

The repository also includes a backward-compatible diagnostic family for
negative transfer and species–developmental-stage confounding. It retains
`shared_heads` as the default, and adds single-task heads, split taxonomy/age
branches, task-specific pooling, joint species–stage sampling, PCGrad, age
supervised-contrastive learning, and a separate exploratory species adversary.

The configurations, architecture diagrams, training commands, output
contracts, and holdout interpretation guide are in
[`docs/GENERALISATION_EXPERIMENTS.md`](docs/GENERALISATION_EXPERIMENTS.md).
Generate completed-run tables, publication figures, and computed explanations
with:

```bash
make generalisation-report
```

Ordinary random splits primarily measure interpolation. The structured
holdouts are the relevant evidence for biological generalisation.

## Complete Genome performance comparison

The complete performance launcher includes matched shared-head, single-task
genus/species/age, split taxonomy-age, isolated mechanism, and full-model
configurations. Every configuration uses the three paper seeds, three
backbones, the original split, and four structured holdouts. Render and inspect
all 630 run specifications without submitting:

```bash
make performance-genome-all-dry-run
```

Submit only after the dry run and runtime-checkout validation succeed:

```bash
make performance-genome-all-submit
```

The list can be restricted without editing the Makefile, for example
`PERFORMANCE_ALL_CONFIGS="shared_heads single_task_age split_taxonomy_age"`.
Each completed run saves its best-checkpoint representation, labels, and a
manifest distinguishing projected age embeddings, age-branch features, and
backbone features from models without an age head. Generate the tables,
performance plots, and matched multi-architecture embedding figure with:

```bash
make performance-genome-report
```
