# Worm Species paper-ablation pipeline

This branch contains only the code, configuration, documentation, and focused
tests reachable from the Genome paper pipeline:

1. build one deterministic segmented-image base cache;
2. precompute deterministic Gaussian blur, patch shuffle, and resolution-loss
   variants on the persistent shared filesystem;
3. train 45 original-image baselines;
4. train 110 matched visual-ablation models;
5. train 20 biological holdout models;
6. collect completed results and rebuild the paper tables and figures.

The pipeline contains 175 model fits. Baselines use five backbones, three seeds,
and three complete loss-weight recipes. Visual and holdout experiments use one
seed and compare with the best baseline.

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
It selects the Genome cluster profile, all three experiment stages, both cache
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
- resolution loss.

Random train augmentation remains live. Each condition has a versioned,
content-addressed directory, manifest, ready marker, file lock, and atomic
publication. Multiple SLURM jobs and nodes can safely read the same persistent
cache. A training task copies only its required condition directory to
node-local scratch and validates the shared ready marker before use.

Saturation remains on-the-fly because it is inexpensive.

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

## Verification

```bash
make test
```

This runs the retained configuration, cache, transform, model, evaluation,
logging, loss, and end-to-end dry-run/report tests. A dry-run proves planning
and rendering, not real training or live cluster execution.
