# Configuration files

Configuration is plain YAML with recursive, mapping-only inheritance through
`extends`. Lists are replaced, not concatenated. Dotted `key=value` overrides
remain supported and the resolved value is always an ordinary Python dictionary.

## Inheritance and precedence

```text
configs/defaults/base.yaml
            |
            v
       config.yaml                 safe one-run quick start
            |
            +-- configs/experiments/standard.yaml
            +-- configs/experiments/hierarchy.yaml
            +-- configs/experiments/dual_cue.yaml
            |        +-- ghpc_dual_cue.yaml
            +-- configs/experiments/colour_ablation.yaml
            |        +-- ghpc_colour_ablation.yaml
            +-- configs/experiments/patch_shuffle_matrix.yaml
            +-- configs/experiments/dinov3_rgb_stress.yaml
            +-- dev/genome_cues_matched.yaml
            |        +-- genome_cues_hierarchy.yaml
            |        +-- genome_rgb_stress.yaml
            +-- configs/experiments/persistent_hierarchy.yaml
                         |
                         +-- persistent_hierarchy_wandb.yaml

configs/clusters/local.yaml        independent machine axis
configs/clusters/genome.yaml
            |
            +-- genome_persistent.yaml
configs/clusters/ghpc.yaml
```

For local training, the child YAML overrides its parent. For a SLURM plan, the
general precedence is:

```text
explicit CLI --override
    > explicitly imported --legacy-env compatibility values
    > cluster profile
    > experiment child
    > config.yaml
    > configs/defaults/base.yaml
```

Ambient historical launcher variables are ignored unless `--legacy-env` is
requested. Cluster files contain machine resources and paths; they must not
change scientific run-spec hashes.

## Canonical scientific layers

New child configs should use four independent layers:

- `preprocessing` for deterministic resize and normalization on every split;
- `augmentation` for train-only random flips and rotation;
- `sweep.parameters` × `sweep.conditions` for external training expansion;
- `evaluation.test_conditions` and `evaluation.condition_matrix` for
  checkpoint evaluation that never creates another fit.

Each `sweep.conditions` item is a complete object with `name`, `feature`,
`transform`, optional `strength`, and transform-specific `parameters`. Numeric
condition families can use `name_template`, `parameter`, and an inclusive
`range`. The planner resolves one object into one run specification and then
disables expansion before invoking the trainer.

Historical condition sections are accepted as migration input, but they are
normalized into these layers and are not uploaded as duplicate W&B config
columns. See [the full configuration reference](../config.md) for syntax and
examples.

## Which file to use

| Goal | Experiment configuration | Typical count |
| --- | --- | ---: |
| One local standard run | `config.yaml` | 1 |
| Two-model standard sweep | `configs/experiments/standard.yaml` | 2 |
| Standard sweep with hierarchy consistency | `configs/experiments/hierarchy.yaml` | 2 |
| Full matched cue and RGB-stress study | `configs/experiments/dual_cue.yaml` | 224 |
| GHPC dual-cue historical W&B settings | `configs/experiments/ghpc_dual_cue.yaml` | 224 |
| Four-model 2×2/4×4 patch matrix | `configs/experiments/patch_shuffle_matrix.yaml` | 12 trainings, 36 evaluation cells |
| DINOv3 ViT/ConvNeXt hierarchy and RGB stress | `configs/experiments/dinov3_rgb_stress.yaml` | 12 trainings, 288 evaluation cells |
| Colour retention 100% through 0% | `configs/experiments/colour_ablation.yaml` | 202 |
| GHPC colour W&B settings | `configs/experiments/ghpc_colour_ablation.yaml` | 202 |
| Genome matched cues with optimizer/loss choices | `dev/genome_cues_matched.yaml` | 7,296 |
| Same matched study with hierarchy weights | `dev/genome_cues_hierarchy.yaml` | 21,888 |
| Original-RGB training plus transformed evaluation | `dev/genome_rgb_stress.yaml` | 64 trainings, 1,536 evaluation cells |
| Genome persistent-cache hierarchy sweep | `configs/experiments/persistent_hierarchy.yaml` | 2 |
| Same persistent sweep with W&B | `configs/experiments/persistent_hierarchy_wandb.yaml` | 2 |
| Complete Genome paper-ablation pipeline | `dev/genome_ablation_pipeline.yaml` | 190 trainings + final report |

Use `local.yaml` for rendering and CPU-only planning, `genome.yaml` for the
Genome dual-cue workflow, `genome_persistent.yaml` for Genome persistent-cache
sweeps, and `ghpc.yaml` for GHPC node-local scratch. GHPC requires an explicit
GPU-node list before rendering or submission.

## One-command paper ablation

The paper pipeline is split into three readable experiment files:

- `dev/genome_ablation_baseline.yaml`: original images, five backbones, and
  three complete genus/species/age loss-weight recipes across seeds 2024,
  2025, and 2026 (45 fits and seed-level confidence intervals);
- `dev/genome_visual_ablation.yaml`: 0% colour, 2x2/4x4/8x8/16x16 patch
  shuffling, Gaussian blur at 10-100%, and resolution loss at 10-100% for all
  five backbones (125 fits). The original baseline supplies severity 0 in the
  final graphs;
- `dev/genome_data_holdouts.yaml`: four plainly worded juvenile,
  species, and genus holdout questions for all five backbones (20 fits).

Every file keeps hierarchy loss disabled. W&B uses the single
`worm-species-paper` project, one group per stage, compact scalar logging, no
model uploads, and no confusion-matrix or large-table logging.

Render the complete dependency chain without submitting:

```bash
make ablation-pipeline
```

Submit baseline, visual ablations, holdouts, and the final graph job:

```bash
make ablation-pipeline PIPELINE_MODE=submit
```

All run outputs, SLURM logs, generated plans, CSV tables, summary JSON, and PNG
figures are stored below `paper_result/`. Genome limits every array to eight
active tasks. A node-level file lock makes one task transfer the ready image
cache into `/tmp`; other tasks scheduled on the same node reuse it.

The report chooses the best baseline configuration by mean validation score
across its three seeds. Single-seed visual and cohort ablations are then shown
against that reference. Visual evaluation compares matched-condition
performance with original-image performance; it does not run a full
condition-by-condition Cartesian test matrix.

The final report also writes a `manuscript_artifacts` readiness checklist into
`paper_result/summary/paper_results_manifest.json`. It covers the dataset
composition, model/training configuration, experimental-ablation and holdout
tables; the workflow, representative-image and transformation-example panels;
the baseline confidence-interval plot; the visual-ablation overview; the
matched-versus-original comparison; and the structured-holdout figure.

After training has finished, tables and figures can be regenerated any number
of times without submitting jobs or loading checkpoints:

```bash
python scripts/build_paper_results.py
```

Edit `dev/paper_report_style.yaml` to change the plot palette, baseline
reference colour, workflow colours, heatmap colormap, font size, or DPI, then
run the same Python command again. Existing files in `paper_result/tables`,
`paper_result/figures`, and `paper_result/summary` are replaced from the
completed run records.

## Preferred commands

Validate and inspect the safe root configuration without touching data:

```bash
PYTHONPATH=src python -m worm_species.config.validate \
  --config config.yaml --workflow training

PYTHONPATH=src python -m worm_species.config.inspect \
  --config config.yaml --workflow training --format yaml
```

Inspect an experiment and cluster plan:

```bash
make validate \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/local.yaml

make inspect \
  CONFIG=configs/experiments/patch_shuffle_matrix.yaml \
  CLUSTER=configs/clusters/local.yaml
```

Render without submission, run one local process, or explicitly submit:

```bash
make dry-run \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/genome.yaml \
  ARTIFACTS_DIR=slurm/generated/dual-cue-check

make train TRAIN_CONFIG=config.yaml

make submit \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/genome.yaml
```

`make dry-run` never calls `sbatch`. Direct equivalents are available through
`PYTHONPATH=src python -m worm_species.slurm launch --dry-run ...` and
`--submit`. Scheduler submission is never implied by rendering.

Use `--check-paths` with the config validator only on a machine where the data
and predefined split paths should exist. Dry-run and cluster planning leave this
off so a login node does not need the full dataset tree.

See [the full configuration reference](../config.md) for every
important switch, condition semantics, resource field, and worked example.
