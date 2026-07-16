# Genome training configuration guide

This guide shows how to define a Genome experiment for ConvNeXt Base and ViT,
multiple learning rates, early stopping, task-loss weights, optional hierarchy
loss, full 0–100% colour retention, and the other supported image cues.

The ready-to-run root experiment is [`devconfig.yaml`](devconfig.yaml). It is
the validated full configuration described below. Use small child files for
alternative loss-weight, hierarchy, or early-stopping choices so those choices
remain explicit and do not accidentally create a much larger Cartesian sweep.

The examples use the canonical configuration model:

```text
sweep.parameters Cartesian product
    × complete sweep.conditions
    = generated run specifications
```

Every generated run specification becomes exactly one SLURM array task, one
canonical trainer invocation, and one model fit. The trainer does not expand a
second sweep internally.

## Recommended file layout

Keep scientific choices separate from the Genome machine profile:

```text
configs/experiments/genome_cues_full.yaml
configs/experiments/genome_cues_hierarchy.yaml
configs/experiments/genome_cues_equal_weights.yaml
configs/experiments/genome_cues_age_weighted.yaml
configs/experiments/genome_cues_pilot.yaml
configs/clusters/genome.yaml
```

The experiment files control models, learning rates, losses, cues, and
evaluation. `configs/clusters/genome.yaml` controls partitions, resources,
paths, scratch, concurrency, and the conda environment.

## Full matched-condition experiment

Save the following as `configs/experiments/genome_cues_full.yaml`:

```yaml
extends: ../../config.yaml

# These values apply to every generated run unless sweep.parameters overrides
# them. Training remains ordinary masked multitask training.
training:
  mode: multitask
  use_masked_labels: true
  epochs: 200
  batch_size: 256
  lr: 0.0005
  weight_decay: 0.0001
  use_amp: true
  class_weight: true
  num_workers: 6
  val_interval: 5

# Early stopping is independent of hierarchy loss and cue selection.
early_stopping:
  enabled: true
  monitor: macro_f1
  mode: max
  patience: 6
  min_delta: 0.0001

# This is the default task-weight recipe. Child files can replace it without
# repeating the model, LR, or condition definitions.
multi_task:
  loss_weights:
    genus: 1.0
    species: 0.5
    age: 2.0
  normalize_loss_by_active_tasks: true
  selection_metric: mean_macro_f1
  hierarchy_loss:
    enabled: false
    parent_task: genus
    child_task: species
    weight: 0.5

preprocessing:
  image_size: 224
  normalisation:
    enabled: true
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

# Augmentation is train-only. Experimental conditions below are applied
# consistently to train, validation, and test for matched-condition training.
augmentation:
  enabled: true
  horizontal_flip: {enabled: true, probability: 0.5}
  vertical_flip: {enabled: true, probability: 0.5}
  rotation: {enabled: true, degrees: 270}

model:
  pretrained: true
  freeze_backbone: false

wandb:
  enabled: true
  project: worm-species-genome-cues
  mode: online
  group: genome-cues-full
  job_type: train

sweep:
  enabled: true

  # This creates 2 models × 2 learning rates = 4 parameter combinations.
  parameters:
    model.name:
      - convnext_base
      - vit_b_16
    training.lr:
      - 0.0005
      - 0.0001

  # Complete objects prevent unrelated transform parameters from being mixed.
  # Saturation is endpoint-inclusive: 1.00, 0.99, ..., 0.01, 0.00.
  conditions:
    - {name: original, feature: baseline, transform: original, strength: 0.0, parameters: {}}

    - name_template: saturation_{percent:03d}pct
      feature: colour
      transform: saturation
      parameter: retention
      range: {start: 1.0, stop: 0.0, step: -0.01}

    - {name: grayscale, feature: colour, transform: grayscale, strength: 1.0, parameters: {}}

    - name: channel_shuffle_201
      feature: colour
      transform: channel_shuffle
      strength: 1.0
      parameters: {order: [2, 0, 1]}

    - name: bilateral_d5_c25_s25
      feature: texture
      transform: bilateral_filter
      strength: 25.0
      parameters: {diameter: 5, sigma_colour: 25.0, sigma_space: 25.0}
    - name: bilateral_d7_c50_s50
      feature: texture
      transform: bilateral_filter
      strength: 50.0
      parameters: {diameter: 7, sigma_colour: 50.0, sigma_space: 50.0}
    - name: bilateral_d9_c100_s100
      feature: texture
      transform: bilateral_filter
      strength: 100.0
      parameters: {diameter: 9, sigma_colour: 100.0, sigma_space: 100.0}

    - {name: gaussian_sigma_0.5, feature: texture, transform: gaussian_blur, strength: 0.5, parameters: {sigma: 0.5}}
    - {name: gaussian_sigma_1, feature: texture, transform: gaussian_blur, strength: 1.0, parameters: {sigma: 1.0}}
    - {name: gaussian_sigma_2, feature: texture, transform: gaussian_blur, strength: 2.0, parameters: {sigma: 2.0}}
    - {name: gaussian_sigma_4, feature: texture, transform: gaussian_blur, strength: 4.0, parameters: {sigma: 4.0}}

    - {name: patch_shuffle_grid_2, feature: shape, transform: patch_shuffle, strength: 2, parameters: {grid_size: 2, seed: 2026}}
    - {name: patch_shuffle_grid_4, feature: shape, transform: patch_shuffle, strength: 4, parameters: {grid_size: 4, seed: 2026}}
    - {name: patch_shuffle_grid_8, feature: shape, transform: patch_shuffle, strength: 8, parameters: {grid_size: 8, seed: 2026}}

# Leave post-training evaluation disabled for a pure matched-condition study.
# Enabling either block evaluates checkpoints; it never adds model fits.
evaluation:
  test_conditions:
    enabled: false
    evaluate_original_training: false
    conditions: []
  condition_matrix:
    enabled: false
    conditions: []
    write_reports: true

output:
  out_dir: outputs

slurm:
  planning:
    experiment_type: genome_cues_full
    external_expansion: sweep
```

This file defines 114 named training conditions:

- 1 explicit original condition;
- 101 saturation values from 100% through 0%, inclusive;
- 1 explicit grayscale condition;
- 1 channel permutation;
- 3 bilateral-filter strengths;
- 4 Gaussian-blur strengths;
- 3 patch-shuffle grid strengths.

With two models and two learning rates, it creates:

```text
2 models × 2 learning rates × 114 conditions = 456 model fits
```

The explicit `original`, `saturation_100pct`, `grayscale`, and
`saturation_000pct` conditions are intentionally separate named experiments,
even though their pixels may be equivalent at the colour endpoints. Remove the
explicit duplicates if that comparison is not scientifically useful.

## Optional hierarchy loss

Do not duplicate the full condition catalogue. Save this small child as
`configs/experiments/genome_cues_hierarchy.yaml`:

```yaml
extends: genome_cues_full.yaml

multi_task:
  hierarchy_loss:
    enabled: true
    parent_task: genus
    child_task: species
    weight: 0.5

wandb:
  group: genome-cues-full-hierarchy

slurm:
  planning:
    experiment_type: genome_cues_full_hierarchy
```

Use `enabled: false` in the parent for ordinary multitask loss and `true` in
the child for genus/species consistency. The hierarchy weight must be
non-negative, and the parent and child tasks must be distinct configured tasks.

## Task-loss weight recipes

Task weights are ordinary configuration, so the cleanest way to compare
correlated recipes is one small child per recipe. This avoids accidentally
creating the Cartesian product of every individual task weight.

Equal task weights, saved as `genome_cues_equal_weights.yaml`:

```yaml
extends: genome_cues_full.yaml

multi_task:
  loss_weights: {genus: 1.0, species: 1.0, age: 1.0}

wandb:
  group: genome-cues-equal-weights

slurm:
  planning:
    experiment_type: genome_cues_equal_weights
```

Age-weighted loss, saved as `genome_cues_age_weighted.yaml`:

```yaml
extends: genome_cues_full.yaml

multi_task:
  loss_weights: {genus: 1.0, species: 0.5, age: 3.0}

wandb:
  group: genome-cues-age-weighted

slurm:
  planning:
    experiment_type: genome_cues_age_weighted
```

To request a true Cartesian task-weight sweep in one file, add lists under
`sweep.parameters`:

```yaml
sweep:
  parameters:
    model.name: [convnext_base, vit_b_16]
    training.lr: [0.0005, 0.0001]
    multi_task.loss_weights.genus: [0.5, 1.0]
    multi_task.loss_weights.species: [0.5, 1.0]
    multi_task.loss_weights.age: [1.0, 2.0]
```

That is eight weight combinations, not two recipes. With the full condition
catalogue it creates `2 × 2 × 2 × 2 × 2 × 114 = 3,648` fits. Inspect the count
before submitting.

## Early-stopping choices

The main example enables early stopping with patience 6 and minimum improvement
0.0001. A more patient child can replace only those choices:

```yaml
extends: genome_cues_full.yaml

early_stopping:
  enabled: true
  patience: 12
  min_delta: 0.00005

wandb:
  group: genome-cues-patient-stopping
```

To compare patience values within one Cartesian sweep:

```yaml
sweep:
  parameters:
    model.name: [convnext_base, vit_b_16]
    training.lr: [0.0005, 0.0001]
    early_stopping.patience: [4, 8, 12]
```

This triples the run count. To disable early stopping, use a separate child
with `early_stopping.enabled: false`; mixing enabled and disabled values with
patience values would create scientifically duplicate disabled combinations.

## Small pilot before the full sweep

A full 456-fit run should first be checked with a small representative
condition list. Save `configs/experiments/genome_cues_pilot.yaml`:

```yaml
extends: genome_cues_full.yaml

training:
  epochs: 3
  batch_size: 32
  num_workers: 2

early_stopping:
  enabled: false

sweep:
  enabled: true
  parameters:
    model.name: [convnext_base, vit_b_16]
    training.lr: [0.0005, 0.0001]
  conditions:
    - {name: original, feature: baseline, transform: original, strength: 0.0, parameters: {}}
    - {name: saturation_050pct, feature: colour, transform: saturation, strength: 0.5, parameters: {retention: 0.5}}
    - {name: grayscale, feature: colour, transform: grayscale, strength: 1.0, parameters: {}}
    - {name: channel_shuffle_201, feature: colour, transform: channel_shuffle, strength: 1.0, parameters: {order: [2, 0, 1]}}
    - {name: bilateral_d7_c50_s50, feature: texture, transform: bilateral_filter, strength: 50.0, parameters: {diameter: 7, sigma_colour: 50.0, sigma_space: 50.0}}
    - {name: gaussian_sigma_2, feature: texture, transform: gaussian_blur, strength: 2.0, parameters: {sigma: 2.0}}
    - {name: patch_shuffle_grid_4, feature: shape, transform: patch_shuffle, strength: 4, parameters: {grid_size: 4, seed: 2026}}

wandb:
  group: genome-cues-pilot

slurm:
  planning:
    experiment_type: genome_cues_pilot
```

The pilot creates `2 models × 2 learning rates × 7 conditions = 28` fits.

## Fixed-RGB stress testing instead of matched training

The conditions in `sweep.conditions` are training conditions. For an
original-RGB model followed by transformed-test evaluation, train only the
original condition and put the stress battery beneath
`evaluation.test_conditions`:

```yaml
sweep:
  enabled: true
  parameters:
    model.name: [convnext_base, vit_b_16]
    training.lr: [0.0005, 0.0001]
  conditions:
    - {name: original, feature: baseline, transform: original, parameters: {}}

evaluation:
  test_conditions:
    enabled: true
    evaluate_original_training: true
    conditions:
      - name_template: saturation_{percent:03d}pct
        feature: colour
        transform: saturation
        parameter: retention
        range: {start: 1.0, stop: 0.0, step: -0.01}
      - {name: grayscale, feature: colour, transform: grayscale, strength: 1.0, parameters: {}}
      - {name: gaussian_sigma_2, feature: texture, transform: gaussian_blur, strength: 2.0, parameters: {sigma: 2.0}}
      - {name: patch_shuffle_grid_4, feature: shape, transform: patch_shuffle, strength: 4, parameters: {grid_size: 4, seed: 2026}}
```

This creates four model fits, not `4 × 104` fits. The test conditions reuse the
four original-RGB checkpoints. Do not silently mix this mode with matched
training; choose the scientific question explicitly.

## Genome validation and submission

First confirm that the Genome paths in `configs/clusters/genome.yaml` match the
actual installation:

```yaml
slurm:
  environment:
    conda_sh: ${HOME}/miniforge3/etc/profile.d/conda.sh
    conda_env: wormspecies
  paths:
    project_root: ${HOME}/worm-species/source
    data_root: ${HOME}/worm-species/data
    metadata_csv: ${HOME}/worm-species/data/01_Segmented/global_metadata.csv
    results_root: outputs_slurm
    cache_root: ${HOME}/worm-species/data/image_cache
```

Then validate, inspect, and render without submitting:

```bash
make validate \
  CONFIG=configs/experiments/genome_cues_pilot.yaml \
  CLUSTER=configs/clusters/genome.yaml

make inspect \
  CONFIG=configs/experiments/genome_cues_pilot.yaml \
  CLUSTER=configs/clusters/genome.yaml

make dry-run \
  CONFIG=configs/experiments/genome_cues_pilot.yaml \
  CLUSTER=configs/clusters/genome.yaml \
  ARTIFACTS_DIR=slurm/generated/genome-cues-pilot-check
```

Inspect the printed model list, condition list, expected run count, internal
run count, result paths, resources, and rendered array command. Only then
submit:

```bash
make submit \
  CONFIG=configs/experiments/genome_cues_pilot.yaml \
  CLUSTER=configs/clusters/genome.yaml
```

After the pilot succeeds, replace the pilot configuration with
`genome_cues_full.yaml` or one of its task-weight/hierarchy children.

Useful operational commands:

```bash
make status RESULTS_ROOT=outputs_slurm
make collect RESULTS_ROOT=outputs_slurm
make dashboard-prepare SLURM_RESULTS_ROOT=outputs_slurm
make dashboard SLURM_RESULTS_ROOT=outputs_slurm
```

## Validation rules and common mistakes

- `training.lr` must be positive.
- Early-stopping patience and minimum delta must be non-negative.
- Task weights must be finite and non-negative, with at least one positive
  selected-task weight.
- Hierarchy parent and child must be different configured tasks.
- Saturation retention must be within `[0, 1]`; the range step is signed and
  non-zero.
- Channel orders must be permutations of `[0, 1, 2]`.
- Bilateral diameters must be positive odd integers; both sigmas must be
  positive.
- Gaussian sigma must be positive.
- Patch grids must be at least 2 and divide `preprocessing.image_size`; 2, 4,
  and 8 all divide 224.
- Every `sweep.parameters` value must be a non-empty list.
- Condition names and output paths must be unique.
- External sweep expansion and internal expansion cannot both remain enabled.
- A dry run does not call `sbatch`; submission requires the explicit submit
  command.

## W&B configuration columns

The canonical logger uploads the resolved configuration once using unique
slash-delimited keys, for example `training/lr`,
`multi_task/loss_weights/age`, and
`input_condition/parameters/retention`. Compatibility aliases are normalized
before upload, so W&B does not receive repeated legacy and canonical columns.
Local `config.json`, metrics, reports, and histories remain the scientific
source of truth.

See [config.md](config.md) for the full key registry and [configs/README.md](configs/README.md)
for the existing experiment and cluster hierarchy.
