# Configuration reference

Worm Species uses ordinary YAML dictionaries. Configuration inheritance,
validation, experiment planning, local training, and SLURM rendering all use the
same resolved dictionary. No configuration framework or hidden training profile
is involved.

The safe starting point is [`config.yaml`](../config.yaml). It inherits detailed
defaults from [`configs/defaults/base.yaml`](../configs/defaults/base.yaml).
Experiment files change scientific choices; cluster files change only machine,
resource, scratch, and path choices.

## Resolution and precedence

`extends` accepts a path relative to the child file. Mapping values are merged
recursively and lists are replaced in full; lists are never concatenated.

```text
configs/defaults/base.yaml
            |
            v
       config.yaml
            |
            +-- configs/experiments/standard.yaml
            +-- configs/experiments/hierarchy.yaml
            +-- configs/experiments/dual_cue.yaml
            +-- configs/experiments/colour_ablation.yaml
            +-- configs/experiments/patch_shuffle_matrix.yaml
            +-- configs/experiments/persistent_hierarchy.yaml
                         |
                         +-- persistent_hierarchy_wandb.yaml

configs/clusters/local.yaml
configs/clusters/genome.yaml
            |
            +-- genome_persistent.yaml
configs/clusters/ghpc.yaml
```

For a submitted plan, later sources win in this order:

```text
explicit CLI --override
    > explicitly requested --legacy-env values
    > cluster configuration
    > experiment child
    > config.yaml
    > configs/defaults/base.yaml
```

Historical environment variables are not imported implicitly. Pass
`--legacy-env` only when deliberately reproducing an old launcher environment.
The result remains a normal Python dictionary and the same dotted override
syntax continues to work.

## Preferred commands

Validate or inspect configuration without starting training:

```bash
PYTHONPATH=src python -m worm_species.config.validate \
  --config config.yaml --workflow training

PYTHONPATH=src python -m worm_species.config.inspect \
  --config config.yaml --workflow training --format yaml
```

Use `--check-paths` only on a machine where the configured dataset and split
paths should already exist. Omitting it is intentional for laptop dry runs,
login-node planning, and cluster submission.

Inspect and render an experiment/cluster combination:

```bash
make validate \
  CONFIG=configs/experiments/patch_shuffle_matrix.yaml \
  CLUSTER=configs/clusters/local.yaml

make inspect \
  CONFIG=configs/experiments/patch_shuffle_matrix.yaml \
  CLUSTER=configs/clusters/local.yaml

make dry-run \
  CONFIG=configs/experiments/patch_shuffle_matrix.yaml \
  CLUSTER=configs/clusters/local.yaml \
  ARTIFACTS_DIR=slurm/generated/patch-matrix-check
```

Run exactly one local training process or explicitly submit a plan:

```bash
make train TRAIN_CONFIG=config.yaml

make submit \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/genome.yaml
```

Direct SLURM equivalents use `python -m worm_species.slurm launch` with one of
the mutually exclusive `--dry-run` and `--submit` flags. Rendering never implies
submission.

## Data, images, labels, and splits

| Key | Type and allowed value | Meaning |
| --- | --- | --- |
| `seed` | integer | Global reproducibility seed. |
| `data.root_dir` | path string | Image-tree root. Checked only when path validation is requested. |
| `data.metadata_csv` | path string | Global metadata table. Checked only when appropriate to the workflow. |
| `data.image_col` | column name | Usually `rel_path_seg`; `rel_path_raw` and `rel_path_segmask` are documented alternatives. |
| `data.mask_col` | column name | Segmentation-mask path column. |
| `data.image_size` | positive integer | Square image side used by transforms and the model. Patch grids must divide it. |
| `data.colour_retention` | number in `[0, 1]` | `1.0` is RGB identity; `0.0` removes saturation. |
| `data.crop_to_foreground` | boolean | Enable mask-derived foreground cropping. |
| `data.crop_pad` | number at least `0` | Fractional padding around a foreground crop. |
| `data.group_col` | column name | Individual-level grouping key used to prevent leakage. |
| `data.strip_final_number_from_group` | boolean | Use only when a numeric barcode suffix is not part of the individual identity. |
| `data.min_individuals_per_class` | positive integer | Default rare-class threshold. |
| `data.min_individuals_per_class_by_task` | mapping of task to threshold | Task-specific rare-class masks. |
| `data.target_cols` | mapping | Canonical task names to metadata columns. Defaults to genus, species, and age. |
| `data.species_requires_binomial` | boolean | Require the validated species-label form. |
| `data.taxonomic_uncertainty` | mapping | Uncertain labels/patterns plus explicit species and life-stage overrides. |

`split.test_size` and `split.val_size` must each be strictly between `0` and
`1`, and their sum must be less than `1`. With
`split.use_predefined_splits: true`, the trainer loads the external CSVs beneath
`split.predefined_split_dir` and checks group separation. The repository's
`split_csv/*.csv` links are external scientific inputs and must not be rewritten.
`split.save_splits` controls whether a newly resolved split is recorded; it does
not authorize changing predefined membership.

## Model, tasks, and loss

```yaml
model:
  name: efficientnet_b0
  pretrained: true
  freeze_backbone: false

multi_task:
  loss_weights:
    genus: 1.0
    species: 0.5
    age: 2.0
  normalize_loss_by_active_tasks: true
  selection_metric: mean_macro_f1
  hierarchy_loss:
    enabled: true
    parent_task: genus
    child_task: species
    weight: 0.5
```

`model.name` must be a supported callable model constructor. Current experiment
files exercise `resnet18`, `resnet50`, `efficientnet_b0`, `convnext_base`, and
`vit_b_16`. `model.pretrained` controls pretrained weights; it is independent of
`model.freeze_backbone`, which controls parameter updates after construction.

Every selected task requires a finite, non-negative loss weight and at least one
task must have positive weight. Hierarchy consistency requires distinct selected
parent and child tasks, a valid child-to-parent mapping, and a non-negative
weight. `normalize_loss_by_active_tasks` prevents a sample with fewer observed
labels from being down-weighted merely because labels are absent. The checkpoint
selection field may be `mean_macro_f1` or a configured task metric such as
`genus_macro_f1`.

## Optimisation, masking, AMP, and stopping

| Key | Constraint | Default purpose |
| --- | --- | --- |
| `training.mode` | currently `multitask` | Selects the algorithm family, not a legacy script. |
| `training.use_masked_labels` | boolean | Mask unavailable task losses/metrics without changing rows or split membership. |
| `training.epochs` | positive integer | Maximum epochs. |
| `training.batch_size` | positive integer | Examples per optimiser step. |
| `training.lr` | positive number | Learning rate. |
| `training.weight_decay` | number at least `0` | Optimiser weight decay. |
| `training.use_amp` | boolean | Enable automatic mixed precision where supported. |
| `training.class_weight` | boolean | Apply training-split class weights. |
| `training.num_workers` | integer at least `0` | Data-loader workers. Use `0` for simple CPU debugging. |
| `training.val_interval` | positive integer | Epoch interval between validation passes. |
| `early_stopping.enabled` | boolean | Enable metric-based stopping. |
| `early_stopping.mode` | `max` or `min` | Improvement direction. |
| `early_stopping.patience` | non-negative integer | Number of non-improving checks tolerated. |
| `early_stopping.min_delta` | non-negative number | Minimum metric improvement. |

When `use_masked_labels` is false, every selected task label is required and a
missing value fails clearly. It never silently drops rows to make a run pass.

## W&B is a feature switch

W&B does not select a trainer. `wandb.enabled: false` makes the canonical trainer
fully local. When enabled, important fields are `project`, `entity`, `group`,
`name`, `job_type`, `tags`, `save_code`, and `log_model`. Valid modes include
`online`, `offline`, `disabled`, `dryrun`, `run`, `shared`, and `null`. Use
`offline` for normal network-free operation and mock or disable W&B in CPU tests.

## Training modes are explicit switches

The following switches replace historical script profiles:

| Switch | Effect |
| --- | --- |
| `training.use_masked_labels` | Mask partially missing task labels. |
| `multi_task.hierarchy_loss.enabled` | Add the configured parent/child consistency term. |
| `wandb.enabled` | Enable W&B logging in the configured mode. |
| `input_condition.enabled` | Apply exactly one already-resolved training condition. |
| `matched_condition_training.enabled` | Ask the external planner to generate matched-condition run specs. |
| `test_cue_suppression.enabled` | After original-RGB training, run the fixed-RGB transformed-test battery. |
| `condition_matrix_evaluation.enabled` | Evaluate selected checkpoints across selected test conditions without fitting again. |
| `colour_ablation.enabled` | Ask the external planner for colour-retention run specs. |
| `sweep.enabled` | Ask the external planner to expand listed hyperparameter values. |

`experiment.type`, where present, is one of `standard`, `matched_condition`,
`rgb_stress_test`, or `matched_and_rgb_stress`. It describes the resolved run;
it must not trigger a second condition expansion inside the trainer.

### Assigned training condition

A generated run specification contains one explicit condition:

```yaml
input_condition:
  enabled: true
  condition: patch_shuffle_grid_4
  feature: spatial_layout
  transform: patch_shuffle
  strength: 4
  seed: 2026
```

Train and validation use this assigned condition for matched-condition runs.
Test-only fixed-RGB stress transforms remain a separate evaluation path. A
resolved submitted task must always report one internal training run.

### Fixed-RGB test allow-list

`test_cue_suppression.enabled` enables post-training stress evaluation for an
original-RGB checkpoint. An optional `condition_names` list restricts evaluation
to named catalogue conditions, for example:

```yaml
test_cue_suppression:
  enabled: true
  condition_names:
    - gaussian_sigma_2
    - patch_shuffle_grid_4
```

The allow-list must be non-empty, unique, and resolve to known conditions. It is
not a list of additional training runs.

## Transform catalogue and parameter validation

The exact canonical transform names are:

| Name | Parameters and constraints | Determinism/identity |
| --- | --- | --- |
| `original` | no parameters | RGB identity. |
| `saturation` | retention or catalogue values in `[0, 1]`; catalogue step greater than `0` | `1.0` is identity; `0.0` is greyscale. |
| `grayscale` | no numeric parameter | Output RGB channels are equal. |
| `channel_shuffle` | `order` is a permutation of `[0, 1, 2]` | Explicit order is deterministic. |
| `bilateral_filter` | positive odd `diameter`; positive `sigma_colour` and `sigma_space` | Preserves image shape and dtype. |
| `gaussian_blur` | `sigma` greater than `0` | Preserves image shape and dtype. |
| `patch_shuffle` | integer `grid_size >= 2` that divides `data.image_size`; integer `seed` | Fixed seed gives a reproducible patch permutation. |

Unknown names and invalid parameter combinations fail validation. The detailed
catalogue beneath `test_cue_suppression` controls available saturation values,
channel orders, bilateral settings, Gaussian sigmas, and patch grids. A
`condition_names` allow-list selects from that catalogue without redefining it.

## Condition-matrix semantics

`condition_matrix_evaluation` is post-training evaluation only:

```yaml
condition_matrix_evaluation:
  enabled: true
  condition_names:
    - original
    - patch_shuffle_grid_2
    - patch_shuffle_grid_4
  write_reports: true
```

For every evaluated train/test pair, the relation is recorded as:

| Relation | Meaning |
| --- | --- |
| `matched` | Training and test condition names are equal. |
| `rgb_stress` | Training used `original`, while testing used a transformed condition. |
| `cross_condition` | A transformed-training checkpoint is evaluated under another condition. |

The matrix never adds optimiser steps or model fits. With four models and the
three conditions `original`, `patch_shuffle_grid_2`, and
`patch_shuffle_grid_4`, the planner creates 12 training processes. Evaluating
each selected checkpoint under all three test conditions produces 36
model/condition cells, or 108 task rows for genus/species/age. This deliberately
tests both matched and mismatched train/test conditions without multiplying the
training sweep.

Use the supplied experiment:

```bash
make inspect \
  CONFIG=configs/experiments/patch_shuffle_matrix.yaml \
  CLUSTER=configs/clusters/local.yaml

make dry-run \
  CONFIG=configs/experiments/patch_shuffle_matrix.yaml \
  CLUSTER=configs/clusters/local.yaml \
  ARTIFACTS_DIR=slurm/generated/patch-shuffle-matrix
```

All four models in that file have `model.pretrained: true`; change it to false
only as an explicit experiment decision. The selected patch grids must continue
to divide the configured image size.

## Sweep ownership and nested-expansion prevention

Sweep values are non-empty lists. The external planner owns all expansion:

```text
one generated run specification
    = one array task
    = one canonical trainer invocation
    = one model fit
```

Generated run configurations resolve the assigned model and condition, then
disable `sweep`, `colour_ablation`, and `matched_condition_training` before
calling the trainer. Planning fails before submission if an external run spec
would retain an internal expansion, a run resolves to more than one fit, run IDs
are duplicated, result paths collide, or the run-spec count differs from the
array size.

The current supplied plans are:

| Experiment | Models/conditions | Training processes | Evaluation note |
| --- | --- | ---: | --- |
| root `config.yaml` | one assigned model, original RGB | 1 | Safe local default. |
| `standard.yaml` | two models | 2 | Standard training. |
| `hierarchy.yaml` | two models | 2 | Same sweep with hierarchy consistency. |
| `dual_cue.yaml` | two models, full cue catalogue | 224 | Matched training plus original-RGB stress testing. |
| `patch_shuffle_matrix.yaml` | four models × three train conditions | 12 | 36 train/test evaluation cells. |
| `colour_ablation.yaml` | two models × 101 retention values | 202 | Values include both 100% and 0%. |
| `persistent_hierarchy.yaml` | two models | 2 | Persistent-cache hierarchy run. |

Counts are validation contracts for these files, not a hard-coded requirement
that every experiment have 224 tasks.

## Output, cache, logs, and generated plans

| Key | Purpose |
| --- | --- |
| `output.out_dir` | Canonical local trainer result root. |
| `slurm.paths.results_root` | Cluster result root, normally `outputs_slurm`. |
| `cache.enabled` | Enable image-cache reads/build integration. |
| `cache.dir` | Image cache path used by training. |
| `cache.root_dir_cache` | Higher-level cache root retained for compatibility. |
| `cache.format` | Cached image format, currently `png`. |
| `cache.rebuild` | Explicitly rebuild rather than reuse. |
| `cache.num_workers` | Positive integer cache worker count. |
| `slurm.logging.directory` | Scheduler stdout/stderr root, normally `logs/slurm`. |

Build or verify a persistent image cache explicitly; cache maintenance is not a
side effect of config inspection:

```bash
PYTHONPATH=src python -m worm_species.cache build \
  --config config.yaml \
  --data-root ../petridish-worm-images \
  --metadata-csv ../petridish-worm-images/01_Segmented/global_metadata.csv \
  --cache-dir ../cache/images

PYTHONPATH=src python -m worm_species.cache verify \
  --cache-dir ../cache/images
```

Dry-run plans are written beneath `slurm/generated/`, which is generated state,
not a scientific result. A plan records resolved configuration, run specs,
resource choices, dependency graph, hashes, and rendered scripts. It never
copies datasets, checkpoints, or existing outputs. `make clean-generated`
removes only generated plans and local dashboard indexes; it refuses unrelated
paths and never removes scientific results.

## Cluster profiles

Cluster profiles are an independent axis. Choose them without changing model,
label, transform, or experiment semantics.

### Local

[`configs/clusters/local.yaml`](../configs/clusters/local.yaml) disables real
SLURM submission and uses one CPU, no GPU, `4G` memory, a one-hour limit, no
scratch copy, and local relative paths. It is the preferred profile for planning
and tests.

### Genome

[`configs/clusters/genome.yaml`](../configs/clusters/genome.yaml) selects the
Genome GPU partitions, one GPU, 12 CPUs, 12,384 MiB memory, a 90-minute limit,
and at most 12 active array tasks. It uses a per-job temporary cache under
`/tmp/${USER}/worm_species` and a `CACHE_READY` marker. Project, data, results,
cache, conda, collection, and logging paths are declared in the profile.

[`configs/clusters/genome_persistent.yaml`](../configs/clusters/genome_persistent.yaml)
inherits Genome, selects persistent-cache scratch semantics, uses eight CPUs,
and carries the historical excluded-node choice. Use it with a persistent-cache
experiment instead of editing the experiment YAML.

### GHPC

[`configs/clusters/ghpc.yaml`](../configs/clusters/ghpc.yaml) uses one GPU and
node-local `/scratch`. Setup and cleanup are per-node jobs. A GHPC plan must
provide a non-empty, explicit GPU-node list, and its scratch root must be unique
per submission. Static or ambiguous scratch cleanup fails before rendering.

Example safe GHPC hierarchy dry run:

```bash
PYTHONPATH=src python -m worm_species.slurm launch --dry-run \
  --config configs/experiments/hierarchy.yaml \
  --cluster-config configs/clusters/ghpc.yaml \
  --override 'slurm.scratch.nodes=[gpu001,gpu002]' \
  --artifacts-dir slurm/generated/ghpc-hierarchy-check
```

Replace the example node names with the exact nodes assigned for the intended
submission. Do not submit the example literally.

### Resource and path validation

The SLURM validator checks supported cluster values, nodes/tasks/CPU/GPU counts,
memory, time-limit syntax, array concurrency, setup/collection/cleanup resources,
scratch mode, unique path semantics, result collisions, and unsupported extra
options. Bare integer memory values are interpreted consistently by the
renderer; unit-bearing strings are also supported where the profile uses them.

Machine paths may contain shell variables such as `${USER}` or `${HOME}`. They
are resolved for the rendered environment rather than required to exist on the
planning host. `--check-paths` is therefore opt-in and should not be combined
with cross-machine dry-run validation unless those paths are locally mounted.

## Worked experiment choices

| Goal | Experiment file | Cluster file |
| --- | --- | --- |
| One local baseline | `config.yaml` | none required |
| Two-model standard | `configs/experiments/standard.yaml` | `configs/clusters/local.yaml` |
| Hierarchy consistency | `configs/experiments/hierarchy.yaml` | local, Genome, or GHPC |
| Full dual-cue study | `configs/experiments/dual_cue.yaml` | `configs/clusters/genome.yaml` |
| Four pretrained patch models, train/test both grids | `configs/experiments/patch_shuffle_matrix.yaml` | local for planning, selected cluster for execution |
| Colour-retention sweep | `configs/experiments/colour_ablation.yaml` | selected cluster |
| Persistent hierarchy | `configs/experiments/persistent_hierarchy.yaml` | `configs/clusters/genome_persistent.yaml` |
| Persistent hierarchy plus W&B | `configs/experiments/persistent_hierarchy_wandb.yaml` | `configs/clusters/genome_persistent.yaml` |

For one explicit local condition, use a child config or dotted overrides and
disable all expansion mechanisms:

```bash
PYTHONPATH=src python -m worm_species.training \
  --config config.yaml --single-run \
  --override \
    input_condition.enabled=true \
    input_condition.condition=gaussian_sigma_2 \
    input_condition.feature=texture \
    input_condition.transform=gaussian_blur \
    input_condition.strength=2.0 \
    sweep.enabled=false \
    colour_ablation.enabled=false \
    matched_condition_training.enabled=false
```

Before a costly run, use `make validate`, `make inspect`, and `make dry-run` in
that order. Confirm the resolved model list, condition list, run count, internal
runs per task, output root, scratch root, and cluster resources. Only then use
the explicitly submitting command.

## Scientific safety contracts

- Training-time conditions and test-only stress conditions remain separate.
- Fixed-RGB stress evaluation begins from an original-RGB trained checkpoint.
- A condition matrix evaluates checkpoints and never expands training.
- Seeds are explicit for stochastic transforms.
- Predefined split membership and external split links are not mutated.
- Every externally generated run spec resolves to one model fit.
- Checkpoint, metric, result, W&B, and filename schemas remain stable across
  configuration-only choices.
- Dry-run, status, collection, and dashboard discovery never load full model
  checkpoints or modify live result directories.

For the compact file map, see [`configs/README.md`](../configs/README.md). For
the repository-wide entry-point map, see [`info.md`](../info.md).
