# Configuration reference

Worm Species uses ordinary YAML dictionaries. Configuration inheritance,
validation, experiment planning, local training, and SLURM rendering all use the
same resolved dictionary. No configuration framework or hidden training profile
is involved.

The safe starting point is [`config.yaml`](config.yaml). It inherits detailed
defaults from [`configs/defaults/base.yaml`](configs/defaults/base.yaml).
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
            |        +-- ghpc_dual_cue.yaml
            +-- configs/experiments/colour_ablation.yaml
            |        +-- ghpc_colour_ablation.yaml
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
| `preprocessing.image_size` | positive integer | Square image side used by transforms and the model. Patch grids must divide it. The legacy `data.image_size` spelling is normalized. |
| `preprocessing.normalisation.enabled` | boolean | Apply the configured mean/std after tensor conversion on every split. |
| `preprocessing.normalisation.mean`, `.std` | three finite numbers; standard deviations positive | RGB channel normalisation applied consistently to train, validation, and test. |
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

### Transform layers and order

The canonical composer keeps responsibilities explicit while preserving the
established pixel pipeline:

```text
resize
  -> train-only horizontal flip
  -> train-only vertical flip
  -> train-only random rotation
  -> optional train-only random Gaussian blur
  -> tensor conversion
  -> the one assigned input condition
  -> normalisation
```

Validation and test omit augmentation. They retain resize, the selected
evaluation condition, and normalization. Setting `augmentation.enabled: false`
disables all random operations; each child operation can also be disabled
independently. Flip and Gaussian-blur probabilities must be in `[0,1]`, rotation
degrees must be non-negative, and Gaussian blur requires a positive odd kernel
size plus an ascending two-value sigma range. A fixed seed preserves the
existing deterministic behavior. Changing an operation or its order is a
scientific change, not a layout choice.

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

`model.name` must be a supported torchvision constructor or DINOv3 model name.
Current experiment files exercise `resnet18`, `resnet50`, `efficientnet_b0`,
`convnext_base`, and `vit_b_16`. DINOv3 accepts Meta-style names including
`dinov3_vits16`, `dinov3_vitb16`, `dinov3_vitl16`, and
`dinov3_convnext_tiny` (plus the corresponding small/base/large variants).
The aliases use LVD-1689M weights; `dinov3_vitl16_sat493m` and
`dinov3_vit7b16_sat493m` select satellite-pretrained weights. Canonical timm
DINOv3 names are accepted as well. `model.pretrained` controls pretrained
weights; it is independent of `model.freeze_backbone`, which controls parameter
updates after construction. DINOv3 requires `timm>=1.0.20`.

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

The adapter uploads one canonical resolved configuration. Every setting has one
slash-delimited W&B key, such as `training/lr` or
`input_condition/parameters/sigma`. Legacy aliases and historical
double-underscore flattened keys are normalized before upload, so they do not
create repeated columns. Canonical values win if both spellings are present.
Metric aliases required by historical dashboards remain logging keys, not
duplicated configuration columns.

## Training modes are explicit switches

The following switches replace historical script profiles:

| Switch | Effect |
| --- | --- |
| `training.use_masked_labels` | Mask partially missing task labels. |
| `multi_task.hierarchy_loss.enabled` | Add the configured parent/child consistency term. |
| `wandb.enabled` | Enable W&B logging in the configured mode. |
| `input_condition.enabled` | Apply exactly one already-resolved training condition. |
| `evaluation.test_conditions.enabled` | Run the configured post-training transformed-test battery. |
| `evaluation.condition_matrix.enabled` | Evaluate selected checkpoints across selected test conditions without fitting again. |
| `sweep.enabled` | Ask the external planner to expand parameter values × complete condition objects. |

`experiment.type`, where present, is one of `standard`, `matched_condition`,
`rgb_stress_test`, or `matched_and_rgb_stress`. It describes the resolved run;
it must not trigger a second condition expansion inside the trainer.

### Assigned training condition

A generated run specification contains one explicit condition:

```yaml
input_condition:
  enabled: true
  name: patch_shuffle_grid_4
  feature: spatial_layout
  transform: patch_shuffle
  strength: 4
  parameters:
    grid_size: 4
    seed: 2026
```

Train and validation use this assigned condition for matched-condition runs.
Test-only fixed-RGB stress transforms remain a separate evaluation path. A
resolved submitted task must always report one internal training run.

### Fixed-RGB test schedule

`evaluation.test_conditions` defines post-training stress evaluation.
Conditions may be complete objects or names referring to `sweep.conditions`.
The schedule is evaluation only and never adds a model fit:

```yaml
evaluation:
  test_conditions:
    enabled: true
    evaluate_original_training: true
    conditions: [gaussian_sigma_2, patch_shuffle_grid_4]
```

The list must be non-empty, unique, and resolve to known conditions. The
`evaluate_original_training` switch lets the planner enable the battery only
for the original-RGB training condition. Historical `test_cue_suppression`
input is accepted and migrated, but new child configs should use this section.

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
| `patch_shuffle` | integer `grid_size >= 2` that divides `preprocessing.image_size`; integer `seed` | Fixed seed gives a reproducible patch permutation. |

Unknown names and invalid parameter combinations fail validation. Complete
condition objects keep their transform parameters together beneath
`parameters`; range objects generate complete conditions before planning.

## Condition-matrix semantics

`evaluation.condition_matrix` is post-training evaluation only:

```yaml
evaluation:
  condition_matrix:
    enabled: true
    conditions: [original, patch_shuffle_grid_2, patch_shuffle_grid_4]
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

The generic plan is the Cartesian product of non-empty parameter lists and the
complete condition list. This child config creates four trainings:

```yaml
extends: ../../config.yaml
sweep:
  enabled: true
  parameters:
    model.name: [resnet18, efficientnet_b0]
  conditions:
    - {name: original, feature: baseline, transform: original, parameters: {}}
    - name: patch_shuffle_grid_4
      feature: shape
      transform: patch_shuffle
      strength: 4
      parameters: {grid_size: 4, seed: 2026}
```

Endpoint-safe numeric ranges are also declarative. This creates 101 complete
saturation conditions from 100% through 0%, inclusive:

```yaml
sweep:
  enabled: true
  conditions:
    - name_template: saturation_{percent:03d}pct
      feature: colour
      transform: saturation
      parameter: retention
      range: {start: 1.0, stop: 0.0, step: -0.01}
```

Range items and explicit items may be mixed. Identifiers must remain unique.
The planner validates the resulting count, run identifiers, output paths, and
one-fit-per-spec invariant before it can submit anything.

Generated run configurations resolve the assigned model and condition, then
disable all expansion controls before calling the trainer. Planning fails
before submission if an external run spec
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

[`configs/clusters/local.yaml`](configs/clusters/local.yaml) disables real
SLURM submission and uses one CPU, no GPU, `4G` memory, a one-hour limit, no
scratch copy, and local relative paths. It is the preferred profile for planning
and tests.

### Genome

[`configs/clusters/genome.yaml`](configs/clusters/genome.yaml) selects the
Genome GPU partitions, one GPU, 12 CPUs, 12,384 MiB memory, a 90-minute limit,
and at most 12 active array tasks. It uses a per-job temporary cache under
`/tmp/${USER}/worm_species` and a `CACHE_READY` marker. Project, data, results,
cache, conda, collection, and logging paths are declared in the profile.

[`configs/clusters/genome_persistent.yaml`](configs/clusters/genome_persistent.yaml)
inherits Genome, selects persistent-cache scratch semantics, uses eight CPUs,
and carries the historical excluded-node choice. Use it with a persistent-cache
experiment instead of editing the experiment YAML.

### GHPC

[`configs/clusters/ghpc.yaml`](configs/clusters/ghpc.yaml) uses one GPU and
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
| GHPC dual-cue parity plan | `configs/experiments/ghpc_dual_cue.yaml` | `configs/clusters/ghpc.yaml` plus explicit nodes |
| Four pretrained patch models, train/test both grids | `configs/experiments/patch_shuffle_matrix.yaml` | local for planning, selected cluster for execution |
| Colour-retention sweep | `configs/experiments/colour_ablation.yaml` | selected cluster |
| GHPC colour parity plan | `configs/experiments/ghpc_colour_ablation.yaml` | `configs/clusters/ghpc.yaml` plus explicit nodes |
| Persistent hierarchy | `configs/experiments/persistent_hierarchy.yaml` | `configs/clusters/genome_persistent.yaml` |
| Persistent hierarchy plus W&B | `configs/experiments/persistent_hierarchy_wandb.yaml` | `configs/clusters/genome_persistent.yaml` |

For one explicit local condition, use a child config or dotted overrides and
disable all expansion mechanisms:

```bash
PYTHONPATH=src python -m worm_species.training \
  --config config.yaml --single-run \
  --override \
    input_condition.enabled=true \
    input_condition.name=gaussian_sigma_2 \
    input_condition.feature=texture \
    input_condition.transform=gaussian_blur \
    input_condition.strength=2.0 \
    input_condition.parameters.sigma=2.0 \
    sweep.enabled=false
```

Before a costly run, use `make validate`, `make inspect`, and `make dry-run` in
that order. Confirm the resolved model list, condition list, run count, internal
runs per task, output root, scratch root, and cluster resources. Only then use
the explicitly submitting command.

## Exact key registry

This registry names the complete public configuration surface. Keys containing
`<task>` or `<name>` are mapping entries rather than fixed literal names.

### Core, data, and splits

| Key | Value |
| --- | --- |
| `extends` | Parent YAML path, resolved relative to the child. |
| `seed` | Integer random seed. |
| `data.root_dir`, `data.metadata_csv` | Dataset and metadata paths. |
| `data.image_col`, `data.mask_col`, `data.barcode_col`, `data.group_col` | Metadata column names. |
| `preprocessing.image_size` | Positive square image size; patch grids must divide it. |
| `preprocessing.normalisation.enabled`, `.mean`, `.std` | Shared split preprocessing and RGB normalization values. |
| `augmentation.enabled` | Master train-only augmentation switch. |
| `augmentation.horizontal_flip.enabled`, `.probability` | Train-only horizontal flip and probability in `[0,1]`. |
| `augmentation.vertical_flip.enabled`, `.probability` | Train-only vertical flip and probability in `[0,1]`. |
| `augmentation.rotation.enabled`, `.degrees` | Train-only random rotation switch and non-negative bound. |
| `augmentation.gaussian_blur.enabled`, `.probability` | Train-only random Gaussian-blur switch and probability in `[0,1]`. |
| `augmentation.gaussian_blur.kernel_size`, `.sigma` | Positive odd kernel and ascending positive sigma range. |
| `data.colour_retention` | Number in `[0, 1]`. |
| `data.crop_to_foreground`, `data.crop_pad` | Foreground crop switch and non-negative pad fraction. |
| `data.strip_final_number_from_group` | Boolean barcode grouping rule. |
| `data.min_individuals_per_class` | Positive default rare-class threshold. |
| `data.min_individuals_per_class_by_task.<task>` | Positive per-task threshold. |
| `data.target_col`, `data.split_target_col` | Primary task and stratification columns. |
| `data.target_cols.<task>` | Task-to-metadata-column mapping. |
| `data.species_requires_binomial` | Boolean species-label validation rule. |
| `data.taxonomic_uncertainty.uncertain_species_labels` | Explicit uncertain-label list. |
| `data.taxonomic_uncertainty.uncertain_species_patterns` | Regular-expression list for uncertain labels. |
| `data.taxonomic_uncertainty.resolved_species_label_overrides.<name>` | Explicit resolved species label. |
| `data.taxonomic_uncertainty.life_stage_overrides.<name>` | Explicit life-stage label. |
| `split.use_predefined_splits`, `split.predefined_split_dir` | Select and locate predefined CSV membership. |
| `split.save_splits` | Permit recording newly generated splits; never rewrites predefined links. |
| `split.test_size`, `split.val_size` | Fractions in `(0, 1)` whose sum is below `1`. |

### Model, optimisation, tasks, and outputs

| Key | Value |
| --- | --- |
| `model.name` | Supported torchvision architecture or DINOv3/timm name. |
| `model.pretrained`, `model.freeze_backbone` | Independent boolean weight and fine-tuning switches. |
| `training.mode` | Currently `multitask`. |
| `training.use_masked_labels` | Boolean missing-label masking switch. |
| `training.epochs`, `training.batch_size`, `training.num_workers`, `training.val_interval` | Positive integers, except workers may be zero. |
| `training.lr` | Positive learning rate. |
| `training.weight_decay` | Non-negative number. |
| `training.use_amp`, `training.class_weight` | Boolean AMP and class-weight switches. |
| `multi_task.loss_weights.<task>` | Finite non-negative task weight; at least one selected task is positive. |
| `multi_task.normalize_loss_by_active_tasks` | Boolean active-task normalization. |
| `multi_task.selection_metric` | Best-checkpoint metric name. |
| `multi_task.hierarchy_loss.enabled` | Boolean hierarchy consistency switch. |
| `multi_task.hierarchy_loss.parent_task`, `multi_task.hierarchy_loss.child_task` | Distinct selected task names. |
| `multi_task.hierarchy_loss.weight` | Non-negative hierarchy weight. |
| `early_stopping.enabled`, `early_stopping.monitor` | Switch and monitored metric. |
| `early_stopping.mode` | `max` or `min`. |
| `early_stopping.patience`, `early_stopping.min_delta` | Non-negative stopping controls. |
| `output.out_dir` | Scientific run-output root. |
| `cache.enabled`, `cache.rebuild` | Cache use and explicit rebuild switches. |
| `cache.dir`, `cache.root_dir_cache`, `cache.format`, `cache.num_workers` | Cache paths, format, and positive worker count. |

### Experiment and transform switches

| Key | Value |
| --- | --- |
| `experiment.type` | `standard`, `matched_condition`, `rgb_stress_test`, or `matched_and_rgb_stress`. |
| `input_condition.enabled` | Apply one resolved training condition. |
| `input_condition.name`, `.feature`, `.transform`, `.strength` | Assigned complete condition identity. |
| `input_condition.parameters.<name>` | Transform-specific parameter such as retention, sigma, order, grid size, or seed. |
| `evaluation.test_conditions.enabled` | Post-training transformed-test switch. |
| `evaluation.test_conditions.conditions` | Non-empty unique condition names or complete condition/range objects. |
| `evaluation.test_conditions.evaluate_original_training` | Restrict the stress battery to original-RGB training when externally planning matched conditions. |
| `evaluation.condition_matrix.enabled`, `.conditions`, `.write_reports` | Cross-evaluation switch, condition list, and report-output switch. |
| `sweep.enabled`, `sweep.parameters.<key>` | External sweep switch and non-empty parameter value lists. |
| `sweep.conditions` | Non-empty complete condition objects and/or declarative range objects. |
| `sweep.conditions[].name`, `.feature`, `.transform`, `.strength`, `.parameters` | One explicit atomic training condition. |
| `sweep.conditions[].name_template`, `.parameter`, `.range` | One endpoint-aware condition range. Range keys are `start`, `stop`, and signed non-zero `step`. |

The loader still accepts `input_condition.condition`,
`test_cue_suppression`, `condition_matrix_evaluation`,
`matched_condition_training`, and `colour_ablation` when reading historical
saved configs. The migration command maps them to the canonical structure:

```bash
PYTHONPATH=src python -m worm_species.config migrate old-config.yaml
```

They are compatibility inputs, not additional active planners, and they are
removed from the normalized dictionary before W&B upload.

### W&B

| Key | Value |
| --- | --- |
| `wandb.enabled` | Independent logging switch. |
| `wandb.project`, `wandb.entity`, `wandb.group`, `wandb.name` | Run identity fields; nullable where supported. |
| `wandb.job_type`, `wandb.tags` | Job label and tag list. |
| `wandb.mode` | `online`, `offline`, `disabled`, `dryrun`, `run`, `shared`, or null. |
| `wandb.save_code`, `wandb.log_model` | Boolean artifact switches. |

### SLURM and cluster keys

| Key family | Values |
| --- | --- |
| `slurm.enabled`, `slurm.cluster_profile` | Scheduler switch and profile name. |
| `slurm.account`, `slurm.partition` | Cluster allocation fields. |
| `slurm.nodes`, `slurm.ntasks`, `slurm.cpus_per_task`, `slurm.gpus_per_task` | Positive resource counts. |
| `slurm.memory`, `slurm.time_limit` | Memory quantity and `HH:MM:SS` limit. |
| `slurm.array.max_active` | Positive array-concurrency limit. |
| `slurm.setup.*`, `slurm.collection.*`, `slurm.cleanup.*` | Each supports `enabled`, `cpus_per_task`, `memory`, `time_limit`; setup/cleanup may use `per_node`, collection may set `partition` and `kind`. |
| `slurm.scratch.mode` | `none`, `job_local`, `node_local`, or `persistent_cache` as supported by the selected profile. |
| `slurm.scratch.root`, `slurm.scratch.nodes` | Scratch path and explicit node list. |
| `slurm.scratch.unique_per_submission`, `slurm.scratch.submission_id` | Unique-root safety controls. |
| `slurm.scratch.copy_project`, `.copy_data`, `.data_include` | Scratch copy switches and include patterns. |
| `slurm.scratch.reuse_ready_cache`, `.ready_marker` | Cache-reuse marker controls. |
| `slurm.scratch.copy_cache_to_tmp`, `.tmp_reserve_gb` | Temporary-cache mode and reserved space. |
| `slurm.scratch.cleanup_after_run` | Cleanup switch; validation still rejects unsafe roots. |
| `slurm.environment.conda_sh`, `.conda_env` | Runtime environment activation. |
| `slurm.paths.project_root`, `.data_root`, `.metadata_csv` | Runtime input paths. |
| `slurm.paths.results_root`, `.cache_root` | Result and cache paths. |
| `slurm.logging.directory`, `.separate_stdout_stderr` | Scheduler log settings. |
| `slurm.monitoring.enabled`, `.interval_seconds` | Resource-profiling switch and positive interval. |
| `slurm.planning.experiment_type`, `.external_expansion` | Planner identity and sole sweep owner. |
| `slurm.submission.extra_sbatch_args`, `.exclude_nodes` | Validated additional scheduler arguments and excluded-node list. |

## Override examples

Override values are parsed as YAML-like scalars; quote lists and shell-sensitive
values. Configuration files are never edited in place.

```bash
# Architecture, image size, optimisation, and task weights.
python train.py --config config.yaml --dry-run --single-run --override \
  model.name=resnet50 \
  model.pretrained=true \
  preprocessing.image_size=224 \
  training.epochs=50 \
  training.lr=0.0001 \
  multi_task.loss_weights.genus=1.0 \
  multi_task.loss_weights.species=1.0 \
  multi_task.loss_weights.age=0.5

# Turn hierarchy and W&B off without changing any other choice.
python train.py --config config.yaml --dry-run --single-run --override \
  multi_task.hierarchy_loss.enabled=false \
  wandb.enabled=false

# Inspect a child config that names two fixed-RGB stress conditions.
PYTHONPATH=src python -m worm_species.config.inspect \
  --config configs/experiments/dual_cue.yaml --workflow training \
  --override \
    'evaluation.test_conditions.conditions=[gaussian_sigma_2,patch_shuffle_grid_4]'

# Change concurrency without editing the experiment or cluster YAML.
make dry-run EXPERIMENT=dual_cue \
  CLUSTER=configs/clusters/genome.yaml MAX_ACTIVE=4
```

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

For the compact file map, see [`configs/README.md`](configs/README.md). For
the repository-wide entry-point map, see [`info.md`](info.md).
