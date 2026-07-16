Please simplify the experiment and transformation configuration before adding further special-purpose planners.

## One generic training sweep

Colour ablation, matched-condition training, model comparison, image-size comparison, and augmentation comparison should all use the same generic external sweep engine.

Remove the need for separate planning mechanisms such as:

```text
colour_ablation
matched_condition_training
special model sweeps
special image-size sweeps
```

These may remain temporarily as compatibility aliases, but they should resolve into one canonical sweep representation.

The canonical rule remains:

```text
one resolved sweep combination
=
one run specification
=
one SLURM task
=
one trainer invocation
=
one model fit
```

Post-training evaluation schedules remain separate because they evaluate existing checkpoints rather than create additional model fits.

## Separate three transformation concepts

Please distinguish clearly between:

### 1. Deterministic preprocessing

Applied to training, validation, and test images:

```yaml
preprocessing:
  image_size: 224

  normalisation:
    enabled: true
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

For backward compatibility, continue accepting:

```yaml
data:
  image_size: 224
```

Internally, there should be one canonical resolved value. Do not maintain two independent image-size settings.

### 2. Stochastic training augmentation

Applied only to the training set by default:

```yaml
augmentation:
  enabled: true

  horizontal_flip:
    enabled: true
    probability: 0.5

  vertical_flip:
    enabled: true
    probability: 0.5

  rotation:
    enabled: true
    degrees: 270
```

This should reproduce the current behaviour:

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.5)
transforms.RandomRotation(degrees=270)
```

Validation and test transforms must not receive these random augmentations unless an explicit diagnostic configuration requests them.

Add validation for:

* probabilities in `[0, 1]`;
* non-negative rotation degrees;
* valid image size;
* valid normalisation vectors;
* equal mean and standard-deviation lengths;
* positive standard deviations.

### 3. Experimental input condition

This is the scientific manipulation being studied:

```yaml
input_condition:
  enabled: true
  name: gaussian_sigma_2
  transform: gaussian_blur
  parameters:
    sigma: 2.0
```

Examples include:

* original;
* saturation;
* greyscale;
* channel shuffle;
* Gaussian blur;
* bilateral filtering;
* patch shuffle.

Experimental conditions are not ordinary data augmentation. They must be reproducible and saved in the resolved run configuration.

For matched-condition training, the assigned condition is applied consistently to training, validation, and testing.

For fixed-RGB stress evaluation, the original model is trained without the condition and the condition is applied only during post-training testing.

## Simplified sweep format

Support normal scalar parameter sweeps:

```yaml
sweep:
  enabled: true

  parameters:
    model.name:
      - convnext_base
      - vit_b_16

    preprocessing.image_size:
      - 224
      - 384

    augmentation.rotation.degrees:
      - 0
      - 90
      - 270
```

This is a Cartesian product.

Also support a list of complete condition objects so incompatible transform parameters are not accidentally combined:

```yaml
sweep:
  enabled: true

  parameters:
    model.name:
      - convnext_base
      - vit_b_16

  conditions:
    - name: original
      transform: original
      parameters: {}

    - name: grayscale
      transform: grayscale
      parameters: {}

    - name: gaussian_sigma_1
      transform: gaussian_blur
      parameters:
        sigma: 1.0

    - name: gaussian_sigma_2
      transform: gaussian_blur
      parameters:
        sigma: 2.0

    - name: patch_shuffle_grid_4
      transform: patch_shuffle
      parameters:
        grid_size: 4
        seed: 2026
```

The planner should calculate:

```text
Cartesian product of sweep.parameters
×
each complete sweep.conditions entry
```

Do not create a Cartesian product between condition names, transform names, and unrelated strength parameters.

## Compact range syntax

For long numerical sweeps such as saturation, support a generic range expansion:

```yaml
sweep:
  conditions:
    - name_template: saturation_{percent:03d}
      transform: saturation
      parameter: retention
      range:
        start: 1.0
        stop: 0.0
        step: -0.01
```

This should produce the same 101 conditions as the previous colour-ablation mechanism.

The range utility should be generic and usable for other numerical parameters where appropriate. It must:

* include both requested endpoints;
* avoid floating-point naming errors;
* reject a zero step;
* reject a step with the wrong direction;
* produce deterministic names;
* detect duplicate condition identifiers.

## Evaluation remains separate

Use one explicit evaluation section:

```yaml
evaluation:
  test_conditions:
    enabled: true

    conditions:
      - original
      - grayscale
      - gaussian_sigma_1
      - gaussian_sigma_2
      - patch_shuffle_grid_4

  condition_matrix:
    enabled: false
```

This section must never add training jobs.

It may evaluate:

* the RGB checkpoint under transformed test sets;
* matched-condition checkpoints;
* selected checkpoints across a condition matrix.

## Preferred compact configuration

A normal experiment should be understandable from one file:

```yaml
model:
  name: convnext_base
  pretrained: true

preprocessing:
  image_size: 224

augmentation:
  enabled: true

  horizontal_flip:
    enabled: true
    probability: 0.5

  vertical_flip:
    enabled: true
    probability: 0.5

  rotation:
    enabled: true
    degrees: 270

sweep:
  enabled: true

  parameters:
    model.name:
      - convnext_base
      - vit_b_16

    preprocessing.image_size:
      - 224
      - 384

  conditions:
    - name: original
      transform: original
      parameters: {}

    - name: grayscale
      transform: grayscale
      parameters: {}

evaluation:
  test_conditions:
    enabled: true
    conditions:
      - original
      - grayscale
```

## Backward compatibility

Continue accepting old keys temporarily:

```text
data.image_size
data.colour_retention
colour_ablation.*
matched_condition_training.*
test_cue_suppression.*
condition_matrix_evaluation.*
```

Resolve them into the canonical structure and produce clear compatibility warnings during inspection, not during normal training output if that would break CLI contracts.

Do not maintain separate internal implementations for the old and new configuration formats.

Add a migration command or report:

```bash
python -m worm_species.config migrate \
  --config configs/experiments/dual_cue.yaml
```

It should print or write the equivalent simplified configuration without modifying the source file automatically.

## Tests

Add tests proving:

* image size can be configured;
* image size can be swept;
* resize output has the expected dimensions;
* disabled augmentation gives deterministic preprocessing;
* horizontal-flip probability validation;
* vertical-flip probability validation;
* rotation validation;
* training augmentation is not applied to validation or test sets;
* matched-condition transforms are applied to all matched splits;
* fixed-RGB stress transforms are test-only;
* explicit condition objects do not create invalid Cartesian combinations;
* saturation range generates exactly 101 conditions;
* two models × 101 conditions creates exactly 202 training processes;
* old colour-ablation configuration resolves to the same run specifications;
* no nested internal expansion occurs;
* evaluation conditions never increase the training-process count.

Please implement this simplification before adding additional experiment-specific configuration branches. Preserve current scientific results, run identifiers, checkpoints, metrics, output schemas, and legacy interfaces.

## Standardise and improve Weights & Biases logging

Please refactor W&B integration so every experiment uses the same clear logging structure, regardless of whether it is standard training, matched-condition training, colour/saturation analysis, or fixed-model cue-suppression evaluation.

W&B must remain an optional logging feature, not a separate trainer or experiment mode.

### Consistent run identity

Every run should have unambiguous metadata:

```text
project
group
job_type
run_name
architecture
training_condition
test_condition
experiment_type
seed
```

Use predictable run names such as:

```text
convnext_base__train-original
vit_b_16__train-grayscale
convnext_base__train-gaussian_sigma_2
```

For post-training evaluations, do not create confusing duplicate training runs. Either:

* log evaluation results into the original training run under clearly prefixed keys; or
* create linked evaluation runs with `job_type=evaluation` and the source training run/checkpoint recorded explicitly.

The chosen strategy must be consistent across all experiments.

### Log the fully resolved configuration

Log the canonical resolved configuration after:

* inheritance;
* cluster-independent experiment resolution;
* compatibility-key migration;
* CLI overrides;
* assigned sweep values;
* assigned training condition.

Important fields should also appear in the W&B summary and filters:

```text
model/name
model/pretrained
preprocessing/image_size
augmentation/horizontal_flip
augmentation/vertical_flip
augmentation/rotation_degrees
training/lr
training/batch_size
training/epochs
training/seed
training_condition/name
training_condition/transform
training_condition/strength
experiment/type
```

Do not log only the original unresolved YAML.

### Standard metric names

Use one stable naming convention:

```text
train/loss
train/genus_loss
train/species_loss
train/age_loss
train/hierarchy_loss

val/loss
val/genus_macro_f1
val/species_macro_f1
val/age_macro_f1
val/mean_macro_f1

test/original/genus_macro_f1
test/original/species_macro_f1
test/original/age_macro_f1
test/original/mean_macro_f1
```

For transformed test conditions, include the condition in the metric path:

```text
test/grayscale/species_macro_f1
test/gaussian_sigma_2/species_macro_f1
test/patch_shuffle_grid_4/species_macro_f1
```

For relative robustness metrics:

```text
robustness/grayscale/species_ratio
robustness/gaussian_sigma_2/species_ratio
robustness/patch_shuffle_grid_4/species_ratio
```

where:

```text
ratio = transformed macro-F1 / original macro-F1
```

Also log:

```text
robustness/<condition>/<task>_relative_drop
comparison/<condition>/<task>_adaptation_gain
```

Avoid simultaneously logging several differently named copies of the same metric.

### Clearly log training and testing conditions

Each run must distinguish:

```text
train_condition
test_condition
condition_relation
```

Allowed relations should include:

```text
original
matched
rgb_stress
cross_condition
```

For example:

```text
train_condition = original
test_condition = gaussian_sigma_2
condition_relation = rgb_stress
```

or:

```text
train_condition = gaussian_sigma_2
test_condition = gaussian_sigma_2
condition_relation = matched
```

This information should appear both in W&B config fields and in result tables.

### Confusion matrices

Log one confusion matrix per task and test condition where labelled examples exist.

Examples:

```text
confusion_matrix/original/genus
confusion_matrix/original/species
confusion_matrix/original/age

confusion_matrix/grayscale/species
confusion_matrix/gaussian_sigma_2/species
```

Requirements:

* use the same class ordering as the saved CSV confusion matrix;
* use human-readable class names;
* log true and predicted labels correctly;
* do not include missing-label samples;
* handle tasks with no valid test labels without crashing;
* avoid overwriting one condition’s confusion matrix with another;
* continue saving the CSV confusion matrices locally as the scientific record.

For large condition batteries, provide configuration controls such as:

```yaml
wandb:
  confusion_matrices:
    enabled: true
    conditions:
      - original
      - grayscale
      - gaussian_sigma_2
    tasks:
      - genus
      - species
      - age
```

Do not upload hundreds of confusion matrices by default without an explicit setting.

### Classification reports and tables

Log structured W&B tables for:

* per-class precision, recall, F1, and support;
* per-condition task metrics;
* robustness ratios;
* matched versus RGB-stress comparisons;
* adaptation gain;
* failed or missing evaluation conditions.

Suggested tables:

```text
tables/test_metrics_by_condition
tables/classification_report_original
tables/classification_report_by_condition
tables/robustness_summary
tables/matched_vs_rgb_stress
```

Each row should include:

```text
model
task
train_condition
test_condition
condition_relation
class_name
precision
recall
f1
support
macro_f1
balanced_accuracy
accuracy
ratio_to_original
adaptation_gain
```

Only include applicable columns for each table.

### Curves and plots

Log useful plots with explicit axes and titles:

* training and validation loss by epoch;
* validation macro-F1 by epoch;
* per-task validation macro-F1;
* saturation retention versus macro-F1;
* saturation retention versus relative performance;
* cue-suppression strength versus macro-F1;
* matched-condition versus RGB-stress performance;
* adaptation gain by condition;
* model-family comparison.

Do not log plots whose axes or conditions are ambiguous.

For saturation curves, retain numeric saturation/retention values as numeric columns so W&B can sort them correctly. Do not rely only on strings such as `saturation_099`.

### Run summaries

At completion, populate stable summary fields:

```text
best_epoch
best_val_score
selection_metric
test_original_mean_macro_f1
test_original_genus_macro_f1
test_original_species_macro_f1
test_original_age_macro_f1
checkpoint_path
run_status
```

For condition experiments, also record:

```text
train_condition
number_of_test_conditions
worst_condition
worst_condition_mean_macro_f1
mean_robustness_ratio
minimum_robustness_ratio
```

Failed or interrupted runs should record a clear status where possible rather than appearing indistinguishable from incomplete logging.

### W&B grouping

Use grouping that makes comparisons easy:

```text
project: worm-species
group: <experiment/submission identifier>
job_type: training or evaluation
```

Use tags for:

```text
architecture
experiment type
training transform
cluster profile
matched training
RGB stress testing
hierarchy loss
pretrained status
image size
```

All runs from one generated SLURM plan should share a submission/group identifier.

### W&B artefacts

When enabled, log lightweight scientific artefacts:

* resolved configuration;
* run specification;
* test metrics;
* classification reports;
* confusion-matrix CSVs;
* condition manifest;
* result summary.

Only upload `best_model.pt` when:

```yaml
wandb:
  log_model: true
```

Do not upload every epoch checkpoint.

Include metadata with a model artefact:

```text
architecture
tasks
class mappings
image size
training condition
seed
best epoch
selection metric
Git commit
```

### Offline and disabled behaviour

The trainer must work correctly when W&B is:

```text
disabled
offline
unavailable
```

Requirements:

* no W&B import failure when logging is disabled;
* local CSV, JSON, checkpoint, and report output remains identical;
* network access is not required for tests;
* W&B failures should not corrupt completed scientific outputs;
* tests should use a mock or disabled W&B client.

### Avoid duplicated logging

Create one canonical W&B adapter, for example:

```text
src/worm_species/logging/wandb_logger.py
```

Training, standard testing, cue-suppression testing, and condition-matrix evaluation should all use this adapter.

Do not retain separate W&B implementations in legacy training scripts.

The adapter should expose functions similar to:

```python
log_epoch_metrics(...)
log_test_condition(...)
log_confusion_matrix(...)
log_classification_report(...)
log_robustness_table(...)
log_artifacts(...)
finalise_run(...)
```

The exact API may differ, but it should provide one consistent implementation.

### Tests

Add tests that verify:

* W&B disabled mode requires no network or installed login;
* canonical metric keys are stable;
* resolved configuration is logged;
* training and test conditions are not confused;
* each task and condition receives a unique confusion-matrix key;
* missing labels are excluded from confusion matrices;
* classification-report tables contain the expected columns;
* saturation strength remains numeric;
* robustness ratios are calculated correctly;
* artefact logging respects `wandb.log_model`;
* legacy W&B entry points use the canonical logger;
* local outputs are identical with W&B enabled, mocked, or disabled;
* post-training evaluation does not create duplicate model-training runs;
* failed logging does not remove or invalidate local results.

Please preserve existing W&B fields where they are already part of the behavioural contract, but add aliases or migration handling where necessary rather than silently breaking old dashboards.

Update the documentation with:

```bash
# Fully local
python train.py --config config.yaml \
  --override wandb.enabled=false

# Offline logging
python train.py --config config.yaml \
  --override wandb.enabled=true wandb.mode=offline

# Online logging
python train.py --config config.yaml \
  --override wandb.enabled=true wandb.mode=online
```

The final W&B interface should make it immediately clear:

* which model was trained;
* under which image and augmentation settings;
* which condition was used for training;
* which condition was used for testing;
* whether the comparison was matched or stress testing;
* how each task performed;
* what the confusion matrices show;
* how robust performance was relative to the original test condition.
 with w and b also updaye the dasboard