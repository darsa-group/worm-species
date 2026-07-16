# Worm species classification

Multitask earthworm image classification for genus, species, and life stage.
The active repository has one canonical trainer, configuration-driven
experiment planning, one SLURM interface, and a read-only results dashboard.

## Setup

```bash
conda env create -f configs/environment.yaml
conda activate wormspecies
```

Run commands from the repository root. The package uses a `src/` layout, so
direct module commands below set `PYTHONPATH=src`; the Make targets do this for
you.

## Quick start

```bash
# Validate the safe one-run configuration.
PYTHONPATH=src python -m worm_species.config.validate \
  --config config.yaml --workflow training

# Inspect the resolved values without loading data or training.
PYTHONPATH=src python -m worm_species.config.inspect \
  --config config.yaml --workflow training --format yaml

# Show the canonical trainer selection and output path.
PYTHONPATH=src python -m worm_species.training \
  --config config.yaml --dry-run --single-run

# Train exactly one model.
make train
```

`config.yaml` is intentionally compact and safe. It inherits detailed defaults
from `configs/defaults/base.yaml`. Scientific variants are child files beneath
`configs/experiments/`; cluster resources and machine paths live independently
beneath `configs/clusters/`.

See [configs/README.md](configs/README.md) for the file map and
[config.md](config.md) for every important key,
constraint, transform, and worked example.
For a ready-to-adapt Genome sweep covering ConvNeXt, ViT, learning rates,
task-loss weights, hierarchy loss, colour retention, and other cues, see
[Genome experiment guide](docs/configuration/genome_experiments.md). The
corresponding matched-training file is
[genome_cues_matched.yaml](dev/genome_cues_matched.yaml).

## One trainer, explicit switches

The preferred command is:

```bash
PYTHONPATH=src python -m worm_species.training \
  --config config.yaml --single-run
```

`train.py` is a convenience entry point for the same implementation:

```bash
python train.py --config config.yaml --single-run
```

Behavior is selected by ordinary configuration values, not script profiles:

```yaml
training:
  use_masked_labels: true

multi_task:
  hierarchy_loss:
    enabled: true

wandb:
  enabled: false

input_condition:
  enabled: false

evaluation:
  test_conditions:
    enabled: false
  condition_matrix:
    enabled: false

sweep:
  enabled: false
```

Useful checks:

```bash
python train.py --config config.yaml --print-resolved-config
python train.py --config config.yaml --dry-run --single-run
```

The dry run reports the model, tasks, loss weights, hierarchy and W&B switches,
assigned training condition, post-training evaluations, output path, and
internal training count. A resolved submitted task must always report one
internal training run.

For one assigned condition:

```bash
PYTHONPATH=src python -m worm_species.training \
  --config config.yaml --single-run --override \
    model.name=convnext_base \
    input_condition.enabled=true \
    input_condition.name=gaussian_sigma_2 \
    input_condition.feature=texture \
    input_condition.transform=gaussian_blur \
    input_condition.strength=2.0 \
    input_condition.parameters.sigma=2.0 \
    sweep.enabled=false
```

Training-time conditions, original-RGB transformed-test stress evaluation, and
post-training condition matrices are separate paths. The trainer never expands
an externally assigned sweep.

## Experiments

The supplied plans are validation contracts:

| Configuration | Training processes | Evaluation behavior |
| --- | ---: | --- |
| `config.yaml` | 1 | Safe original-RGB baseline |
| `configs/experiments/standard.yaml` | 2 | Two-model standard sweep |
| `configs/experiments/hierarchy.yaml` | 2 | Hierarchy consistency enabled |
| `configs/experiments/dual_cue.yaml` | 224 | 112 conditions × two models; RGB stress only for original-trained checkpoints |
| `configs/experiments/colour_ablation.yaml` | 202 | Two models × 101 endpoint-inclusive colour values |
| `configs/experiments/patch_shuffle_matrix.yaml` | 12 | Four pretrained models × three train conditions; 36 train/test evaluation cells |
| `configs/experiments/persistent_hierarchy.yaml` | 2 | Persistent-cache hierarchy workflow |

The patch plan trains on original RGB, 2×2 patch shuffle, and 4×4 patch
shuffle, then evaluates every trained checkpoint on all three test conditions.
That is 12 fits and 36 evaluation cells, not 36 fits.

## Make and SLURM

```bash
make help
make validate EXPERIMENT=standard
make inspect EXPERIMENT=patch_shuffle_matrix
make dry-run EXPERIMENT=dual_cue \
  CLUSTER=configs/clusters/genome.yaml
make submit EXPERIMENT=dual_cue \
  CLUSTER=configs/clusters/genome.yaml
make status RESULTS_ROOT=outputs_slurm/EXPERIMENT
make collect RESULTS_ROOT=outputs_slurm/EXPERIMENT
```

The Makefile is only orchestration. Planning, validation, rendering,
collection, and submission live in `worm_species.slurm`.

Direct dry run:

```bash
PYTHONPATH=src python -m worm_species.slurm launch --dry-run \
  --config configs/experiments/dual_cue.yaml \
  --cluster-config configs/clusters/genome.yaml \
  --artifacts-dir slurm/generated/dual-cue-check
```

Direct submission replaces `--dry-run` with `--submit`. A dry run never calls
`sbatch`.

GHPC node-local plans require an explicit node list and unique scratch root;
missing or ambiguous values fail before rendering:

```bash
PYTHONPATH=src python -m worm_species.slurm launch --dry-run \
  --config configs/experiments/ghpc_dual_cue.yaml \
  --cluster-config configs/clusters/ghpc.yaml \
  --override 'slurm.scratch.nodes=[gpu001,gpu002]' \
  --artifacts-dir slurm/generated/ghpc-dual-check
```

Use `configs/experiments/ghpc_colour_ablation.yaml` for the GHPC colour plan.
Replace the example node names with the actual assigned nodes.

Every generated run specification maps to one array task, one canonical trainer
invocation, and one model fit. Duplicate IDs, result collisions, array-count
mismatches, and nested expansion fail validation before submission.

## Persistent cache

Cache maintenance is explicit and does not occur during inspection:

```bash
PYTHONPATH=src python -m worm_species.cache build \
  --config config.yaml \
  --data-root /path/to/petridish-worm-images \
  --metadata-csv /path/to/global_metadata.csv \
  --cache-dir /path/to/cache/images

PYTHONPATH=src python -m worm_species.cache verify \
  --cache-dir /path/to/cache/images
```

## Dashboard and derived results

The dashboard combines SLURM multitask outputs and historical single-task
outputs without modifying either tree or loading checkpoint bodies.

Prepare per-task metrics and combined confusion-matrix images first:

```bash
make dashboard-prepare \
  SLURM_RESULTS_ROOT=outputs_slurm \
  SINGLE_TASK_RESULTS_ROOT=single_task/outputs
```

Then launch:

```bash
make dashboard \
  SLURM_RESULTS_ROOT=outputs_slurm \
  SINGLE_TASK_RESULTS_ROOT=single_task/outputs
```

The browser exposes source type, run status, architecture, tasks, conditions,
configured epochs, learning rate, weight decay, image size, pretrained/frozen
state, class weighting, per-task loss weights, hierarchy settings, best epoch
and validation score, test metrics, curves, reports, confusion matrices, W&B
fields, logs, and result paths. A separate condition-matrix view shows model,
train/test condition, relation, task filters, the 12/8/16 patch relation counts,
and train-by-test macro-F1 heatmaps. Missing or malformed files become warnings.

Dashboard indexes and derived figures live under
`.cache/worm-species-dashboard/`, outside scientific result directories.

## Notebooks and outputs

Notebooks are active scientific code under:

```text
notebooks/analysis/
notebooks/data/
notebooks/diagnostics/
notebooks/interpretability/
```

Modernized notebooks locate the repository root, add `PROJECT_ROOT/src`, and
import `worm_species.*`. New presentation artifacts are routed to workflow
subdirectories beneath `figures/` and `tables/`; live run directories remain
read-only scientific inputs. Notebook migration preserves existing cell
outputs, metadata, execution counts, run identifiers, filenames, and DPI.

`split_csv/` contains externally linked predefined splits. Do not rewrite or
move those links. Likewise, do not modify datasets, checkpoints, W&B artifacts,
or live `outputs_slurm/` runs during maintenance.

## Tests

```bash
make test-unit
make test-contracts
make test-integration
make test
```

Focused direct examples:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  python -m unittest tests.test_training_cli_contracts

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  python -m unittest tests.test_slurm_planning tests.test_ghpc_dual_colour

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  python -m unittest tests.test_dashboard_condition_matrix tests.test_notebook_migration
```

The standard suite is CPU-only and uses temporary synthetic data. It does not
require the 75 GB data tree, a GPU, W&B network access, or a live SLURM cluster.

For the compact repository map, see [info.md](info.md).
