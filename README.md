# Worm species classification

Multi-task earthworm image classification with genus, species, and life-stage
heads. The repository supports ordinary training, hierarchy consistency,
matched colour/cue training, fixed-RGB cue stress evaluation, local sweeps, and
two cluster-specific SLURM workflows.

## Environment

```bash
conda env create -f configs/environment.yaml
conda activate wormspecies
```

PyTorch, torchvision, and optional W&B support must be present in the runtime
environment. `config.yaml` contains the current data, split, training, cue,
W&B, cache, and sweep settings.

## Canonical training

Use the single canonical trainer for new work:

```bash
python train.py --config config.yaml

# Resolve and validate without loading data or training.
python train.py --config config.yaml --dry-run
python train.py --config config.yaml --print-resolved-config

# Run one externally assigned condition. This can never expand a sweep.
python train.py --config config.yaml --profile cue_suppression \
  --single-run --override \
  model.name=convnext_base \
  sweep.enabled=false \
  matched_condition_training.enabled=false \
  input_condition.enabled=true \
  input_condition.condition=gaussian_blur_sigma_2 \
  input_condition.feature=texture \
  input_condition.transform=gaussian_blur \
  input_condition.sigma=2.0 \
  test_cue_suppression.enabled=false
```

The package form is also supported when `src/` is importable:

```bash
PYTHONPATH=src python -m worm_species.training --config config.yaml --dry-run
```

Training profiles resolve historical defaults into an explicit configuration.
The available profiles are `masked`, `masked_hloss`,
`masked_hloss_wandb`, `colour_ablation`, and `cue_suppression`. A submitted run
specification must use `--single-run`; one specification then means one trainer
invocation and one model fit.

The historical entry points remain thin compatibility wrappers:

| Historical path | Canonical profile |
| --- | --- |
| `train_multitask_masked.py` | `masked` |
| `train_multitask_masked_hloss.py` | `masked_hloss` |
| `train_multitask_masked_hloss_wandb.py` | `masked_hloss_wandb` |
| `train_multitask_colour_ablation.py` | `colour_ablation` |
| `train_multitask_cue_suppression.py` | `cue_suppression` |

Their paths, flags, help text, exit behaviour, output layout, checkpoint schema,
metrics, and W&B fields are retained for existing local and cluster workflows.

All accept the existing dotted overrides and optional internal sweep syntax:

```bash
python train.py --profile masked_hloss \
  --config config.yaml \
  --override training.epochs=5 model.pretrained=false

python train.py --profile masked_hloss \
  --config config.yaml \
  --sweep model.name=resnet18,vit_b_16 training.lr=0.0005,0.0001
```

Do not use `--sweep` for generated run specifications. Configuration validation
rejects external and internal expansion of the same condition.

## Configuration validation

Configuration remains dictionary-based and accepts the existing dotted
overrides. Validate or inspect it without checking cluster-only data paths:

```bash
PYTHONPATH=src python -m worm_species.config.validate --config config.yaml
PYTHONPATH=src python -m worm_species.config.inspect --config config.yaml
```

Add `--check-paths` only on a machine where the configured data and split paths
should exist. Experiment profiles live under `configs/experiments/`; machine
resources live separately under `configs/clusters/`.

## Matched conditions versus fixed-RGB stress testing

These are intentionally separate scientific workflows:

- matched-condition training applies one deterministic condition to train,
  validation, and test and trains a separate model;
- the fixed-RGB stress battery keeps the selected original model checkpoint
  fixed and evaluates deterministic transformed versions of its test split.

`generate_dual_cue_run_specs.py` deduplicates RGB/saturation-100% and
grayscale/saturation-0% endpoints. Generated specs disable internal matched
expansion, and SLURM launchers pass `sweep.enabled=false`, so each array task
trains exactly one intended configuration.

```bash
python generate_dual_cue_run_specs.py \
  config.yaml /tmp/worm_run_specs /tmp/worm_sweep_plan.tsv

python collect_dual_cue_results.py RESULTS_ROOT
```

There is currently no independent root CLI for applying an arbitrary existing
checkpoint to a new dataset. Existing checkpoints are consumed by the analysis
notebooks, while the cue trainer reloads its selected `best_model.pt` before
the configured fixed-RGB stress battery. This limitation is documented rather
than hidden behind a new unvalidated interface.

## Canonical SLURM workflow

The preferred interface is deliberately small:

```bash
make validate
make inspect
make dry-run
make train
make submit
make status RESULTS_ROOT=outputs_slurm/EXPERIMENT
make collect RESULTS_ROOT=outputs_slurm/EXPERIMENT
make dashboard RESULTS_ROOT=outputs_slurm
make test
```

Variables become explicit CLI overrides; configuration files are never edited in
place:

```bash
make dry-run \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/genome.yaml \
  ARTIFACTS_DIR=/tmp/worm-dual-cue-plan

make submit \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/genome.yaml \
  MAX_ACTIVE=4
```

A dry run writes resolved configuration, run specifications, a submission plan,
rendered scripts, checksums, and the dependency graph, but never calls `sbatch`.
The equivalent direct command is:

```bash
PYTHONPATH=src python -m worm_species.slurm launch --dry-run \
  --config configs/experiments/dual_cue.yaml \
  --cluster-config configs/clusters/genome.yaml \
  --artifacts-dir /tmp/worm-dual-cue-plan
```

`status` combines filesystem evidence with `squeue`/`sacct` when available and
falls back to filesystem-only reporting elsewhere. `collect` currently delegates
exactly to the schema-stable dual-cue collector; standard and colour-ablation
collection continue to use their historical collectors until exact adapters are
validated.

Historical SLURM filenames remain available. They are retained as compatibility
entry points where environment-variable, cache-building, or profiling semantics
are not yet fully represented by the canonical planner. Review account,
partition, Conda, data, scratch, and W&B settings before a real submission.
Their byte-preserved implementations now live under `legacy/slurm/`; the root
and `scripts/slurm/` paths remain compatible symlinks. The historical
`config_old.yaml` and ordinary sweep generator are archived similarly under
`legacy/configs/` and `legacy/python/`. See `legacy/README.md` for the complete
mapping and for the active files intentionally kept outside the archive.

## Results dashboard

The dashboard discovers old, new, partial, malformed, and nested runs without
modifying `outputs_slurm/` or loading checkpoints:

```bash
streamlit run dashboard/app.py -- --results-root outputs_slurm
# or
make dashboard RESULTS_ROOT=outputs_slurm
```

Its SQLite index is stored outside the scientific result tree and invalidated by
path, size, and modification time. Missing or malformed artifacts become visible
warnings rather than fatal errors.

## Notebooks, figures, and tables

Notebooks are grouped under
`notebooks/{analysis,diagnostics,interpretability,data}/`. Migrated notebooks
resolve the repository root from their current location and use canonical
`src.worm_species` imports where behavior is equivalent. Existing cell outputs,
run identifiers, paths to scientific inputs, filenames, and rendering parameters
were preserved.

Future notebook-generated figures and tables are written to workflow-specific
subdirectories beneath `figures/` and `tables/`. Existing artifacts in historical
run directories were not moved. One tracked zero-byte notebook remains untouched
because it is not a valid notebook document.

## Tests and refactor documentation

```bash
make test
make test-unit
make test-contracts
make test-integration

# Equivalent direct command
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  python -m unittest discover -s tests -p 'test_*.py'
```

The bounded inventory, pre-refactor report, migration map, notebook status,
and executed contract results are under `docs/refactor/`. Notebooks are
grouped under `notebooks/{analysis,diagnostics,interpretability,data}/`, and
operational shell scripts are grouped under `scripts/` while their historical
root paths remain available. Generated
outputs, datasets, checkpoints, caches, W&B artifacts, and SLURM result trees
are outside the source audit and are not modified by the refactor.
