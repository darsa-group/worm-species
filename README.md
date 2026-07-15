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

## Local training

The historical entry points remain supported:

```bash
python train_multitask_masked.py --config config.yaml
python train_multitask_masked_hloss.py --config config.yaml
python train_multitask_masked_hloss_wandb.py --config config.yaml
python train_multitask_colour_ablation.py --config config.yaml
python train_multitask_cue_suppression.py --config config.yaml
```

Their canonical implementations live under `scripts/training/`; the root
paths are compatibility wrappers retained for existing local and cluster
commands.

All accept the existing dotted overrides and optional internal sweep syntax:

```bash
python train_multitask_masked_hloss.py \
  --config config.yaml \
  --override training.epochs=5 model.pretrained=false

python train_multitask_masked_hloss.py \
  --config config.yaml \
  --sweep model.name=resnet18,vit_b_16 training.lr=0.0005,0.0001
```

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

## Persistent cache and SLURM

Genome cluster cache workflow:

```bash
sbatch 01_build_persistent_cache_resolved.sh
bash 02_submit_sweep_cache_to_tmp_resolved.sh
bash run_persistent_cache_sweep_wandb.sh
bash submit_dual_cue_experiment_genome.sh
```

GHPC node-local workflow:

```bash
GPU_NODES="nodeA nodeB" bash submit_worm_node_local_scratch_sweep.sh
GPU_NODES="nodeA nodeB" bash submit_worm_node_local_scratch_sweep_hloss.sh
GPU_NODES="nodeA nodeB" bash submit_colour_ablation_sweep.sh
GPU_NODES="nodeA nodeB" bash submit_dual_cue_experiment.sh
```

Review launcher defaults for accounts, partitions, Conda paths, data roots,
scratch locations, and W&B environment variables before submission.

## Tests and refactor documentation

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The bounded inventory, pre-refactor report, migration map, notebook status,
and executed contract results are under `docs/refactor/`. Notebooks are
grouped under `notebooks/{analysis,diagnostics,interpretability,data}/`, and
operational shell scripts are grouped under `scripts/` while their historical
root paths remain available. Generated
outputs, datasets, checkpoints, caches, W&B artifacts, and SLURM result trees
are outside the source audit and are not modified by the refactor.
